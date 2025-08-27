// master_hackrf.c — Полный C-бэкенд для Panorama (HackRF sweep + FFT + EMA + частотное сглаживание)

#include "hackrf_master.h"

#include <libhackrf/hackrf.h>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include <errno.h>

// ====== Защита от конфликта с заголовком ======
#ifndef MIN_FFT_SIZE
#define MIN_FFT_SIZE 16
#endif
#ifndef DEFAULT_FFT_SIZE
#define DEFAULT_FFT_SIZE 32
#endif
#ifndef MAX_FFT_SIZE
#define MAX_FFT_SIZE 256
#endif

// ====== Локальные константы ======
#define DEFAULT_SAMPLE_RATE_HZ             20000000u   // 20 MHz
#define DEFAULT_BASEBAND_FILTER_BANDWIDTH  15000000u   // 15 MHz
#define OFFSET_HZ                          7500000u    // 7.5 MHz LO offset
#define TUNE_STEP_HZ                       20000000u   // 20 MHz шаг тюнера (равен SR для sweep)
#define FREQ_ONE_MHZ                       1000000u
#define BLOCKS_PER_TRANSFER                16
#define BYTES_PER_BLOCK                    16384
#define MAX_SPECTRUM_POINTS                100000
#define INTERLEAVED                        1



#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ================== Структуры ==================
typedef struct {
    // FFT
    fftwf_complex* in;
    fftwf_complex* out;
    fftwf_plan plan;
    float* window;
    float window_loss_db;
    float enbw_corr_db;

    // Текущее окно
    float*  pwr_dbm;    // [fft_size]
    double* freqs_hz;   // [fft_size]

    // Предвычисления
    double* bin_offsets_quarter_hz; // [fft_size/4]

    // Параметры FFT
    int    fft_size;
    double sample_rate;
    double bin_width;

    // EMA (временное сглаживание при записи в глобальную сетку)
    float ema_alpha;

    // Частотное сглаживание (скользящее среднее)
    int    freq_smooth_enabled;   // 0/1
    int    freq_smooth_window;    // нечетное >=1; 1 = выключено
    float* pwr_dbm_smoothed;      // [fft_size]

    // Детектор (на будущее)
    float detection_threshold_db;
    int   min_width_bins;
    int   min_sweeps;
    float timeout_sec;
} ProcessingContext;

typedef struct {
    hackrf_device* dev;
    ProcessingContext* proc;

    // «Глобальный» спектр (равномерная сетка)
    double freq_start_hz;
    double freq_stop_hz;
    double freq_step_hz;

    double* spectrum_freqs;   // [N]
    float*  spectrum_powers;  // [N]
    int*    spectrum_counts;  // [N]
    size_t  spectrum_points;
    pthread_mutex_t spectrum_mutex;

    // Статус
    volatile int running;
    volatile int stop_requested;
    struct timeval start_time;
    
    // Поток свипа
    volatile int is_running_thread;
    pthread_t sweep_thread;

    // Калибровка
    float calibration_offset_db;

    // Режим сегментов и FFT
    int segment_mode;       // 2 или 4
    int fft_size_override;  // 0 = авто

    // Колбэки (оставляем только multi_cb на будущее)
    hq_multi_segment_cb multi_cb;
    void* user_data;
} MasterContext;

// ================== Прототипы (NEW) ==================
static int setup_global_spectrum_grid(MasterContext* m, double f_start_hz, double f_stop_hz, double freq_step_hz);

// ================== Глобальные ==================
static MasterContext* g_master = NULL;
static char g_last_error[256] = {0};

// Буферы сегментов для multi_cb
static hq_segment_data_t g_segments[MAX_SEGMENTS];
static double* g_seg_freqs[4] = {NULL, NULL, NULL, NULL};
static float*  g_seg_powers[4] = {NULL, NULL, NULL, NULL};
static int     g_seg_caps[4]   = {0, 0, 0, 0};

// Калибровка
static struct {
    float freq_mhz[MAX_CALIBRATION_ENTRIES];
    int   lna_db[MAX_CALIBRATION_ENTRIES];
    int   vga_db[MAX_CALIBRATION_ENTRIES];
    int   amp_on[MAX_CALIBRATION_ENTRIES];
    float offset_db[MAX_CALIBRATION_ENTRIES];
    int   count;
    int   enabled;
} g_calibration = {0};

// ================== Утилиты/ошибки ==================
static void set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, args);
    va_end(args);
}
const char* hq_last_error(void) { return g_last_error; }

// ================== Математика окна ==================
static void calculate_window_corrections(const float* window, int N, float* power_loss, float* enbw_corr) {
    double sum_linear = 0.0, sum_squared = 0.0;
    for (int i = 0; i < N; i++) {
        sum_linear  += window[i];
        sum_squared += (double)window[i] * (double)window[i];
    }
    double coherent_gain   = sum_linear / N;
    double processing_gain = sum_squared / N;

    *power_loss = -10.0f * log10f((float)processing_gain);
    double enbw = N * processing_gain / (coherent_gain * coherent_gain);
    *enbw_corr = 10.0f * log10f((float)enbw);
}

static inline float calculate_power_dbm(float re, float im, float correction_db) {
    float mag2 = re*re + im*im;
    if (mag2 < 1e-20f) mag2 = 1e-20f;
    float dbfs = 10.0f * log10f(mag2);
    return dbfs + correction_db - 10.0f; // -10 dB базовая поправка HackRF
}

// ================== Калибровка ==================
static float get_calibration_offset_db(double freq_mhz) {
    if (!g_calibration.enabled || g_calibration.count == 0) return 0.0f;
    float best = 0.0f;
    float min_d = 1e9f;
    for (int i = 0; i < g_calibration.count; i++) {
        float d = fabsf((float)freq_mhz - g_calibration.freq_mhz[i]);
        if (d < min_d) { min_d = d; best = g_calibration.offset_db[i]; }
    }
    return best;
}

// ================== Частотное сглаживание ==================
static inline void smooth_boxcar_float(const float* in, float* out, int N, int W) {
    if (W <= 1 || N <= 0) {
        if (in != out) memcpy(out, in, sizeof(float) * (size_t)N);
        return;
    }
    if ((W & 1) == 0) W += 1; // нечетное окно
    int r = W / 2;

    // реализация через клэмп краёв
    double acc = 0.0;
    int L = 0, R = 0;

    // Инициализация для i=0
    L = 0; R = (r < N-1) ? r : (N-1);
    for (int k = L; k <= R; ++k) acc += in[k];
    out[0] = (float)(acc / (double)(R - L + 1));

    for (int i = 1; i < N; ++i) {
        int newL = i - r; if (newL < 0) newL = 0;
        int newR = i + r; if (newR >= N) newR = N - 1;

        // сдвиг окна
        while (L < newL) { acc -= in[L]; ++L; }
        while (R < newR) { ++R; acc += in[R]; }
        while (L > newL) { --L; acc += in[L]; }
        while (R > newR) { acc -= in[R]; --R; }

        out[i] = (float)(acc / (double)(R - L + 1));
    }
}

// ================== ProcessingContext ==================
static ProcessingContext* create_processing_context(int fft_size, double sample_rate) {
    ProcessingContext* ctx = (ProcessingContext*)calloc(1, sizeof(ProcessingContext));
    if (!ctx) return NULL;

    ctx->fft_size    = fft_size;
    ctx->sample_rate = sample_rate;
    ctx->bin_width   = sample_rate / fft_size;

    ctx->ema_alpha   = 0.25f;

    ctx->freq_smooth_enabled = 0;
    ctx->freq_smooth_window  = 5; // по умолчанию лёгкое сглаживание
    ctx->detection_threshold_db = -80.0f;
    ctx->min_width_bins = 3;
    ctx->min_sweeps = 2;
    ctx->timeout_sec = 5.0f;

    ctx->in   = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    ctx->out  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    ctx->pwr_dbm  = (float*) malloc(sizeof(float)  * fft_size);
    ctx->pwr_dbm_smoothed = (float*) malloc(sizeof(float) * fft_size);
    ctx->freqs_hz = (double*)malloc(sizeof(double) * fft_size);
    ctx->window   = (float*) malloc(sizeof(float)  * fft_size);
    ctx->bin_offsets_quarter_hz = (double*)malloc(sizeof(double) * (fft_size/4));

    if (!ctx->in || !ctx->out || !ctx->pwr_dbm || !ctx->pwr_dbm_smoothed || !ctx->freqs_hz || !ctx->window || !ctx->bin_offsets_quarter_hz) {
        if (ctx->in) fftwf_free(ctx->in);
        if (ctx->out) fftwf_free(ctx->out);
        free(ctx->pwr_dbm);
        free(ctx->pwr_dbm_smoothed);
        free(ctx->freqs_hz);
        free(ctx->window);
        free(ctx->bin_offsets_quarter_hz);
        free(ctx);
        return NULL;
    }

    for (int i = 0; i < fft_size; i++)
        ctx->window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (fft_size - 1)));

    calculate_window_corrections(ctx->window, fft_size, &ctx->window_loss_db, &ctx->enbw_corr_db);
    ctx->plan = fftwf_plan_dft_1d(fft_size, ctx->in, ctx->out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int i = 0; i < fft_size/4; i++)
        ctx->bin_offsets_quarter_hz[i] = (i + 0.5) * ctx->bin_width;

    return ctx;
}

static void destroy_processing_context(ProcessingContext* ctx) {
    if (!ctx) return;
    if (ctx->plan) fftwf_destroy_plan(ctx->plan);
    if (ctx->in)   fftwf_free(ctx->in);
    if (ctx->out)  fftwf_free(ctx->out);
    free(ctx->pwr_dbm);
    free(ctx->pwr_dbm_smoothed);
    free(ctx->freqs_hz);
    free(ctx->window);
    free(ctx->bin_offsets_quarter_hz);
    free(ctx);
}

// ================== Глобальный спектр ==================
static inline void spectrum_update_ema(MasterContext* m, double freq_hz, float val_dbm, float alpha) {
    if (!m || !m->spectrum_freqs || !m->spectrum_powers || !m->spectrum_counts) return;  // защита от неинициализированных массивов
    if (freq_hz < m->freq_start_hz || freq_hz > m->freq_stop_hz) return;
    if (m->freq_step_hz <= 0) return;

    int idx = (int)floor((freq_hz - m->freq_start_hz) / m->freq_step_hz + 0.5);
    if (idx < 0 || (size_t)idx >= m->spectrum_points) return;

    if (m->spectrum_counts[idx] == 0) {
        m->spectrum_powers[idx] = val_dbm;
    } else {
        m->spectrum_powers[idx] = (1.0f - alpha) * m->spectrum_powers[idx] + alpha * val_dbm;
    }
    m->spectrum_counts[idx]++;
}

// ================== Обработка sweep-блока ==================
static void process_sweep_block(MasterContext* master, uint8_t* buffer, int valid_length) {
    if (!master || !master->proc || !buffer || valid_length < 16) return;
    if (master->stop_requested) return;  // защита от остановки
    ProcessingContext* proc = master->proc;

    // сигнатура sweep-пакета
    if (buffer[0] != 0x7F || buffer[1] != 0x7F) return;

    // центральная частота (LE)
    uint64_t center_hz = 0;
    for (int i = 0; i < 8; i++) center_hz |= ((uint64_t)buffer[2 + i]) << (8 * i);

    // IQ данные
    int8_t* iq = (int8_t*)(buffer + 16);
    int samples_available = (valid_length - 16) / 2; // попарно I,Q
    if (samples_available < proc->fft_size) return;

    // окно + нормализация
    const float scale = 1.0f / 128.0f;
    for (int i = 0; i < proc->fft_size; i++) {
        float I = iq[2*i]     * scale;
        float Q = iq[2*i + 1] * scale;
        proc->in[i][0] = I * proc->window[i];
        proc->in[i][1] = Q * proc->window[i];
    }

    // FFT
    fftwf_execute(proc->plan);

    // Коррекции мощности
    const float fft_norm   = 1.0f / proc->fft_size;
    const double center_mhz = (double)center_hz / 1e6;
    const float cal_off    = get_calibration_offset_db(center_mhz);
    const float total_corr = proc->window_loss_db + proc->enbw_corr_db + cal_off;

    // Заполняем мощности и частоты бинов окна
    const double Fs = proc->sample_rate;
    const double f0 = (double)center_hz - 0.5 * Fs;
    for (int i = 0; i < proc->fft_size; i++) {
        float re = proc->out[i][0] * fft_norm;
        float im = proc->out[i][1] * fft_norm;
        proc->pwr_dbm[i] = calculate_power_dbm(re, im, total_corr);
        proc->freqs_hz[i] = f0 + (i + 0.5) * proc->bin_width; // центр бина
    }

    // Частотное сглаживание (перед EMA и колбэками)
    const float* pwr_src = proc->pwr_dbm;
    if (proc->freq_smooth_enabled && proc->freq_smooth_window > 1) {
        int W = proc->freq_smooth_window;
        if ((W & 1) == 0) W += 1; // нечетное окно
        smooth_boxcar_float(proc->pwr_dbm, proc->pwr_dbm_smoothed, proc->fft_size, W);
        pwr_src = proc->pwr_dbm_smoothed;
    }

    // Обновляем «глобальную» сетку (EMA)
    pthread_mutex_lock(&master->spectrum_mutex);
    for (int i = 0; i < proc->fft_size; i++) {
        spectrum_update_ema(master, proc->freqs_hz[i], pwr_src[i], proc->ema_alpha);
    }

    // Подготовка данных сегментов
    if (master->multi_cb) {
        int seg_count = (master->segment_mode == HQ_SEGMENT_MODE_2) ? 2 : 4;
        int N = proc->fft_size;

        if (seg_count == 4) {
            int Q = N / 4;
            int parts[4][2] = {
                {0, Q}, {Q, Q}, {2*Q, Q}, {3*Q, N - 3*Q}
            };
            for (int s = 0; s < 4; s++) {
                int start = parts[s][0];
                int len   = parts[s][1];
                if (g_seg_caps[s] < len) {
                    double* nf = (double*)realloc(g_seg_freqs[s],  sizeof(double)*len);
                    float*  np = (float*) realloc(g_seg_powers[s], sizeof(float) *len);
                    if (!nf || !np) { if (nf) g_seg_freqs[s]=nf; if (np) g_seg_powers[s]=np; len=0; }
                    else { g_seg_freqs[s]=nf; g_seg_powers[s]=np; g_seg_caps[s]=len; }
                }
                g_segments[s].segment_id = s;
                g_segments[s].count      = len;
                g_segments[s].freqs_hz   = g_seg_freqs[s];
                g_segments[s].data_dbm   = g_seg_powers[s];
                for (int i = 0; i < len; i++) {
                    g_seg_freqs[s][i]  = proc->freqs_hz[start + i];
                    g_seg_powers[s][i] = pwr_src[start + i];
                }
            }
        } else { // seg_count == 2
            int half = N / 2;
            int lens[2] = { half, N - half };
            int starts[2] = { 0, half };
            for (int s = 0; s < 2; s++) {
                int len   = lens[s];
                int start = starts[s];
                if (g_seg_caps[s] < len) {
                    double* nf = (double*)realloc(g_seg_freqs[s],  sizeof(double)*len);
                    float*  np = (float*) realloc(g_seg_powers[s], sizeof(float) *len);
                    if (!nf || !np) { if (nf) g_seg_freqs[s]=nf; if (np) g_seg_powers[s]=np; len=0; }
                    else { g_seg_freqs[s]=nf; g_seg_powers[s]=np; g_seg_caps[s]=len; }
                }
                g_segments[s].segment_id = s;
                g_segments[s].count      = len;
                g_segments[s].freqs_hz   = g_seg_freqs[s];
                g_segments[s].data_dbm   = g_seg_powers[s];
                for (int i = 0; i < len; i++) {
                    g_seg_freqs[s][i]  = proc->freqs_hz[start + i];
                    g_seg_powers[s][i] = pwr_src[start + i];
                }
            }
        }

        master->multi_cb(g_segments, seg_count, proc->bin_width, center_hz, master->user_data);
    }

    pthread_mutex_unlock(&master->spectrum_mutex);
}

// ================== RX Callback ==================
static int rx_callback(hackrf_transfer* transfer) {
    if (!g_master || !g_master->running || g_master->stop_requested) return -1;

    uint8_t* buffer = transfer->buffer;
    int total = transfer->valid_length; // обычно BLOCKS_PER_TRANSFER * BYTES_PER_BLOCK

    // обрабатываем пачкой по BYTES_PER_BLOCK
    int blocks = total / BYTES_PER_BLOCK;
    if (blocks <= 0) blocks = BLOCKS_PER_TRANSFER; // запасной вариант
    for (int i = 0; i < blocks; i++) {
        process_sweep_block(g_master, buffer + i * BYTES_PER_BLOCK, BYTES_PER_BLOCK);
    }
    return 0;
}

// ================== Поток-воркер (sweep) ==================
static void* worker_thread_fn(void* arg) {
    MasterContext* master = (MasterContext*)arg;

    // fprintf(stdout, "[HackRF Master] Worker thread started\n");
    // fprintf(stdout, "[HackRF Master] Range: %.1f–%.1f MHz, bin(step)=%.1f kHz\n",
    //         master->freq_start_hz/1e6, master->freq_stop_hz/1e6, master->freq_step_hz/1e3);

    // Настройка SDR
    int r = hackrf_set_sample_rate_manual(master->dev, DEFAULT_SAMPLE_RATE_HZ, 1);
    if (r != HACKRF_SUCCESS) { set_error("hackrf_set_sample_rate_manual: %s", hackrf_error_name(r)); return NULL; }
    r = hackrf_set_baseband_filter_bandwidth(master->dev, DEFAULT_BASEBAND_FILTER_BANDWIDTH);
    if (r != HACKRF_SUCCESS) { set_error("hackrf_set_baseband_filter_bandwidth: %s", hackrf_error_name(r)); return NULL; }

    // Выбор FFT
    int fft_size = DEFAULT_FFT_SIZE;
    if (master->fft_size_override > 0) {
        fft_size = master->fft_size_override;
    } else if (master->freq_step_hz > 0) {
        int auto_size = (int)(DEFAULT_SAMPLE_RATE_HZ / master->freq_step_hz);
        if (auto_size < MIN_FFT_SIZE) auto_size = MIN_FFT_SIZE;
        if (auto_size > MAX_FFT_SIZE) auto_size = MAX_FFT_SIZE;
        int p2 = 1; while ((p2 << 1) <= auto_size) p2 <<= 1;
        fft_size = p2;
    }
    // fprintf(stdout, "[HackRF Master] Using FFT size: %d (bin=%.1f kHz)\n",
    //         fft_size, (DEFAULT_SAMPLE_RATE_HZ/(double)fft_size)/1e3);

    master->proc = create_processing_context(fft_size, DEFAULT_SAMPLE_RATE_HZ);
    if (!master->proc) { set_error("create_processing_context failed"); return NULL; }

    // Sweep init — диапазоны в МГц кратно 20
    uint32_t fmin_mhz = (uint32_t)(master->freq_start_hz / 1e6);
    uint32_t fmax_mhz = (uint32_t)(master->freq_stop_hz  / 1e6);
    fmin_mhz = (fmin_mhz / 20) * 20;
    fmax_mhz = ((fmax_mhz + 19) / 20) * 20;
    if (fmax_mhz <= fmin_mhz) fmax_mhz = fmin_mhz + 20;

    uint16_t freqs_mhz[2] = { (uint16_t)fmin_mhz, (uint16_t)fmax_mhz };
    // fprintf(stdout, "[HackRF Master] Sweep init: %u–%u MHz, step=%u, offset=%u, interleaved=%d\n",
    //         freqs_mhz[0], freqs_mhz[1], TUNE_STEP_HZ, OFFSET_HZ, INTERLEAVED);

    r = hackrf_init_sweep(master->dev,
                          freqs_mhz,       // пары [start, stop] в МГц
                          1,               // число пар
                          BYTES_PER_BLOCK, // размер блока в callback
                          TUNE_STEP_HZ,    // частотный шаг тюнера
                          OFFSET_HZ,       // LO offset
                          INTERLEAVED);    // interleaved формат
    if (r != HACKRF_SUCCESS) {
        set_error("hackrf_init_sweep failed: %s", hackrf_error_name(r));
        destroy_processing_context(master->proc); master->proc = NULL;
        return NULL;
    }

    r = hackrf_start_rx_sweep(master->dev, rx_callback, NULL);
    if (r != HACKRF_SUCCESS) {
        set_error("hackrf_start_rx_sweep failed: %s", hackrf_error_name(r));
        destroy_processing_context(master->proc); master->proc = NULL;
        return NULL;
    }

    // fprintf(stdout, "[HackRF Master] Sweep started\n");

    while (master->running && !master->stop_requested) {
        usleep(100000);
    }

    // fprintf(stdout, "[HackRF Master] Stopping sweep\n");
    hackrf_stop_rx(master->dev);

    destroy_processing_context(master->proc);
    master->proc = NULL;

    // fprintf(stdout, "[HackRF Master] Worker thread finished\n");
    return NULL;
}

// ================== API ==================
int hq_open(const char* serial_suffix) {
    int r = hackrf_init();
    if (r != HACKRF_SUCCESS) { 
        set_error("hackrf_init: %s", hackrf_error_name(r)); 
        return r; 
    }

    // Закрываем предыдущий контекст если есть
    if (g_master) {
        hq_close();
    }

    // Создаем новый контекст
    g_master = (MasterContext*)calloc(1, sizeof(MasterContext));
    if (!g_master) { 
        hackrf_exit(); 
        set_error("calloc MasterContext failed"); 
        return -1; 
    }

    // Инициализируем мьютекс
    if (pthread_mutex_init(&g_master->spectrum_mutex, NULL) != 0) {
        set_error("pthread_mutex_init failed");
        free(g_master);
        g_master = NULL;
        hackrf_exit();
        return -1;
    }

    // Открываем устройство
    if (!serial_suffix || !serial_suffix[0]) {
        r = hackrf_open(&g_master->dev);
        if (r != HACKRF_SUCCESS) {
            set_error("hackrf_open: %s", hackrf_error_name(r));
        }
    } else {
        // Поиск по суффиксу серийника
        hackrf_device_list_t* list = hackrf_device_list();
        if (!list) { 
            set_error("hackrf_device_list() failed"); 
            pthread_mutex_destroy(&g_master->spectrum_mutex); 
            free(g_master); 
            g_master = NULL; 
            hackrf_exit(); 
            return -1; 
        }
        
        int found = -1;
        size_t suf_len = strlen(serial_suffix);
        for (int i = 0; i < list->devicecount; i++) {
            const char* ser = list->serial_numbers[i];
            if (!ser) continue;
            size_t len = strlen(ser);
            if (len >= suf_len && strcmp(ser + len - suf_len, serial_suffix) == 0) {
                found = i; 
                break;
            }
        }
        
        if (found >= 0) {
            r = hackrf_open_by_serial(list->serial_numbers[found], &g_master->dev);
            if (r != HACKRF_SUCCESS) {
                set_error("hackrf_open_by_serial: %s", hackrf_error_name(r));
            }
        } else {
            r = HACKRF_ERROR_NOT_FOUND;
            set_error("Device with suffix '%s' not found", serial_suffix);
        }
        
        hackrf_device_list_free(list);
    }

    // Проверяем результат открытия
    if (r != HACKRF_SUCCESS) {
        pthread_mutex_destroy(&g_master->spectrum_mutex);
        free(g_master); 
        g_master = NULL; 
        hackrf_exit();
        return r;
    }

    // Устанавливаем дефолты
    g_master->segment_mode = HQ_SEGMENT_MODE_4;
    g_master->fft_size_override = 0;
    g_master->calibration_offset_db = 0.0f;
    g_master->running = 0;
    g_master->stop_requested = 0;
    g_master->is_running_thread = 0;

    return 0;
}

void hq_close(void) {
    if (!g_master) return;
    MasterContext* m = g_master;
    g_master = NULL; // обнуляем сразу, чтобы повторные вызовы были безопасны

    // сначала стоп
    hq_stop();

    // освобождение ресурсов
    if (m->dev) {
        hackrf_close(m->dev);
        m->dev = NULL;
    }
    hackrf_exit();

    // освободить буферы спектра/контексты
    if (m->proc) {
        destroy_processing_context(m->proc);
        m->proc = NULL;
    }
    if (m->spectrum_freqs)  { free(m->spectrum_freqs);  m->spectrum_freqs  = NULL; }
    if (m->spectrum_powers) { free(m->spectrum_powers); m->spectrum_powers = NULL; }
    if (m->spectrum_counts) { free(m->spectrum_counts); m->spectrum_counts = NULL; }

    // маленький «дыхательный» промежуток, чтобы ядро отпустило интерфейс
    usleep(150 * 1000); // 150 мс

    free(m);
}

int hq_configure(double f_start_mhz, double f_stop_mhz, double bin_hz,
                  int lna_db, int vga_db, int amp_on)
{
    if (!g_master) {
        set_error("Master not opened");
        return -1;
    }
    MasterContext* m = g_master;

    // Приведём входы
    if (f_stop_mhz <= f_start_mhz) {
        set_error("Invalid range: stop <= start");
        return -1;
    }
    const double f_start_hz = f_start_mhz * 1e6;
    const double f_stop_hz  = f_stop_mhz  * 1e6;

    // Выбор/уточнение FFT размера относительно bin_hz
    // Если bin_hz == 0, используем дефолтный размер
    double sr = (double)DEFAULT_SAMPLE_RATE_HZ;
    int fft_size = m->fft_size_override > 0 ? m->fft_size_override : DEFAULT_FFT_SIZE;
    if (bin_hz > 0.0) {
        // Подгоняем FFT так, чтобы bin ≈ bin_hz (до ближайшей степени двойки)
        double ideal_n = sr / bin_hz;
        int n = 1;
        while (n < (int)ideal_n && n < MAX_FFT_SIZE) n <<= 1;
        // Выбираем ближайшую степень двойки
        int n_prev = n >> 1;
        if (n_prev >= MIN_FFT_SIZE && fabs(sr/n_prev - bin_hz) < fabs(sr/n - bin_hz)) {
            fft_size = n_prev;
        } else {
            fft_size = n;
        }
        if (fft_size < MIN_FFT_SIZE) fft_size = MIN_FFT_SIZE;
        if (fft_size > MAX_FFT_SIZE) fft_size = MAX_FFT_SIZE;
    }

    // Уничтожаем прежний proc (если есть) и создаём новый с указанным FFT size
    if (m->proc) {
        destroy_processing_context(m->proc);
        m->proc = NULL;
    }
    m->proc = create_processing_context(fft_size, sr);
    if (!m->proc) {
        set_error("Failed to create processing context");
        return -1;
    }

    // ВАЖНО: шаг глобальной сетки = реальный шаг FFT (NEW/CRITICAL)
    m->freq_step_hz = m->proc->bin_width; // <<— ключевая фиксация

    // Сохраняем границы
    m->freq_start_hz = f_start_hz;
    m->freq_stop_hz  = f_stop_hz;

    // Создаём глобальную сетку спектра строго с шагом bin_width (NEW)
    if (setup_global_spectrum_grid(m, f_start_hz, f_stop_hz, m->freq_step_hz) != 0) {
        set_error("Failed to allocate spectrum grid");
        return -1;
    }

    // Настройки усилений/AMP — по твоей текущей логике (не показана в отрывке)
    // TODO: здесь поставить hackrf_set_amp_enable(dev, amp_on), hackrf_set_lna_gain, hackrf_set_vga_gain
    if (m->dev) {
        hackrf_set_lna_gain(m->dev, lna_db);
        hackrf_set_vga_gain(m->dev, vga_db);
        hackrf_set_amp_enable(m->dev, amp_on);
    }

    return 0;
}

// ================== Построение глобальной сетки (NEW) ==================
static int setup_global_spectrum_grid(MasterContext* m, double f_start_hz, double f_stop_hz, double freq_step_hz)
{
    if (!m || freq_step_hz <= 0.0) return -1;

    // Освобождаем предыдущие буферы (если были)
    if (m->spectrum_freqs) { free(m->spectrum_freqs); m->spectrum_freqs = NULL; }
    if (m->spectrum_powers){ free(m->spectrum_powers); m->spectrum_powers = NULL; }
    if (m->spectrum_counts){ free(m->spectrum_counts); m->spectrum_counts = NULL; }
    
    // Уничтожаем предыдущий мьютекс (если был)
    pthread_mutex_destroy(&m->spectrum_mutex);

    // Размер сетки (включая правую границу)
    double span_hz = f_stop_hz - f_start_hz;
    if (span_hz <= 0.0) return -1;

    size_t points = (size_t)floor(span_hz / freq_step_hz + 1.0);
    if (points < 2) points = 2; // на всякий случай
    if (points > MAX_SPECTRUM_POINTS) points = MAX_SPECTRUM_POINTS; // защита

    m->spectrum_points = points;
    m->spectrum_freqs  = (double*)malloc(points * sizeof(double));
    m->spectrum_powers = (float*) malloc(points * sizeof(float));
    m->spectrum_counts = (int*)   malloc(points * sizeof(int));
    if (!m->spectrum_freqs || !m->spectrum_powers || !m->spectrum_counts) {
        if (m->spectrum_freqs)  free(m->spectrum_freqs);
        if (m->spectrum_powers) free(m->spectrum_powers);
        if (m->spectrum_counts) free(m->spectrum_counts);
        m->spectrum_freqs = NULL; m->spectrum_powers = NULL; m->spectrum_counts = NULL;
        m->spectrum_points = 0;
        return -1;
    }

    for (size_t i = 0; i < points; ++i) {
        m->spectrum_freqs[i]  = f_start_hz + (double)i * freq_step_hz;
        m->spectrum_powers[i] = -INFINITY; // или, например, -200.f
        m->spectrum_counts[i] = 0;
    }

    pthread_mutex_init(&m->spectrum_mutex, NULL);
    return 0;
}

int hq_start_multi_segment(hq_multi_segment_cb cb, void* user) {
    if (!g_master || !g_master->dev) { set_error("Device not opened"); return -1; }
    if (g_master->running) { set_error("Already running"); return -1; }

    g_master->multi_cb = cb;
    g_master->user_data = user;
    g_master->running = 1;
    g_master->stop_requested = 0;

    gettimeofday(&g_master->start_time, NULL);

    if (pthread_create(&g_master->sweep_thread, NULL, worker_thread_fn, g_master) != 0) {
        set_error("pthread_create failed: %s", strerror(errno));
        g_master->running = 0;
        return -1;
    }
    g_master->is_running_thread = 1;
    return 0;
}

// Старт без каких-либо колбэков — только наполняем глобальную сетку,
// которую далее читают через hq_get_master_spectrum()
int hq_start_no_cb(void) {
    if (!g_master) { set_error("Not configured"); return -1; }
    if (g_master->running) return 0;
    g_master->multi_cb  = NULL;
    g_master->user_data = NULL;
    g_master->stop_requested = 0;
    // запуск потока свипа ...
    // (оставь ту же логику запуска, что и в hq_start_multi_segment)
    int rc = pthread_create(&g_master->sweep_thread, NULL, worker_thread_fn, g_master);
    if (rc != 0) { set_error("pthread_create failed"); return -1; }
    g_master->is_running_thread = 1;
    g_master->running = 1;
    return 0;
}

int hq_stop(void) {
    if (!g_master) return 0; // уже закрыт/не открыт
    MasterContext* m = g_master;

    // просим тред остановиться
    m->stop_requested = 1;

    // если идёт стриминг — остановим
    if (m->dev) {
        // игнорируем код возврата: если уже не стримит — это нормально
        hackrf_stop_rx(m->dev);
    }

    // дождаться рабочего треда, если он запущен
    if (m->is_running_thread) {
        void* rc = NULL;
        pthread_join(m->sweep_thread, &rc);
        m->is_running_thread = 0;
    }

    m->running = 0;
    return 0;
}

// ================== Доступ к глобальному спектру ==================
int hq_get_master_spectrum(double* freqs_hz, float* powers_dbm, int max_points) {
    if (!g_master || !freqs_hz || !powers_dbm || max_points <= 0) return 0;
    if (!g_master->spectrum_freqs || !g_master->spectrum_powers) return 0;  // защита от неинициализированных массивов
    pthread_mutex_lock(&g_master->spectrum_mutex);
    int n = (int)((size_t)max_points < g_master->spectrum_points ? max_points : g_master->spectrum_points);
    memcpy(freqs_hz,   g_master->spectrum_freqs,  sizeof(double) * n);
    memcpy(powers_dbm, g_master->spectrum_powers, sizeof(float)  * n);
    pthread_mutex_unlock(&g_master->spectrum_mutex);
    return n;
}

// ================== Настройки обработки ==================
void hq_set_ema_alpha(float alpha) {
    if (!g_master || !g_master->proc) return;
    if (alpha < 0.01f) alpha = 0.01f;
    if (alpha > 1.0f)  alpha = 1.0f;
    g_master->proc->ema_alpha = alpha;
}

void hq_set_detector_params(float threshold_offset_db, int min_width_bins,
                            int min_sweeps, float timeout_sec) {
    if (!g_master || !g_master->proc) return;
    g_master->proc->detection_threshold_db = threshold_offset_db;
    g_master->proc->min_width_bins = min_width_bins;
    g_master->proc->min_sweeps = min_sweeps;
    g_master->proc->timeout_sec = timeout_sec;
}

int hq_set_segment_mode(int mode) {
    if (mode != HQ_SEGMENT_MODE_2 && mode != HQ_SEGMENT_MODE_4) {
        set_error("Invalid segment mode. Use 2 or 4");
        return -1;
    }
    if (g_master) g_master->segment_mode = mode;
    return 0;
}
int hq_get_segment_mode(void) {
    return g_master ? g_master->segment_mode : HQ_SEGMENT_MODE_4;
}

int hq_set_fft_size(int size) {
    if (size < MIN_FFT_SIZE || size > MAX_FFT_SIZE) {
        set_error("FFT size out of range [%d..%d]", MIN_FFT_SIZE, MAX_FFT_SIZE);
        return -1;
    }
    // проверим, что степень 2
    int v = size;
    while ((v & 1) == 0) v >>= 1;
    if (v != 1) { set_error("FFT size must be power of 2"); return -1; }

    if (g_master) g_master->fft_size_override = size;
    return 0;
}
int hq_get_fft_size(void) {
    if (!g_master) return DEFAULT_FFT_SIZE;
    if (g_master->fft_size_override > 0) return g_master->fft_size_override;
    return DEFAULT_FFT_SIZE;
}

// ====== Частотное сглаживание (API) ======
void hq_set_freq_smoothing(int enabled, int window_bins) {
    if (!g_master) return;
    if (window_bins < 1) window_bins = 1;
    if ((window_bins & 1) == 0) window_bins += 1; // нечетным

    if (g_master->proc) {
        g_master->proc->freq_smooth_enabled = enabled ? 1 : 0;
        g_master->proc->freq_smooth_window  = window_bins;
    }
}
int  hq_get_freq_smoothing_enabled(void) {
    if (!g_master || !g_master->proc) return 0;
    return g_master->proc->freq_smooth_enabled;
}
int  hq_get_freq_smoothing_window(void) {
    if (!g_master || !g_master->proc) return 1;
    return g_master->proc->freq_smooth_window;
}

// ================== Калибровка ==================
int hq_load_calibration(const char* csv_path) {
    if (!csv_path) { set_error("CSV path is NULL"); return -1; }
    FILE* f = fopen(csv_path, "r");
    if (!f) { set_error("Cannot open calibration file: %s", csv_path); return -1; }

    char line[256];
    g_calibration.count = 0;

    // пропускаем заголовок (если есть)
    (void)fgets(line, sizeof(line), f);

    while (fgets(line, sizeof(line), f) && g_calibration.count < MAX_CALIBRATION_ENTRIES) {
        float freq_mhz, offset_db;
        int   lna_db, vga_db, amp_on;
        if (sscanf(line, "%f,%d,%d,%d,%f", &freq_mhz, &lna_db, &vga_db, &amp_on, &offset_db) == 5) {
            int i = g_calibration.count++;
            g_calibration.freq_mhz[i] = freq_mhz;
            g_calibration.lna_db[i]   = lna_db;
            g_calibration.vga_db[i]   = vga_db;
            g_calibration.amp_on[i]   = amp_on;
            g_calibration.offset_db[i]= offset_db;
        }
    }
    fclose(f);

    if (g_master && g_calibration.count > 0) {
        float sum = 0.0f;
        for (int i = 0; i < g_calibration.count; i++) sum += g_calibration.offset_db[i];
        g_master->calibration_offset_db = sum / g_calibration.count;
    }
    return 0;
}

int hq_enable_calibration(int enable) {
    g_calibration.enabled = enable ? 1 : 0;
    if (g_master) {
        if (!enable) {
            g_master->calibration_offset_db = 0.0f;
        } else if (g_calibration.count > 0) {
            float sum = 0.0f;
            for (int i = 0; i < g_calibration.count; i++) sum += g_calibration.offset_db[i];
            g_master->calibration_offset_db = sum / g_calibration.count;
        }
    }
    return 0;
}

int hq_get_calibration_status(void) {
    return (g_calibration.enabled && g_calibration.count > 0) ? 1 : 0;
}

// ================== Перечисление устройств ==================
int hq_device_count(void) {
    int r = hackrf_init();
    if (r != HACKRF_SUCCESS) return 0;
    hackrf_device_list_t* list = hackrf_device_list();
    if (!list) { hackrf_exit(); return 0; }
    int count = list->devicecount;
    hackrf_device_list_free(list);
    hackrf_exit();
    return count;
}

int hq_get_device_serial(int idx, char* out, int cap) {
    if (!out || cap <= 0) return -1;

    int r = hackrf_init();
    if (r != HACKRF_SUCCESS) return -1;

    hackrf_device_list_t* list = hackrf_device_list();
    if (!list) { hackrf_exit(); return -1; }

    if (idx < 0 || idx >= list->devicecount) {
        hackrf_device_list_free(list); hackrf_exit(); return -1;
    }

    const char* s = list->serial_numbers[idx];
    if (s) {
        strncpy(out, s, cap-1);
        out[cap-1] = '\0';
    } else {
        out[0] = '\0';
    }

    hackrf_device_list_free(list);
    hackrf_exit();
    return 0;
}
