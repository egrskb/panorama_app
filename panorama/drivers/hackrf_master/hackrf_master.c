
// hackrf_master.c - Полноценный C-бэкенд для Panorama с расчетами спектра

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

// ================== Константы ==================
#define DEFAULT_SAMPLE_RATE_HZ    20000000u   // 20 MHz
#define DEFAULT_BASEBAND_FILTER_BANDWIDTH 15000000u   // 15 MHz
#define OFFSET_HZ                 7500000u    // 7.5 MHz LO offset
#define TUNE_STEP_HZ              20000000u   // 20 MHz step
#define FREQ_ONE_MHZ              1000000u
#define BLOCKS_PER_TRANSFER       16
#define BYTES_PER_BLOCK           16384
#define MAX_SPECTRUM_POINTS       100000
#define INTERLEAVED               1

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ================== Структуры данных ==================

typedef struct {
    // FFT
    fftwf_complex* in;
    fftwf_complex* out;
    fftwf_plan plan;
    float* window;
    float window_loss_db;
    float enbw_corr_db;
    
    // Буферы мощности
    float* pwr_dbm;
    double* bin_offsets_hz;
    
    // Параметры
    int fft_size;
    double sample_rate;
    double bin_width;
    
    // EMA фильтрация
    float ema_alpha;
    
    // Детектор пиков
    float detection_threshold_db;
    int min_width_bins;
    int min_sweeps;
    float timeout_sec;
} ProcessingContext;

typedef struct {
    hackrf_device* dev;
    ProcessingContext* proc;
    
    // Конфигурация sweep
    double freq_start_hz;
    double freq_stop_hz;
    double freq_step_hz;
    
    // Глобальный спектр
    double* spectrum_freqs;
    float* spectrum_powers;
    int* spectrum_counts;
    size_t spectrum_points;
    pthread_mutex_t spectrum_mutex;
    
    // Статус
    volatile int running;
    volatile int stop_requested;
    struct timeval start_time;
    
    // Калибровка
    float calibration_offset_db;
    
    // Настройки
    int segment_mode;
    int fft_size_override;  // 0 = auto, иначе фиксированный размер
    
    // Колбэки
    hq_multi_segment_cb multi_cb;
    hq_segment_cb legacy_cb;
    void* user_data;
} MasterContext;

// ================== Глобальные переменные ==================
static MasterContext* g_master = NULL;
static char g_last_error[256] = {0};
static pthread_t g_worker_thread;

// Буферы для сегментов
static hq_segment_data_t g_segments[MAX_SEGMENTS];

// Калибровка
static struct {
    float freq_mhz[MAX_CALIBRATION_ENTRIES];
    int lna_db[MAX_CALIBRATION_ENTRIES];
    int vga_db[MAX_CALIBRATION_ENTRIES];
    int amp_on[MAX_CALIBRATION_ENTRIES];
    float offset_db[MAX_CALIBRATION_ENTRIES];
    int count;
    int enabled;
} g_calibration = {0};

// ================== Утилиты ==================

const char* hq_last_error(void) { 
    return g_last_error; 
}

static void set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, args);
    va_end(args);
}

// Вычисление корректировок окна Ханна
static void calculate_window_corrections(float* window, int N, float* power_loss, float* enbw_corr) {
    double sum_linear = 0.0;
    double sum_squared = 0.0;
    
    for (int i = 0; i < N; i++) {
        sum_linear += window[i];
        sum_squared += window[i] * window[i];
    }
    
    double coherent_gain = sum_linear / N;
    double processing_gain = sum_squared / N;
    
    // Потери мощности из-за окна
    *power_loss = -10.0f * log10f(processing_gain);
    
    // ENBW коррекция
    double enbw = N * processing_gain / (coherent_gain * coherent_gain);
    *enbw_corr = 10.0f * log10f(enbw);
}

// Правильный расчет мощности в dBm
static inline float calculate_power_dbm(float re, float im, float correction) {
    float magnitude_squared = re * re + im * im;
    if (magnitude_squared < 1e-20f) {
        magnitude_squared = 1e-20f;
    }
    
    float power_dbfs = 10.0f * log10f(magnitude_squared);
    float power_dbm = power_dbfs + correction - 10.0f;  // -10 dB базовая калибровка HackRF
    
    return power_dbm;
}

// Получение калибровочной поправки для частоты
static float get_calibration_offset(double freq_mhz) {
    if (!g_calibration.enabled || g_calibration.count == 0) {
        return 0.0f;
    }
    
    // Находим ближайшую точку калибровки
    float min_diff = 1e9;
    float offset = 0.0f;
    
    for (int i = 0; i < g_calibration.count; i++) {
        float diff = fabs(g_calibration.freq_mhz[i] - freq_mhz);
        if (diff < min_diff) {
            min_diff = diff;
            offset = g_calibration.offset_db[i];
        }
    }
    
    return offset;
}

// ================== Обработка FFT ==================

static ProcessingContext* create_processing_context(int fft_size, double sample_rate) {
    ProcessingContext* ctx = (ProcessingContext*)calloc(1, sizeof(ProcessingContext));
    if (!ctx) return NULL;
    
    ctx->fft_size = fft_size;
    ctx->sample_rate = sample_rate;
    ctx->bin_width = sample_rate / fft_size;
    ctx->ema_alpha = 0.25f;
    ctx->detection_threshold_db = -80.0f;
    ctx->min_width_bins = 3;
    ctx->min_sweeps = 2;
    ctx->timeout_sec = 5.0f;
    
    // Выделяем память для FFT
    ctx->in = fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    ctx->out = fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    ctx->pwr_dbm = malloc(sizeof(float) * fft_size);
    ctx->window = malloc(sizeof(float) * fft_size);
    ctx->bin_offsets_hz = malloc(sizeof(double) * (fft_size/4));
    
    if (!ctx->in || !ctx->out || !ctx->pwr_dbm || !ctx->window || !ctx->bin_offsets_hz) {
        // Освобождаем память в случае ошибки
        if (ctx->in) fftwf_free(ctx->in);
        if (ctx->out) fftwf_free(ctx->out);
        free(ctx->pwr_dbm);
        free(ctx->window);
        free(ctx->bin_offsets_hz);
        free(ctx);
        return NULL;
    }
    
    // Создаем окно Ханна
    for (int i = 0; i < fft_size; i++) {
        ctx->window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (fft_size - 1)));
    }
    
    // Вычисляем корректировки
    calculate_window_corrections(ctx->window, fft_size, 
                                &ctx->window_loss_db, &ctx->enbw_corr_db);
    
    printf("[HackRF Master] FFT size: %d, bin width: %.1f kHz\n", 
           fft_size, ctx->bin_width/1000);
    printf("[HackRF Master] Window corrections: loss=%.2f dB, ENBW=%.2f dB\n", 
           ctx->window_loss_db, ctx->enbw_corr_db);
    
    // Создаем план FFT
    ctx->plan = fftwf_plan_dft_1d(fft_size, ctx->in, ctx->out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    // Предвычисляем смещения частот для четверти окна
    for (int i = 0; i < fft_size/4; i++) {
        ctx->bin_offsets_hz[i] = (i + 0.5) * ctx->bin_width;
    }
    
    return ctx;
}

static void destroy_processing_context(ProcessingContext* ctx) {
    if (!ctx) return;
    
    if (ctx->plan) fftwf_destroy_plan(ctx->plan);
    if (ctx->in) fftwf_free(ctx->in);
    if (ctx->out) fftwf_free(ctx->out);
    free(ctx->pwr_dbm);
    free(ctx->window);
    free(ctx->bin_offsets_hz);
    free(ctx);
}

// ================== Обработка блока данных ==================

static void process_sweep_block(MasterContext* master, uint8_t* buffer, int valid_length) {
    if (!master || !master->proc || !buffer || valid_length < 16) return;
    
    ProcessingContext* proc = master->proc;
    
    // Проверяем сигнатуру sweep-пакета (0x7F 0x7F)
    if (buffer[0] != 0x7F || buffer[1] != 0x7F) {
        return;
    }
    
    // Извлекаем центральную частоту
    uint64_t center_hz = 0;
    for (int i = 0; i < 8; i++) {
        center_hz |= ((uint64_t)buffer[2 + i]) << (8 * i);
    }
    
    // IQ данные начинаются с 16-го байта
    int8_t* iq_data = (int8_t*)(buffer + 16);
    int samples_available = (valid_length - 16) / 2;
    
    if (samples_available < proc->fft_size) {
        return;
    }
    
    // Применяем окно и нормализуем
    const float scale = 1.0f / 128.0f;
    for (int i = 0; i < proc->fft_size; i++) {
        float I = iq_data[2*i] * scale;
        float Q = iq_data[2*i + 1] * scale;
        proc->in[i][0] = I * proc->window[i];
        proc->in[i][1] = Q * proc->window[i];
    }
    
    // Выполняем FFT
    fftwf_execute(proc->plan);
    
    // Вычисляем мощность с правильной нормализацией
    const float fft_norm = 1.0f / proc->fft_size;
    const double center_mhz = center_hz / 1e6;
    const float cal_offset = get_calibration_offset(center_mhz);
    const float total_correction = proc->window_loss_db + proc->enbw_corr_db + cal_offset;
    
    for (int i = 0; i < proc->fft_size; i++) {
        proc->pwr_dbm[i] = calculate_power_dbm(
            proc->out[i][0] * fft_norm,
            proc->out[i][1] * fft_norm,
            total_correction
        );
    }
    
    // Обновляем глобальный спектр согласно режиму сегментов
    pthread_mutex_lock(&master->spectrum_mutex);
    
    const int quarter = proc->fft_size / 4;
    if (master->segment_mode == 4) {
            // 4-сегментный режим (как в эталоне)
            
            // Сегмент A: [center - OFFSET, center - OFFSET + Fs/4]
            uint64_t seg_a_start = center_hz - OFFSET_HZ;
            for (int i = 0; i < quarter; i++) {
                double freq = seg_a_start + proc->bin_offsets_hz[i];
                if (freq >= master->freq_start_hz && freq <= master->freq_stop_hz) {
                    int idx = (int)((freq - master->freq_start_hz) / master->freq_step_hz);
                    if (idx >= 0 && idx < master->spectrum_points) {
                        // Применяем EMA фильтрацию
                        if (master->spectrum_counts[idx] == 0) {
                            master->spectrum_powers[idx] = proc->pwr_dbm[i + 5*quarter/8 + 1];
                        } else {
                            master->spectrum_powers[idx] = 
                                (1.0f - proc->ema_alpha) * master->spectrum_powers[idx] + 
                                proc->ema_alpha * proc->pwr_dbm[i + 5*quarter/8 + 1];
                        }
                        master->spectrum_counts[idx]++;
                    }
                }
            }
            
            // Сегмент B: [center + OFFSET + Fs/2, center + OFFSET + 3Fs/4]
            uint64_t seg_b_start = center_hz + OFFSET_HZ + proc->sample_rate/2;
            for (int i = 0; i < quarter; i++) {
                double freq = seg_b_start + proc->bin_offsets_hz[i];
                if (freq >= master->freq_start_hz && freq <= master->freq_stop_hz) {
                    int idx = (int)((freq - master->freq_start_hz) / master->freq_step_hz);
                    if (idx >= 0 && idx < master->spectrum_points) {
                        if (master->spectrum_counts[idx] == 0) {
                            master->spectrum_powers[idx] = proc->pwr_dbm[i + quarter/8 + 1];
                        } else {
                            master->spectrum_powers[idx] = 
                                (1.0f - proc->ema_alpha) * master->spectrum_powers[idx] + 
                                proc->ema_alpha * proc->pwr_dbm[i + quarter/8 + 1];
                        }
                        master->spectrum_counts[idx]++;
                    }
                }
            }
            
            // Сегмент C: [center + OFFSET + Fs/4, center + OFFSET + Fs/2]
            uint64_t seg_c_start = center_hz + OFFSET_HZ + proc->sample_rate/4;
            for (int i = 0; i < quarter; i++) {
                double freq = seg_c_start + proc->bin_offsets_hz[i];
                if (freq >= master->freq_start_hz && freq <= master->freq_stop_hz) {
                    int idx = (int)((freq - master->freq_start_hz) / master->freq_step_hz);
                    if (idx >= 0 && idx < master->spectrum_points) {
                        if (master->spectrum_counts[idx] == 0) {
                            master->spectrum_powers[idx] = proc->pwr_dbm[i];
                        } else {
                            master->spectrum_powers[idx] = 
                                (1.0f - proc->ema_alpha) * master->spectrum_powers[idx] + 
                                proc->ema_alpha * proc->pwr_dbm[i];
                        }
                        master->spectrum_counts[idx]++;
                    }
                }
            }
            
            // Сегмент D: [center + OFFSET + 3Fs/4, center + OFFSET + Fs]
            uint64_t seg_d_start = center_hz + OFFSET_HZ + 3*proc->sample_rate/4;
            for (int i = 0; i < quarter; i++) {
                double freq = seg_d_start + proc->bin_offsets_hz[i];
                if (freq >= master->freq_start_hz && freq <= master->freq_stop_hz) {
                    int idx = (int)((freq - master->freq_start_hz) / master->freq_step_hz);
                    if (idx >= 0 && idx < master->spectrum_points) {
                        if (master->spectrum_counts[idx] == 0) {
                            master->spectrum_powers[idx] = proc->pwr_dbm[i + 3*quarter/8];
                        } else {
                            master->spectrum_powers[idx] = 
                                (1.0f - proc->ema_alpha) * master->spectrum_powers[idx] + 
                                proc->ema_alpha * proc->pwr_dbm[i + 3*quarter/8];
                        }
                        master->spectrum_counts[idx]++;
                    }
                }
            }
    pthread_mutex_unlock(&master->spectrum_mutex);
}

// ================== RX Callback ==================

static int rx_callback(hackrf_transfer* transfer) {
    if (!g_master || !g_master->running || g_master->stop_requested) {
        return -1;  // Останавливаем прием
    }
    
    uint8_t* buffer = transfer->buffer;
    int valid_length = transfer->valid_length;
    
    // Обрабатываем каждый блок в трансфере
    for (int i = 0; i < BLOCKS_PER_TRANSFER; i++) {
        process_sweep_block(g_master, buffer + i * BYTES_PER_BLOCK, BYTES_PER_BLOCK);
    }
    
    return 0;
}

// ================== Worker Thread ==================

static void* worker_thread_fn(void* arg) {
    MasterContext* master = (MasterContext*)arg;
    
    printf("[HackRF Master] Worker thread started\n");
    printf("[HackRF Master] Range: %.1f-%.1f MHz, step: %.1f kHz\n",
           master->freq_start_hz/1e6, master->freq_stop_hz/1e6, master->freq_step_hz/1e3);
    
    // Настройка железа
    hackrf_set_sample_rate_manual(master->dev, DEFAULT_SAMPLE_RATE_HZ, 1);
    hackrf_set_baseband_filter_bandwidth(master->dev, DEFAULT_BASEBAND_FILTER_BANDWIDTH);
    
    // Определяем размер FFT
    int fft_size = DEFAULT_FFT_SIZE;
    
    if (master->fft_size_override > 0) {
        // Используем фиксированный размер если задан
        fft_size = master->fft_size_override;
    } else if (master->freq_step_hz > 0) {
        // Автоматически подбираем размер FFT под желаемый шаг
        fft_size = (int)(DEFAULT_SAMPLE_RATE_HZ / master->freq_step_hz);
        
        // Ограничиваем разумными пределами
        if (fft_size < MIN_FFT_SIZE) fft_size = MIN_FFT_SIZE;
        if (fft_size > MAX_FFT_SIZE) fft_size = MAX_FFT_SIZE;
        
        // Округляем до ближайшей степени 2 для оптимальной FFT
        int power_of_2 = 1;
        while (power_of_2 < fft_size) {
            power_of_2 <<= 1;
        }
        // Если ближайшая степень 2 слишком большая, используем предыдущую
        if (power_of_2 > MAX_FFT_SIZE) {
            power_of_2 >>= 1;
        }
        fft_size = power_of_2;
    }
    
    printf("[HackRF Master] Using FFT size: %d (bin width: %.1f kHz)\n", 
           fft_size, (DEFAULT_SAMPLE_RATE_HZ / (double)fft_size) / 1000.0);
    
    master->proc = create_processing_context(fft_size, DEFAULT_SAMPLE_RATE_HZ);
    if (!master->proc) {
        set_error("Failed to create processing context");
        return NULL;
    }
    
    // Конфигурация sweep
    uint32_t freq_min_mhz = (uint32_t)(master->freq_start_hz / FREQ_ONE_MHZ);
    uint32_t freq_max_mhz = (uint32_t)(master->freq_stop_hz / FREQ_ONE_MHZ);
    
    // Округляем до 20 МГц
    freq_min_mhz = (freq_min_mhz / 20) * 20;
    freq_max_mhz = ((freq_max_mhz + 19) / 20) * 20;
    
    uint16_t freq_list[2] = {freq_min_mhz, freq_max_mhz};
    
    int r = hackrf_init_sweep(master->dev, freq_list, 1, 
                              BYTES_PER_BLOCK, TUNE_STEP_HZ, 
                              OFFSET_HZ, INTERLEAVED);
    if (r != HACKRF_SUCCESS) {
        set_error("hackrf_init_sweep failed: %s", hackrf_error_name(r));
        destroy_processing_context(master->proc);
        master->proc = NULL;
        return NULL;
    }
    
    // Запускаем sweep
    r = hackrf_start_rx_sweep(master->dev, rx_callback, NULL);
    if (r != HACKRF_SUCCESS) {
        set_error("hackrf_start_rx_sweep failed: %s", hackrf_error_name(r));
        destroy_processing_context(master->proc);
        master->proc = NULL;
        return NULL;
    }
    
    printf("[HackRF Master] Sweep started: %u-%u MHz\n", freq_min_mhz, freq_max_mhz);
    
    // Основной цикл
    while (master->running && !master->stop_requested) {
        usleep(100000);  // 100 мс
    }
    
    printf("[HackRF Master] Stopping sweep\n");
    hackrf_stop_rx(master->dev);
    
    destroy_processing_context(master->proc);
    master->proc = NULL;
    
    printf("[HackRF Master] Worker thread finished\n");
    return NULL;
}

// ================== API функции ==================

int hq_open(const char* serial_suffix) {
    int r = hackrf_init();
    if (r != HACKRF_SUCCESS) {
        set_error("hackrf_init failed: %s", hackrf_error_name(r));
        return r;
    }
    
    // Закрываем предыдущее устройство если открыто
    if (g_master) {
        hq_close();
    }
    
    g_master = (MasterContext*)calloc(1, sizeof(MasterContext));
    if (!g_master) {
        set_error("Memory allocation failed");
        hackrf_exit();
        return -1;
    }
    
    // Инициализируем мьютекс
    pthread_mutex_init(&g_master->spectrum_mutex, NULL);
    
    // Открываем устройство
    if (!serial_suffix || !serial_suffix[0]) {
        r = hackrf_open(&g_master->dev);
    } else {
        // Поиск по суффиксу серийного номера
        hackrf_device_list_t* list = hackrf_device_list();
        if (!list) {
            set_error("Failed to get device list");
            free(g_master);
            g_master = NULL;
            hackrf_exit();
            return -1;
        }
        
        int found = -1;
        size_t suffix_len = strlen(serial_suffix);
        for (int i = 0; i < list->devicecount; i++) {
            const char* serial = list->serial_numbers[i];
            if (serial) {
                size_t serial_len = strlen(serial);
                if (serial_len >= suffix_len) {
                    if (strcmp(serial + (serial_len - suffix_len), serial_suffix) == 0) {
                        found = i;
                        break;
                    }
                }
            }
        }
        
        if (found >= 0) {
            r = hackrf_open_by_serial(list->serial_numbers[found], &g_master->dev);
        } else {
            set_error("Device with suffix '%s' not found", serial_suffix);
            r = HACKRF_ERROR_NOT_FOUND;
        }
        
        hackrf_device_list_free(list);
    }
    
    if (r != HACKRF_SUCCESS) {
        set_error("Failed to open device: %s", hackrf_error_name(r));
        free(g_master);
        g_master = NULL;
        hackrf_exit();
        return r;
    }
    
    // Устанавливаем значения по умолчанию
    g_master->segment_mode = 4;
    g_master->calibration_offset_db = 0.0f;
    g_master->fft_size_override = 0;  // Auto
    
    return 0;
}

void hq_close(void) {
    if (!g_master) return;
    
    // Останавливаем worker thread если работает
    if (g_master->running) {
        hq_stop();
    }
    
    // Освобождаем память спектра
    free(g_master->spectrum_freqs);
    free(g_master->spectrum_powers);
    free(g_master->spectrum_counts);
    
    // Уничтожаем мьютекс
    pthread_mutex_destroy(&g_master->spectrum_mutex);
    
    // Закрываем устройство
    if (g_master->dev) {
        hackrf_close(g_master->dev);
    }
    
    free(g_master);
    g_master = NULL;
    
    hackrf_exit();
}

int hq_configure(double f_start_mhz, double f_stop_mhz, double bin_hz,
                 int lna_db, int vga_db, int amp_on) {
    if (!g_master || !g_master->dev) {
        set_error("Device not opened");
        return -1;
    }
    
    // Сохраняем параметры
    g_master->freq_start_hz = f_start_mhz * 1e6;
    g_master->freq_stop_hz = f_stop_mhz * 1e6;
    g_master->freq_step_hz = bin_hz;
    
    // Настраиваем усиление
    hackrf_set_lna_gain(g_master->dev, lna_db);
    hackrf_set_vga_gain(g_master->dev, vga_db);
    hackrf_set_amp_enable(g_master->dev, amp_on);
    
    // Выделяем память для глобального спектра
    size_t n_points = (size_t)((g_master->freq_stop_hz - g_master->freq_start_hz) / bin_hz) + 1;
    if (n_points > MAX_SPECTRUM_POINTS) {
        n_points = MAX_SPECTRUM_POINTS;
    }
    
    g_master->spectrum_points = n_points;
    g_master->spectrum_freqs = (double*)realloc(g_master->spectrum_freqs, sizeof(double) * n_points);
    g_master->spectrum_powers = (float*)realloc(g_master->spectrum_powers, sizeof(float) * n_points);
    g_master->spectrum_counts = (int*)realloc(g_master->spectrum_counts, sizeof(int) * n_points);
    
    // Инициализируем сетку частот
    for (size_t i = 0; i < n_points; i++) {
        g_master->spectrum_freqs[i] = g_master->freq_start_hz + i * bin_hz;
        g_master->spectrum_powers[i] = -120.0f;
        g_master->spectrum_counts[i] = 0;
    }
    
    printf("[HackRF Master] Configured: %.1f-%.1f MHz, %zu points, bin=%.1f kHz\n",
           f_start_mhz, f_stop_mhz, n_points, bin_hz/1000);
    
    return 0;
}

int hq_start_multi_segment(hq_multi_segment_cb cb, void* user) {
    if (!g_master || !g_master->dev) {
        set_error("Device not opened");
        return -1;
    }
    
    if (g_master->running) {
        set_error("Already running");
        return -1;
    }
    
    g_master->multi_cb = cb;
    g_master->legacy_cb = NULL;
    g_master->user_data = user;
    g_master->running = 1;
    g_master->stop_requested = 0;
    
    gettimeofday(&g_master->start_time, NULL);
    
    // Запускаем worker thread
    if (pthread_create(&g_worker_thread, NULL, worker_thread_fn, g_master) != 0) {
        set_error("Failed to create worker thread");
        g_master->running =

        