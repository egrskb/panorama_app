// hackrf_master.c - Полноценный C-бэкенд для Panorama с расчетами спектра
// Основано на эталонных файлах hq_master.c и hq_rssi.c

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
    double freq_hz;
    float power_dbm;
    float snr_db;
} DetectedPeak;

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
    
    // Режим сегментов (2 или 4)
    int segment_mode;
    
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
    const float total_correction = proc->window_loss_db + proc->enbw_corr_db + 
                                  master->calibration_offset_db;
    
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
        
        // Подготавливаем данные сегментов для колбэка
        if (master->multi_cb) {
            // Заполняем структуры сегментов
            for (int seg = 0; seg < 4; seg++) {
                g_segments[seg].segment_id = seg;
                g_segments[seg].count = quarter;
                // Здесь нужно будет заполнить freqs_hz и data_dbm
                // из соответствующих частей спектра
            }
            
            master->multi_cb(g_segments, 4, proc->bin_width, center_hz, master->user_data);
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
    
    // Создаем контекст обработки
    int fft_size = DEFAULT_FFT_SIZE;
    if (master->freq_step_hz > 0) {
        // Подбираем размер FFT под желаемый шаг
        double desired_bin = master->freq_step_hz;
        if (desired_bin < 2445.0) desired_bin = 2445.0;
        if (desired_bin > 5e6) desired_bin = 5e6;
        
        fft_size = (int)(DEFAULT_SAMPLE_RATE_HZ / desired_bin);
        if (fft_size < 4) fft_size = 4;
        if (fft_size > 8192) fft_size = 8192;
        
        // Выравниваем на 8
        while ((fft_size + 4) % 8) fft_size++;
    }
    
    master->proc = create_processing_context(fft_size, DEFAULT_SAMPLE_RATE_HZ);
    if (!master->proc) {
        set_error("Failed to create processing context");
        return NULL;
    }
    
    printf("[HackRF Master] FFT size: %d, bin width: %.1f kHz\n", 
           fft_size, master->proc->bin_width/1000);
    
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
        
        // Здесь можно добавить периодическую обработку,
        // например, детекцию пиков или обновление статистики
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
        g_master->running = 0;
        return -1;
    }
    
    return 0;
}

int hq_start(hq_segment_cb cb, void* user) {
    if (!g_master || !g_master->dev) {
        set_error("Device not opened");
        return -1;
    }
    
    g_master->legacy_cb = cb;
    g_master->multi_cb = NULL;
    g_master->user_data = user;
    g_master->running = 1;
    g_master->stop_requested = 0;
    
    if (pthread_create(&g_worker_thread, NULL, worker_thread_fn, g_master) != 0) {
        set_error("Failed to create worker thread");
        g_master->running = 0;
        return -1;
    }
    
    return 0;
}

int hq_stop(void) {
    if (!g_master || !g_master->running) {
        return 0;
    }
    
    g_master->stop_requested = 1;
    
    // Ждем завершения потока
    pthread_join(g_worker_thread, NULL);
    
    g_master->running = 0;
    g_master->stop_requested = 0;
    
    return 0;
}

// ================== Новые API функции для полной интеграции ==================

int hq_get_master_spectrum(double* freqs_hz, float* powers_dbm, int max_points) {
    if (!g_master || !freqs_hz || !powers_dbm || max_points <= 0) {
        return 0;
    }
    
    pthread_mutex_lock(&g_master->spectrum_mutex);
    
    int points_to_copy = (max_points < g_master->spectrum_points) ? 
                         max_points : g_master->spectrum_points;
    
    memcpy(freqs_hz, g_master->spectrum_freqs, sizeof(double) * points_to_copy);
    memcpy(powers_dbm, g_master->spectrum_powers, sizeof(float) * points_to_copy);
    
    pthread_mutex_unlock(&g_master->spectrum_mutex);
    
    return points_to_copy;
}

void hq_set_ema_alpha(float alpha) {
    if (!g_master || !g_master->proc) return;
    
    if (alpha < 0.01f) alpha = 0.01f;
    if (alpha > 1.0f) alpha = 1.0f;
    
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
    if (mode != 2 && mode != 4) {
        set_error("Invalid segment mode (must be 2 or 4)");
        return -1;
    }
    
    if (g_master) {
        g_master->segment_mode = mode;
    }
    
    return 0;
}

int hq_get_segment_mode(void) {
    return g_master ? g_master->segment_mode : 4;
}

// ================== Калибровка ==================

static struct {
    float freq_mhz[MAX_CALIBRATION_ENTRIES];
    int lna_db[MAX_CALIBRATION_ENTRIES];
    int vga_db[MAX_CALIBRATION_ENTRIES];
    int amp_on[MAX_CALIBRATION_ENTRIES];
    float offset_db[MAX_CALIBRATION_ENTRIES];
    int count;
    int enabled;
} g_calibration = {0};

int hq_load_calibration(const char* csv_path) {
    if (!csv_path) {
        set_error("CSV path is NULL");
        return -1;
    }
    
    FILE* f = fopen(csv_path, "r");
    if (!f) {
        set_error("Cannot open calibration file: %s", csv_path);
        return -1;
    }
    
    char line[256];
    int line_num = 0;
    g_calibration.count = 0;
    
    // Пропускаем заголовок
    if (fgets(line, sizeof(line), f)) {
        line_num++;
    }
    
    while (fgets(line, sizeof(line), f) && g_calibration.count < MAX_CALIBRATION_ENTRIES) {
        line_num++;
        
        float freq_mhz, offset_db;
        int lna_db, vga_db, amp_on;
        
        if (sscanf(line, "%f,%d,%d,%d,%f", 
                   &freq_mhz, &lna_db, &vga_db, &amp_on, &offset_db) == 5) {
            int idx = g_calibration.count;
            g_calibration.freq_mhz[idx] = freq_mhz;
            g_calibration.lna_db[idx] = lna_db;
            g_calibration.vga_db[idx] = vga_db;
            g_calibration.amp_on[idx] = amp_on;
            g_calibration.offset_db[idx] = offset_db;
            g_calibration.count++;
        }
    }
    
    fclose(f);
    
    printf("[HackRF Master] Loaded %d calibration entries from %s\n", 
           g_calibration.count, csv_path);
    
    // Автоматически применяем калибровку к текущему устройству
    if (g_master && g_calibration.count > 0) {
        // Находим ближайшую калибровку для текущей частоты
        // (упрощенно - берем среднее)
        float sum = 0;
        for (int i = 0; i < g_calibration.count; i++) {
            sum += g_calibration.offset_db[i];
        }
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
            // Применяем усредненную калибровку
            float sum = 0;
            for (int i = 0; i < g_calibration.count; i++) {
                sum += g_calibration.offset_db[i];
            }
            g_master->calibration_offset_db = sum / g_calibration.count;
        }
    }
    
    return 0;
}

int hq_get_calibration_status(void) {
    return g_calibration.enabled && g_calibration.count > 0;
}

// ================== Перечисление устройств ==================

int hq_device_count(void) {
    int r = hackrf_init();
    if (r != HACKRF_SUCCESS) return 0;
    
    hackrf_device_list_t* list = hackrf_device_list();
    if (!list) {
        hackrf_exit();
        return 0;
    }
    
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
    if (!list) {
        hackrf_exit();
        return -1;
    }
    
    if (idx < 0 || idx >= list->devicecount) {
        hackrf_device_list_free(list);
        hackrf_exit();
        return -1;
    }
    
    const char* serial = list->serial_numbers[idx];
    if (serial) {
        strncpy(out, serial, cap - 1);
        out[cap - 1] = '\0';
    } else {
        out[0] = '\0';
    }
    
    hackrf_device_list_free(list);
    hackrf_exit();
    
    return 0;
}