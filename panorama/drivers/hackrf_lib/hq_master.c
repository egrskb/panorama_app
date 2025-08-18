// hq_master.c - ИСПРАВЛЕННАЯ ВЕРСИЯ
// Правильный расчет RSSI и работа без зависаний

#include "hq_master.h"
#include "hq_init.h"
#include "hq_grouping.h"
#include "hq_rssi.h"

#include <libhackrf/hackrf.h>
#include <fftw3.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/time.h>
#include <inttypes.h>

#ifndef BLOCKS_PER_TRANSFER
#define BLOCKS_PER_TRANSFER 16
#endif

#ifndef BYTES_PER_BLOCK
#define BYTES_PER_BLOCK 16384
#endif

#define DEFAULT_SAMPLE_RATE_HZ            20000000u   // 20 MHz
#define DEFAULT_BASEBAND_FILTER_BANDWIDTH 15000000u   // 15 MHz

#define FREQ_ONE_MHZ 1000000ull

#ifndef INTERLEAVED
#define INTERLEAVED 1
#endif

#define TUNE_STEP_HZ 20000000u
#define OFFSET_HZ     7500000u

// Внешние функции
extern void hq_on_peak_detected(SdrCtx* ctx, double f_hz, float dbm);

// ================== Внутреннее состояние мастера ==================

typedef struct {
    SdrCtx* ctx;

    // sweep-план
    double f_start;
    double f_stop;
    double f_step;

    // FFT
    int N;
    double Fs;
    double bin_w;
    fftwf_complex* in;
    fftwf_complex* out;
    fftwf_plan plan;
    float* window;
    float* pwr_db;

    // Спектр
    double* freqs;
    float*  powers;
    int*    counts;
    size_t  n_points;

    // Коррекции для правильного RSSI
    float window_power_loss_db;  // Потери из-за окна
    float enbw_correction_db;    // Коррекция ENBW

    // Тайминг
    float ema_alpha;
    struct timeval usb_tv;

    // Предвычисления
    double* bin_offsets_hz;
    int peak_decim;
    
    // Флаг остановки
    volatile bool stop_requested;
} MasterData;

// ================== Утилиты ==================

static inline float calculate_power_dbm(float re, float im, float correction) {
    float magnitude_squared = re * re + im * im;
    if (magnitude_squared < 1e-20f) {
        magnitude_squared = 1e-20f;
    }
    
    // Правильный расчет мощности в dBm
    // 10*log10(magnitude^2) = 20*log10(magnitude)
    float power_dbfs = 10.0f * log10f(magnitude_squared);
    
    // Применяем коррекции и калибровку для перевода в dBm
    // Типичная калибровка HackRF: 0 dBFS ≈ -10 dBm при LNA=24, VGA=20
    float power_dbm = power_dbfs + correction - 10.0f;  // Базовая калибровка
    
    return power_dbm;
}

static void calculate_window_corrections(float* window, int N, float* power_loss, float* enbw_corr) {
    // Вычисляем коррекции для окна Ханна
    double sum_linear = 0.0;
    double sum_squared = 0.0;
    
    for (int i = 0; i < N; i++) {
        sum_linear += window[i];
        sum_squared += window[i] * window[i];
    }
    
    // Coherent gain (для амплитуды)
    double coherent_gain = sum_linear / N;
    
    // Processing gain (для мощности)
    double processing_gain = sum_squared / N;
    
    // Потери мощности из-за окна
    *power_loss = -10.0f * log10f(processing_gain);
    
    // ENBW коррекция
    double enbw = N * processing_gain / (coherent_gain * coherent_gain);
    *enbw_corr = 10.0f * log10f(enbw);
}

// ================== RX callback ==================

static int master_rx_cb(hackrf_transfer* t) {
    MasterData* d = (MasterData*)t->rx_ctx;
    SdrCtx* ctx = d->ctx;
    
    if (!ctx || !ctx->running || d->stop_requested) {
        return -1;  // Останавливаем прием
    }

    // Инициализируем время первой пачки
    if (d->usb_tv.tv_sec == 0 && d->usb_tv.tv_usec == 0) {
        gettimeofday(&d->usb_tv, NULL);
    }

    int8_t* buf = (int8_t*)t->buffer;
    const int N = d->N;
    const int q = N / 4;
    
    // Коррекция для правильного RSSI
    float total_correction = d->window_power_loss_db + d->enbw_correction_db;

    for (int j = 0; j < BLOCKS_PER_TRANSFER; ++j) {
        uint8_t* ubuf = (uint8_t*)buf;

        // Проверяем заголовок блока
        if (!(ubuf[0] == 0x7F && ubuf[1] == 0x7F)) {
            buf += BYTES_PER_BLOCK;
            continue;
        }

        // Частота настройки блока
        uint64_t frequency =
            ((uint64_t)ubuf[9] << 56) | ((uint64_t)ubuf[8] << 48) |
            ((uint64_t)ubuf[7] << 40) | ((uint64_t)ubuf[6] << 32) |
            ((uint64_t)ubuf[5] << 24) | ((uint64_t)ubuf[4] << 16) |
            ((uint64_t)ubuf[3] <<  8) |  (uint64_t)ubuf[2];

        // Берём ПОСЛЕДНИЕ N I/Q отсчётов из блока
        int8_t* samp = buf + BYTES_PER_BLOCK - (N * 2);

        // Применяем окно и нормализуем
        for (int i = 0; i < N; ++i) {
            d->in[i][0] = (float)samp[2*i]   * d->window[i] / 128.0f;
            d->in[i][1] = (float)samp[2*i+1] * d->window[i] / 128.0f;
        }

        // FFT
        fftwf_execute(d->plan);
        
        // Вычисляем мощность с правильной нормализацией
        float norm_factor = 1.0f / (float)N;  // Нормализация FFT
        for (int i = 0; i < N; ++i) {
            d->pwr_db[i] = calculate_power_dbm(
                d->out[i][0] * norm_factor,
                d->out[i][1] * norm_factor,
                total_correction
            );
        }

        // const double bw = d->bin_w;  // Unused variable

        // Секция A: [frequency, frequency + Fs/4)
        {
            uint64_t hz_low = frequency;
            int bin0 = 1 + (5*N)/8;
            
            for (int i = 0; i < q; ++i) {
                int bin = bin0 + i;
                float dbm = d->pwr_db[bin];
                double f_hz = (double)hz_low + d->bin_offsets_hz[i];

                // Маппинг в общую сетку
                long idx = (long)((f_hz - d->f_start) / d->f_step);
                if (idx >= 0 && (size_t)idx < d->n_points) {
                    float* dst = &d->powers[idx];
                    int*   cnt = &d->counts[idx];
                    
                    if (*cnt == 0) {
                        *dst = dbm;
                    } else {
                        // EMA для сглаживания
                        *dst = (1.0f - d->ema_alpha) * (*dst) + d->ema_alpha * dbm;
                    }
                    (*cnt)++;

                    // Детекция пиков для watchlist
                    // Decimate peak insertion to reduce queue pressure
                    if ((i % d->peak_decim) == 0 && dbm > -80.0f) {
                        add_peak(f_hz, dbm);
                    }
                }
            }
        }

        // Секция B: [frequency + Fs/2, frequency + 3Fs/4)
        {
            uint64_t hz_low = frequency + (uint64_t)(d->Fs / 2.0);
            int bin0 = 1 + (N/8);
            
            for (int i = 0; i < q; ++i) {
                int bin = bin0 + i;
                float dbm = d->pwr_db[bin];
                double f_hz = (double)hz_low + d->bin_offsets_hz[i];

                long idx = (long)((f_hz - d->f_start) / d->f_step);
                if (idx >= 0 && (size_t)idx < d->n_points) {
                    float* dst = &d->powers[idx];
                    int*   cnt = &d->counts[idx];
                    
                    if (*cnt == 0) {
                        *dst = dbm;
                    } else {
                        *dst = (1.0f - d->ema_alpha) * (*dst) + d->ema_alpha * dbm;
                    }
                    (*cnt)++;

                    if ((i % d->peak_decim) == 0 && dbm > -80.0f) {
                        add_peak(f_hz, dbm);
                    }
                }
            }
        }

        buf += BYTES_PER_BLOCK;
    }

    return 0;
}

// ================== Поток мастера ==================

static void* master_thread_fn(void* arg) {
    MasterData* d = (MasterData*)arg;
    SdrCtx* ctx = d->ctx;

    printf("Master: Initializing with range %.1f-%.1f MHz\n", 
           d->f_start/1e6, d->f_stop/1e6);

    // Настройка железа
    d->Fs = (double)DEFAULT_SAMPLE_RATE_HZ;
    hackrf_set_sample_rate_manual(ctx->dev, DEFAULT_SAMPLE_RATE_HZ, 1);
    hackrf_set_baseband_filter_bandwidth(ctx->dev, DEFAULT_BASEBAND_FILTER_BANDWIDTH);
    hackrf_set_vga_gain(ctx->dev, ctx->vga_db);
    hackrf_set_lna_gain(ctx->dev, ctx->lna_db);
    hackrf_set_amp_enable(ctx->dev, ctx->amp_on ? 1 : 0);

    // Выбор N по желаемому шагу сетки
    double requested_bin = (d->f_step > 0.0) ? d->f_step : (d->Fs / 8192.0);
    if (requested_bin < 2445.0) requested_bin = 2445.0;
    if (requested_bin > 5e6)    requested_bin = 5e6;

    int N = (int)floor(d->Fs / requested_bin);
    if (N < 4)    N = 4;
    if (N > 8180) N = 8180;
    while ((N + 4) % 8) N++;
    d->N = N;
    d->bin_w = d->Fs / (double)d->N;

    printf("Master: FFT size = %d, bin width = %.1f kHz\n", N, d->bin_w/1000);

    // FFT ресурсы
    d->in  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * d->N);
    d->out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * d->N);
    d->pwr_db = (float*)malloc(sizeof(float) * d->N);
    d->window = (float*)malloc(sizeof(float) * d->N);

    // Окно Ханна
    for (int i = 0; i < d->N; ++i) {
        d->window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (d->N - 1)));
    }

    // Вычисляем коррекции окна
    calculate_window_corrections(d->window, d->N, 
                                &d->window_power_loss_db, 
                                &d->enbw_correction_db);
    
    printf("Master: Window corrections: power loss = %.2f dB, ENBW = %.2f dB\n",
           d->window_power_loss_db, d->enbw_correction_db);

    d->plan = fftwf_plan_dft_1d(d->N, d->in, d->out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(d->plan);  // Прогрев

    // Предвычисления
    const int q = d->N / 4;
    d->bin_offsets_hz = (double*)malloc(sizeof(double) * q);
    for (int i = 0; i < q; ++i) {
        d->bin_offsets_hz[i] = (i + 0.5) * d->bin_w;
    }

    d->ema_alpha = 0.25f;
    d->peak_decim = 4;
    memset(&d->usb_tv, 0, sizeof(d->usb_tv));

    // Итоговая сетка
    if (d->f_start < 0) d->f_start = 0;
    if (d->f_stop < d->f_start) d->f_stop = d->f_start + d->Fs/4.0;

    d->f_step = d->bin_w;
    uint64_t span_hz = (uint64_t)llround(d->f_stop - d->f_start);
    d->n_points = (size_t)ceil((double)span_hz / d->f_step);
    
    // Ограничиваем количество точек
    if (d->n_points > MAX_SPECTRUM_POINTS) {
        printf("Master: Limiting points from %zu to %d\n", d->n_points, MAX_SPECTRUM_POINTS);
        d->n_points = MAX_SPECTRUM_POINTS;
    }

    d->freqs  = (double*)malloc(sizeof(double) * d->n_points);
    d->powers = (float*) calloc(d->n_points, sizeof(float));
    d->counts = (int*)   calloc(d->n_points, sizeof(int));
    
    for (size_t i = 0; i < d->n_points; ++i) {
        d->freqs[i] = d->f_start + (double)i * d->f_step;
    }

    // Конфигурация hackrf_sweep
    uint32_t fmin_mhz = (uint32_t)(d->f_start / FREQ_ONE_MHZ);
    uint32_t fmax_mhz = (uint32_t)(d->f_stop  / FREQ_ONE_MHZ);
    uint32_t steps = (uint32_t)((fmax_mhz - fmin_mhz + 19) / 20);
    fmax_mhz = fmin_mhz + steps * 20;

    uint16_t freqs_mhz[2];
    freqs_mhz[0] = (uint16_t)fmin_mhz;
    freqs_mhz[1] = (uint16_t)fmax_mhz;

    printf("Master: Starting sweep %u-%u MHz\n", fmin_mhz, fmax_mhz);

    int r = hackrf_init_sweep(ctx->dev,
                              freqs_mhz,
                              1,
                              BYTES_PER_BLOCK,
                              TUNE_STEP_HZ,
                              OFFSET_HZ,
                              INTERLEAVED);
    if (r != HACKRF_SUCCESS) {
        fprintf(stderr, "Master: hackrf_init_sweep failed: %s (%d)\n", 
                hackrf_error_name(r), r);
        goto cleanup;
    }

    ctx->running = true;
    d->stop_requested = false;
    
    r = hackrf_start_rx_sweep(ctx->dev, master_rx_cb, d);
    if (r != HACKRF_SUCCESS) {
        fprintf(stderr, "Master: hackrf_start_rx_sweep failed: %s (%d)\n", 
                hackrf_error_name(r), r);
        goto cleanup;
    }

    printf("Master: Sweep started successfully\n");

    // Периодически отправляем спектр в UI
    struct timeval t_prev = {0}, t_now = {0};
    gettimeofday(&t_prev, NULL);
    
    int update_count = 0;
    int no_data_count = 0;  // Счетчик циклов без данных

    while (ctx->running && !d->stop_requested && hackrf_is_streaming(ctx->dev) == HACKRF_TRUE) {
        usleep(100000); // 100 ms
        
        // Дополнительная проверка состояния устройства
        if (hackrf_is_streaming(ctx->dev) != HACKRF_TRUE) {
            printf("Master: Device stopped streaming, breaking loop\n");
            break;
        }
        
        gettimeofday(&t_now, NULL);
        double dt = (t_now.tv_sec - t_prev.tv_sec) + 1e-6 * (t_now.tv_usec - t_prev.tv_usec);
        
        if (dt >= 0.10) { // Каждые 100 ms
            // Проверяем, есть ли данные для обновления
            bool has_data = false;
            for (size_t i = 0; i < d->n_points; i++) {
                if (d->counts[i] > 0) {
                    has_data = true;
                    break;
                }
            }
            
            if (has_data) {
                // Обновляем глобальный спектр
                hq_update_spectrum(d->freqs, d->powers, (int)d->n_points);
                t_prev = t_now;
                update_count++;
                no_data_count = 0;  // Сбрасываем счетчик
                
                // Логируем каждые 10 обновлений (1 секунда)
                if (update_count % 10 == 0) {
                    // Находим максимум для диагностики
                    float max_power = -200.0f;
                    for (size_t i = 0; i < d->n_points; i++) {
                        if (d->powers[i] > max_power) {
                            max_power = d->powers[i];
                        }
                    }
                    printf("Master: Update #%d, max power: %.1f dBm\n", 
                           update_count, max_power);
                }
            } else {
                no_data_count++;
                // Если долго нет данных, логируем предупреждение
                if (no_data_count % 50 == 0) {  // Каждые 5 секунд
                    printf("Master: Warning - no data for %d cycles\n", no_data_count);
                }
            }
        }
        
        // Проверяем флаг остановки
        if (!ctx->running) {
            d->stop_requested = true;
            break;
        }
        
        // Защита от зависания - если слишком долго нет данных, выходим
        if (no_data_count > 200) {  // 20 секунд без данных
            printf("Master: Timeout - no data for too long, stopping\n");
            break;
        }
    }

    printf("Master: Stopping sweep\n");
    hackrf_stop_rx(ctx->dev);

cleanup:
    printf("Master: Cleanup\n");
    
    if (d->plan) fftwf_destroy_plan(d->plan);
    if (d->in)   fftwf_free(d->in);
    if (d->out)  fftwf_free(d->out);
    free(d->pwr_db);
    free(d->window);
    free(d->bin_offsets_hz);
    free(d->freqs);
    free(d->powers);
    free(d->counts);
    
    ctx->thread_data = NULL;
    ctx->running = false;
    
    free(d);
    
    printf("Master: Thread finished\n");
    return NULL;
}

// ================== API ==================

int start_master(SdrCtx* ctx, const SweepPlan* plan) {
    if (!ctx || !ctx->dev || !plan) {
        fprintf(stderr, "Master: Invalid parameters\n");
        return -1;
    }
    
    if (ctx->running) {
        printf("Master: Already running\n");
        return 0;
    }

    MasterData* d = (MasterData*)calloc(1, sizeof(MasterData));
    if (!d) {
        fprintf(stderr, "Master: Failed to allocate data\n");
        return -2;
    }

    d->ctx     = ctx;
    d->f_start = (double)plan->start_hz;
    d->f_stop  = (double)plan->stop_hz;
    d->f_step  = (double)plan->step_hz;
    d->stop_requested = false;

    ctx->thread_data = d;
    ctx->running = true;

    if (pthread_create(&ctx->thread, NULL, master_thread_fn, d) != 0) {
        fprintf(stderr, "Master: Failed to create thread\n");
        ctx->running = false;
        ctx->thread_data = NULL;
        free(d);
        return -3;
    }
    
    printf("Master: Thread created\n");
    return 0;
}

void hq_master_stop(SdrCtx* ctx) {
    if (!ctx) return;
    if (!ctx->running) return;
    
    printf("Master: Stopping\n");
    
    // Сигнализируем остановку
    ctx->running = false;
    
    // Если есть данные потока, устанавливаем флаг
    if (ctx->thread_data) {
        MasterData* d = (MasterData*)ctx->thread_data;
        d->stop_requested = true;
    }
    
    // Ждем завершения потока
    pthread_join(ctx->thread, NULL);
    
    printf("Master: Stopped\n");
}