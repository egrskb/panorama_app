// hq_slave.c - ПОЛНАЯ ЗАМЕНА ФАЙЛА
#include "hq_slave.h"
#include "hq_rssi.h"
#include "hq_grouping.h"
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <fftw3.h>

typedef struct {
    SdrCtx* ctx;
    int slave_idx;
    
    // FFT processing
    fftwf_complex* fft_in;
    fftwf_complex* fft_out;
    fftwf_plan fft_plan;
    int fft_size;
    float* window;
    
    // Target tracking
    double target_freq_hz;
    double bandwidth_hz;
    bool has_target;
    uint64_t last_target_update_ns;
} SlaveData;

// Создание окна для FFT
static void create_window(float* window, int size) {
    for (int i = 0; i < size; i++) {
        double a0 = 0.35875;
        double a1 = 0.48829;
        double a2 = 0.14128;
        double a3 = 0.01168;
        double phase = 2.0 * M_PI * i / (size - 1);
        window[i] = a0 - a1 * cos(phase) + a2 * cos(2 * phase) - a3 * cos(3 * phase);
    }
}

static int slave_rx_callback(hackrf_transfer* transfer) {
    SlaveData* data = (SlaveData*)transfer->rx_ctx;
    SdrCtx* ctx = data->ctx;
    
    if (!ctx->running) {
        return -1;
    }
    
    // Обрабатываем только если есть цель
    if (!data->has_target) {
        return 0;
    }
    
    // Преобразуем I/Q в комплексные числа
    int8_t* buf = (int8_t*)transfer->buffer;
    int samples = transfer->valid_length / 2;
    
    // Заполняем FFT буфер
    for (int i = 0; i < samples && i < data->fft_size; i++) {
        float i_val = (float)buf[i*2] / 128.0f;
        float q_val = (float)buf[i*2 + 1] / 128.0f;
        
        data->fft_in[i][0] = i_val * data->window[i];
        data->fft_in[i][1] = q_val * data->window[i];
    }
    
    // FFT
    fftwf_execute(data->fft_plan);
    
    // Анализируем спектр в области цели
    float bin_width_hz = (float)ctx->samp_rate / data->fft_size;
    float max_power = -120.0f;
    double peak_freq = data->target_freq_hz;
    
    for (int i = 0; i < data->fft_size; i++) {
        int idx = (i + data->fft_size/2) % data->fft_size;
        
        float re = data->fft_out[idx][0];
        float im = data->fft_out[idx][1];
        float power = 10.0f * log10f(re*re + im*im + 1e-20f);
        
        double freq_hz = ctx->center_hz + (i - data->fft_size/2) * bin_width_hz;
        
        // Проверяем попадание в область цели
        if (fabs(freq_hz - data->target_freq_hz) < data->bandwidth_hz / 2) {
            if (power > max_power) {
                max_power = power;
                peak_freq = freq_hz;
            }
        }
    }
    
    // Обновляем измерение для цели
    if (max_power > -100.0f) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t now_ns = ts.tv_sec * 1000000000ULL + ts.tv_nsec;
        
        // Обновляем в watchlist
        pthread_mutex_lock(&g_watchlist_mutex);
        
        for (size_t i = 0; i < g_watchlist_count; i++) {
            if (fabs(g_watchlist[i].f_center_hz - peak_freq) < data->bandwidth_hz) {
                // Обновляем RSSI с EMA
                g_watchlist[i].rssi_ema = rssi_apply_ema(
                    g_watchlist[i].rssi_ema, max_power, g_ema_alpha
                );
                g_watchlist[i].last_ns = now_ns;
                g_watchlist[i].hit_count++;
                
                // Добавляем в очередь пиков для трилатерации
                Peak peak = {
                    .f_hz = peak_freq,
                    .rssi_dbm = max_power,
                    .last_ns = now_ns
                };
                peak_queue_push(g_peaks_queue, &peak);
                break;
            }
        }
        
        pthread_mutex_unlock(&g_watchlist_mutex);
    }
    
    return 0;
}

static void* slave_thread_fn(void* arg) {
    SlaveData* data = (SlaveData*)arg;
    SdrCtx* ctx = data->ctx;
    int slave_idx = data->slave_idx;
    
    printf("Slave %d thread started\n", slave_idx);
    
    // Инициализация FFT
    data->fft_size = 4096;  // Меньше чем у master для скорости
    data->fft_in = fftwf_alloc_complex(data->fft_size);
    data->fft_out = fftwf_alloc_complex(data->fft_size);
    // Создание плана FFTW не потокобезопасно — защищаем глобальным мьютексом
    pthread_mutex_lock(&g_fftw_planner_mutex);
    data->fft_plan = fftwf_plan_dft_1d(data->fft_size, data->fft_in, data->fft_out,
                                        FFTW_FORWARD, FFTW_ESTIMATE);
    pthread_mutex_unlock(&g_fftw_planner_mutex);
    
    data->window = malloc(data->fft_size * sizeof(float));
    create_window(data->window, data->fft_size);
    
    // Начальные параметры
    data->has_target = false;
    data->bandwidth_hz = 8e6;  // 8 МГц полоса обзора
    
    // Запускаем RX
    int r = hackrf_start_rx(ctx->dev, slave_rx_callback, data);
    if (r != HACKRF_SUCCESS) {
        fprintf(stderr, "Slave %d: start_rx failed: %s\n", 
                slave_idx, hackrf_error_name(r));
        goto cleanup;
    }
    
    struct timespec ts;
    // double current_center = 0;  // Unused variable
    
    while (ctx->running) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t now_ns = ts.tv_sec * 1000000000ULL + ts.tv_nsec;
        
        // Получаем цель из watchlist
        pthread_mutex_lock(&g_watchlist_mutex);
        
        if (g_watchlist_count > 0) {
            // Выбираем цель для этого slave
            // Slave 1 берет нечетные индексы, Slave 2 - четные
            int target_idx = -1;
            
            for (size_t i = 0; i < g_watchlist_count; i++) {
                if ((slave_idx == 1 && (i % 2) == 1) ||
                    (slave_idx == 2 && (i % 2) == 0)) {
                    
                    // Проверяем актуальность цели
                    if ((now_ns - g_watchlist[i].last_ns) < 2000000000ULL) {  // 2 секунды
                        target_idx = i;
                        break;
                    }
                }
            }
            
            if (target_idx >= 0) {
                double new_freq = g_watchlist[target_idx].f_center_hz;
                
                // Перестраиваемся если нужно
                if (!data->has_target || fabs(new_freq - data->target_freq_hz) > 1e6) {
                    data->target_freq_hz = new_freq;
                    data->has_target = true;
                    data->last_target_update_ns = now_ns;
                    
                    // Центрируемся на цели
                    ctx->center_hz = new_freq;
                    r = hackrf_set_freq(ctx->dev, (uint64_t)new_freq);
                    if (r != HACKRF_SUCCESS) {
                        fprintf(stderr, "Slave %d: retune to %.1f MHz failed\n",
                                slave_idx, new_freq/1e6);
                    } else {
                        printf("Slave %d: tracking target at %.1f MHz (RSSI: %.1f dBm)\n",
                               slave_idx, new_freq/1e6, g_watchlist[target_idx].rssi_ema);
                    }
                    
                    // current_center = new_freq;  // Unused variable
                    ctx->retune_count++;
                }
            } else {
                // Нет подходящей цели
                if (data->has_target && (now_ns - data->last_target_update_ns) > 5000000000ULL) {
                    printf("Slave %d: lost target\n", slave_idx);
                    data->has_target = false;
                }
            }
        } else {
            data->has_target = false;
        }
        
        pthread_mutex_unlock(&g_watchlist_mutex);
        
        // Небольшая задержка
        usleep(50000);  // 50ms
    }
    
    // Останавливаем RX
    hackrf_stop_rx(ctx->dev);
    
cleanup:
    // Очистка ресурсов
    if (data->fft_in) fftwf_free(data->fft_in);
    if (data->fft_out) fftwf_free(data->fft_out);
    if (data->fft_plan) {
        pthread_mutex_lock(&g_fftw_planner_mutex);
        fftwf_destroy_plan(data->fft_plan);
        pthread_mutex_unlock(&g_fftw_planner_mutex);
    }
    if (data->window) free(data->window);
    
    printf("Slave %d thread stopped\n", slave_idx);
    free(data);
    return NULL;
}

int start_slave(SdrCtx* slave, int slave_idx) {
    if (!slave || slave->running) {
        return -1;
    }
    
    SlaveData* data = malloc(sizeof(SlaveData));
    if (!data) {
        return -2;
    }
    
    memset(data, 0, sizeof(SlaveData));
    data->ctx = slave;
    data->slave_idx = slave_idx;
    
    slave->thread_data = data;
    slave->running = true;
    
    if (pthread_create(&slave->thread, NULL, slave_thread_fn, data) != 0) {
        slave->running = false;
        free(data);
        return -3;
    }
    
    return 0;
}

void stop_slave(SdrCtx* slave) {
    if (!slave || !slave->running) return;
    
    slave->running = false;
    pthread_join(slave->thread, NULL);
    slave->thread_data = NULL;
}