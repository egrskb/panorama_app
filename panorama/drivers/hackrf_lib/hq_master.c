// hq_master.c
#include "hq_master.h"
#include "hq_rssi.h"
#include "hq_grouping.h"
#include "hq_init.h"  // Для функций спектра
#include <libhackrf/hackrf.h>  // Для hackrf_transfer
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Добавляем недостающие определения из hackrf_sweep.c
#define BYTES_PER_BLOCK 16384
#define MAX_SWEEP_RANGES 10

// Глобальные переменные
extern double g_grouping_tolerance_hz;

typedef struct {
    SdrCtx* ctx;
    SweepPlan plan;
    volatile int* rx_count;
    volatile float* last_rssi;  // Добавляем указатель на последний RSSI
} MasterData;

static int master_rx_callback(hackrf_transfer* transfer) {
    MasterData* data = (MasterData*)transfer->rx_ctx;
    SdrCtx* ctx = data->ctx;
    
    if (!ctx->running) {
        return -1;  // Stop receiving
    }
    
    // Estimate RSSI from received data
    float rssi = rssi_estimate_power((const int8_t*)transfer->buffer, transfer->valid_length / 2);
    
    // If signal is above threshold, add to peak queue
    if (rssi > -80.0f) {
        add_peak(ctx->center_hz, rssi);
    }
    
    // Увеличиваем счетчик принятых пакетов
    if (data->rx_count) {
        (*data->rx_count)++;
    }
    
    // Сохраняем последний RSSI
    if (data->last_rssi) {
        *data->last_rssi = rssi;
    }
    
    // Логируем для отладки
    if (transfer->valid_length > 0) {
        printf("RX callback: %d bytes, rssi=%.1f dBm, center=%.1f MHz\n", 
               transfer->valid_length, rssi, ctx->center_hz/1e6);
    }
    
    return 0;  // Continue receiving
}

static void* master_thread_fn(void* arg) {
    MasterData* data = (MasterData*)arg;
    SdrCtx* ctx = data->ctx;
    SweepPlan* plan = &data->plan;
    
    printf("Master thread started: %.1f-%.1f MHz, step %.1f MHz\n", 
           plan->start_hz/1e6, plan->stop_hz/1e6, plan->step_hz/1e6);
    
    // Вычисляем количество точек для полного спектра
    int n_points = (int)((plan->stop_hz - plan->start_hz) / plan->step_hz) + 1;
    if (n_points > MAX_SPECTRUM_POINTS) {
        n_points = MAX_SPECTRUM_POINTS;
        printf("Limiting spectrum to %d points for stability\n", n_points);
    }
    
    // Создаем буферы для полного спектра
    double* freqs = malloc(n_points * sizeof(double));
    float* powers = malloc(n_points * sizeof(float));
    
    if (!freqs || !powers) {
        fprintf(stderr, "Failed to allocate spectrum buffers\n");
        return NULL;
    }
    
    // Инициализируем частоты
    for (int i = 0; i < n_points; i++) {
        freqs[i] = plan->start_hz + i * plan->step_hz;
        powers[i] = -120.0f;  // Начальный уровень шума
    }
    
    double current_freq = plan->start_hz;
    struct timespec ts_start, ts_end;
    volatile int rx_count = 0;
    volatile float last_rssi = -120.0f;
    data->rx_count = &rx_count;
    data->last_rssi = &last_rssi;
    
    // Start RX
    printf("Starting RX on device %p...\n", (void*)ctx->dev);
    int r = hackrf_start_rx(ctx->dev, master_rx_callback, data);
    if (r != HACKRF_SUCCESS) {
        fprintf(stderr, "Master: hackrf_start_rx failed: %s (%d)\n", hackrf_error_name(r), r);
        free(freqs);
        free(powers);
        return NULL;
    }
    
    printf("✓ RX started successfully\n");
    printf("Generating continuous spectrum with %d points\n", n_points);
    
    // Небольшая задержка для стабилизации RX
    usleep(100000);  // 100ms
    
    while (ctx->running) {
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        rx_count = 0;
        last_rssi = -120.0f;  // Сбрасываем RSSI для новой частоты
        
        // Retune to new frequency
        ctx->center_hz = current_freq;
        printf("Tuning to %.1f MHz...\n", current_freq/1e6);
        r = hackrf_set_freq(ctx->dev, (uint64_t)current_freq);
        if (r != HACKRF_SUCCESS) {
            fprintf(stderr, "Master retune failed: %s (%d)\n", hackrf_error_name(r), r);
            continue;
        }
        printf("✓ Tuned to %.1f MHz\n", current_freq/1e6);
        
        // Wait for dwell time
        usleep(plan->dwell_ms * 1000);
        
        // Находим индекс текущей частоты в спектре
        int freq_idx = (int)((current_freq - plan->start_hz) / plan->step_hz);
        if (freq_idx >= 0 && freq_idx < n_points) {
            // Оцениваем мощность на текущей частоте
            float rssi = -120.0f;
            
            // Используем rx_count для оценки активности
            if (rx_count > 0) {
                // Используем реальный RSSI из callback
                if (last_rssi > -120.0f) {
                    rssi = last_rssi;  // Используем реальный RSSI
                    printf("Frequency %.1f MHz: rx_count=%d, real_rssi=%.1f dBm (ACTIVE)\n", 
                           current_freq/1e6, rx_count, rssi);
                } else {
                    rssi = -50.0f;  // Fallback значение
                    printf("Frequency %.1f MHz: rx_count=%d, fallback_rssi=%.1f dBm (ACTIVE)\n", 
                           current_freq/1e6, rx_count, rssi);
                }
            } else {
                printf("Frequency %.1f MHz: no data received (rx_count=0)\n", current_freq/1e6);
            }
            
            // Обновляем спектр
            powers[freq_idx] = rssi;
            
            // Обновляем глобальный буфер спектра ПОСЛЕ каждого измерения
            hq_update_spectrum(freqs, powers, n_points);
            
            // Логируем каждые 20 МГц для отладки
            if (freq_idx % 100 == 0) {
                printf("Updated spectrum at %.1f MHz (%.1f dBm, rx_count=%d)\n", 
                       current_freq/1e6, rssi, rx_count);
            }
        }
        
        // Process accumulated peaks
        regroup_frequencies(g_grouping_tolerance_hz);
        
        // Next frequency
        current_freq += plan->step_hz;
        if (current_freq > plan->stop_hz) {
            // Завершили полный проход
            printf("Completed full spectrum sweep: %d points\n", n_points);
            current_freq = plan->start_hz;  // Wrap around
        }
        
        // Update statistics
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        uint64_t elapsed_ns = (ts_end.tv_sec - ts_start.tv_sec) * 1000000000ULL +
                             (ts_end.tv_nsec - ts_start.tv_nsec);
        ctx->last_retune_ns = elapsed_ns;
        ctx->retune_count++;
    }
    
    // Stop RX
    hackrf_stop_rx(ctx->dev);
    
    // Очищаем память
    free(freqs);
    free(powers);
    
    printf("Master thread stopped\n");
    free(data);
    return NULL;
}

int start_master(SdrCtx* master, const SweepPlan* plan) {
    if (!master || !plan || master->running) {
        return -1;
    }
    
    MasterData* data = malloc(sizeof(MasterData));
    if (!data) {
        return -2;
    }
    
    data->ctx = master;
    data->plan = *plan;
    master->thread_data = data;
    master->running = true;
    
    if (pthread_create(&master->thread, NULL, master_thread_fn, data) != 0) {
        master->running = false;
        free(data);
        return -3;
    }
    
    return 0;
}

void stop_master(SdrCtx* master) {
    if (!master || !master->running) return;
    
    master->running = false;
    pthread_join(master->thread, NULL);
    master->thread_data = NULL;
}