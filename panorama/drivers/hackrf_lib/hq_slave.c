// hq_slave.c
#include "hq_slave.h"
#include "hq_rssi.h"
#include "hq_grouping.h"
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    SdrCtx* ctx;
    int slave_idx;
    volatile int* rx_count;
} SlaveData;

static int slave_rx_callback(hackrf_transfer* transfer) {
    SlaveData* data = (SlaveData*)transfer->rx_ctx;
    SdrCtx* ctx = data->ctx;
    
    if (!ctx->running) {
        return -1;  // Stop receiving
    }
    
    // Estimate RSSI
    float rssi = rssi_estimate_power(transfer->buffer, transfer->valid_length / 2);
    
    // Update watchlist items in current band
    pthread_mutex_lock(&g_watchlist_mutex);
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t now_ns = ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    
    for (size_t i = 0; i < g_watchlist_count; i++) {
        // Check if target is within ±4 MHz of current center
        if (fabs(g_watchlist[i].f_center_hz - ctx->center_hz) < 4e6) {
            // Simple model: adjust RSSI based on offset from center
            float offset_mhz = (g_watchlist[i].f_center_hz - ctx->center_hz) / 1e6;
            float adjusted_rssi = rssi - fabs(offset_mhz) * 0.5f;
            
            // Update with EMA
            g_watchlist[i].rssi_ema = rssi_apply_ema(
                g_watchlist[i].rssi_ema, adjusted_rssi, g_ema_alpha
            );
            g_watchlist[i].last_ns = now_ns;
            g_watchlist[i].hit_count++;
        }
    }
    
    pthread_mutex_unlock(&g_watchlist_mutex);
    
    (*data->rx_count)++;
    
    return 0;
}

static void* slave_thread_fn(void* arg) {
    SlaveData* data = (SlaveData*)arg;
    SdrCtx* ctx = data->ctx;
    int slave_idx = data->slave_idx;
    
    printf("Slave %d thread started\n", slave_idx);
    
    struct timespec ts_start, ts_end;
    double current_center = 0;
    WatchItem local_targets[MAX_WATCHLIST];
    size_t num_targets = 0;
    volatile int rx_count = 0;
    data->rx_count = &rx_count;
    
    // Start RX
    int r = hackrf_start_rx(ctx->dev, slave_rx_callback, data);
    if (r != HACKRF_SUCCESS) {
        fprintf(stderr, "Slave %d: hackrf_start_rx failed: %s\n", 
                slave_idx, hackrf_error_name(r));
        return NULL;
    }
    
    while (ctx->running) {
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        rx_count = 0;
        
        // Get snapshot of targets to monitor
        pthread_mutex_lock(&g_watchlist_mutex);
        num_targets = g_watchlist_count < MAX_WATCHLIST ? g_watchlist_count : MAX_WATCHLIST;
        if (num_targets > 0) {
            memcpy(local_targets, g_watchlist, num_targets * sizeof(WatchItem));
        }
        pthread_mutex_unlock(&g_watchlist_mutex);
        
        if (num_targets == 0) {
            usleep(50000);  // 50ms if no targets
            continue;
        }
        
        // Find best center frequency to cover maximum targets
        double best_center = 0;
        int max_coverage = 0;
        
        for (size_t i = 0; i < num_targets; i++) {
            double f = local_targets[i].f_center_hz;
            int coverage = 0;
            
            // Count how many targets fall within ±4 MHz
            for (size_t j = 0; j < num_targets; j++) {
                if (fabs(local_targets[j].f_center_hz - f) < 4e6) {
                    coverage++;
                }
            }
            
            if (coverage > max_coverage) {
                max_coverage = coverage;
                best_center = f;
            }
        }
        
        // Retune if necessary
        if (best_center > 0 && fabs(current_center - best_center) > 100e3) {
            current_center = best_center;
            ctx->center_hz = current_center;
            
            r = hackrf_set_freq(ctx->dev, (uint64_t)current_center);
            if (r != HACKRF_SUCCESS) {
                fprintf(stderr, "Slave %d retune failed: %s\n", 
                        slave_idx, hackrf_error_name(r));
                continue;
            }
            
            usleep(2000);  // 2ms for stabilization
        }
        
        // Wait for some data collection
        usleep(20000);  // 20ms
        
        // Update statistics
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        uint64_t elapsed_ns = (ts_end.tv_sec - ts_start.tv_sec) * 1000000000ULL +
                             (ts_end.tv_nsec - ts_start.tv_nsec);
        ctx->last_retune_ns = elapsed_ns;
        ctx->retune_count++;
        
        // Target 50 Hz update rate
        if (elapsed_ns < 20000000) {  // Less than 20ms
            usleep((20000000 - elapsed_ns) / 1000);
        }
    }
    
    // Stop RX
    hackrf_stop_rx(ctx->dev);
    
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