// hq_master.c
#include "hq_master.h"
#include "hq_rssi.h"
#include "hq_grouping.h"
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    SdrCtx* ctx;
    SweepPlan plan;
    volatile int* rx_count;
} MasterData;

static int master_rx_callback(hackrf_transfer* transfer) {
    MasterData* data = (MasterData*)transfer->rx_ctx;
    SdrCtx* ctx = data->ctx;
    
    if (!ctx->running) {
        return -1;  // Stop receiving
    }
    
    // Estimate RSSI from received data
    float rssi = rssi_estimate_power(transfer->buffer, transfer->valid_length / 2);
    
    // If signal is above threshold, add to peak queue
    if (rssi > -80.0f) {
        add_peak(ctx->center_hz, rssi);
    }
    
    (*data->rx_count)++;
    
    return 0;  // Continue receiving
}

static void* master_thread_fn(void* arg) {
    MasterData* data = (MasterData*)arg;
    SdrCtx* ctx = data->ctx;
    SweepPlan* plan = &data->plan;
    
    printf("Master thread started: %.1f-%.1f MHz, step %.1f MHz\n", 
           plan->start_hz/1e6, plan->stop_hz/1e6, plan->step_hz/1e6);
    
    double current_freq = plan->start_hz;
    struct timespec ts_start, ts_end;
    volatile int rx_count = 0;
    data->rx_count = &rx_count;
    
    // Start RX
    int r = hackrf_start_rx(ctx->dev, master_rx_callback, data);
    if (r != HACKRF_SUCCESS) {
        fprintf(stderr, "Master: hackrf_start_rx failed: %s\n", hackrf_error_name(r));
        return NULL;
    }
    
    while (ctx->running) {
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        rx_count = 0;
        
        // Retune to new frequency
        ctx->center_hz = current_freq;
        r = hackrf_set_freq(ctx->dev, (uint64_t)current_freq);
        if (r != HACKRF_SUCCESS) {
            fprintf(stderr, "Master retune failed: %s\n", hackrf_error_name(r));
            continue;
        }
        
        // Wait for dwell time
        usleep(plan->dwell_ms * 1000);
        
        // Process accumulated peaks
        regroup_frequencies(g_grouping_tolerance_hz);
        
        // Next frequency
        current_freq += plan->step_hz;
        if (current_freq > plan->stop_hz) {
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