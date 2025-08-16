// hq_grouping.c
#include "hq_grouping.h"
#include "hq_rssi.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

PeakQueue* peak_queue_create(size_t capacity) {
    PeakQueue* q = malloc(sizeof(PeakQueue));
    if (!q) return NULL;
    
    q->buffer = malloc(capacity * sizeof(Peak));
    if (!q->buffer) {
        free(q);
        return NULL;
    }
    
    q->capacity = capacity;
    q->head = 0;
    q->tail = 0;
    pthread_mutex_init(&q->mutex, NULL);
    
    return q;
}

void peak_queue_destroy(PeakQueue* q) {
    if (!q) return;
    
    pthread_mutex_destroy(&q->mutex);
    free(q->buffer);
    free(q);
}

int peak_queue_push(PeakQueue* q, const Peak* peak) {
    if (!q || !peak) return -1;
    
    pthread_mutex_lock(&q->mutex);
    
    size_t next = (q->tail + 1) % q->capacity;
    if (next == q->head) {
        // Queue full - overwrite oldest
        q->head = (q->head + 1) % q->capacity;
    }
    
    q->buffer[q->tail] = *peak;
    q->tail = next;
    
    pthread_mutex_unlock(&q->mutex);
    return 0;
}

int peak_queue_pop(PeakQueue* q, Peak* peak) {
    if (!q || !peak) return -1;
    
    pthread_mutex_lock(&q->mutex);
    
    if (q->head == q->tail) {
        // Queue empty
        pthread_mutex_unlock(&q->mutex);
        return -1;
    }
    
    *peak = q->buffer[q->head];
    q->head = (q->head + 1) % q->capacity;
    
    pthread_mutex_unlock(&q->mutex);
    return 0;
}

size_t peak_queue_size(PeakQueue* q) {
    if (!q) return 0;
    
    pthread_mutex_lock(&q->mutex);
    size_t size = (q->tail >= q->head) ? 
                  (q->tail - q->head) : 
                  (q->capacity - q->head + q->tail);
    pthread_mutex_unlock(&q->mutex);
    
    return size;
}

void add_peak(double f_hz, float rssi_dbm) {
    if (!g_peaks_queue) return;
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    
    Peak peak = {
        .f_hz = f_hz,
        .rssi_dbm = rssi_dbm,
        .last_ns = ts.tv_sec * 1000000000ULL + ts.tv_nsec
    };
    
    peak_queue_push(g_peaks_queue, &peak);
}

void regroup_frequencies(double delta_hz) {
    if (!g_peaks_queue) return;
    
    Peak peaks[1000];
    int n_peaks = 0;
    
    // Collect all peaks from queue
    Peak p;
    while (peak_queue_pop(g_peaks_queue, &p) == 0 && n_peaks < 1000) {
        peaks[n_peaks++] = p;
    }
    
    if (n_peaks == 0) return;
    
    // Simple bubble sort for small arrays
    for (int i = 0; i < n_peaks - 1; i++) {
        for (int j = 0; j < n_peaks - i - 1; j++) {
            if (peaks[j].f_hz > peaks[j+1].f_hz) {
                Peak tmp = peaks[j];
                peaks[j] = peaks[j+1];
                peaks[j+1] = tmp;
            }
        }
    }
    
    pthread_mutex_lock(&g_watchlist_mutex);
    
    // Group nearby frequencies
    for (int i = 0; i < n_peaks; i++) {
        // Find existing WatchItem
        int found_idx = -1;
        for (size_t j = 0; j < g_watchlist_count; j++) {
            if (fabs(g_watchlist[j].f_center_hz - peaks[i].f_hz) < delta_hz) {
                found_idx = j;
                break;
            }
        }
        
        if (found_idx >= 0) {
            // Update existing
            WatchItem* item = &g_watchlist[found_idx];
            item->rssi_ema = rssi_apply_ema(item->rssi_ema, peaks[i].rssi_dbm, g_ema_alpha);
            item->last_ns = peaks[i].last_ns;
            item->hit_count++;
        } else if (g_watchlist_count < MAX_WATCHLIST) {
            // Add new
            WatchItem* item = &g_watchlist[g_watchlist_count++];
            item->f_center_hz = peaks[i].f_hz;
            item->bw_hz = 8e6;  // Default 8 MHz
            item->rssi_ema = peaks[i].rssi_dbm;
            item->last_ns = peaks[i].last_ns;
            item->hit_count = 1;
        }
    }
    
    // Remove stale entries (TTL > 5 seconds)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t now_ns = ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    uint64_t ttl_ns = 5000000000ULL;  // 5 seconds
    
    size_t write_idx = 0;
    for (size_t i = 0; i < g_watchlist_count; i++) {
        if ((now_ns - g_watchlist[i].last_ns) < ttl_ns) {
            if (write_idx != i) {
                g_watchlist[write_idx] = g_watchlist[i];
            }
            write_idx++;
        }
    }
    g_watchlist_count = write_idx;
    
    pthread_mutex_unlock(&g_watchlist_mutex);
}