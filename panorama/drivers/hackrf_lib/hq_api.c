// hq_api.c
#include "hq_api.h"
#include "hq_init.h"
#include "hq_master.h"
#include "hq_slave.h"
#include "hq_grouping.h"
#include "hq_rssi.h"
#include "hq_scheduler.h"
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

// Global state
static SdrCtx g_devs[MAX_DEVICES];
static int g_num_devs = 0;
static _Atomic bool g_running = false;

// Queues and lists
extern PeakQueue* g_peaks_queue;
extern WatchItem g_watchlist[];
extern size_t g_watchlist_count;
extern pthread_mutex_t g_watchlist_mutex;
extern double g_grouping_tolerance_hz;
extern float g_ema_alpha;

// Background grouping thread
static pthread_t g_grouping_thread;
static _Atomic bool g_grouping_running = false;

// Parameters
static SdrParams g_config = {
    .samp_rate = 12000000,
    .bb_filter_bw = 8000000,
    .lna_db = 24,
    .vga_db = 20,
    .amp_on = false,
    .dc_offset_enable = true,    // По умолчанию включаем DC offset
    .dc_offset_auto = true       // По умолчанию автоматическая коррекция
};

// Глобальные переменные для конфигурации диапазона
static double g_sweep_start_hz = 2400000000.0;  // 2.4 GHz по умолчанию
static double g_sweep_stop_hz = 2500000000.0;   // 2.5 GHz по умолчанию
static double g_sweep_step_hz = 5000000.0;      // 5 MHz по умолчанию
static uint32_t g_sweep_dwell_ms = 2;           // 2 ms по умолчанию

// Background grouping thread function
static void* grouping_thread_fn(void* arg) {
    (void)arg;  // Unused
    
    printf("Grouping thread started\n");
    
    while (g_grouping_running) {
        // Process peaks every 100ms
        usleep(100000);
        
        // Regroup frequencies
        regroup_frequencies(g_grouping_tolerance_hz);
    }
    
    printf("Grouping thread stopped\n");
    return NULL;
}

int hq_open_all(int num_expected) {
    if (g_num_devs > 0) {
        fprintf(stderr, "Devices already open\n");
        return -EBUSY;
    }
    
    if (num_expected < 1 || num_expected > MAX_DEVICES) {
        fprintf(stderr, "Invalid number of devices: %d\n", num_expected);
        return -EINVAL;
    }
    
    printf("Opening %d devices...\n", num_expected);
    
    int r = init_devices(g_devs, num_expected, &g_config);
    if (r == 0) {
        g_num_devs = num_expected;
        
        // Create peak queue
        g_peaks_queue = peak_queue_create(4096);
        if (!g_peaks_queue) {
            fprintf(stderr, "Failed to create peak queue\n");
            teardown_devices(g_devs, g_num_devs);
            g_num_devs = 0;
            return -ENOMEM;
        }
        
        // Start grouping thread
        g_grouping_running = true;
        if (pthread_create(&g_grouping_thread, NULL, grouping_thread_fn, NULL) != 0) {
            fprintf(stderr, "Failed to create grouping thread\n");
            peak_queue_destroy(g_peaks_queue);
            g_peaks_queue = NULL;
            teardown_devices(g_devs, g_num_devs);
            g_num_devs = 0;
            g_grouping_running = false;
            return -1;
        }
        
        printf("Successfully opened %d devices\n", num_expected);
    } else {
        fprintf(stderr, "Failed to initialize devices: %d\n", r);
    }
    
    return r;
}

void hq_close_all(void) {
    printf("Closing all devices...\n");
    
    hq_stop();
    
    // Stop grouping thread
    if (g_grouping_running) {
        g_grouping_running = false;
        pthread_join(g_grouping_thread, NULL);
    }
    
    // Destroy peak queue
    if (g_peaks_queue) {
        peak_queue_destroy(g_peaks_queue);
        g_peaks_queue = NULL;
    }
    
    // Clear watchlist
    pthread_mutex_lock(&g_watchlist_mutex);
    g_watchlist_count = 0;
    pthread_mutex_unlock(&g_watchlist_mutex);
    
    // Close devices
    teardown_devices(g_devs, g_num_devs);
    g_num_devs = 0;
    
    printf("All devices closed\n");
}

int hq_config_set_rates(uint32_t samp_rate_hz, uint32_t bb_bw_hz) {
    if (g_running) {
        fprintf(stderr, "Cannot change rates while running\n");
        return -EBUSY;
    }
    
    g_config.samp_rate = samp_rate_hz;
    g_config.bb_filter_bw = bb_bw_hz;
    
    // Apply to opened devices
    for (int i = 0; i < g_num_devs; i++) {
        if (g_devs[i].dev) {
            hackrf_set_sample_rate(g_devs[i].dev, samp_rate_hz);
            set_bb_filter(g_devs[i].dev, bb_bw_hz);
            g_devs[i].samp_rate = samp_rate_hz;
        }
    }
    
    printf("Rates set: sample=%u Hz, BB filter=%u Hz\n", samp_rate_hz, bb_bw_hz);
    return 0;
}

int hq_config_set_gains(uint32_t lna_db, uint32_t vga_db, bool amp_on) {
    if (g_running) {
        fprintf(stderr, "Cannot change gains while running\n");
        return -EBUSY;
    }
    
    // Clamp gains
    lna_db = lna_db < 0 ? 0 : (lna_db > 40 ? 40 : lna_db - (lna_db % 8));
    vga_db = vga_db < 0 ? 0 : (vga_db > 62 ? 62 : vga_db - (vga_db % 2));
    
    g_config.lna_db = lna_db;
    g_config.vga_db = vga_db;
    g_config.amp_on = amp_on;
    
    // Apply to devices
    for (int i = 0; i < g_num_devs; i++) {
        if (g_devs[i].dev) {
            hackrf_set_lna_gain(g_devs[i].dev, lna_db);
            hackrf_set_vga_gain(g_devs[i].dev, vga_db);
            hackrf_set_amp_enable(g_devs[i].dev, amp_on ? 1 : 0);
            
            g_devs[i].lna_db = lna_db;
            g_devs[i].vga_db = vga_db;
            g_devs[i].amp_on = amp_on;
        }
    }
    
    printf("Gains set: LNA=%u dB, VGA=%u dB, AMP=%s\n", 
           lna_db, vga_db, amp_on ? "ON" : "OFF");
    return 0;
}

int hq_start(void) {
    if (g_running) {
        fprintf(stderr, "Already running\n");
        return -EBUSY;
    }
    
    if (g_num_devs < 1) {
        fprintf(stderr, "No devices open\n");
        return -EINVAL;
    }
    
    printf("Starting multi-SDR operation with %d devices...\n", g_num_devs);
    
    g_running = true;
    
    // Start master sweep (device 0)
    if (g_num_devs >= 1) {
        SweepPlan plan = {
            .start_hz = g_sweep_start_hz,
            .stop_hz = g_sweep_stop_hz,
            .step_hz = g_sweep_step_hz,
            .dwell_ms = g_sweep_dwell_ms
        };
        
        int r = start_master(&g_devs[MASTER_IDX], &plan);
        if (r != 0) {
            fprintf(stderr, "Failed to start master: %d\n", r);
            g_running = false;
            return r;
        }
        printf("Master started\n");
    }
    
    // Start slaves (devices 1 and 2)
    for (int i = 1; i < g_num_devs && i < MAX_DEVICES; i++) {
        int r = start_slave(&g_devs[i], i);
        if (r != 0) {
            fprintf(stderr, "Failed to start slave %d: %d\n", i, r);
            
            // Stop already started devices
            for (int j = 0; j < i; j++) {
                if (g_devs[j].running) {
                    g_devs[j].running = false;
                    pthread_join(g_devs[j].thread, NULL);
                }
            }
            
            g_running = false;
            return r;
        }
        printf("Slave %d started\n", i);
    }
    
    printf("All devices started successfully\n");
    return 0;
}

void hq_stop(void) {
    if (!g_running) {
        return;
    }
    
    printf("Stopping all devices...\n");
    
    g_running = false;
    
    // Stop all devices
    for (int i = 0; i < g_num_devs; i++) {
        if (g_devs[i].running) {
            g_devs[i].running = false;
            
            // Wait for thread to finish
            pthread_join(g_devs[i].thread, NULL);
            
            printf("Device %d stopped\n", i);
        }
    }
    
    printf("All devices stopped\n");
}

int hq_get_watchlist_snapshot(WatchItem* out, int max_items) {
    if (!out || max_items <= 0) return 0;
    
    pthread_mutex_lock(&g_watchlist_mutex);
    
    int count = (g_watchlist_count < (size_t)max_items) ? g_watchlist_count : max_items;
    if (count > 0) {
        memcpy(out, g_watchlist, count * sizeof(WatchItem));
    }
    
    pthread_mutex_unlock(&g_watchlist_mutex);
    
    return count;
}

int hq_get_recent_peaks(Peak* out, int max_items) {
    if (!out || max_items <= 0 || !g_peaks_queue) return 0;
    
    int count = 0;
    Peak peak;
    
    while (count < max_items && peak_queue_pop(g_peaks_queue, &peak) == 0) {
        out[count++] = peak;
    }
    
    return count;
}

int hq_get_master_spectrum(double* freqs_hz, float* powers_dbm, int max_points) {
    if (!freqs_hz || !powers_dbm || max_points <= 0) {
        printf("hq_get_master_spectrum: Invalid parameters\n");
        return -1;
    }
    
    // Используем функцию из hq_init.c
    extern int hq_get_spectrum(double* freqs, float* powers, int max_points);
    int result = hq_get_spectrum(freqs_hz, powers_dbm, max_points);
    
    printf("hq_get_master_spectrum: returned %d points\n", result);
    if (result > 0) {
        printf("First few values: freq[0]=%.1f MHz, power[0]=%.1f dBm\n", 
               freqs_hz[0]/1e6, powers_dbm[0]);
    }
    
    return result;
}

void hq_set_grouping_tolerance_hz(double delta_hz) {
    if (delta_hz > 0) {
        g_grouping_tolerance_hz = delta_hz;
        printf("Grouping tolerance set to %.0f Hz\n", delta_hz);
    }
}

void hq_set_ema_alpha(float alpha) {
    if (alpha > 0.0f && alpha <= 1.0f) {
        g_ema_alpha = alpha;
        printf("EMA alpha set to %.2f\n", alpha);
    }
}

int hq_get_status(HqStatus* out) {
    if (!out) return -EINVAL;
    
    memset(out, 0, sizeof(HqStatus));
    
    // Check device status
    out->master_running = (g_num_devs > 0 && g_devs[MASTER_IDX].running) ? 1 : 0;
    
    for (int i = 1; i < g_num_devs && i < 3; i++) {
        out->slave_running[i-1] = g_devs[i].running ? 1 : 0;
    }
    
    // Calculate average retune time
    uint64_t total_retune_ns = 0;
    int retune_count = 0;
    
    for (int i = 0; i < g_num_devs; i++) {
        if (g_devs[i].retune_count > 0) {
            total_retune_ns += g_devs[i].last_retune_ns;
            retune_count++;
        }
    }
    
    if (retune_count > 0) {
        out->retune_ms_avg = (double)(total_retune_ns / retune_count) / 1e6;
    }
    
    // Get watchlist count
    pthread_mutex_lock(&g_watchlist_mutex);
    out->watch_items = g_watchlist_count;
    pthread_mutex_unlock(&g_watchlist_mutex);
    
    return 0;
}

int hq_get_device_count(void) {
    // Просто возвращаем количество устройств, которые мы можем открыть
    return MAX_DEVICES;
}

int hq_list_devices(char* serials[], int max_count) {
    if (!serials || max_count <= 0) return 0;
    
    int r = hackrf_init();
    if (r != HACKRF_SUCCESS) {
        return 0;
    }
    
    hackrf_device_list_t* list = hackrf_device_list();
    if (!list) {
        hackrf_exit();
        return 0;
    }
    
    int count = (list->devicecount < max_count) ? list->devicecount : max_count;
    
    for (int i = 0; i < count; i++) {
        const char* serial = list->serial_numbers[i];
        if (serial && strlen(serial) > 0) {
            // Выделяем память и копируем серийник
            serials[i] = strdup(serial);
        } else {
            serials[i] = NULL;
        }
    }
    
    hackrf_device_list_free(list);
    hackrf_exit();
    
    return count;
}

int hq_config_set_freq_range(double start_hz, double stop_hz, double step_hz) {
    if (start_hz < 0 || stop_hz <= start_hz || step_hz <= 0) {
        return -EINVAL;
    }
    
    g_sweep_start_hz = start_hz;
    g_sweep_stop_hz = stop_hz;
    g_sweep_step_hz = step_hz;
    
    printf("Frequency range configured: %.0f - %.0f Hz, step %.0f Hz\n", 
           start_hz, stop_hz, step_hz);
    return 0;
}

int hq_config_set_dwell_time(uint32_t dwell_ms) {
    if (dwell_ms < 1 || dwell_ms > 1000) {
        return -EINVAL;
    }
    
    g_sweep_dwell_ms = dwell_ms;
    printf("Dwell time set to %u ms\n", dwell_ms);
    return 0;
}

// DC Offset функции
int hq_config_set_dc_offset(bool enable, bool auto_correct, int16_t i_offset, int16_t q_offset) {
    if (g_num_devs == 0) {
        fprintf(stderr, "No devices open\n");
        return -ENODEV;
    }
    
    // Обновляем глобальную конфигурацию
    g_config.dc_offset_enable = enable;
    g_config.dc_offset_auto = auto_correct;
    
    // Применяем настройки ко всем открытым устройствам
    for (int i = 0; i < g_num_devs; i++) {
        if (g_devs[i].dev) {
            int r = set_dc_offset(g_devs[i].dev, enable, auto_correct, i_offset, q_offset);
            if (r != HACKRF_SUCCESS) {
                fprintf(stderr, "Device %d: set_dc_offset failed: %s\n", 
                        i, hackrf_error_name(r));
                return r;
            }
            
            // Обновляем локальные настройки устройства
            g_devs[i].dc_offset_enable = enable;
            g_devs[i].dc_offset_auto = auto_correct;
            if (!auto_correct) {
                g_devs[i].dc_offset_i = i_offset;
                g_devs[i].dc_offset_q = q_offset;
            }
        }
    }
    
    printf("DC offset configured: enable=%d, auto=%d, I=%d, Q=%d\n", 
           enable, auto_correct, i_offset, q_offset);
    return 0;
}

int hq_config_calibrate_dc_offset(int device_idx) {
    if (g_num_devs == 0) {
        fprintf(stderr, "No devices open\n");
        return -ENODEV;
    }
    
    if (device_idx < 0 || device_idx >= g_num_devs) {
        fprintf(stderr, "Invalid device index: %d\n", device_idx);
        return -EINVAL;
    }
    
    if (!g_devs[device_idx].dev) {
        fprintf(stderr, "Device %d not open\n", device_idx);
        return -ENODEV;
    }
    
    // Выполняем калибровку DC offset
    int16_t i_offset, q_offset;
    int r = calibrate_dc_offset(g_devs[device_idx].dev, &i_offset, &q_offset);
    if (r != HACKRF_SUCCESS) {
        fprintf(stderr, "Device %d: DC offset calibration failed: %s\n", 
                device_idx, hackrf_error_name(r));
        return r;
    }
    
    // Обновляем локальные настройки устройства
    g_devs[device_idx].dc_offset_i = i_offset;
    g_devs[device_idx].dc_offset_q = q_offset;
    
    printf("Device %d: DC offset calibrated (I:%d, Q:%d)\n", 
           device_idx, i_offset, q_offset);
    return 0;
}