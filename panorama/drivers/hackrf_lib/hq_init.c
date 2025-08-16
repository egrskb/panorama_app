#include "hq_init.h"
#include "hq_grouping.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  // Для usleep

// Global variables
PeakQueue* g_peaks_queue = NULL;
WatchItem g_watchlist[MAX_WATCHLIST];
size_t g_watchlist_count = 0;
pthread_mutex_t g_watchlist_mutex = PTHREAD_MUTEX_INITIALIZER;
float g_ema_alpha = 0.1f;
double g_grouping_tolerance_hz = 1000000.0; // 1 MHz default

// Глобальные переменные для непрерывного спектра от Master SDR
#define MAX_SPECTRUM_POINTS 50000
static double g_spectrum_freqs[MAX_SPECTRUM_POINTS];
static float g_spectrum_powers[MAX_SPECTRUM_POINTS];
static int g_spectrum_points = 0;
static pthread_mutex_t g_spectrum_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool g_spectrum_ready = false;

static int clamp_lna(int lna) {
    lna = lna < 0 ? 0 : (lna > 40 ? 40 : lna);
    return lna - (lna % 8);
}

static int clamp_vga(int vga) {
    vga = vga < 0 ? 0 : (vga > 62 ? 62 : vga);
    return vga - (vga % 2);
}

int set_bb_filter(hackrf_device* dev, uint32_t bw_hz) {
    uint32_t actual_bw = hackrf_compute_baseband_filter_bw(bw_hz);
    return hackrf_set_baseband_filter_bandwidth(dev, actual_bw);
}

// Функция настройки DC offset
int set_dc_offset(hackrf_device* dev, bool enable, bool auto_correct, int16_t i_offset, int16_t q_offset) {
    if (!dev) return -1;
    
    // Проверяем доступность функций DC offset
    #ifdef HACKRF_HAVE_DC_OFFSET
    int r;
    
    // Включаем/выключаем DC offset
    r = hackrf_set_dc_offset_enable(dev, enable ? 1 : 0);
    if (r != HACKRF_SUCCESS) {
        fprintf(stderr, "set_dc_offset_enable failed: %s (%d)\n", hackrf_error_name(r), r);
        return r;
    }
    
    if (enable) {
        // Включаем/выключаем автоматическую коррекцию
        r = hackrf_set_dc_offset_auto(dev, auto_correct ? 1 : 0);
        if (r != HACKRF_SUCCESS) {
            fprintf(stderr, "set_dc_offset_auto failed: %s (%d)\n", hackrf_error_name(r), r);
            return r;
        }
        
        // Если автоматическая коррекция выключена, устанавливаем ручные значения
        if (!auto_correct) {
            r = hackrf_set_dc_offset_manual(dev, i_offset, q_offset);
            if (r != HACKRF_SUCCESS) {
                fprintf(stderr, "set_dc_offset_manual failed: %s (%d)\n", hackrf_error_name(r), r);
                return r;
            }
        }
    }
    
    return HACKRF_SUCCESS;
    #else
    // DC offset не поддерживается в этой версии libhackrf
    fprintf(stderr, "Warning: DC offset functions not available in this libhackrf version\n");
    return HACKRF_SUCCESS; // Возвращаем успех, но ничего не делаем
    #endif
}

// Функция калибровки DC offset
int calibrate_dc_offset(hackrf_device* dev, int16_t* i_offset, int16_t* q_offset) {
    if (!dev || !i_offset || !q_offset) return -1;
    
    #ifdef HACKRF_HAVE_DC_OFFSET
    // Включаем автоматическую калибровку
    int r = hackrf_set_dc_offset_auto(dev, 1);
    if (r != HACKRF_SUCCESS) {
        fprintf(stderr, "calibrate_dc_offset: set_dc_offset_auto failed: %s (%d)\n", hackrf_error_name(r), r);
        return r;
    }
    
    // Ждем немного для завершения калибровки
    usleep(100000);  // 100ms
    
    // Получаем текущие значения DC offset
    r = hackrf_get_dc_offset_manual(dev, i_offset, q_offset);
    if (r != HACKRF_SUCCESS) {
        fprintf(stderr, "calibrate_dc_offset: get_dc_offset_manual failed: %s (%d)\n", hackrf_error_name(r), r);
        return r;
    }
    
    printf("DC offset calibrated: I=%d, Q=%d\n", *i_offset, *q_offset);
    return HACKRF_SUCCESS;
    #else
    // DC offset не поддерживается в этой версии libhackrf
    fprintf(stderr, "Warning: DC offset calibration not available in this libhackrf version\n");
    *i_offset = 0;
    *q_offset = 0;
    return HACKRF_SUCCESS; // Возвращаем успех, но ничего не делаем
    #endif
}

int init_devices(SdrCtx* devs, int n, const SdrParams* defaults) {
    if (!devs || n <= 0 || n > MAX_DEVICES) {
        fprintf(stderr, "Invalid parameters: devs=%p, n=%d\n", devs, n);
        return -1;
    }
    
    int r = hackrf_init();
    if (r != HACKRF_SUCCESS) {
        fprintf(stderr, "hackrf_init failed: %s (%d)\n", hackrf_error_name(r), r);
        return r;
    }
    
    hackrf_device_list_t* list = hackrf_device_list();
    if (!list) {
        fprintf(stderr, "Failed to get device list\n");
        hackrf_exit();
        return -2;
    }
    
    if (list->devicecount < n) {
        fprintf(stderr, "Found %d devices, need %d\n", list->devicecount, n);
        hackrf_device_list_free(list);
        hackrf_exit();
        return -3;
    }
    
    memset(devs, 0, sizeof(SdrCtx) * n);
    
    for (int i = 0; i < n; i++) {
        const char* serial = list->serial_numbers[i];
        printf("Opening device %d: %s\n", i, serial);
        
        r = hackrf_open_by_serial(serial, &devs[i].dev);
        if (r != HACKRF_SUCCESS) {
            fprintf(stderr, "Failed to open device %s: %s (%d)\n", 
                    serial, hackrf_error_name(r), r);
            
            // Cleanup already opened devices
            for (int j = 0; j < i; j++) {
                if (devs[j].dev) {
                    hackrf_close(devs[j].dev);
                    devs[j].dev = NULL;
                }
            }
            hackrf_device_list_free(list);
            hackrf_exit();
            return r;
        }
        
        // Apply defaults
        devs[i].samp_rate = defaults->samp_rate;
        devs[i].lna_db = clamp_lna(defaults->lna_db);
        devs[i].vga_db = clamp_vga(defaults->vga_db);
        devs[i].amp_on = defaults->amp_on;
        
        // DC Offset настройки
        devs[i].dc_offset_enable = defaults->dc_offset_enable;
        devs[i].dc_offset_auto = defaults->dc_offset_auto;
        devs[i].dc_offset_i = 0;
        devs[i].dc_offset_q = 0;
        
        // Configure hardware
        r = hackrf_set_sample_rate(devs[i].dev, devs[i].samp_rate);
        if (r != HACKRF_SUCCESS) {
            fprintf(stderr, "Device %d: set_sample_rate failed: %s\n", 
                    i, hackrf_error_name(r));
        }
        
        r = set_bb_filter(devs[i].dev, defaults->bb_filter_bw);
        if (r != HACKRF_SUCCESS) {
            fprintf(stderr, "Device %d: set_bb_filter failed: %s\n", 
                    i, hackrf_error_name(r));
        }
        
        r = hackrf_set_lna_gain(devs[i].dev, devs[i].lna_db);
        if (r != HACKRF_SUCCESS) {
            fprintf(stderr, "Device %d: set_lna_gain failed: %s\n", 
                    i, hackrf_error_name(r));
        }
        
        r = hackrf_set_vga_gain(devs[i].dev, devs[i].vga_db);
        if (r != HACKRF_SUCCESS) {
            fprintf(stderr, "Device %d: set_vga_gain failed: %s\n", 
                    i, hackrf_error_name(r));
        }
        
        r = hackrf_set_amp_enable(devs[i].dev, devs[i].amp_on ? 1 : 0);
        if (r != HACKRF_SUCCESS) {
            fprintf(stderr, "Device %d: set_amp_enable failed: %s\n", 
                    i, hackrf_error_name(r));
        }
        
        // Настройка DC offset
        if (devs[i].dc_offset_enable) {
            if (devs[i].dc_offset_auto) {
                // Автоматическая калибровка DC offset
                r = calibrate_dc_offset(devs[i].dev, &devs[i].dc_offset_i, &devs[i].dc_offset_q);
                if (r != HACKRF_SUCCESS) {
                    fprintf(stderr, "Device %d: DC offset calibration failed: %s\n", 
                            i, hackrf_error_name(r));
                } else {
                    printf("Device %d: DC offset auto-calibrated (I:%d, Q:%d)\n", 
                           i, devs[i].dc_offset_i, devs[i].dc_offset_q);
                }
            } else {
                // Ручная настройка DC offset
                r = set_dc_offset(devs[i].dev, true, false, devs[i].dc_offset_i, devs[i].dc_offset_q);
                if (r != HACKRF_SUCCESS) {
                    fprintf(stderr, "Device %d: set_dc_offset failed: %s\n", 
                            i, hackrf_error_name(r));
                } else {
                    printf("Device %d: DC offset manually set (I:%d, Q:%d)\n", 
                           i, devs[i].dc_offset_i, devs[i].dc_offset_q);
                }
            }
        } else {
            // Отключаем DC offset
            r = set_dc_offset(devs[i].dev, false, false, 0, 0);
            if (r != HACKRF_SUCCESS) {
                fprintf(stderr, "Device %d: disable_dc_offset failed: %s\n", 
                        i, hackrf_error_name(r));
            } else {
                printf("Device %d: DC offset disabled\n", i);
            }
        }
        
        // Allocate RX buffer
        devs[i].buffer_size = 262144;  // 256KB
        devs[i].rx_buffer = malloc(devs[i].buffer_size);
        if (!devs[i].rx_buffer) {
            fprintf(stderr, "Device %d: Failed to allocate RX buffer\n", i);
        }
        
        devs[i].running = false;
        devs[i].retune_count = 0;
        devs[i].last_retune_ns = 0;
        
        printf("Device %d initialized: %s (LNA:%d VGA:%d AMP:%d)\n",
               i, serial, devs[i].lna_db, devs[i].vga_db, devs[i].amp_on);
    }
    
    hackrf_device_list_free(list);
    printf("All %d devices initialized successfully\n", n);
    return 0;
}

// Функции для работы с непрерывным спектром
void hq_update_spectrum(double* freqs, float* powers, int n_points) {
    pthread_mutex_lock(&g_spectrum_mutex);
    
    if (n_points > MAX_SPECTRUM_POINTS) {
        n_points = MAX_SPECTRUM_POINTS;
    }
    
    memcpy(g_spectrum_freqs, freqs, n_points * sizeof(double));
    memcpy(g_spectrum_powers, powers, n_points * sizeof(float));
    g_spectrum_points = n_points;
    g_spectrum_ready = true;
    
    pthread_mutex_unlock(&g_spectrum_mutex);
}

int hq_get_spectrum(double* freqs, float* powers, int max_points) {
    pthread_mutex_lock(&g_spectrum_mutex);
    
    if (!g_spectrum_ready) {
        pthread_mutex_unlock(&g_spectrum_mutex);
        printf("hq_get_spectrum: spectrum not ready\n");
        return 0;
    }
    
    int n_points = (g_spectrum_points < max_points) ? g_spectrum_points : max_points;
    printf("hq_get_spectrum: copying %d points (ready: %d, max: %d)\n", 
           n_points, g_spectrum_points, max_points);
    
    memcpy(freqs, g_spectrum_freqs, n_points * sizeof(double));
    memcpy(powers, g_spectrum_powers, n_points * sizeof(float));
    
    pthread_mutex_unlock(&g_spectrum_mutex);
    return n_points;
}

void teardown_devices(SdrCtx* devs, int n) {
    if (!devs) return;
    
    printf("Tearing down %d devices\n", n);
    
    for (int i = 0; i < n; i++) {
        if (devs[i].running) {
            devs[i].running = false;
            if (devs[i].dev) {
                hackrf_stop_rx(devs[i].dev);
            }
            pthread_join(devs[i].thread, NULL);
        }
        
        if (devs[i].dev) {
            hackrf_close(devs[i].dev);
            devs[i].dev = NULL;
        }
        
        if (devs[i].rx_buffer) {
            free(devs[i].rx_buffer);
            devs[i].rx_buffer = NULL;
        }
    }
    
    hackrf_exit();
    printf("All devices closed\n");
}