#include "hq_init.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

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