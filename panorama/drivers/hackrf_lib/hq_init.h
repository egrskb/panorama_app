#ifndef HQ_INIT_H
#define HQ_INIT_H

#include <libhackrf/hackrf.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#define MAX_DEVICES 3
#define MASTER_IDX 0
#define SLAVE1_IDX 1
#define SLAVE2_IDX 2

typedef struct {
    uint32_t samp_rate;
    uint32_t bb_filter_bw;
    uint32_t lna_db;
    uint32_t vga_db;
    bool amp_on;
} SdrParams;

typedef struct {
    hackrf_device* dev;
    pthread_t thread;
    _Atomic bool running;
    
    double center_hz;
    uint32_t samp_rate;
    uint32_t lna_db;
    uint32_t vga_db;
    bool amp_on;
    
    // Буферы и статистика
    int8_t* rx_buffer;
    size_t buffer_size;
    uint64_t retune_count;
    uint64_t last_retune_ns;
    
    // Для передачи данных в поток
    void* thread_data;
} SdrCtx;

int init_devices(SdrCtx* devs, int n, const SdrParams* defaults);
void teardown_devices(SdrCtx* devs, int n);
int set_bb_filter(hackrf_device* dev, uint32_t bw_hz);

#endif // HQ_INIT_H