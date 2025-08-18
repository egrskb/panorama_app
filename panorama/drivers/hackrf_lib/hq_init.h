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
    bool dc_offset_enable;      // Включение/выключение DC offset
    bool dc_offset_auto;        // Автоматическая коррекция DC offset
} SdrParams;

typedef struct {
    hackrf_device* dev;
    pthread_t thread;
    bool running;
    pthread_mutex_t running_mutex;  // Мьютекс для синхронизации доступа к running
    
    double center_hz;
    uint32_t samp_rate;
    uint32_t lna_db;
    uint32_t vga_db;
    bool amp_on;
    
    // DC Offset настройки
    bool dc_offset_enable;
    bool dc_offset_auto;
    int16_t dc_offset_i;       // DC offset для I канала
    int16_t dc_offset_q;       // DC offset для Q канала
    
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

// DC Offset функции
int set_dc_offset(hackrf_device* dev, bool enable, bool auto_correct, int16_t i_offset, int16_t q_offset);
int calibrate_dc_offset(hackrf_device* dev, int16_t* i_offset, int16_t* q_offset);

// Функции для работы с непрерывным спектром
void hq_update_spectrum(double* freqs, float* powers, int n_points);
int hq_get_spectrum(double* freqs, float* powers, int max_points);

// Mutex protecting FFTW plan creation/destruction
extern pthread_mutex_t g_fftw_mutex;

#endif // HQ_INIT_H