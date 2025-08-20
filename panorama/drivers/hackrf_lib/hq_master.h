// hq_master.h - Master SDR для широкого sweep и поиска пиков
#ifndef HQ_MASTER_H
#define HQ_MASTER_H

#include <libhackrf/hackrf.h>
#include <fftw3.h>
#include <pthread.h>
#include <stdbool.h>
#include "hq_watchlist.h"

// Параметры sweep
typedef struct {
    double start_hz;
    double stop_hz;
    double step_hz;
    uint32_t sample_rate;
    uint32_t bandwidth;
    uint32_t dwell_ms;
} MasterConfig;

// Параметры детектора пиков
typedef struct {
    double threshold_db;      // Порог над baseline
    double min_distance_hz;   // Минимальное расстояние между пиками
    int min_width_bins;       // Минимальная ширина пика
    double baseline_alpha;    // Коэффициент для EMA baseline
} PeakDetectorConfig;

// Состояние master
typedef struct {
    hackrf_device* device;
    pthread_t thread;
    bool running;
    
    // Конфигурация
    MasterConfig config;
    PeakDetectorConfig detector;
    
    // FFT
    int fft_size;
    fftwf_complex* fft_in;
    fftwf_complex* fft_out;
    fftwf_plan fft_plan;
    float* window;
    float* psd;
    
    // Baseline для адаптивного порога
    float* baseline;
    int baseline_initialized;
    
    // Watchlist
    Watchlist* watchlist;
    
    // Callbacks
    hq_on_peak_cb peak_callback;
    void* peak_callback_data;
    
    // Статистика
    uint64_t sweep_count;
    uint64_t peak_count;
    double coverage_percent;
} MasterState;

// API функции
int master_init(MasterState* state, hackrf_device* device, Watchlist* watchlist);
void master_destroy(MasterState* state);

int master_configure(MasterState* state, const MasterConfig* config);
int master_set_detector(MasterState* state, const PeakDetectorConfig* detector);
int master_set_peak_callback(MasterState* state, hq_on_peak_cb cb, void* user_data);

int master_start(MasterState* state);
int master_stop(MasterState* state);
bool master_is_running(MasterState* state);

// Внутренние функции (для callback)
void master_process_sweep_block(MasterState* state, const uint8_t* buffer, int len);
void master_detect_peaks(MasterState* state, const float* psd, int n_bins, double freq_start_hz);
void master_update_baseline(MasterState* state, const float* psd, int n_bins);

#endif // HQ_MASTER_H