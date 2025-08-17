#ifndef HQ_API_H
#define HQ_API_H

#include <stdint.h>
#include <stdbool.h>
#include <errno.h>

// Макросы для экспорта функций
#ifdef _WIN32
    #define HQ_EXPORT __declspec(dllexport)
#else
    #define HQ_EXPORT __attribute__((visibility("default")))
#endif

// Публичные структуры для FFI
typedef struct {
    double f_center_hz;
    double bw_hz;
    float rssi_ema;
    uint64_t last_ns;
    int hit_count;
} WatchItem;

typedef struct {
    double f_hz;
    float rssi_dbm;
    uint64_t last_ns;
} Peak;

typedef struct {
    int master_running;
    int slave_running[2];
    double retune_ms_avg;
    int watch_items;
} HqStatus;

// API функции
HQ_EXPORT int  hq_open_all(int num_expected);
HQ_EXPORT void hq_close_all(void);

HQ_EXPORT int  hq_config_set_rates(uint32_t samp_rate_hz, uint32_t bb_bw_hz);
HQ_EXPORT int  hq_config_set_gains(uint32_t lna_db, uint32_t vga_db, bool amp_on);

// DC Offset настройки
HQ_EXPORT int  hq_config_set_dc_offset(bool enable, bool auto_correct, int16_t i_offset, int16_t q_offset);
HQ_EXPORT int  hq_config_calibrate_dc_offset(int device_idx);

// Настройка диапазона частот
HQ_EXPORT int  hq_config_set_freq_range(double start_hz, double stop_hz, double step_hz);
HQ_EXPORT int  hq_config_set_dwell_time(uint32_t dwell_ms);

// НОВОЕ: Настройка параметров детектора
HQ_EXPORT void hq_set_detector_params(float threshold_offset_db, int min_width_bins,
                           int min_sweeps, float timeout_sec);

HQ_EXPORT int  hq_start(void);
HQ_EXPORT void hq_stop(void);

HQ_EXPORT int  hq_get_watchlist_snapshot(WatchItem* out, int max_items);
HQ_EXPORT int  hq_get_recent_peaks(Peak* out, int max_items);

// Чтение непрерывного спектра от Master SDR
HQ_EXPORT int  hq_get_master_spectrum(double* freqs_hz, float* powers_dbm, int max_points);

HQ_EXPORT void hq_set_grouping_tolerance_hz(double delta_hz);
HQ_EXPORT void hq_set_ema_alpha(float alpha);

HQ_EXPORT int  hq_get_status(HqStatus* out);

// Device enumeration
HQ_EXPORT int  hq_list_devices(char* serials[], int max_count);
HQ_EXPORT int  hq_get_device_count(void);

#endif // HQ_API_H