// hackrf_master.h — Заголовок для master_hackrf.c
#ifndef HACKRF_MASTER_H
#define HACKRF_MASTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ================== Константы FFT ==================
#define MIN_FFT_SIZE       16
#define DEFAULT_FFT_SIZE   4096   // Увеличено для поддержки 5 кГц бина при 20 МГц SR
#define MAX_FFT_SIZE       65536  // максимум для FFTW

// ================== Константы сегментов ==================
#define HQ_SEGMENT_MODE_2  2
#define HQ_SEGMENT_MODE_4  4
#define MAX_SEGMENTS       4

// ================== Калибровка ==================
#define MAX_CALIBRATION_ENTRIES 256

// ================== Типы данных ==================

// Данные одного сегмента спектра
typedef struct {
    int     segment_id;   // 0..3
    int     count;        // количество бинов
    double* freqs_hz;     // массив частот [count]
    float*  data_dbm;     // мощности [count]
} hq_segment_data_t;

// Колбэк: несколько сегментов (2 или 4)
typedef void (*hq_multi_segment_cb)(
    const hq_segment_data_t* segments,
    int segment_count,
    double bin_width_hz,
    uint64_t center_hz,
    void* user_data);



// ================== API ==================

// --- Ошибки ---
const char* hq_last_error(void);

// --- Управление устройством ---
int  hq_open(const char* serial_suffix); // "" или NULL = первое устройство
void hq_close(void);
int  hq_device_count(void);
int  hq_get_device_serial(int idx, char* out, int cap);

// --- Конфигурация ---
int  hq_configure(double f_start_mhz, double f_stop_mhz, double bin_hz,
                  int lna_db, int vga_db, int amp_on);

// --- Старт/стоп ---
int  hq_start_multi_segment(hq_multi_segment_cb cb, void* user);
int  hq_start_no_cb(void);
int  hq_stop(void);

// --- Доступ к спектру (EMA-сетка) ---
int  hq_get_master_spectrum(double* freqs_hz, float* powers_dbm, int max_points);
// Возвращает фактическую ширину бина (шаг сетки) в Гц
double hq_get_fft_bin_hz(void);

// --- Настройки обработки ---
void hq_set_ema_alpha(float alpha);
void hq_set_detector_params(float threshold_offset_db, int min_width_bins,
                            int min_sweeps, float timeout_sec);

int  hq_set_segment_mode(int mode); // 2 или 4
int  hq_get_segment_mode(void);

int  hq_set_fft_size(int size); // power-of-2, в пределах MIN..MAX
int  hq_get_fft_size(void);

// --- Частотное сглаживание ---
void hq_set_freq_smoothing(int enabled, int window_bins);
int  hq_get_freq_smoothing_enabled(void);
int  hq_get_freq_smoothing_window(void);

// --- Калибровка ---
int  hq_load_calibration(const char* csv_path);
int  hq_enable_calibration(int enable);
int  hq_get_calibration_status(void);

#ifdef __cplusplus
}
#endif

#endif // HACKRF_MASTER_H