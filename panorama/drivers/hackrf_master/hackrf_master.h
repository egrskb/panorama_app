// hackrf_master.h — C API для бэкенда sweep (совместим с CFFI)

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Константы для FFT
#define MIN_FFT_SIZE 16
#define DEFAULT_FFT_SIZE 32      // Для bin ~800 кГц при 20 MHz SR
#define MAX_FFT_SIZE 256         // Максимум для детального анализа

// Константы для многосекционного режима
#define MAX_SEGMENTS 4
#define MAX_CALIBRATION_ENTRIES 1000

// Структура для передачи данных сегмента
typedef struct {
    double* freqs_hz;      // массив частот для сегмента
    float*  data_dbm;      // массив мощностей для сегмента
    int     count;         // количество бинов в сегменте
    int     segment_id;    // идентификатор сегмента (A/B/C/D)
    uint64_t hz_low;       // нижняя частота сегмента
    uint64_t hz_high;      // верхняя частота сегмента
} hq_segment_data_t;

// Колбэки
typedef void (*hq_multi_segment_cb)(
    const hq_segment_data_t* segments,
    int                       segment_count,
    double                    fft_bin_width_hz,
    uint64_t                  center_hz,
    void*                     user
);

typedef void (*hq_segment_cb)(
    const double* freqs_hz,
    const float*  data_dbm,
    int           count,
    double        fft_bin_width_hz,
    uint64_t      hz_low,
    uint64_t      hz_high,
    void*         user
);

// Основные функции
int  hq_open(const char* serial_suffix_or_null);
void hq_close(void);

int  hq_configure(double f_start_mhz, double f_stop_mhz, double bin_hz,
                  int lna_db, int vga_db, int amp_on);

// Запуск sweep
int  hq_start_multi_segment(hq_multi_segment_cb cb, void* user);
int  hq_start(hq_segment_cb cb, void* user);  // для обратной совместимости
int  hq_stop(void);

// Получение готового спектра
int  hq_get_master_spectrum(double* freqs_hz, float* powers_dbm, int max_points);

// Настройки обработки
void hq_set_ema_alpha(float alpha);
void hq_set_detector_params(float threshold_offset_db, int min_width_bins,
                            int min_sweeps, float timeout_sec);
int  hq_set_segment_mode(int mode);  // 2 или 4 сегмента
int  hq_get_segment_mode(void);
int  hq_set_fft_size(int size);      // Установить размер FFT
int  hq_get_fft_size(void);          // Получить текущий размер FFT

// Калибровка
int  hq_load_calibration(const char* csv_path);
int  hq_enable_calibration(int enable);
int  hq_get_calibration_status(void);

// Ошибки и статус
const char* hq_last_error(void);

// Перечисление устройств
int  hq_device_count(void);
int  hq_get_device_serial(int idx, char* out, int cap);

#ifdef __cplusplus
}
#endif