// hackrf_master.h — C API для бэкенда sweep (совместим с CFFI)

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Константы для многосекционного режима
#define MAX_SEGMENTS 4
#define SEGMENT_A 0  // [f-OFFSET, f-OFFSET+Fs/4]
#define SEGMENT_B 1  // [f+OFFSET+Fs/2, f+OFFSET+3Fs/4]
#define SEGMENT_C 2  // [f+OFFSET+Fs/4, f+OFFSET+Fs/2]
#define SEGMENT_D 3  // [f+OFFSET+3Fs/4, f+OFFSET+Fs]

// Структура для передачи данных сегмента
typedef struct {
    double* freqs_hz;      // массив частот для сегмента
    float*  data_dbm;      // массив мощностей для сегмента
    int     count;         // количество бинов в сегменте
    int     segment_id;    // идентификатор сегмента (A/B/C/D)
    uint64_t hz_low;       // нижняя частота сегмента
    uint64_t hz_high;      // верхняя частота сегмента
} hq_segment_data_t;

// Новый колбэк для многосекционного режима
// segments - массив сегментов, segment_count - количество сегментов
typedef void (*hq_multi_segment_cb)(
    const hq_segment_data_t* segments,
    int                       segment_count,
    double                    fft_bin_width_hz,
    uint64_t                  center_hz,
    void*                     user
);

// Устаревший колбэк для обратной совместимости
typedef void (*hq_segment_cb)(
    const double* freqs_hz,
    const float*  data_dbm,
    int           count,
    double        fft_bin_width_hz,
    uint64_t      hz_low,
    uint64_t      hz_high,
    void*         user
);

// Открытие строго по серийному номеру (или его суффиксу).
// Если serial_suffix_or_null == NULL или пустая строка — возвращает ошибку.
int  hq_open(const char* serial_suffix_or_null);
void hq_close(void);

int  hq_configure(double f_start_mhz, double f_stop_mhz, double bin_hz,
                  int lna_db, int vga_db, int amp_on);

// Новые функции для многосекционного режима
int  hq_start_multi_segment(hq_multi_segment_cb cb, void* user);
int  hq_start(hq_segment_cb cb, void* user);  // для обратной совместимости
int  hq_stop(void);

// Функции калибровки
int  hq_load_calibration(const char* csv_path);
int  hq_enable_calibration(int enable);
int  hq_get_calibration_status(void);

// Функции для настройки режима
        int  hq_set_segment_mode(int mode);  // только 4 сегмента
int  hq_get_segment_mode(void);

const char* hq_last_error(void);

// Перечисление устройств (через hackrf_device_list)
int  hq_device_count(void);
// Возвращает 0 при успехе, пишет нуль-терминированную строку серийника
int  hq_get_device_serial(int idx, char* out, int cap);

#ifdef __cplusplus
}
#endif
