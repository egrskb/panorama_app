// hackrf_master.h — C API для бэкенда sweep (совместим с CFFI)

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Колбэк одного сегмента (четверть окна как в hackrf_sweep)
// freqs_hz[count], data_dbm[count], ширина бина, и низ/верх сегмента (Hz)
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

int  hq_start(hq_segment_cb cb, void* user);
int  hq_stop(void);

const char* hq_last_error(void);

// Перечисление устройств (через hackrf_device_list)
int  hq_device_count(void);
// Возвращает 0 при успехе, пишет нуль-терминированную строку серийника
int  hq_get_device_serial(int idx, char* out, int cap);

#ifdef __cplusplus
}
#endif
