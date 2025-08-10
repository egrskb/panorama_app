#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*hq_segment_cb)(
    const double* freqs_hz,     // длина = count
    const float*  pwr_dbm,      // длина = count
    int           count,
    double        bin_hz,       // фактическая бин-ширина (= 20e6/fftSize)
    uint64_t      hz_low,       // границы сегмента как в hackrf_sweep CSV
    uint64_t      hz_high,
    void*         user          // прокидываем handle обратно в Python
);

int  hq_open(const char* serial_suffix /*NULL=первое устройство*/);
int  hq_configure(double f_start_mhz, double f_stop_mhz,
                  double requested_bin_hz,
                  int lna_db, int vga_db, int amp_enable);
/* запускает RX sweep на отдельном USB потоке libhackrf;
   колбэк вызывается на нативном потоке libusb */
int  hq_start(hq_segment_cb cb, void* user);
int  hq_stop(void);
void hq_close(void);

/* последнее текстовое описание ошибки (не потокобезопасно) */
const char* hq_last_error(void);

#ifdef __cplusplus
}
#endif

/* device enumeration */
int hq_device_count(void);
/* buf_len >= 64 вполне достаточно для полного серийника */
int hq_get_device_serial(int idx, char* buf, int buf_len);