// hq_sweep.h
#pragma once
#include <stdint.h>

#define BYTES_PER_BLOCK  16384   // типичное значение для sweep FW
#define HQ_INTERLEAVED   1       // данные IQ идут I,Q,I,Q,...

// колбэк: отдадим массив частот (Гц) и мощностей (dBm)
typedef void (*hq_segment_cb)(
    const double* freqs_hz,
    const float*  data_dbm,
    int           count,
    double        fft_bin_width_hz,
    uint64_t      hz_low,
    uint64_t      hz_high,
    void*         user
);

// API
const char* hq_last_error(void);

int  hq_open(const char* serial_suffix);
int  hq_configure(double f_start_mhz, double f_stop_mhz,
                  double requested_bin_hz,
                  int lna_db, int vga_db, int amp_enable);
int  hq_start(hq_segment_cb cb, void* user);
int  hq_stop(void);
void hq_close(void);

// калибровка из CSV
int  hq_load_calibration(const char* csv_path); // freq_mhz,lna,vga,amp,offset_db
void hq_enable_calibration(int enable);
int  hq_calibration_loaded(void);
