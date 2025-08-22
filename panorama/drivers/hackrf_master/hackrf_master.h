#pragma once
#include <stdint.h>

#define BYTES_PER_BLOCK (16384)
/* было INTERLEAVED, но чтобы не конфликтовать с чужими заголовками — HQ_INTERLEAVED */
#define HQ_INTERLEAVED 1

typedef void (*hq_segment_cb)(
    const double* freqs_hz,
    const float*  data_dbm,
    int count,
    double fft_bin_width_hz,
    uint64_t hz_low, uint64_t hz_high,
    void* user
);

int  hq_open(const char* serial_suffix);
int  hq_configure(double f_start_mhz, double f_stop_mhz, double bin_hz,
                  int lna_db, int vga_db, int amp_on);
int  hq_start(hq_segment_cb cb, void* user);
int  hq_stop(void);
void hq_close(void);

float cal_lookup_db(float freq_mhz, int lna_db, int vga_db, int amp_on);

const char* hq_last_error(void);

int  hq_device_count(void);
int  hq_get_device_serial(int idx, char* out, int cap);
