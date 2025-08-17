// hq_sweep.h - Заголовочный файл для sweep режима HackRF
#ifndef HQ_SWEEP_H
#define HQ_SWEEP_H

#include <stdint.h>

// Размер блока в байтах, приходящего из прошивки sweep
#define BYTES_PER_BLOCK 16384

// Interleaved режим (обычно 1)
#define INTERLEAVED 1

// Callback для sweep сегментов
typedef void (*hq_segment_cb)(const double* freqs_hz,
                              const float*  data_dbm,
                              int count,
                              double fft_bin_width_hz,
                              uint64_t hz_low, uint64_t hz_high,
                              void* user);

#endif // HQ_SWEEP_H
