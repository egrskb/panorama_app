// hackrf_master.c - sweep backend для Panorama (API hackrf 0.9 / 2024.02.x)
// Работает как hackrf_sweep: hackrf_init_sweep + hackrf_start_rx_sweep,
// в rx_callback парсим 0x7f 0x7f + center_freq и отдаём 1/4 окна через hq_segment_cb.

#include "hackrf_master.h"
#include <libhackrf/hackrf.h>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define DEFAULT_SAMPLE_RATE_HZ 20000000
#define DEFAULT_FFT_SIZE       8192
#define OFFSET                 7500000          // 7.5 MHz
#define TUNE_STEP_MHZ          20               // шаг тюнинга (МГц)
#define FREQ_ONE_MHZ           (1000000U)
#define SWEEP_STYLE_INTERLEAVED 0               // соответствует HACKRF_SWEEP_STYLE_INTERLEAVED

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static hackrf_device* dev = NULL;
static volatile int running = 0;
static hq_segment_cb g_cb = NULL;
static void* g_user = NULL;
static char g_err[256] = {0};

static fftwf_complex* fft_in = NULL;
static fftwf_complex* fft_out = NULL;
static fftwf_plan fft_plan;
static float* window = NULL;
static float* pwr = NULL;
static int fft_size = DEFAULT_FFT_SIZE;
static double fft_bin_width = 0.0;

static double g_bin_hz = 0.0;

// ---- утилиты ошибок ----
const char* hq_last_error(void) { return g_err; }
static void set_err(const char* s) { snprintf(g_err, sizeof(g_err), "%s", s); }
static void set_errf(const char* fmt, const char* a) { snprintf(g_err, sizeof(g_err), fmt, a); }

// ---- калибровка (плейсхолдер) ----
static float cal_lookup_db(float freq_mhz, int lna_db, int vga_db, int amp_on) {
    (void)freq_mhz; (void)lna_db; (void)vga_db; (void)amp_on;
    return 0.0f;
}

// ---- перечисление устройств ----
int hq_device_count(void) {
    struct hackrf_device_list* lst = hackrf_device_list();
    if (!lst) return 0;
    int n = lst->devicecount;
    hackrf_device_list_free(lst);
    return n;
}

int hq_get_device_serial(int idx, char* out, int cap) {
    if (!out || cap <= 0) return -1;
    struct hackrf_device_list* lst = hackrf_device_list();
    if (!lst) return -2;
    if (idx < 0 || idx >= lst->devicecount) {
        hackrf_device_list_free(lst);
        return -3;
    }
    const char* s = lst->serial_numbers[idx];
    if (!s) s = "";
    snprintf(out, cap, "%s", s);
    hackrf_device_list_free(lst);
    return 0;
}

// ---- открытие/закрытие ----
int hq_open(const char* serial_suffix) {
    int r = hackrf_init();
    if (r) { set_err("hackrf_init failed"); return r; }

    if (!serial_suffix || !serial_suffix[0]) {
        set_err("Serial is required (no fallback to first device)");
        return -100; // специально наш код ошибки
    }

    // Поддержка «суффикса»: ищем устройство, серийник которого оканчивается на заданную строку
    struct hackrf_device_list* lst = hackrf_device_list();
    if (!lst) { set_err("hackrf_device_list failed"); return -101; }

    int found = -1;
    size_t want_len = strlen(serial_suffix);
    for (int i = 0; i < lst->devicecount; ++i) {
        const char* s = lst->serial_numbers[i] ? lst->serial_numbers[i] : "";
        size_t sl = strlen(s);
        if (sl >= want_len) {
            if (strcmp(s + (sl - want_len), serial_suffix) == 0) {
                found = i; break;
            }
        }
    }

    if (found < 0) {
        hackrf_device_list_free(lst);
        set_errf("Device with serial suffix not found: %s", serial_suffix);
        return -102;
    }

    r = hackrf_device_list_open(lst, found, &dev);
    hackrf_device_list_free(lst);
    if (r) { set_err("hackrf_open (by list index) failed"); return r; }

    // Ставим sample rate — используется при FFT
    hackrf_set_sample_rate(dev, DEFAULT_SAMPLE_RATE_HZ);
    return 0;
}

void hq_close(void) {
    running = 0;
    if (dev) { hackrf_close(dev); dev = NULL; }
    hackrf_exit();

    if (fft_in)      { fftwf_free(fft_in);  fft_in = NULL; }
    if (fft_out)     { fftwf_free(fft_out); fft_out = NULL; }
    if (pwr)         { free(pwr);           pwr = NULL; }
    if (window)      { free(window);        window = NULL; }
    if (fft_plan)    { fftwf_destroy_plan(fft_plan); fft_plan = NULL; }
}

// ---- конфигурация свипа ----
int hq_configure(double f_start_mhz, double f_stop_mhz, double bin_hz,
                 int lna_db, int vga_db, int amp_on) {
    if (!dev) { set_err("device not opened"); return -1; }
    g_bin_hz = bin_hz;

    // Усиления
    hackrf_set_lna_gain(dev, lna_db);
    hackrf_set_vga_gain(dev, vga_db);
    hackrf_set_amp_enable(dev, amp_on);

    // FFT init
    fft_size = (int)(DEFAULT_SAMPLE_RATE_HZ / bin_hz);
    if (fft_size > 8192) fft_size = 8192;
    if (fft_size < 256)  fft_size = 256;
    fft_bin_width = (double)DEFAULT_SAMPLE_RATE_HZ / (double)fft_size;

    if (fft_in)  { fftwf_free(fft_in);  fft_in = NULL; }
    if (fft_out) { fftwf_free(fft_out); fft_out = NULL; }
    if (pwr)     { free(pwr);           pwr = NULL; }
    if (window)  { free(window);        window = NULL; }

    fft_in  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    fft_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    pwr     = (float*)malloc(sizeof(float) * fft_size);
    window  = (float*)malloc(sizeof(float) * fft_size);
    for (int i = 0; i < fft_size; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (fft_size - 1)));
    }
    fft_plan = fftwf_plan_dft_1d(fft_size, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);

    // sweep init (как в hackrf_sweep.c)
    uint16_t freqs[2];
    if (f_stop_mhz < f_start_mhz) {
        double tmp = f_start_mhz; f_start_mhz = f_stop_mhz; f_stop_mhz = tmp;
    }
    freqs[0] = (uint16_t)floor(f_start_mhz);
    freqs[1] = (uint16_t)ceil(f_stop_mhz);

    int r = hackrf_init_sweep(
        dev,
        freqs, 1,
        BYTES_PER_BLOCK,                          // размер блока из libhackrf.h
        TUNE_STEP_MHZ * FREQ_ONE_MHZ,            // шаг 20 МГц
        OFFSET,                                  // offset 7.5 МГц
        SWEEP_STYLE_INTERLEAVED                  // interleaved режим (0)
    );
    if (r != HACKRF_SUCCESS) {
        set_err("hackrf_init_sweep failed");
        return r;
    }
    return 0;
}

// ---- обработка блока ----
static void process_block(int8_t* buf, int valid_length, uint64_t center_hz) {
    int count = fft_size;
    if (valid_length/2 < count) return;

    const float scale = 1.0f/128.0f;
    for (int i = 0; i < count; i++) {
        float I = buf[2*i]   * scale;
        float Q = buf[2*i+1] * scale;
        fft_in[i][0] = I * window[i];
        fft_in[i][1] = Q * window[i];
    }
    fftwf_execute(fft_plan);

    for (int i = 0; i < count; i++) {
        float re = fft_out[i][0], im = fft_out[i][1];
        float mag = re*re + im*im;
        if (mag <= 1e-12f) mag = 1e-12f;
        pwr[i] = 10.0f * log10f(mag) + cal_lookup_db((float)center_hz/1e6f, 0, 0, 0);
    }

    // как в hackrf_sweep — отдаём 1/4 окна (нижнюю четверть)
    int quarter = count/4;

    static double* freqs = NULL;
    static int cap = 0;
    if (quarter > cap) {
        free(freqs);
        cap = quarter;
        freqs = (double*)malloc(sizeof(double)*cap);
    }

    double start_hz = (double)center_hz - (double)DEFAULT_SAMPLE_RATE_HZ/2.0;
    for (int i = 0; i < quarter; i++) {
        freqs[i] = start_hz + (i + 0.5) * fft_bin_width;
    }

    if (g_cb) {
        g_cb(freqs, pwr, quarter, fft_bin_width,
             (uint64_t)freqs[0], (uint64_t)freqs[quarter-1], g_user);
    }
}

// ---- RX колбэк ----
static int rx_callback(hackrf_transfer* transfer) {
    if (!running) return 0;
    uint8_t* ubuf = (uint8_t*)transfer->buffer;

    // сигнатура sweep-пакета
    if (ubuf[0]==0x7f && ubuf[1]==0x7f) {
        uint64_t freq=0;
        for (int i=0; i<8; i++) freq |= ((uint64_t)ubuf[2+i]) << (8*i);
        int8_t* iq = (int8_t*)(ubuf+16);
        process_block(iq, transfer->valid_length-16, freq);
    }
    return 0;
}

// ---- старт/стоп ----
int hq_start(hq_segment_cb cb, void* user) {
    if (!dev) { set_err("device not opened"); return -1; }
    g_cb = cb; g_user = user; running = 1;
    int r = hackrf_start_rx_sweep(dev, rx_callback, NULL);
    if (r != HACKRF_SUCCESS) {
        set_err("hackrf_start_rx_sweep failed");
        running = 0;
        return r;
    }
    return 0;
}

int hq_stop(void) {
    running = 0;
    return hackrf_stop_rx(dev);
}
