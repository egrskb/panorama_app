#define _GNU_SOURCE
#include "hackrf_master.h"
#include <libhackrf/hackrf.h>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <ctype.h>

/* -------- Константы (как в hackrf_sweep) -------- */
#define DEFAULT_SAMPLE_RATE_HZ            (20000000u)   /* 20 MHz */
#define DEFAULT_BASEBAND_FILTER_BANDWIDTH (15000000u)   /* 15 MHz */
#define TUNE_STEP_MHZ                     (20u)
#define OFFSET_HZ                         (7500000u)
#define BLOCKS_PER_TRANSFER               16

/* -------- Глобалы устройства/состояния -------- */
static hackrf_device* dev = NULL;
static volatile int running = 0;
static hq_segment_cb g_cb = NULL;
static void*         g_user = NULL;
static char g_err[256] = {0};

/* -------- FFT/окно -------- */
static int fftSize = 0;
static double fft_bin_width = 0.0;
static fftwf_complex* fftwIn = NULL;
static fftwf_complex* fftwOut = NULL;
static fftwf_plan fftwPlan = NULL;
static float* pwr = NULL;
static float* window = NULL;

/* -------- Sweep config -------- */
static uint16_t freq_ranges[2] = {0,0};
static uint32_t step_count = 0;

/* -------- Текущие усиления -------- */
static int s_lna_db = 0;
static int s_vga_db = 0;
static int s_amp_on = 0;

/* -------- Коррекции окна -------- */
static float s_corr_window_power_db = 0.0f;
static float s_corr_enbw_db = 0.0f;

/* =========================================================
 *             Калибровка LUT (CSV)
 * ========================================================= */

typedef struct {
    float f_mhz;
    float k_db;
} kpt_t;

typedef struct {
    int   lna, vga, amp;
    kpt_t* pts;
    int   n;
    int   cap;
} kprof_t;

static kprof_t* s_profiles = NULL;
static int      s_profiles_n = 0;
static int      s_cal_enabled = 0;

static void cal_clear(void) {
    if (s_profiles) {
        for (int i=0;i<s_profiles_n;++i) {
            free(s_profiles[i].pts);
        }
        free(s_profiles);
    }
    s_profiles = NULL;
    s_profiles_n = s_cal_enabled = 0;
}

static int cal_find_profile(int lna, int vga, int amp) {
    for (int i=0;i<s_profiles_n;++i) {
        if (s_profiles[i].lna==lna && s_profiles[i].vga==vga && s_profiles[i].amp==amp)
            return i;
    }
    return -1;
}

float cal_lookup_db(float f_mhz, int lna, int vga, int amp) {
    if (!s_cal_enabled || s_profiles_n == 0) return 0.0f;
    int idx = cal_find_profile(lna, vga, amp);
    if (idx < 0) return 0.0f;
    kprof_t* p = &s_profiles[idx];
    if (p->n <= 0) return 0.0f;
    if (f_mhz <= p->pts[0].f_mhz)   return p->pts[0].k_db;
    if (f_mhz >= p->pts[p->n-1].f_mhz) return p->pts[p->n-1].k_db;
    for (int j=0;j<p->n-1;++j) {
        float f0 = p->pts[j].f_mhz;
        float f1 = p->pts[j+1].f_mhz;
        if (f_mhz >= f0 && f_mhz <= f1) {
            float t = (f_mhz - f0) / (f1 - f0);
            return p->pts[j].k_db + t * (p->pts[j+1].k_db - p->pts[j].k_db);
        }
    }
    return 0.0f;
}

/* =========================================================
 *                    Утилиты
 * ========================================================= */
static inline float logPower(fftwf_complex in, float scale) {
    float re = in[0] * scale;
    float im = in[1] * scale;
    float magsq = re*re + im*im;
    return (float)(10.0f * log10f(magsq));
}

static void set_err(const char* fmt, int code) {
    snprintf(g_err, sizeof(g_err), fmt, code);
}

const char* hq_last_error(void) { return g_err; }

/* =========================================================
 *                    API
 * ========================================================= */

int hq_open(const char* serial_suffix) {
    int r = hackrf_init();
    if (r) { set_err("hackrf_init failed: %d", r); return r; }
    if (serial_suffix && serial_suffix[0]) {
        r = hackrf_open_by_serial(serial_suffix, &dev);
    } else {
        r = hackrf_open(&dev);
    }
    if (r) { set_err("hackrf_open failed: %d", r); hackrf_exit(); return r; }
    r = hackrf_set_sample_rate(dev, DEFAULT_SAMPLE_RATE_HZ);
    if (r) { set_err("hackrf_set_sample_rate failed: %d", r); return r; }
    uint32_t bw = hackrf_compute_baseband_filter_bw(DEFAULT_BASEBAND_FILTER_BANDWIDTH);
    r = hackrf_set_baseband_filter_bandwidth(dev, bw);
    if (r) { set_err("hackrf_set_baseband_filter_bandwidth failed: %d", r); return r; }
    return 0;
}

int hq_configure(double f_start_mhz, double f_stop_mhz,
                 double requested_bin_hz,
                 int lna_db, int vga_db, int amp_enable)
{
    if (!dev) return -1;
    if (lna_db < 0) lna_db = 0; if (lna_db > 40) lna_db = 40; lna_db -= (lna_db % 8);
    if (vga_db < 0) vga_db = 0; if (vga_db > 62) vga_db = 62; vga_db -= (vga_db % 2);

    hackrf_set_lna_gain(dev, lna_db);
    hackrf_set_vga_gain(dev, vga_db);
    hackrf_set_amp_enable(dev, amp_enable ? 1 : 0);
    s_lna_db = lna_db; s_vga_db = vga_db; s_amp_on = amp_enable ? 1 : 0;

    if (requested_bin_hz < 2445.0) requested_bin_hz = 2445.0;
    if (requested_bin_hz > 5e6)    requested_bin_hz = 5e6;
    int N = (int)floor((double)DEFAULT_SAMPLE_RATE_HZ / requested_bin_hz);
    if (N < 4) N = 4; if (N > 8180) N = 8180;
    while ((N + 4) % 8) N++;
    fftSize = N;
    fft_bin_width = (double)DEFAULT_SAMPLE_RATE_HZ / (double)fftSize;

    if (fftwIn) { fftwf_free(fftwIn); fftwIn=NULL; }
    if (fftwOut){ fftwf_free(fftwOut); fftwOut=NULL; }
    if (pwr)    { fftwf_free(pwr);    pwr=NULL; }
    if (window) { fftwf_free(window); window=NULL; }
    if (fftwPlan){ fftwf_destroy_plan(fftwPlan); fftwPlan=NULL; }

    fftwIn  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*fftSize);
    fftwOut = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*fftSize);
    pwr     = (float*)fftwf_malloc(sizeof(float)*fftSize);
    window  = (float*)fftwf_malloc(sizeof(float)*fftSize);

    for (int i=0;i<fftSize;i++)
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (fftSize - 1)));
    fftwPlan = fftwf_plan_dft_1d(fftSize, fftwIn, fftwOut, FFTW_FORWARD, FFTW_ESTIMATE);

    s_corr_window_power_db = 10.0f * log10f(4.0f);
    s_corr_enbw_db         = 10.0f * log10f(1.5f);

    freq_ranges[0] = (uint16_t)floor(f_start_mhz);
    freq_ranges[1] = (uint16_t)ceil (f_stop_mhz);
    step_count = (uint32_t)((freq_ranges[1] - freq_ranges[0]) / TUNE_STEP_MHZ) + 1u;

    return 0;
}

static int rx_callback(hackrf_transfer* transfer)
{
    if (!running || !g_cb) return 0;

    int8_t* buf = (int8_t*)transfer->buffer;
    for (int j=0; j<BLOCKS_PER_TRANSFER; ++j) {
        uint8_t* ubuf = (uint8_t*)buf;
        if (!(ubuf[0]==0x7F && ubuf[1]==0x7F)) { buf += BYTES_PER_BLOCK; continue; }

        uint64_t frequency =
            ((uint64_t)ubuf[9] << 56) | ((uint64_t)ubuf[8] << 48) |
            ((uint64_t)ubuf[7] << 40) | ((uint64_t)ubuf[6] << 32) |
            ((uint64_t)ubuf[5] << 24) | ((uint64_t)ubuf[4] << 16) |
            ((uint64_t)ubuf[3] << 8)  | (uint64_t)ubuf[2];

        /* последние fftSize I/Q с конца блока */
        buf += BYTES_PER_BLOCK - (fftSize*2);
        for (int i=0;i<fftSize;i++){
            fftwIn[i][0] = buf[i*2]   * window[i] * (1.0f/128.0f);
            fftwIn[i][1] = buf[i*2+1] * window[i] * (1.0f/128.0f);
        }
        buf += fftSize*2;

        fftwf_execute(fftwPlan);
        for (int i=0;i<fftSize;i++) pwr[i] = logPower(fftwOut[i], 1.0f/fftSize);

        const int q = fftSize/4;

        /* сегмент A: [f, f+Fs/4], берем бины начиная с 1+(5/8*N) */
        {
            double* freqs = (double*)malloc(sizeof(double)*q);
            float*  data  = (float*) malloc(sizeof(float )*q);
            if (!freqs || !data) { if(freqs)free(freqs); if(data)free(data); return 0; }

            uint64_t hz_low  = frequency;
            uint64_t hz_high = frequency + DEFAULT_SAMPLE_RATE_HZ/4;
            for (int i=0;i<q;i++){
                float   raw_db = pwr[i + 1 + (fftSize*5)/8];
                double  f_hz   = (double)hz_low + (i + 0.5) * fft_bin_width;
                float   f_mhz  = (float)(f_hz / 1.0e6);
                float   k_db   = cal_lookup_db(f_mhz, s_lna_db, s_vga_db, s_amp_on);
                data[i]  = raw_db + s_corr_window_power_db + s_corr_enbw_db + k_db;
                freqs[i] = f_hz;
            }
            g_cb(freqs, data, q, fft_bin_width, hz_low, hz_high, g_user);
            free(freqs); free(data);
        }
        /* сегмент B: [f+Fs/2, f+3/4*Fs], берем бины начиная с 1+(1/8*N) */
        {
            double* freqs = (double*)malloc(sizeof(double)*q);
            float*  data  = (float*) malloc(sizeof(float )*q);
            if (!freqs || !data) { if(freqs)free(freqs); if(data)free(data); return 0; }

            uint64_t hz_low  = frequency + DEFAULT_SAMPLE_RATE_HZ/2;
            uint64_t hz_high = frequency + (DEFAULT_SAMPLE_RATE_HZ*3)/4;
            for (int i=0;i<q;i++){
                float   raw_db = pwr[i + 1 + (fftSize/8)];
                double  f_hz   = (double)hz_low + (i + 0.5) * fft_bin_width;
                float   f_mhz  = (float)(f_hz / 1.0e6);
                float   k_db   = cal_lookup_db(f_mhz, s_lna_db, s_vga_db, s_amp_on);
                data[i]  = raw_db + s_corr_window_power_db + s_corr_enbw_db + k_db;
                freqs[i] = f_hz;
            }
            g_cb(freqs, data, q, fft_bin_width, hz_low, hz_high, g_user);
            free(freqs); free(data);
        }

        /* сегмент C: [f+Fs/4, f+Fs/2] (с модульной индексацией по FFT) */
        {
            double* freqs = (double*)malloc(sizeof(double)*q);
            float*  data  = (float*) malloc(sizeof(float )*q);
            if (!freqs || !data) { if(freqs)free(freqs); if(data)free(data); return 0; }

            uint64_t hz_low  = frequency + DEFAULT_SAMPLE_RATE_HZ/4;
            uint64_t hz_high = frequency + DEFAULT_SAMPLE_RATE_HZ/2;
            int base = 1 + (fftSize*7)/8; /* смещение, затем wrap по модулю */
            for (int i=0;i<q;i++){
                int idx   = (base + i) % fftSize;
                float raw_db = pwr[idx];
                double  f_hz   = (double)hz_low + (i + 0.5) * fft_bin_width;
                float   f_mhz  = (float)(f_hz / 1.0e6);
                float   k_db   = cal_lookup_db(f_mhz, s_lna_db, s_vga_db, s_amp_on);
                data[i]  = raw_db + s_corr_window_power_db + s_corr_enbw_db + k_db;
                freqs[i] = f_hz;
            }
            g_cb(freqs, data, q, fft_bin_width, hz_low, hz_high, g_user);
            free(freqs); free(data);
        }

        /* сегмент D: [f+3/4*Fs, f+Fs] (с модульной индексацией по FFT) */
        {
            double* freqs = (double*)malloc(sizeof(double)*q);
            float*  data  = (float*) malloc(sizeof(float )*q);
            if (!freqs || !data) { if(freqs)free(freqs); if(data)free(data); return 0; }

            uint64_t hz_low  = frequency + (DEFAULT_SAMPLE_RATE_HZ*3)/4;
            uint64_t hz_high = frequency + DEFAULT_SAMPLE_RATE_HZ;
            int base = 1 + (fftSize*3)/8;
            for (int i=0;i<q;i++){
                int idx   = (base + i) % fftSize;
                float raw_db = pwr[idx];
                double  f_hz   = (double)hz_low + (i + 0.5) * fft_bin_width;
                float   f_mhz  = (float)(f_hz / 1.0e6);
                float   k_db   = cal_lookup_db(f_mhz, s_lna_db, s_vga_db, s_amp_on);
                data[i]  = raw_db + s_corr_window_power_db + s_corr_enbw_db + k_db;
                freqs[i] = f_hz;
            }
            g_cb(freqs, data, q, fft_bin_width, hz_low, hz_high, g_user);
            free(freqs); free(data);
        }
    }
    return 0;
}

int hq_start(hq_segment_cb cb, void* user)
{
    if (!dev) return -1;
    g_cb = cb; g_user = user;
    running = 1;

    int r = hackrf_init_sweep(
        dev,
        freq_ranges,
        1,
        BYTES_PER_BLOCK,
        TUNE_STEP_MHZ * 1000000u,
        OFFSET_HZ,
        HQ_INTERLEAVED
    );
    if (r) { set_err("hackrf_init_sweep failed: %d", r); running=0; return r; }

    r = hackrf_start_rx_sweep(dev, rx_callback, NULL);
    if (r) { set_err("hackrf_start_rx_sweep failed: %d", r); running=0; return r; }
    return 0;
}

int hq_device_count(void)
{
    int r = hackrf_init();
    if (r) { set_err("hackrf_init failed: %d", r); return 0; }
    hackrf_device_list_t* lst = hackrf_device_list();
    if (!lst) return 0;
    int n = lst->devicecount;
    hackrf_device_list_free(lst);
    return n;
}

int hq_get_device_serial(int idx, char* buf, int buf_len)
{
    if (!buf || buf_len <= 0) return -1;
    int r = hackrf_init();
    if (r) { set_err("hackrf_init failed: %d", r); return r; }
    hackrf_device_list_t* lst = hackrf_device_list();
    if (!lst) return -2;
    int n = lst->devicecount;
    if (idx < 0 || idx >= n) { hackrf_device_list_free(lst); return -3; }
    const char* s = lst->serial_numbers[idx];
    if (!s) s = "";
    snprintf(buf, buf_len, "%s", s);
    hackrf_device_list_free(lst);
    return 0;
}

int hq_stop(void)
{
    if (!dev) return 0;
    running = 0;
    hackrf_stop_rx(dev);
    return 0;
}

void hq_close(void)
{
    if (dev) {
        hackrf_close(dev);
        dev = NULL;
        hackrf_exit();
    }
    if (fftwPlan) { fftwf_destroy_plan(fftwPlan); fftwPlan=NULL; }
    if (fftwIn)   { fftwf_free(fftwIn); fftwIn=NULL; }
    if (fftwOut)  { fftwf_free(fftwOut); fftwOut=NULL; }
    if (pwr)      { fftwf_free(pwr); pwr=NULL; }
    if (window)   { fftwf_free(window); window=NULL; }
    cal_clear();
    g_cb=NULL; g_user=NULL;
}