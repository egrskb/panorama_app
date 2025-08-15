#define _GNU_SOURCE
#include "hq_sweep.h"
#include <libhackrf/hackrf.h>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <ctype.h>

/* -------- Константы соответствуют hackrf_sweep -------- */
#define DEFAULT_SAMPLE_RATE_HZ            (20000000u)   /* 20 MHz */
#define DEFAULT_BASEBAND_FILTER_BANDWIDTH (15000000u)   /* 15 MHz */
#define TUNE_STEP_MHZ                     (20u)         /* Fs / 1e6 */
#define OFFSET_HZ                         (7500000u)    /* 7.5 MHz */
#define BLOCKS_PER_TRANSFER               16

/* -------- Глобалы устройства/состояние -------- */
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

/* -------- Sweep config (один сплошной диапазон, как и было) -------- */
static uint16_t freq_ranges[2] = {0,0};
static uint32_t step_count = 0;

/* -------- Текущие усиления (для выбора профиля LUT) -------- */
static int s_lna_db = 0;
static int s_vga_db = 0;
static int s_amp_on = 0;

/* -------- Коррекции окна/ENBW для Hann -------- */
static float s_corr_window_power_db = 0.0f;   /* ≈ +6.0206 dB */
static float s_corr_enbw_db = 0.0f;           /* ≈ +1.7609 dB */

/* =========================================================
 *             БЛОК КАЛИБРОВКИ (CSV как в SDR Console)
 * ========================================================= */

/* Точка калибровки для профиля */
typedef struct {
    float f_mhz;   /* частота, МГц */
    float k_db;    /* оффсет, dB */
} kpt_t;

/* Профиль по усилениям */
typedef struct {
    int   lna, vga, amp;
    kpt_t* pts;    /* динамический массив */
    int   n;       /* число точек */
    int   cap;     /* вместимость */
} kprof_t;

static kprof_t* s_profiles = NULL;
static int      s_profiles_n = 0;
static int      s_profiles_cap = 0;
static int      s_cal_enabled = 0; /* применяем ли LUT; включается при успешной загрузке */

/* Очистка LUT из памяти */
static void cal_clear(void) {
    if (s_profiles) {
        for (int i=0;i<s_profiles_n;++i) {
            free(s_profiles[i].pts);
            s_profiles[i].pts = NULL;
            s_profiles[i].n = s_profiles[i].cap = 0;
        }
        free(s_profiles);
        s_profiles = NULL;
    }
    s_profiles_n = s_profiles_cap = 0;
    s_cal_enabled = 0;
}

/* Утилита: trim пробелы */
static char* trim(char* s) {
    if (!s) return s;
    while (*s && isspace((unsigned char)*s)) ++s;
    char* end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) --end;
    *end = '\0';
    return s;
}

/* Найти профиль по усилениям */
static int cal_find_profile(int lna, int vga, int amp) {
    for (int i=0;i<s_profiles_n;++i) {
        if (s_profiles[i].lna==lna && s_profiles[i].vga==vga && s_profiles[i].amp==amp)
            return i;
    }
    return -1;
}

/* Добавить (или получить) профиль */
static int cal_ensure_profile(int lna, int vga, int amp) {
    int idx = cal_find_profile(lna,vga,amp);
    if (idx >= 0) return idx;
    /* новый */
    if (s_profiles_n == s_profiles_cap) {
        int new_cap = s_profiles_cap ? s_profiles_cap*2 : 4;
        kprof_t* np = (kprof_t*)realloc(s_profiles, new_cap*sizeof(kprof_t));
        if (!np) return -1;
        s_profiles = np;
        s_profiles_cap = new_cap;
    }
    idx = s_profiles_n++;
    s_profiles[idx].lna = lna;
    s_profiles[idx].vga = vga;
    s_profiles[idx].amp = amp;
    s_profiles[idx].pts = NULL;
    s_profiles[idx].n   = 0;
    s_profiles[idx].cap = 0;
    return idx;
}

/* Добавить точку в профиль */
static int cal_add_point(int idx, float f_mhz, float k_db) {
    if (idx < 0 || idx >= s_profiles_n) return -1;
    kprof_t* p = &s_profiles[idx];
    if (p->n == p->cap) {
        int new_cap = p->cap ? p->cap*2 : 8;
        kpt_t* np = (kpt_t*)realloc(p->pts, new_cap*sizeof(kpt_t));
        if (!np) return -1;
        p->pts = np;
        p->cap = new_cap;
    }
    p->pts[p->n].f_mhz = f_mhz;
    p->pts[p->n].k_db  = k_db;
    p->n += 1;
    return 0;
}

/* Сортировка точек профиля по частоте */
static int cmp_kpt(const void* a, const void* b) {
    const kpt_t* pa = (const kpt_t*)a;
    const kpt_t* pb = (const kpt_t*)b;
    if (pa->f_mhz < pb->f_mhz) return -1;
    if (pa->f_mhz > pb->f_mhz) return 1;
    return 0;
}
static void cal_sort_profiles(void) {
    for (int i=0;i<s_profiles_n;++i) {
        if (s_profiles[i].n > 1) {
            qsort(s_profiles[i].pts, s_profiles[i].n, sizeof(kpt_t), cmp_kpt);
        }
    }
}

/* Линейная интерполяция в профиле */
static float cal_lookup_db(float f_mhz, int lna, int vga, int amp) {
    if (!s_cal_enabled || s_profiles_n == 0) return 0.0f;
    int idx = cal_find_profile(lna, vga, amp);
    if (idx < 0) return 0.0f;
    kprof_t* p = &s_profiles[idx];
    if (p->n <= 0) return 0.0f;
    if (p->n == 1) return p->pts[0].k_db;

    if (f_mhz <= p->pts[0].f_mhz)   return p->pts[0].k_db;
    if (f_mhz >= p->pts[p->n-1].f_mhz) return p->pts[p->n-1].k_db;

    /* бинарный поиск можно, но линейного тоже достаточно при небольших N */
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

/* ------ Публичные функции для калибровки ------ */

/* Загрузка CSV: freq_mhz,lna_db,vga_db,amp,offset_db
 * Возвращает 0 при успехе.
 */
int hq_load_calibration(const char* csv_path)
{
    cal_clear();

    FILE* f = fopen(csv_path, "r");
    if (!f) { snprintf(g_err, sizeof(g_err), "cal open: %s", csv_path); return -1; }

    char line[512];
    int lineno = 0, added = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char* s = trim(line);
        if (*s == '\0' || *s == '#' || *s == ';') continue;

        /* Разбор CSV — простейший, через strtok */
        char* tok;
        char* rest = s;
        char* fields[5] = {0};
        int nf = 0;
        while ((tok = strtok(rest, ",;"))) {
            rest = NULL;
            fields[nf++] = trim(tok);
            if (nf >= 5) break;
        }
        if (nf < 5) {
            /* Пропустим кривую строку */
            continue;
        }
        float f_mhz = (float)atof(fields[0]);
        int   lna   = atoi(fields[1]);
        int   vga   = atoi(fields[2]);
        int   amp   = atoi(fields[3]);
        float k_db  = (float)atof(fields[4]);

        int pidx = cal_ensure_profile(lna, vga, amp);
        if (pidx < 0) { fclose(f); cal_clear(); snprintf(g_err,sizeof(g_err),"cal oom"); return -2; }
        if (cal_add_point(pidx, f_mhz, k_db) != 0) { fclose(f); cal_clear(); snprintf(g_err,sizeof(g_err),"cal oom"); return -3; }
        added++;
    }
    fclose(f);

    if (added == 0) { snprintf(g_err,sizeof(g_err),"cal empty"); return -4; }

    cal_sort_profiles();
    s_cal_enabled = 1;
    return 0;
}

/* Включить/выключить применение LUT (по умолчанию включается после load) */
void hq_enable_calibration(int enable) { s_cal_enabled = (enable ? 1 : 0); }

/* Проверка, загружена ли калибровка */
int hq_calibration_loaded(void) { return (s_cal_enabled && s_profiles_n>0) ? 1 : 0; }

/* =========================================================
 *                    Остальной код
 * ========================================================= */

static inline float logPower(fftwf_complex in, float scale) {
    float re = in[0] * scale;
    float im = in[1] * scale;
    float magsq = re*re + im*im;
    return (float)(log2f(magsq) * 10.0f / log2f(10.0f));
}

static void set_err(const char* fmt, int code) {
    snprintf(g_err, sizeof(g_err), fmt, code);
}

const char* hq_last_error(void) { return g_err; }

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

    /* сетки gain как в hackrf_sweep */
    if (lna_db < 0) lna_db = 0; 
    if (lna_db > 40) lna_db = 40; 
    lna_db -= (lna_db % 8);
    
    if (vga_db < 0) vga_db = 0; 
    if (vga_db > 62) vga_db = 62; 
    vga_db -= (vga_db % 2);

    int r = hackrf_set_lna_gain(dev, lna_db);
    if (r) { set_err("hackrf_set_lna_gain failed: %d", r); return r; }
    r = hackrf_set_vga_gain(dev, vga_db);
    if (r) { set_err("hackrf_set_vga_gain failed: %d", r); return r; }
    r = hackrf_set_amp_enable(dev, amp_enable ? 1 : 0);
    if (r) { set_err("hackrf_set_amp_enable failed: %d", r); return r; }

    s_lna_db = lna_db; s_vga_db = vga_db; s_amp_on = amp_enable ? 1 : 0;

    /* FFT: Fs / requested_bin_hz, доводим до (N+4)%8==0 */
    if (requested_bin_hz < 2445.0) requested_bin_hz = 2445.0;
    if (requested_bin_hz > 5e6)    requested_bin_hz = 5e6;

    int N = (int)floor((double)DEFAULT_SAMPLE_RATE_HZ / requested_bin_hz);
    if (N < 4) N = 4; 
    if (N > 8180) N = 8180;
    while ((N + 4) % 8) N++;
    fftSize = N;
    fft_bin_width = (double)DEFAULT_SAMPLE_RATE_HZ / (double)fftSize;

    /* FFTW буферы и окно (Hann) */
    if (fftwIn) { fftwf_free(fftwIn); fftwIn=NULL; }
    if (fftwOut){ fftwf_free(fftwOut); fftwOut=NULL; }
    if (pwr)     { fftwf_free(pwr);     pwr=NULL; }
    if (window)  { fftwf_free(window);  window=NULL; }
    if (fftwPlan){ fftwf_destroy_plan(fftwPlan); fftwPlan=NULL; }

    fftwIn  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*fftSize);
    fftwOut = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*fftSize);
    pwr     = (float*)fftwf_malloc(sizeof(float)*fftSize);
    window  = (float*)fftwf_malloc(sizeof(float)*fftSize);
    if (!fftwIn || !fftwOut || !pwr || !window) { set_err("fftw malloc", 0); return -2; }

    for (int i=0;i<fftSize;i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (fftSize - 1)));
    }
    fftwPlan = fftwf_plan_dft_1d(fftSize, fftwIn, fftwOut, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(fftwPlan);

    /* Оконные коррекции (Hann) */
    s_corr_window_power_db = 10.0f * log10f(4.0f);  /* ≈ +6.0206 dB */
    s_corr_enbw_db         = 10.0f * log10f(1.5f);  /* ≈ +1.7609 dB */

    /* Диапазон sweep (целые МГц) */
    if (f_stop_mhz <= f_start_mhz) return -3;
    if (f_start_mhz < 0) f_start_mhz = 0;
    if (f_stop_mhz  > 7250) f_stop_mhz = 7250;

    freq_ranges[0] = (uint16_t)floor(f_start_mhz);
    freq_ranges[1] = (uint16_t)ceil (f_stop_mhz);
    
    /* Проверка на переполнение uint16_t */
    if (freq_ranges[1] > 7250) freq_ranges[1] = 7250;
    
    step_count = (uint32_t)((freq_ranges[1] - freq_ranges[0]) / TUNE_STEP_MHZ) + 1u;
    
    /* Отладочная информация */
    fprintf(stderr, "hq_configure: %u-%u MHz, bin=%d Hz, steps=%u\n", 
            freq_ranges[0], freq_ranges[1], fftSize, step_count);

    return 0;
}

static int rx_callback(hackrf_transfer* transfer)
{
    if (!running || !g_cb) return 0;
    
    static uint64_t last_frequency = 0;
    static int blocks_processed = 0;

    int8_t* buf = (int8_t*)transfer->buffer;
    for (int j=0; j<BLOCKS_PER_TRANSFER; ++j) {
        uint8_t* ubuf = (uint8_t*)buf;
        
        /* Проверяем заголовок блока */
        if (!(ubuf[0]==0x7F && ubuf[1]==0x7F)) { 
            buf += BYTES_PER_BLOCK; 
            continue; 
        }

        uint64_t frequency =
            ((uint64_t)ubuf[9] << 56) | ((uint64_t)ubuf[8] << 48) |
            ((uint64_t)ubuf[7] << 40) | ((uint64_t)ubuf[6] << 32) |
            ((uint64_t)ubuf[5] << 24) | ((uint64_t)ubuf[4] << 16) |
            ((uint64_t)ubuf[3] << 8)  | (uint64_t)ubuf[2];
        
        /* Отладка для высоких частот */
        if (frequency > 5900000000ULL && frequency < 6100000000ULL) {
            if (frequency != last_frequency) {
                fprintf(stderr, "Processing freq: %.1f MHz\n", frequency / 1e6);
                last_frequency = frequency;
            }
        }
        
        blocks_processed++;

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
        freq_ranges, /* одна пара [start, stop] в МГц */
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
    hackrf_device_list_t* lst = hackrf_device_list();
    if (!lst) return 0;
    int n = lst->devicecount;
    hackrf_device_list_free(lst);
    return n;
}

int hq_get_device_serial(int idx, char* buf, int buf_len)
{
    if (!buf || buf_len <= 0) return -1;

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