// hq_init.c — DC offset as in hackrf_sweep (no-ops)
// Мы не трогаем DC на устройстве и не делаем DSP-коррекцию.
// Все вызовы API — «успешные» заглушки без предупреждений.

#include "hq_init.h"
#include "hq_grouping.h"

#include <libhackrf/hackrf.h>
#include <pthread.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <math.h>
#include <errno.h>
#include <fftw3.h>

/* ===================== Глобалы ===================== */

PeakQueue* g_peaks_queue = NULL;

WatchItem g_watchlist[MAX_WATCHLIST];
size_t g_watchlist_count = 0;
pthread_mutex_t g_watchlist_mutex = PTHREAD_MUTEX_INITIALIZER;

float  g_ema_alpha = 0.1f;
double g_grouping_tolerance_hz = 1e6; // 1 MHz по умолчанию

// Глобальные буферы непрерывного спектра мастера
#define MAX_SPECTRUM_POINTS 50000
static double g_spectrum_freqs[MAX_SPECTRUM_POINTS];
static float  g_spectrum_powers[MAX_SPECTRUM_POINTS];
static int    g_spectrum_points = 0;
static _Atomic bool g_spectrum_ready = false;
static pthread_mutex_t g_spectrum_mutex = PTHREAD_MUTEX_INITIALIZER;
// Protects FFTW plan creation/destruction on systems without
// fftwf_make_planner_thread_safe (FFTW <3.3.8 or without threads lib).
pthread_mutex_t g_fftw_mutex = PTHREAD_MUTEX_INITIALIZER;

/* ===================== Вспомогалочки ===================== */

static int clamp_lna(int lna) {
    if (lna < 0) lna = 0;
    if (lna > 40) lna = 40;
    return lna - (lna % 8);
}

static int clamp_vga(int vga) {
    if (vga < 0) vga = 0;
    if (vga > 62) vga = 62;
    return vga - (vga % 2);
}

int set_bb_filter(hackrf_device* dev, uint32_t bw_hz) {
    uint32_t actual_bw = hackrf_compute_baseband_filter_bw(bw_hz);
    return hackrf_set_baseband_filter_bandwidth(dev, actual_bw);
}

/* ===================== DC offset: NO-OP как в hackrf_sweep ===================== */

int set_dc_offset(hackrf_device* dev, bool enable, bool auto_correct,
                  int16_t i_offset, int16_t q_offset)
{
    (void)dev; (void)enable; (void)auto_correct; (void)i_offset; (void)q_offset;
    // Ничего не делаем — как делает hackrf_sweep (вообще не настраивает DC).
    return HACKRF_SUCCESS;
}

int calibrate_dc_offset(hackrf_device* dev, int16_t* i_offset, int16_t* q_offset)
{
    (void)dev;
    if (i_offset) *i_offset = 0;
    if (q_offset) *q_offset = 0;
    // Возвращаем «калибровку по нулям» без предупреждений.
    return HACKRF_SUCCESS;
}

/* ===================== Инициализация устройств ===================== */

static void apply_defaults_to_ctx(SdrCtx* c, const SdrParams* d) {
    c->samp_rate = d->samp_rate;
    c->lna_db    = clamp_lna(d->lna_db);
    c->vga_db    = clamp_vga(d->vga_db);
    c->amp_on    = d->amp_on;

    c->dc_offset_enable = d->dc_offset_enable; // не используется девайсом, оставим для UI
    c->dc_offset_auto   = d->dc_offset_auto;
    c->dc_offset_i = 0;
    c->dc_offset_q = 0;

    c->rx_buffer   = NULL;
    c->buffer_size = 0;

    c->running = false;
    pthread_mutex_init(&c->running_mutex, NULL);
    c->retune_count = 0;
    c->last_retune_ns = 0;
    c->thread_data = NULL;
}

int init_devices(SdrCtx* devs, int n, const SdrParams* defaults) {
    if (!devs || n <= 0 || n > MAX_DEVICES) {
        fprintf(stderr, "Invalid parameters: devs=%p, n=%d\n", devs, n);
        return -EINVAL;
    }

    // FFTW's planner is not thread safe on older builds.  We guard plan
    // creation with a mutex instead of relying on
    // fftwf_make_planner_thread_safe(), which may be missing.

    int r = hackrf_init();
    if (r != HACKRF_SUCCESS) {
        fprintf(stderr, "hackrf_init failed: %s (%d)\n", hackrf_error_name(r), r);
        return r;
    }

    hackrf_device_list_t* list = hackrf_device_list();
    if (!list) {
        fprintf(stderr, "Failed to get device list\n");
        hackrf_exit();
        return -EIO;
    }

    if (list->devicecount < n) {
        fprintf(stderr, "Found %d devices, need %d\n", list->devicecount, n);
        hackrf_device_list_free(list);
        hackrf_exit();
        return -ENODEV;
    }

    memset(devs, 0, sizeof(SdrCtx) * n);

    for (int i = 0; i < n; ++i) {
        const char* serial = list->serial_numbers[i];
        printf("Opening device %d: %s\n", i, serial ? serial : "(null)");

        r = hackrf_open_by_serial(serial, &devs[i].dev);
        if (r != HACKRF_SUCCESS) {
            fprintf(stderr, "Failed to open device %s: %s (%d)\n",
                    serial ? serial : "(null)", hackrf_error_name(r), r);

            for (int j = 0; j < i; ++j) {
                if (devs[j].dev) {
                    hackrf_close(devs[j].dev);
                    devs[j].dev = NULL;
                }
            }
            hackrf_device_list_free(list);
            hackrf_exit();
            return r;
        }

        apply_defaults_to_ctx(&devs[i], defaults);

        // Настройка «железа» (без DC!)
        r = hackrf_set_sample_rate(devs[i].dev, devs[i].samp_rate);
        if (r != HACKRF_SUCCESS)
            fprintf(stderr, "Device %d: set_sample_rate failed: %s\n", i, hackrf_error_name(r));

        r = set_bb_filter(devs[i].dev, defaults->bb_filter_bw);
        if (r != HACKRF_SUCCESS)
            fprintf(stderr, "Device %d: set_bb_filter failed: %s\n", i, hackrf_error_name(r));

        r = hackrf_set_lna_gain(devs[i].dev, devs[i].lna_db);
        if (r != HACKRF_SUCCESS)
            fprintf(stderr, "Device %d: set_lna_gain failed: %s\n", i, hackrf_error_name(r));

        r = hackrf_set_vga_gain(devs[i].dev, devs[i].vga_db);
        if (r != HACKRF_SUCCESS)
            fprintf(stderr, "Device %d: set_vga_gain failed: %s\n", i, hackrf_error_name(r));

        r = hackrf_set_amp_enable(devs[i].dev, devs[i].amp_on ? 1 : 0);
        if (r != HACKRF_SUCCESS)
            fprintf(stderr, "Device %d: set_amp_enable failed: %s\n", i, hackrf_error_name(r));

        // DC offset — делаем ничего (как hackrf_sweep)
        if (devs[i].dc_offset_enable) {
            int16_t I=0, Q=0;
            calibrate_dc_offset(devs[i].dev, &I, &Q);   // вернёт 0/0, без лога
            set_dc_offset(devs[i].dev, true, devs[i].dc_offset_auto, I, Q);
        } else {
            set_dc_offset(devs[i].dev, false, false, 0, 0);
        }

        // RX-буфер
        devs[i].buffer_size = 262144; // 256 KB
        devs[i].rx_buffer = (int8_t*)malloc(devs[i].buffer_size);
        if (!devs[i].rx_buffer) {
            fprintf(stderr, "Device %d: Failed to allocate RX buffer\n", i);
        }

        printf("Device %d initialized: %s (LNA:%d VGA:%d AMP:%d)\n",
               i, serial ? serial : "(null)", devs[i].lna_db, devs[i].vga_db, devs[i].amp_on);
    }

    hackrf_device_list_free(list);
    printf("All %d devices initialized successfully\n", n);
    return 0;
}

void teardown_devices(SdrCtx* devs, int n) {
    if (!devs) return;
    printf("Tearing down %d devices\n", n);

    for (int i = 0; i < n; ++i) {
        pthread_mutex_lock(&devs[i].running_mutex);
        if (devs[i].running && devs[i].dev) {
            pthread_mutex_unlock(&devs[i].running_mutex);
            hackrf_stop_rx(devs[i].dev);
            pthread_join(devs[i].thread, NULL);
            pthread_mutex_lock(&devs[i].running_mutex);
            devs[i].running = false;
            pthread_mutex_unlock(&devs[i].running_mutex);
        } else {
            pthread_mutex_unlock(&devs[i].running_mutex);
        }
        if (devs[i].dev) {
            hackrf_close(devs[i].dev);
            devs[i].dev = NULL;
        }
        pthread_mutex_destroy(&devs[i].running_mutex);
        free(devs[i].rx_buffer);
        devs[i].rx_buffer = NULL;
        devs[i].buffer_size = 0;
    }

    hackrf_exit();
    fftwf_cleanup();
}

/* ===================== Непрерывный спектр мастера ===================== */

void hq_update_spectrum(double* freqs, float* powers, int n_points) {
    if (!freqs || !powers || n_points <= 0) return;

    pthread_mutex_lock(&g_spectrum_mutex);

    if (n_points > MAX_SPECTRUM_POINTS) n_points = MAX_SPECTRUM_POINTS;

    memcpy(g_spectrum_freqs,  freqs,  n_points * sizeof(double));
    memcpy(g_spectrum_powers, powers, n_points * sizeof(float));
    g_spectrum_points = n_points;
    g_spectrum_ready  = true;

    pthread_mutex_unlock(&g_spectrum_mutex);
}

int hq_get_spectrum(double* freqs, float* powers, int max_points) {
    pthread_mutex_lock(&g_spectrum_mutex);

    if (!g_spectrum_ready) {
        pthread_mutex_unlock(&g_spectrum_mutex);
        return 0;
    }

    int n = (g_spectrum_points < max_points) ? g_spectrum_points : max_points;
    memcpy(freqs,  g_spectrum_freqs,  n * sizeof(double));
    memcpy(powers, g_spectrum_powers, n * sizeof(float));

    pthread_mutex_unlock(&g_spectrum_mutex);
    return n;
}

/* ===================== Пики/лог ===================== */

void hq_on_peak_detected(SdrCtx* ctx, double f_hz, float dbm) {
    (void)ctx;
    add_peak(f_hz, dbm);
}

void hq_log(const char* lvl, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("[%s] ", lvl ? lvl : "INFO");
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}
