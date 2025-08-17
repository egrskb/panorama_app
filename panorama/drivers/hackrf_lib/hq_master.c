// hq_master.c — быстрый мастер на основе логики hackrf_sweep/example.c
// - N подбирается из желаемого шага сетки (bin_width = Fs/N) как в sweep
// - обработка двух четвертей спектра из каждого блока: 1+5N/8 и 1+N/8
// - частота берётся из заголовка блока 0x7F 0x7F (байты 2..9) — как в hackrf_sweep.c
// - без каких-либо DC-коррекций (точно как утилита hackrf_sweep)
// - экспортирует start_master, совместим с hq_api.c

#include "hq_master.h"   // объявляет SweepPlan
#include "hq_init.h"     // объявляет SdrCtx, hq_update_spectrum(...)
#include "hq_grouping.h"
#include "hq_rssi.h"

#include <libhackrf/hackrf.h>
#include <fftw3.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/time.h>
#include <inttypes.h>

#ifndef BLOCKS_PER_TRANSFER
#define BLOCKS_PER_TRANSFER 16
#endif

// hackrf.h уже определяет BYTES_PER_BLOCK = 16384
// используем то, что даёт libhackrf
#ifndef BYTES_PER_BLOCK
#define BYTES_PER_BLOCK 16384
#endif

#define DEFAULT_SAMPLE_RATE_HZ            20000000u   // 20 MHz
#define DEFAULT_BASEBAND_FILTER_BANDWIDTH 15000000u

#define FREQ_ONE_MHZ 1000000ull

#ifndef INTERLEAVED
#define INTERLEAVED 1
#endif

#define TUNE_STEP_HZ 20000000u
#define OFFSET_HZ     7500000u

// внешние функции из проекта
extern void hq_on_peak_detected(SdrCtx* ctx, double f_hz, float dbm);

// ================== Внутреннее состояние мастера ==================

typedef struct {
    SdrCtx* ctx;

    // sweep-план
    double f_start;
    double f_stop;
    double f_step;   // желаемая дискретизация (если 0 — берём Fs/N)

    // FFT
    int N;
    double Fs;
    double bin_w;
    fftwf_complex* in;
    fftwf_complex* out;
    fftwf_plan plan;
    float* window;
    float* pwr_db;

    // итоговый «полный» спектр на сетке [f_start..f_stop] с шагом f_step≈bin_w
    double* freqs;
    float*  powers;
    int*    counts;
    size_t  n_points;

    // тайминг/EMA
    float ema_alpha;
    struct timeval usb_tv;

    // предвычисления
    double* bin_offsets_hz; // (N/4) элементов: (i+0.5)*bin_w
    int peak_decim;         // разрежение детекции пиков (напр. 4)
} MasterData;

// ================== Утилиты ==================

static inline float pwr_db(float re, float im) {
    float m = re*re + im*im;
    if (m < 1e-20f) m = 1e-20f;
    return 10.0f * log10f(m);
}

// ================== RX callback (как в hackrf_sweep.c) ==================

static int master_rx_cb(hackrf_transfer* t) {
    MasterData* d = (MasterData*)t->rx_ctx;
    SdrCtx* ctx = d->ctx;
    if (!ctx || !ctx->running) return -1;

    // инициализируем время первой пачки
    if (d->usb_tv.tv_sec == 0 && d->usb_tv.tv_usec == 0)
        gettimeofday(&d->usb_tv, NULL);

    int8_t* buf = (int8_t*)t->buffer;
    const int N = d->N;
    const int q = N/4;

    for (int j = 0; j < BLOCKS_PER_TRANSFER; ++j) {
        uint8_t* ubuf = (uint8_t*)buf;

        // заголовок блока
        if (!(ubuf[0] == 0x7F && ubuf[1] == 0x7F)) {
            buf += BYTES_PER_BLOCK;
            continue;
        }

        // частота «настройки» блока (как в hackrf_sweep.c)
        uint64_t frequency =
            ((uint64_t)ubuf[9] << 56) | ((uint64_t)ubuf[8] << 48) |
            ((uint64_t)ubuf[7] << 40) | ((uint64_t)ubuf[6] << 32) |
            ((uint64_t)ubuf[5] << 24) | ((uint64_t)ubuf[4] << 16) |
            ((uint64_t)ubuf[3] <<  8) |  (uint64_t)ubuf[2];

        // берём ПОСЛЕДНИЕ N I/Q отсчётов из блока (как в hackrf_sweep.c)
        int8_t* samp = buf + BYTES_PER_BLOCK - (N * 2);

        // окно Ханна + нормализация в [-1..1]
        for (int i = 0; i < N; ++i) {
            d->in[i][0] = (float)samp[2*i]   * d->window[i] / 128.0f;
            d->in[i][1] = (float)samp[2*i+1] * d->window[i] / 128.0f;
        }

        // FFT
        fftwf_execute(d->plan);
        for (int i = 0; i < N; ++i)
            d->pwr_db[i] = pwr_db(d->out[i][0], d->out[i][1]);

        const double bw = d->bin_w;

        // Секция A: [frequency, frequency + Fs/4) -> биновые индексы 1 + 5N/8 .. (1+5N/8) + q-1
        {
            uint64_t hz_low = frequency;
            int bin0 = 1 + (5*N)/8;
            for (int i = 0; i < q; ++i) {
                int bin = bin0 + i;
                float dbm = d->pwr_db[bin];
                double f_hz = (double)hz_low + d->bin_offsets_hz[i];

                // маппинг в общую сетку
                long idx = (long)((f_hz - d->f_start) / d->f_step);
                if (idx >= 0 && (size_t)idx < d->n_points) {
                    float* dst = &d->powers[idx];
                    int*   cnt = &d->counts[idx];
                    if (*cnt == 0) *dst = dbm;
                    else *dst = (1.0f - d->ema_alpha) * (*dst) + d->ema_alpha * dbm;
                    (*cnt)++;

                    // разрежённая детекция пиков (не грузим CPU)
                    if (((i & (d->peak_decim - 1)) == 0) && dbm > -75.0f) {
                        hq_on_peak_detected(ctx, f_hz, dbm);
                    }
                }
            }
        }

        // Секция B: [frequency + Fs/2, frequency + 3Fs/4)
        {
            uint64_t hz_low = frequency + (uint64_t)(d->Fs / 2.0);
            int bin0 = 1 + (N/8);
            for (int i = 0; i < q; ++i) {
                int bin = bin0 + i;
                float dbm = d->pwr_db[bin];
                double f_hz = (double)hz_low + d->bin_offsets_hz[i];

                long idx = (long)((f_hz - d->f_start) / d->f_step);
                if (idx >= 0 && (size_t)idx < d->n_points) {
                    float* dst = &d->powers[idx];
                    int*   cnt = &d->counts[idx];
                    if (*cnt == 0) *dst = dbm;
                    else *dst = (1.0f - d->ema_alpha) * (*dst) + d->ema_alpha * dbm;
                    (*cnt)++;

                    if (((i & (d->peak_decim - 1)) == 0) && dbm > -75.0f) {
                        hq_on_peak_detected(ctx, f_hz, dbm);
                    }
                }
            }
        }

        buf += BYTES_PER_BLOCK;
    }

    return 0;
}

// ================== Поток мастера ==================

static void* master_thread_fn(void* arg) {
    MasterData* d = (MasterData*)arg;
    SdrCtx* ctx = d->ctx;

    // Настройка «железа» как в hackrf_sweep
    d->Fs = (double)DEFAULT_SAMPLE_RATE_HZ;
    hackrf_set_sample_rate_manual(ctx->dev, DEFAULT_SAMPLE_RATE_HZ, 1);
    hackrf_set_baseband_filter_bandwidth(ctx->dev, DEFAULT_BASEBAND_FILTER_BANDWIDTH);
    hackrf_set_vga_gain(ctx->dev, ctx->vga_db);
    hackrf_set_lna_gain(ctx->dev, ctx->lna_db);
    hackrf_set_amp_enable(ctx->dev, ctx->amp_on ? 1 : 0);

    // Выбор N по желаемому шагу сетки (как в sweep: N = Fs / bin_width)
    double requested_bin = (d->f_step > 0.0) ? d->f_step : (d->Fs / 8192.0);
    if (requested_bin < 2445.0) requested_bin = 2445.0;
    if (requested_bin > 5e6)    requested_bin = 5e6;

    int N = (int)floor(d->Fs / requested_bin);
    if (N < 4)    N = 4;
    if (N > 8180) N = 8180;
    while ((N + 4) % 8) N++;
    d->N = N;
    d->bin_w = d->Fs / (double)d->N;

    // FFT ресурсы
    d->in  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * d->N);
    d->out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * d->N);
    d->pwr_db = (float*)malloc(sizeof(float) * d->N);
    d->window = (float*)malloc(sizeof(float) * d->N);

    // Окно Ханна
    for (int i = 0; i < d->N; ++i)
        d->window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (d->N - 1)));

    d->plan = fftwf_plan_dft_1d(d->N, d->in, d->out, FFTW_FORWARD, FFTW_ESTIMATE);
    // один раз прогоним план (как советуют в hackrf_sweep)
    fftwf_execute(d->plan);

    // Предвычислим смещения частот для q=N/4 бинов
    const int q = d->N / 4;
    d->bin_offsets_hz = (double*)malloc(sizeof(double) * q);
    for (int i = 0; i < q; ++i)
        d->bin_offsets_hz[i] = (i + 0.5) * d->bin_w;

    d->ema_alpha = 0.25f;
    d->peak_decim = 4;
    memset(&d->usb_tv, 0, sizeof(d->usb_tv));

    // Итоговая сетка: шаг = d->bin_w (почти равен желаемому d->f_step)
    if (d->f_start < 0) d->f_start = 0;
    if (d->f_stop  < d->f_start) d->f_stop = d->f_start + d->Fs/4.0;

    d->f_step = d->bin_w;
    uint64_t span_hz = (uint64_t) llround(d->f_stop - d->f_start);
    d->n_points = (size_t)ceil((double)span_hz / d->f_step);
    if (d->n_points < 1) d->n_points = 1;

    d->freqs  = (double*)malloc(sizeof(double) * d->n_points);
    d->powers = (float*) calloc(d->n_points, sizeof(float));
    d->counts = (int*)   calloc(d->n_points, sizeof(int));
    for (size_t i = 0; i < d->n_points; ++i)
        d->freqs[i] = d->f_start + (double)i * d->f_step;

    // Конфигурация hackrf_sweep (одна пара диапазонов)
    uint32_t fmin_mhz = (uint32_t)(d->f_start / FREQ_ONE_MHZ);
    uint32_t fmax_mhz = (uint32_t)(d->f_stop  / FREQ_ONE_MHZ);
    // округлим верх до кратного 20 МГц, чтобы шаги тюнинга укладывались
    uint32_t steps = (uint32_t)((fmax_mhz - fmin_mhz + 19) / 20);
    fmax_mhz = fmin_mhz + steps * 20;

    uint16_t freqs_mhz[2];
    freqs_mhz[0] = (uint16_t)fmin_mhz;
    freqs_mhz[1] = (uint16_t)fmax_mhz;

    int r = hackrf_init_sweep(ctx->dev,
                              freqs_mhz,
                              1,
                              BYTES_PER_BLOCK,
                              TUNE_STEP_HZ,
                              OFFSET_HZ,
                              INTERLEAVED);
    if (r != HACKRF_SUCCESS) {
        fprintf(stderr, "hackrf_init_sweep failed: %s (%d)\n", hackrf_error_name(r), r);
        goto cleanup;
    }

    ctx->running = true;
    r = hackrf_start_rx_sweep(ctx->dev, master_rx_cb, d);
    if (r != HACKRF_SUCCESS) {
        fprintf(stderr, "hackrf_start_rx_sweep failed: %s (%d)\n", hackrf_error_name(r), r);
        goto cleanup;
    }

    // периодически отправляем спектр в UI
    struct timeval t_prev = {0}, t_now = {0};
    gettimeofday(&t_prev, NULL);

    while (ctx->running && hackrf_is_streaming(ctx->dev) == HACKRF_TRUE) {
        usleep(50000); // 50 ms
        gettimeofday(&t_now, NULL);
        double dt = (t_now.tv_sec - t_prev.tv_sec) + 1e-6 * (t_now.tv_usec - t_prev.tv_usec);
        if (dt >= 0.10) { // каждые ~100 ms
            hq_update_spectrum(d->freqs, d->powers, (int)d->n_points);
            t_prev = t_now;
        }
    }

    hackrf_stop_rx(ctx->dev);

cleanup:
    if (d->plan) fftwf_destroy_plan(d->plan);
    if (d->in)   fftwf_free(d->in);
    if (d->out)  fftwf_free(d->out);
    free(d->pwr_db);
    free(d->window);
    free(d->bin_offsets_hz);
    free(d->freqs);
    free(d->powers);
    free(d->counts);
    free(d);
    ctx->thread_data = NULL;
    ctx->running = false;
    return NULL;
}

// ================== API ==================

// ЯВНО экспортируем символ, который ждёт твой hq_api.c
__attribute__((visibility("default")))
int start_master(SdrCtx* ctx, const SweepPlan* plan) {
    if (!ctx || !ctx->dev || !plan) return -1;
    if (ctx->running) return 0;

    MasterData* d = (MasterData*)calloc(1, sizeof(MasterData));
    if (!d) return -2;

    d->ctx     = ctx;
    d->f_start = (double)plan->start_hz;
    d->f_stop  = (double)plan->stop_hz;
    d->f_step  = (double)plan->step_hz; // 0 => выберем из Fs/N

    ctx->thread_data = d;
    ctx->running = true;

    if (pthread_create(&ctx->thread, NULL, master_thread_fn, d) != 0) {
        ctx->running = false;
        ctx->thread_data = NULL;
        free(d);
        return -3;
    }
    return 0;
}

void hq_master_stop(SdrCtx* ctx) {
    if (!ctx) return;
    if (!ctx->running) return;
    ctx->running = false;
    pthread_join(ctx->thread, NULL);
}
