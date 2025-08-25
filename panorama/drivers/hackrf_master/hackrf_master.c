// hackrf_master.c - sweep backend для Panorama (API hackrf 0.9 / 2024.02.x)
// Работает как hackrf_sweep: hackrf_init_sweep + hackrf_start_rx_sweep,
// в rx_callback парсим 0x7f 0x7f + center_freq и отдаём 2-4 сегмента через hq_multi_segment_cb.

#include "hackrf_master.h"
#include <libhackrf/hackrf.h>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#define DEFAULT_SAMPLE_RATE_HZ 20000000
#define DEFAULT_FFT_SIZE       8192
#define OFFSET_HZ              7500000          // 7.5 MHz
#define TUNE_STEP_MHZ          20               // шаг тюнинга (МГц)
#define FREQ_ONE_MHZ           (1000000U)
#define SWEEP_STYLE_INTERLEAVED 0               // соответствует HACKRF_SWEEP_STYLE_INTERLEAVED
#define MAX_CALIBRATION_ENTRIES 1000            // максимальное количество калибровочных записей

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Глобальные переменные
static hackrf_device* dev = NULL;
static volatile int running = 0;
static hq_multi_segment_cb g_multi_cb = NULL;
static hq_segment_cb g_legacy_cb = NULL;  // для обратной совместимости
static void* g_user = NULL;
static char g_err[256] = {0};

// FFT и обработка сигнала
static fftwf_complex* fft_in = NULL;
static fftwf_complex* fft_out = NULL;
static fftwf_plan fft_plan;
static float* window = NULL;
static float* pwr = NULL;
static int fft_size = DEFAULT_FFT_SIZE;
static double fft_bin_width = 0.0;
static double g_bin_hz = 0.0;

// Настройки режима
static int g_segment_mode = 4;  // по умолчанию 4 сегмента
static int g_lna_db = 24;
static int g_vga_db = 20;
static int g_amp_on = 0;

// Калибровка
static int g_calibration_enabled = 0;
static struct {
    float freq_mhz;
    int lna_db;
    int vga_db;
    int amp_on;
    float offset_db;
} g_calibration_table[MAX_CALIBRATION_ENTRIES];
static int g_calibration_count = 0;

// Буферы для сегментов
static double* g_segment_freqs[MAX_SEGMENTS] = {NULL};
static float* g_segment_pwr[MAX_SEGMENTS] = {NULL};
static int g_segment_cap[MAX_SEGMENTS] = {0};

// ---- утилиты ошибок ----
const char* hq_last_error(void) { return g_err; }
static void set_err(const char* s) { snprintf(g_err, sizeof(g_err), "%s", s); }
static void set_errf(const char* fmt, ...) { 
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, args);
    va_end(args);
}

// ---- функции калибровки ----
static float cal_lookup_db(float freq_mhz, int lna_db, int vga_db, int amp_on) {
    if (!g_calibration_enabled || g_calibration_count == 0) {
        return 0.0f;
    }
    
    // Ищем ближайшую запись в таблице калибровки
    float min_distance = 1e9f;
    float best_offset = 0.0f;
    
    for (int i = 0; i < g_calibration_count; i++) {
        float freq_dist = fabsf(freq_mhz - g_calibration_table[i].freq_mhz);
        int lna_dist = abs(lna_db - g_calibration_table[i].lna_db);
        int vga_dist = abs(vga_db - g_calibration_table[i].vga_db);
        int amp_dist = abs(amp_on - g_calibration_table[i].amp_on);
        
        // Взвешенное расстояние (частота важнее)
        float total_dist = freq_dist * 0.7f + lna_dist * 0.1f + vga_dist * 0.1f + amp_dist * 0.1f;
        
        if (total_dist < min_distance) {
            min_distance = total_dist;
            best_offset = g_calibration_table[i].offset_db;
        }
    }
    
    return best_offset;
}

int hq_load_calibration(const char* csv_path) {
    if (!csv_path) {
        set_err("CSV path is NULL");
        return -1;
    }
    
    FILE* f = fopen(csv_path, "r");
    if (!f) {
        set_errf("Cannot open calibration file: %s", csv_path);
        return -2;
    }
    
    char line[256];
    int line_num = 0;
    g_calibration_count = 0;
    
    // Пропускаем заголовок
    if (fgets(line, sizeof(line), f)) {
        line_num++;
    }
    
    while (fgets(line, sizeof(line), f) && g_calibration_count < MAX_CALIBRATION_ENTRIES) {
        line_num++;
        
        // Парсим CSV: freq_mhz,lna,vga,amp,offset_db
        float freq_mhz, offset_db;
        int lna_db, vga_db, amp_on;
        
        if (sscanf(line, "%f,%d,%d,%d,%f", &freq_mhz, &lna_db, &vga_db, &amp_on, &offset_db) == 5) {
            g_calibration_table[g_calibration_count].freq_mhz = freq_mhz;
            g_calibration_table[g_calibration_count].lna_db = lna_db;
            g_calibration_table[g_calibration_count].vga_db = vga_db;
            g_calibration_table[g_calibration_count].amp_on = amp_on;
            g_calibration_table[g_calibration_count].offset_db = offset_db;
            g_calibration_count++;
        } else {
            printf("Warning: Invalid calibration line %d: %s", line_num, line);
        }
    }
    
    fclose(f);
    printf("Loaded %d calibration entries from %s\n", g_calibration_count, csv_path);
    return 0;
}

int hq_enable_calibration(int enable) {
    g_calibration_enabled = enable ? 1 : 0;
    return 0;
}

int hq_get_calibration_status(void) {
    return g_calibration_enabled;
}

// ---- настройка режима сегментов ----
int hq_set_segment_mode(int mode) {
    if (mode != 4) {
        set_err("Segment mode must be 4");
        return -1;
    }
    g_segment_mode = mode;
    return 0;
}

int hq_get_segment_mode(void) {
    return g_segment_mode;
}

// ---- перечисление устройств ----
int hq_device_count(void) {
    int r = hackrf_init();
    if (r != HACKRF_SUCCESS) return 0;
    
    struct hackrf_device_list* lst = hackrf_device_list();
    if (!lst) {
        hackrf_exit();
        return 0;
    }
    int n = lst->devicecount;
    hackrf_device_list_free(lst);
    hackrf_exit();
    return n;
}

int hq_get_device_serial(int idx, char* out, int cap) {
    if (!out || cap <= 0) return -1;
    
    int r = hackrf_init();
    if (r != HACKRF_SUCCESS) return -2;
    
    struct hackrf_device_list* lst = hackrf_device_list();
    if (!lst) {
        hackrf_exit();
        return -3;
    }
    
    if (idx < 0 || idx >= lst->devicecount) {
        hackrf_device_list_free(lst);
        hackrf_exit();
        return -4;
    }
    
    const char* s = lst->serial_numbers[idx];
    if (!s) s = "";
    snprintf(out, cap, "%s", s);
    
    hackrf_device_list_free(lst);
    hackrf_exit();
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
        set_errf("Device with serial suffix '%s' not found", serial_suffix);
        return -102;
    }

    r = hackrf_open_by_serial(lst->serial_numbers[found], &dev);
    hackrf_device_list_free(lst);
    if (r != HACKRF_SUCCESS) {
        set_errf("hackrf_open_by_serial failed: %s", hackrf_error_name(r));
        return r;
    }

    // Инициализируем FFT
    fft_in  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    fft_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    pwr     = (float*)malloc(sizeof(float) * fft_size);
    window  = (float*)malloc(sizeof(float) * fft_size);
    
    if (!fft_in || !fft_out || !pwr || !window) {
        set_err("Memory allocation failed");
        return -103;
    }
    
    // Создаем окно Хэмминга
    for (int i = 0; i < fft_size; i++) {
        window[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (fft_size - 1));
    }
    
    fft_plan = fftwf_plan_dft_1d(fft_size, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    if (!fft_plan) {
        set_err("FFT plan creation failed");
        return -104;
    }

    return 0;
}

void hq_close(void) {
    if (fft_plan) fftwf_destroy_plan(fft_plan);
    if (fft_in) fftwf_free(fft_in);
    if (fft_out) fftwf_free(fft_out);
    if (pwr) free(pwr);
    if (window) free(window);
    
    // Освобождаем буферы сегментов
    for (int i = 0; i < MAX_SEGMENTS; i++) {
        if (g_segment_freqs[i]) free(g_segment_freqs[i]);
        if (g_segment_pwr[i]) free(g_segment_pwr[i]);
        g_segment_freqs[i] = NULL;
        g_segment_pwr[i] = NULL;
        g_segment_cap[i] = 0;
    }
    
    if (dev) {
        hackrf_close(dev);
        dev = NULL;
    }
    hackrf_exit();
}

int hq_configure(double f_start_mhz, double f_stop_mhz, double bin_hz,
                  int lna_db, int vga_db, int amp_on) {
    if (!dev) { set_err("device not opened"); return -1; }

    g_lna_db = lna_db;
    g_vga_db = vga_db;
    g_amp_on = amp_on;
    g_bin_hz = bin_hz;
    fft_bin_width = bin_hz;

    // Устанавливаем параметры HackRF
    int r = hackrf_set_lna_gain(dev, lna_db);
    if (r != HACKRF_SUCCESS) {
        set_errf("hackrf_set_lna_gain failed: %s", hackrf_error_name(r));
        return r;
    }

    r = hackrf_set_vga_gain(dev, vga_db);
    if (r != HACKRF_SUCCESS) {
        set_errf("hackrf_set_vga_gain failed: %s", hackrf_error_name(r));
        return r;
    }

    r = hackrf_set_amp_enable(dev, amp_on);
    if (r != HACKRF_SUCCESS) {
        set_errf("hackrf_set_amp_enable failed: %s", hackrf_error_name(r));
        return r;
    }

    // sweep init (как в hackrf_sweep.c)
    uint16_t freqs[2];
    if (f_stop_mhz < f_start_mhz) {
        double tmp = f_start_mhz; f_start_mhz = f_stop_mhz; f_stop_mhz = tmp;
    }
    freqs[0] = (uint16_t)floor(f_start_mhz);
    freqs[1] = (uint16_t)ceil(f_stop_mhz);

    r = hackrf_init_sweep(
        dev,
        freqs, 1,
        BYTES_PER_BLOCK,                          // размер блока из libhackrf.h
        TUNE_STEP_MHZ * FREQ_ONE_MHZ,            // шаг 20 МГц
        OFFSET_HZ,                               // offset 7.5 МГц
        SWEEP_STYLE_INTERLEAVED                  // interleaved режим (0)
    );
    if (r != HACKRF_SUCCESS) {
        set_errf("hackrf_init_sweep failed: %s", hackrf_error_name(r));
        return r;
    }
    
    // Вычисляем корректировки окна
    float window_loss_db = -1.76f;  // для окна Хэмминга
    float enbw_corr_db = 1.85f;    // эквивалентная ширина полосы для окна Хэмминга
    
    printf("Window corrections: loss=%.2f dB, ENBW=%.2f dB\n", window_loss_db, enbw_corr_db);
    
    return 0;
}

// ---- обработка блока с многосекционным режимом ----
static void process_block_multi_segment(int8_t* buf, int valid_length, uint64_t center_hz) {
    int count = fft_size;
    if (valid_length/2 < count) return;

    // Нормализация входных данных
    const float scale = 1.0f/128.0f;
    for (int i = 0; i < count; i++) {
        float I = buf[2*i]   * scale;
        float Q = buf[2*i+1] * scale;
        fft_in[i][0] = I * window[i];
        fft_in[i][1] = Q * window[i];
    }
    
    fftwf_execute(fft_plan);

    // Вычисляем мощность с нормализацией FFT
    const float fft_norm = 1.0f / (float)fft_size;
    const float window_loss_db = -1.76f;  // для окна Хэмминга
    const float enbw_corr_db = 1.85f;    // эквивалентная ширина полосы
    
    for (int i = 0; i < count; i++) {
        float re = fft_out[i][0] * fft_norm, im = fft_out[i][1] * fft_norm;
        float mag = re*re + im*im;
        if (mag <= 1e-12f) mag = 1e-12f;
        
        // Нормализация мощности: dBFS + поправки окна + калибровка
        float power_dbfs = 10.0f * log10f(mag);
        float power_dbm = power_dbfs + (window_loss_db + enbw_corr_db) + 
                         cal_lookup_db((float)center_hz/1e6f, g_lna_db, g_vga_db, g_amp_on);
        
        pwr[i] = power_dbm;
    }

    // Создаем сегменты согласно выбранному режиму
    int segments_per_pass = g_segment_mode;
    int bins_per_segment = count / segments_per_pass;
    
    // Вычисляем частоты для каждого сегмента
    for (int seg = 0; seg < segments_per_pass; seg++) {
        // Убеждаемся, что у нас достаточно памяти для сегмента
        if (bins_per_segment > g_segment_cap[seg]) {
            if (g_segment_freqs[seg]) free(g_segment_freqs[seg]);
            if (g_segment_pwr[seg]) free(g_segment_pwr[seg]);
            g_segment_cap[seg] = bins_per_segment;
            g_segment_freqs[seg] = (double*)malloc(sizeof(double) * bins_per_segment);
            g_segment_pwr[seg] = (float*)malloc(sizeof(float) * bins_per_segment);
        }
        
        // Вычисляем частотный диапазон сегмента
        uint64_t seg_start_hz, seg_end_hz;
        int seg_start_bin, seg_end_bin;
        
        if (seg == SEGMENT_A) {
            // [f-OFFSET, f-OFFSET+Fs/4]
            seg_start_hz = center_hz - OFFSET_HZ;
            seg_end_hz = center_hz - OFFSET_HZ + DEFAULT_SAMPLE_RATE_HZ/4;
            seg_start_bin = 0;
            seg_end_bin = bins_per_segment;
        } else if (seg == SEGMENT_B) {
            // [f+OFFSET+Fs/2, f+OFFSET+3Fs/4]
            seg_start_hz = center_hz + OFFSET_HZ + DEFAULT_SAMPLE_RATE_HZ/2;
            seg_end_hz = center_hz + OFFSET_HZ + 3*DEFAULT_SAMPLE_RATE_HZ/4;
            seg_start_bin = bins_per_segment;
            seg_end_bin = 2*bins_per_segment;
        } else if (seg == SEGMENT_C && g_segment_mode == 4) {
            // [f+OFFSET+Fs/4, f+OFFSET+Fs/2]
            seg_start_hz = center_hz + OFFSET_HZ + DEFAULT_SAMPLE_RATE_HZ/4;
            seg_end_hz = center_hz + OFFSET_HZ + DEFAULT_SAMPLE_RATE_HZ/2;
            seg_start_bin = 2*bins_per_segment;
            seg_end_bin = 3*bins_per_segment;
        } else if (seg == SEGMENT_D && g_segment_mode == 4) {
            // [f+OFFSET+3Fs/4, f+OFFSET+Fs]
            seg_start_hz = center_hz + OFFSET_HZ + 3*DEFAULT_SAMPLE_RATE_HZ/4;
            seg_end_hz = center_hz + OFFSET_HZ + DEFAULT_SAMPLE_RATE_HZ;
            seg_start_bin = 3*bins_per_segment;
            seg_end_bin = 4*bins_per_segment;
        } else {
            continue;  // пропускаем неиспользуемые сегменты
        }
        
        // Заполняем данные сегмента
        for (int i = 0; i < bins_per_segment; i++) {
            int src_bin = seg_start_bin + i;
            if (src_bin < count) {
                g_segment_freqs[seg][i] = seg_start_hz + (i + 0.5) * fft_bin_width;
                g_segment_pwr[seg][i] = pwr[src_bin];
            }
        }
    }
    
    // Вызываем новый колбэк с многосегментными данными
    if (g_multi_cb) {
        hq_segment_data_t segments[MAX_SEGMENTS];
        int valid_segments = 0;
        
        for (int seg = 0; seg < segments_per_pass; seg++) {
            if (g_segment_freqs[seg] && g_segment_pwr[seg]) {
                segments[valid_segments].freqs_hz = g_segment_freqs[seg];
                segments[valid_segments].data_dbm = g_segment_pwr[seg];
                segments[valid_segments].count = bins_per_segment;
                segments[valid_segments].segment_id = seg;
                segments[valid_segments].hz_low = (uint64_t)g_segment_freqs[seg][0];
                segments[valid_segments].hz_high = (uint64_t)g_segment_freqs[seg][bins_per_segment-1];
                valid_segments++;
            }
        }
        
        if (valid_segments > 0) {
            g_multi_cb(segments, valid_segments, fft_bin_width, center_hz, g_user);
        }
    }
}

// ---- обработка блока для обратной совместимости ----
static void process_block_legacy(int8_t* buf, int valid_length, uint64_t center_hz) {
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
        pwr[i] = 10.0f * log10f(mag) + cal_lookup_db((float)center_hz/1e6f, g_lna_db, g_vga_db, g_amp_on);
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

    if (g_legacy_cb) {
        g_legacy_cb(freqs, pwr, quarter, fft_bin_width,
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
        
        if (g_multi_cb) {
            process_block_multi_segment(iq, transfer->valid_length-16, freq);
        } else if (g_legacy_cb) {
            process_block_legacy(iq, transfer->valid_length-16, freq);
        }
    }
    return 0;
}

// ---- старт/стоп ----
int hq_start_multi_segment(hq_multi_segment_cb cb, void* user) {
    if (!dev) { set_err("device not opened"); return -1; }
    g_multi_cb = cb; g_legacy_cb = NULL; g_user = user; running = 1;
    int r = hackrf_start_rx_sweep(dev, rx_callback, NULL);
    if (r != HACKRF_SUCCESS) {
        set_errf("hackrf_start_rx_sweep failed: %s", hackrf_error_name(r));
        running = 0;
        return r;
    }
    return 0;
}

int hq_start(hq_segment_cb cb, void* user) {
    if (!dev) { set_err("device not opened"); return -1; }
    g_legacy_cb = cb; g_multi_cb = NULL; g_user = user; running = 1;
    int r = hackrf_start_rx_sweep(dev, rx_callback, NULL);
    if (r != HACKRF_SUCCESS) {
        set_errf("hackrf_start_rx_sweep failed: %s", hackrf_error_name(r));
        running = 0;
        return r;
    }
    return 0;
}

int hq_stop(void) {
    running = 0;
    return hackrf_stop_rx(dev);
}
