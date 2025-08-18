// hq_grouping.c - ИСПРАВЛЕННАЯ ВЕРСИЯ
// Правильная логика детектора с параметрами из UI

#include "hq_grouping.h"
#include "hq_rssi.h"
#include "hq_master.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

// Параметры группировки (должны настраиваться из UI)
static float g_threshold_offset_db = 20.0f;    // Порог: baseline + offset
static int g_min_width_bins = 3;               // Минимальная ширина сигнала
static int g_min_sweeps = 3;                   // Минимум свипов для подтверждения
static float g_signal_timeout_sec = 2.0f;      // Таймаут сигнала

// Буфер для расчета baseline
#define BASELINE_HISTORY_SIZE 10
static float g_baseline_history[BASELINE_HISTORY_SIZE][MAX_SPECTRUM_POINTS];
static int g_baseline_history_index = 0;
static int g_baseline_history_count = 0;
static float g_baseline[MAX_SPECTRUM_POINTS];
static bool g_baseline_valid = false;

// Глобальные переменные (объявлены как extern в hq_init.c)
extern PeakQueue* g_peaks_queue;
extern WatchItem g_watchlist[];
extern size_t g_watchlist_count;
extern pthread_mutex_t g_watchlist_mutex;
extern double g_grouping_tolerance_hz;
extern float g_ema_alpha;

// ========== Функции очереди пиков ==========

PeakQueue* peak_queue_create(size_t capacity) {
    PeakQueue* q = malloc(sizeof(PeakQueue));
    if (!q) return NULL;
    
    q->buffer = malloc(capacity * sizeof(Peak));
    if (!q->buffer) {
        free(q);
        return NULL;
    }
    
    q->capacity = capacity;
    q->head = 0;
    q->tail = 0;
    pthread_mutex_init(&q->mutex, NULL);
    
    return q;
}

void peak_queue_destroy(PeakQueue* q) {
    if (!q) return;
    pthread_mutex_destroy(&q->mutex);
    free(q->buffer);
    free(q);
}

int peak_queue_push(PeakQueue* q, const Peak* peak) {
    if (!q || !peak) return -1;
    
    pthread_mutex_lock(&q->mutex);
    
    size_t next = (q->tail + 1) % q->capacity;
    if (next == q->head) {
        // Очередь полная - перезаписываем старое
        q->head = (q->head + 1) % q->capacity;
    }
    
    q->buffer[q->tail] = *peak;
    q->tail = next;
    
    pthread_mutex_unlock(&q->mutex);
    return 0;
}

int peak_queue_pop(PeakQueue* q, Peak* peak) {
    if (!q || !peak) return -1;
    
    pthread_mutex_lock(&q->mutex);
    
    if (q->head == q->tail) {
        pthread_mutex_unlock(&q->mutex);
        return -1;
    }
    
    *peak = q->buffer[q->head];
    q->head = (q->head + 1) % q->capacity;
    
    pthread_mutex_unlock(&q->mutex);
    return 0;
}

size_t peak_queue_size(PeakQueue* q) {
    if (!q) return 0;
    
    pthread_mutex_lock(&q->mutex);
    size_t size = (q->tail >= q->head) ? 
                  (q->tail - q->head) : 
                  (q->capacity - q->head + q->tail);
    pthread_mutex_unlock(&q->mutex);
    
    return size;
}

// ========== Функции для установки параметров детектора ==========

void hq_set_detector_params_impl(float threshold_offset_db, int min_width_bins, 
                                int min_sweeps, float timeout_sec) {
    g_threshold_offset_db = threshold_offset_db;
    g_min_width_bins = min_width_bins;
    g_min_sweeps = min_sweeps;
    g_signal_timeout_sec = timeout_sec;
    
    printf("Grouping: Detector params updated: threshold +%.1f dB, width %d bins, %d sweeps, %.1f sec timeout\n",
           threshold_offset_db, min_width_bins, min_sweeps, timeout_sec);
}

// ========== Обновление baseline ==========

static void update_baseline(const float* spectrum, int n_points) {
    if (n_points <= 0 || n_points > MAX_SPECTRUM_POINTS) return;
    
    // Добавляем в историю
    memcpy(g_baseline_history[g_baseline_history_index], spectrum, 
           n_points * sizeof(float));
    
    g_baseline_history_index = (g_baseline_history_index + 1) % BASELINE_HISTORY_SIZE;
    if (g_baseline_history_count < BASELINE_HISTORY_SIZE) {
        g_baseline_history_count++;
    }
    
    // Нужно минимум 3 строки для baseline
    if (g_baseline_history_count < 3) {
        g_baseline_valid = false;
        return;
    }
    
    // Вычисляем медиану по истории для каждого бина
    for (int bin = 0; bin < n_points; bin++) {
        float values[BASELINE_HISTORY_SIZE];
        int count = g_baseline_history_count;
        
        // Собираем значения для этого бина
        for (int i = 0; i < count; i++) {
            values[i] = g_baseline_history[i][bin];
        }
        
        // Простая сортировка для медианы
        for (int i = 0; i < count - 1; i++) {
            for (int j = i + 1; j < count; j++) {
                if (values[i] > values[j]) {
                    float tmp = values[i];
                    values[i] = values[j];
                    values[j] = tmp;
                }
            }
        }
        
        // Медиана
        if (count % 2 == 0) {
            g_baseline[bin] = (values[count/2 - 1] + values[count/2]) / 2.0f;
        } else {
            g_baseline[bin] = values[count/2];
        }
    }
    
    g_baseline_valid = true;
}

// ========== Добавление пика ==========

void add_peak(double f_hz, float rssi_dbm) {
    if (!g_peaks_queue) return;
    
    // Проверяем размер очереди перед добавлением
    size_t queue_size = peak_queue_size(g_peaks_queue);
    if (queue_size > 3000) {  // Если очередь почти полная, пропускаем
        return;
    }
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    
    Peak peak = {
        .f_hz = f_hz,
        .rssi_dbm = rssi_dbm,
        .last_ns = ts.tv_sec * 1000000000ULL + ts.tv_nsec
    };
    
    peak_queue_push(g_peaks_queue, &peak);
}

// ========== Основная функция группировки с детектором ==========

void regroup_frequencies(double delta_hz) {
    if (!g_peaks_queue) return;
    
    // Собираем все пики из очереди с ограничением
    Peak peaks[1000];
    int n_peaks = 0;
    
    Peak p;
    int max_pops = 1000;  // Ограничиваем количество извлекаемых пиков
    int pop_count = 0;
    
    while (peak_queue_pop(g_peaks_queue, &p) == 0 && n_peaks < 1000 && pop_count < max_pops) {
        peaks[n_peaks++] = p;
        pop_count++;
    }
    
    if (n_peaks == 0) return;
    
    // Если извлекли слишком много пиков, логируем предупреждение
    if (pop_count >= max_pops) {
        printf("Grouping: Warning - too many peaks in queue, processing limited batch\n");
    }
    
    // Сортируем по частоте (с ограничением на время сортировки)
    for (int i = 0; i < n_peaks - 1 && i < 100; i++) {  // Ограничиваем сортировку
        for (int j = 0; j < n_peaks - i - 1 && j < 100; j++) {
            if (peaks[j].f_hz > peaks[j+1].f_hz) {
                Peak tmp = peaks[j];
                peaks[j] = peaks[j+1];
                peaks[j+1] = tmp;
            }
        }
    }
    
    pthread_mutex_lock(&g_watchlist_mutex);
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t now_ns = ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    
    // ИСПРАВЛЕНИЕ: Группируем пики в широкополосные сигналы
    int i = 0;
    int max_groups = 100;  // Ограничиваем количество групп
    int group_count = 0;
    
    while (i < n_peaks && group_count < max_groups) {
        // Начало группы
        double group_start_hz = peaks[i].f_hz;
        double group_end_hz = peaks[i].f_hz;
        float max_rssi = peaks[i].rssi_dbm;
        double peak_freq = peaks[i].f_hz;
        int group_size = 1;
        float rssi_sum = peaks[i].rssi_dbm;
        
        // ВАЖНО: Расширяем группу, собирая все близкие пики
        int j = i + 1;
        int max_group_size = 50;  // Ограничиваем размер группы
        
        while (j < n_peaks && group_size < max_group_size) {
            // Проверяем, входит ли следующий пик в группу
            double gap = peaks[j].f_hz - group_end_hz;
            
            // Если разрыв меньше delta_hz, добавляем в группу
            if (gap < delta_hz) {
                group_end_hz = peaks[j].f_hz;
                rssi_sum += peaks[j].rssi_dbm;
                group_size++;
                
                if (peaks[j].rssi_dbm > max_rssi) {
                    max_rssi = peaks[j].rssi_dbm;
                    peak_freq = peaks[j].f_hz;
                }
                j++;
            } else {
                // Разрыв слишком большой, завершаем группу
                break;
            }
        }
        
        // Проверяем минимальную ширину группы
        double bandwidth_hz = group_end_hz - group_start_hz;
        
        // Для узкополосных сигналов (один пик)
        if (group_size == 1) {
            bandwidth_hz = delta_hz / 10;  // Минимальная ширина для одиночного пика
        }
        
        // Вычисляем центр группы (взвешенный по мощности)
        double weighted_center = 0;
        double weight_sum = 0;
        int max_weight_calc = 20;  // Ограничиваем вычисления весов
        for (int k = i; k < j && (k - i) < max_weight_calc; k++) {
            double weight = pow(10, peaks[k].rssi_dbm / 10.0);
            weighted_center += peaks[k].f_hz * weight;
            weight_sum += weight;
        }
        double center_hz = (weight_sum > 0) ? (weighted_center / weight_sum) : peak_freq;
        
        // Ищем существующую запись в watchlist
        int found_idx = -1;
        size_t max_watchlist_search = 50;  // Ограничиваем поиск в watchlist
        for (size_t k = 0; k < g_watchlist_count && k < max_watchlist_search; k++) {
            // Проверяем перекрытие диапазонов
            double existing_start = g_watchlist[k].f_center_hz - g_watchlist[k].bw_hz / 2;
            double existing_end = g_watchlist[k].f_center_hz + g_watchlist[k].bw_hz / 2;
            
            // Если есть перекрытие, считаем это тем же сигналом
            if ((group_start_hz <= existing_end && group_end_hz >= existing_start) ||
                (fabs(center_hz - g_watchlist[k].f_center_hz) < delta_hz / 2)) {
                found_idx = k;
                break;
            }
        }
        
        if (found_idx >= 0) {
            // Обновляем существующий сигнал
            WatchItem* item = &g_watchlist[found_idx];
            
            // Обновляем границы если сигнал расширился
            double new_start = fmin(group_start_hz, item->f_center_hz - item->bw_hz / 2);
            double new_end = fmax(group_end_hz, item->f_center_hz + item->bw_hz / 2);
            double new_bandwidth = new_end - new_start;
            
            // EMA для центра и мощности
            item->f_center_hz = item->f_center_hz * 0.7 + center_hz * 0.3;
            item->rssi_ema = rssi_apply_ema(item->rssi_ema, max_rssi, g_ema_alpha);
            item->bw_hz = fmax(new_bandwidth, bandwidth_hz);
            
            item->last_ns = now_ns;
            item->hit_count++;
            
            // Логируем значимые широкополосные сигналы
            if (item->bw_hz > 500000 && item->hit_count % 10 == 0) {  // > 500 кГц
                printf("Wideband signal: %.3f MHz, BW: %.1f MHz, RSSI: %.1f dBm\n",
                       item->f_center_hz/1e6, item->bw_hz/1e6, item->rssi_ema);
            }
            
        } else if (g_watchlist_count < MAX_WATCHLIST) {
            // Добавляем новый сигнал
            WatchItem* item = &g_watchlist[g_watchlist_count++];
            item->f_center_hz = center_hz;
            item->bw_hz = fmax(bandwidth_hz, 50000.0);  // Минимум 50 кГц
            item->rssi_ema = max_rssi;
            item->last_ns = now_ns;
            item->hit_count = 1;
            
            // Логируем новые широкополосные сигналы
            if (bandwidth_hz > 500000) {
                printf("New wideband signal detected: %.3f MHz, BW: %.1f MHz\n",
                       center_hz/1e6, bandwidth_hz/1e6);
            }
        }
        
        // Переходим к следующей группе
        i = j;
        group_count++;
    }
    
    // Если достигли лимита групп, логируем предупреждение
    if (group_count >= max_groups) {
        printf("Grouping: Warning - too many groups, processing limited\n");
    }
    
    // Удаляем устаревшие записи
    size_t write_idx = 0;
    uint64_t timeout_ns = (uint64_t)(g_signal_timeout_sec * 1e9);
    int max_cleanup_iterations = 100;  // Ограничиваем количество итераций очистки
    int cleanup_count = 0;
    
    for (size_t i = 0; i < g_watchlist_count && cleanup_count < max_cleanup_iterations; i++) {
        if ((now_ns - g_watchlist[i].last_ns) < timeout_ns) {
            if (write_idx != i) {
                g_watchlist[write_idx] = g_watchlist[i];
            }
            write_idx++;
        } else {
            // Логируем потерю сигнала
            if (g_watchlist[i].bw_hz > 500000) {
                printf("Lost wideband signal: %.3f MHz\n", g_watchlist[i].f_center_hz/1e6);
            }
        }
        cleanup_count++;
    }
    
    // Если достигли лимита итераций очистки, логируем предупреждение
    if (cleanup_count >= max_cleanup_iterations) {
        printf("Grouping: Warning - cleanup limited, some items may remain\n");
    }
    
    g_watchlist_count = write_idx;
    
    pthread_mutex_unlock(&g_watchlist_mutex);
}

// ========== Функция обработки спектра от мастера ==========

void process_master_spectrum(const double* freqs_hz, const float* powers_dbm, int n_points) {
    if (!freqs_hz || !powers_dbm || n_points <= 0) return;
    
    // Обновляем baseline
    update_baseline(powers_dbm, n_points);
    
    if (!g_baseline_valid) {
        // Еще недостаточно данных для baseline
        return;
    }
    
    // Детектируем сигналы выше порога
    float threshold_dbm;
    
    for (int i = 0; i < n_points; i++) {
        // Адаптивный порог для каждого бина
        threshold_dbm = g_baseline[i] + g_threshold_offset_db;
        
        // Проверяем превышение порога
        if (powers_dbm[i] > threshold_dbm) {
            // Добавляем пик для группировки
            add_peak(freqs_hz[i], powers_dbm[i]);
        }
    }
    
    // Запускаем группировку
    regroup_frequencies(g_grouping_tolerance_hz);
}