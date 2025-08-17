// hq_grouping.c - ПОЛНАЯ ЗАМЕНА ФАЙЛА
#include "hq_grouping.h"
#include "hq_rssi.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

// Параметры группировки
#define DEFAULT_GROUP_TOLERANCE_HZ 1000000.0  // 1 МГц по умолчанию
#define MIN_SIGNAL_WIDTH_HZ 100000.0          // Минимальная ширина сигнала 100 кГц
#define MAX_SIGNAL_WIDTH_HZ 20000000.0        // Максимальная ширина 20 МГц (видео)
#define SIGNAL_TIMEOUT_NS 3000000000ULL       // 3 секунды таймаут

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
        // Queue full - overwrite oldest
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
        // Queue empty
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

void add_peak(double f_hz, float rssi_dbm) {
    if (!g_peaks_queue) return;
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    
    Peak peak = {
        .f_hz = f_hz,
        .rssi_dbm = rssi_dbm,
        .last_ns = ts.tv_sec * 1000000000ULL + ts.tv_nsec
    };
    
    peak_queue_push(g_peaks_queue, &peak);
    
    // Логируем для отладки
    if (rssi_dbm > -70.0f) {
        printf("Peak detected: %.3f MHz, %.1f dBm\n", f_hz/1e6, rssi_dbm);
    }
}

// Улучшенная группировка для широкополосных сигналов
void regroup_frequencies(double delta_hz) {
    if (!g_peaks_queue) return;
    
    // Собираем все пики из очереди
    Peak peaks[2000];  // Увеличили размер для широкополосных сигналов
    int n_peaks = 0;
    
    Peak p;
    while (peak_queue_pop(g_peaks_queue, &p) == 0 && n_peaks < 2000) {
        peaks[n_peaks++] = p;
    }
    
    if (n_peaks == 0) return;
    
    // Сортируем по частоте
    for (int i = 0; i < n_peaks - 1; i++) {
        for (int j = 0; j < n_peaks - i - 1; j++) {
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
    
    // Группируем соседние пики в широкополосные сигналы
    int i = 0;
    while (i < n_peaks) {
        // Начало группы
        double group_start_hz = peaks[i].f_hz;
        double group_end_hz = peaks[i].f_hz;
        float max_rssi = peaks[i].rssi_dbm;
        double peak_freq = peaks[i].f_hz;
        int group_size = 1;
        float rssi_sum = peaks[i].rssi_dbm;
        
        // Расширяем группу пока пики близки
        int j = i + 1;
        while (j < n_peaks && (peaks[j].f_hz - group_end_hz) < delta_hz) {
            group_end_hz = peaks[j].f_hz;
            rssi_sum += peaks[j].rssi_dbm;
            group_size++;
            
            // Находим самый сильный пик в группе
            if (peaks[j].rssi_dbm > max_rssi) {
                max_rssi = peaks[j].rssi_dbm;
                peak_freq = peaks[j].f_hz;
            }
            j++;
        }
        
        // Вычисляем характеристики группы
        double bandwidth_hz = group_end_hz - group_start_hz;
        double center_hz = (group_start_hz + group_end_hz) / 2.0;
        float avg_rssi = rssi_sum / group_size;
        
        // Корректируем центр на пиковую частоту для точной трилатерации
        if (bandwidth_hz > MIN_SIGNAL_WIDTH_HZ) {
            // Для широкополосных сигналов используем взвешенный центр
            center_hz = peak_freq;
        }
        
        // Определяем тип сигнала по ширине полосы
        const char* signal_type = "Unknown";
        if (bandwidth_hz < 25000) {
            signal_type = "Narrowband";  // < 25 кГц
        } else if (bandwidth_hz < 200000) {
            signal_type = "Voice/Data";  // 25-200 кГц
        } else if (bandwidth_hz < 2000000) {
            signal_type = "Wideband";    // 200 кГц - 2 МГц
        } else if (bandwidth_hz < 10000000) {
            signal_type = "Video/WiFi";  // 2-10 МГц
        } else {
            signal_type = "Ultra-Wide";  // > 10 МГц
        }
        
        // Ищем существующую запись в watchlist
        int found_idx = -1;
        for (size_t k = 0; k < g_watchlist_count; k++) {
            // Проверяем перекрытие диапазонов
            double overlap_start = fmax(g_watchlist[k].f_center_hz - g_watchlist[k].bw_hz/2, 
                                       center_hz - bandwidth_hz/2);
            double overlap_end = fmin(g_watchlist[k].f_center_hz + g_watchlist[k].bw_hz/2,
                                     center_hz + bandwidth_hz/2);
            
            if (overlap_end > overlap_start) {
                // Есть перекрытие - это тот же сигнал
                found_idx = k;
                break;
            }
        }
        
        if (found_idx >= 0) {
            // Обновляем существующий сигнал
            WatchItem* item = &g_watchlist[found_idx];
            
            // Обновляем центральную частоту с учетом нового пика
            item->f_center_hz = (item->f_center_hz * 0.7 + center_hz * 0.3);
            
            // Расширяем полосу если нужно
            if (bandwidth_hz > item->bw_hz) {
                item->bw_hz = bandwidth_hz;
            }
            
            // Обновляем RSSI с учетом максимального значения
            item->rssi_ema = rssi_apply_ema(item->rssi_ema, max_rssi, g_ema_alpha);
            
            item->last_ns = now_ns;
            item->hit_count++;
            
            printf("Updated signal: %.3f MHz, BW: %.1f kHz, RSSI: %.1f dBm, Type: %s\n",
                   item->f_center_hz/1e6, item->bw_hz/1e3, item->rssi_ema, signal_type);
            
        } else if (g_watchlist_count < MAX_WATCHLIST) {
            // Добавляем новый сигнал
            WatchItem* item = &g_watchlist[g_watchlist_count++];
            item->f_center_hz = center_hz;
            item->bw_hz = fmax(bandwidth_hz, MIN_SIGNAL_WIDTH_HZ);
            item->rssi_ema = max_rssi;
            item->last_ns = now_ns;
            item->hit_count = 1;
            
            printf("New signal detected: %.3f MHz, BW: %.1f kHz, RSSI: %.1f dBm, Type: %s\n",
                   center_hz/1e6, bandwidth_hz/1e3, max_rssi, signal_type);
        }
        
        // Переходим к следующей группе
        i = j;
    }
    
    // Удаляем устаревшие записи
    size_t write_idx = 0;
    for (size_t i = 0; i < g_watchlist_count; i++) {
        if ((now_ns - g_watchlist[i].last_ns) < SIGNAL_TIMEOUT_NS) {
            // Сигнал еще активен
            if (write_idx != i) {
                g_watchlist[write_idx] = g_watchlist[i];
            }
            write_idx++;
        } else {
            printf("Signal timeout: %.3f MHz (last seen %.1f sec ago)\n",
                   g_watchlist[i].f_center_hz/1e6,
                   (now_ns - g_watchlist[i].last_ns) / 1e9);
        }
    }
    g_watchlist_count = write_idx;
    
    pthread_mutex_unlock(&g_watchlist_mutex);
    
    // Логируем статистику
    if (g_watchlist_count > 0) {
        printf("Active signals: %zu\n", g_watchlist_count);
    }
}

// Функция для получения центральной частоты широкополосного сигнала
double get_signal_center_frequency(double freq_hz, double tolerance_hz) {
    pthread_mutex_lock(&g_watchlist_mutex);
    
    for (size_t i = 0; i < g_watchlist_count; i++) {
        // Проверяем попадание частоты в диапазон сигнала
        if (fabs(g_watchlist[i].f_center_hz - freq_hz) < (g_watchlist[i].bw_hz / 2 + tolerance_hz)) {
            double center = g_watchlist[i].f_center_hz;
            pthread_mutex_unlock(&g_watchlist_mutex);
            return center;
        }
    }
    
    pthread_mutex_unlock(&g_watchlist_mutex);
    return freq_hz;  // Не найден в группе - возвращаем исходную частоту
}

// Функция для получения самого сильного сигнала в диапазоне
int get_strongest_signal_in_range(double start_hz, double end_hz, WatchItem* result) {
    if (!result) return -1;
    
    pthread_mutex_lock(&g_watchlist_mutex);
    
    float max_rssi = -200.0f;
    int found_idx = -1;
    
    for (size_t i = 0; i < g_watchlist_count; i++) {
        // Проверяем попадание в диапазон
        if (g_watchlist[i].f_center_hz >= start_hz && 
            g_watchlist[i].f_center_hz <= end_hz) {
            
            if (g_watchlist[i].rssi_ema > max_rssi) {
                max_rssi = g_watchlist[i].rssi_ema;
                found_idx = i;
            }
        }
    }
    
    if (found_idx >= 0) {
        *result = g_watchlist[found_idx];
        pthread_mutex_unlock(&g_watchlist_mutex);
        return 0;
    }
    
    pthread_mutex_unlock(&g_watchlist_mutex);
    return -1;
}