// hq_watchlist.c - Реализация потокобезопасного watchlist
#include "hq_watchlist.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

// Вспомогательные функции
static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static void spinlock_acquire(atomic_int* lock) {
    int expected = 0;
    while (!atomic_compare_exchange_weak(lock, &expected, 1)) {
        expected = 0;
        __builtin_ia32_pause(); // Hint для процессора
    }
}

static void spinlock_release(atomic_int* lock) {
    atomic_store(lock, 0);
}

// ==================== Инициализация/очистка ====================

int watchlist_init(Watchlist* wl, int max_size) {
    if (!wl || max_size <= 0 || max_size > MAX_WATCHLIST) {
        return -1;
    }
    
    memset(wl, 0, sizeof(Watchlist));
    
    // Инициализация параметров
    wl->max_size = max_size;
    wl->grouping_eps_hz = 250000.0;  // 250 кГц по умолчанию
    wl->eviction_timeout_ns = 10ULL * 1000000000ULL; // 10 секунд
    
    // Инициализация блокировок
    if (pthread_rwlock_init(&wl->rwlock, NULL) != 0) {
        return -2;
    }
    
    // Инициализация элементов
    for (int i = 0; i < MAX_WATCHLIST; i++) {
        atomic_init(&wl->items[i].lock, 0);
        wl->items[i].active = false;
    }
    
    atomic_init(&wl->count, 0);
    atomic_init(&wl->total_added, 0);
    atomic_init(&wl->total_removed, 0);
    atomic_init(&wl->total_hits, 0);
    
    return 0;
}

void watchlist_destroy(Watchlist* wl) {
    if (!wl) return;
    
    pthread_rwlock_destroy(&wl->rwlock);
    memset(wl, 0, sizeof(Watchlist));
}

// ==================== Поиск элемента ====================

int watchlist_find_index(Watchlist* wl, double freq_hz) {
    int count = atomic_load(&wl->count);
    
    for (int i = 0; i < MAX_WATCHLIST && i < count; i++) {
        if (!wl->items[i].active) continue;
        
        double center = wl->items[i].item.center_freq_hz;
        double diff = fabs(center - freq_hz);
        
        // Проверяем попадание в группу
        if (diff <= wl->grouping_eps_hz) {
            return i;
        }
    }
    
    return -1;
}

// ==================== Добавление/обновление ====================

int watchlist_upsert(Watchlist* wl, double freq_hz, double peak_dbm) {
    if (!wl) return -1;
    
    pthread_rwlock_wrlock(&wl->rwlock);
    
    uint64_t now_ns = get_time_ns();
    
    // Ищем существующий элемент
    int idx = watchlist_find_index(wl, freq_hz);
    
    if (idx >= 0) {
        // Обновляем существующий
        WatchItemInternal* item = &wl->items[idx];
        
        spinlock_acquire(&item->lock);
        
        // Обновляем частоту (взвешенное среднее)
        double weight = 0.3; // Вес новой частоты
        item->item.center_freq_hz = 
            item->item.center_freq_hz * (1.0 - weight) + freq_hz * weight;
        
        // Обновляем пик
        item->item.peak_dbm_last = peak_dbm;
        item->item.last_update_ts = now_ns;
        item->item.hit_count++;
        
        // Обновляем историю пиков
        item->peak_history[item->peak_history_idx] = peak_dbm;
        item->peak_history_idx = (item->peak_history_idx + 1) % 10;
        
        // Обновляем приоритет
        item->priority_score = watchlist_calculate_priority(item);
        item->last_access_ns = now_ns;
        
        spinlock_release(&item->lock);
        
        atomic_fetch_add(&wl->total_hits, 1);
        
    } else {
        // Добавляем новый элемент
        int current_count = atomic_load(&wl->count);
        
        if (current_count >= wl->max_size) {
            // Нужно вытеснить старый элемент
            watchlist_evict_stale(wl);
            current_count = atomic_load(&wl->count);
            
            if (current_count >= wl->max_size) {
                pthread_rwlock_unlock(&wl->rwlock);
                return -2; // Нет места
            }
        }
        
        // Находим свободный слот
        int free_idx = -1;
        for (int i = 0; i < MAX_WATCHLIST; i++) {
            if (!wl->items[i].active) {
                free_idx = i;
                break;
            }
        }
        
        if (free_idx < 0) {
            pthread_rwlock_unlock(&wl->rwlock);
            return -3;
        }
        
        // Инициализируем новый элемент
        WatchItemInternal* item = &wl->items[free_idx];
        
        memset(item, 0, sizeof(WatchItemInternal));
        
        item->item.center_freq_hz = freq_hz;
        item->item.span_hz = 10000000.0; // 10 МГц по умолчанию
        item->item.peak_dbm_last = peak_dbm;
        item->item.rms_dbm = -120.0;
        item->item.last_update_ts = now_ns;
        item->item.hit_count = 1;
        item->item.group_size = 1;
        
        item->priority_score = 100.0; // Высокий приоритет для новых
        item->last_access_ns = now_ns;
        item->active = true;
        
        item->peak_history[0] = peak_dbm;
        item->peak_history_idx = 1;
        
        atomic_fetch_add(&wl->count, 1);
        atomic_fetch_add(&wl->total_added, 1);
    }
    
    // Пытаемся слить близкие частоты
    watchlist_merge_nearby(wl);
    
    pthread_rwlock_unlock(&wl->rwlock);
    
    return 0;
}

// ==================== Обновление RMS ====================

int watchlist_update_rms(Watchlist* wl, double freq_hz, double rms_dbm) {
    if (!wl) return -1;
    
    pthread_rwlock_rdlock(&wl->rwlock);
    
    int idx = watchlist_find_index(wl, freq_hz);
    
    if (idx < 0) {
        pthread_rwlock_unlock(&wl->rwlock);
        return -1;
    }
    
    WatchItemInternal* item = &wl->items[idx];
    
    spinlock_acquire(&item->lock);
    
    // Обновляем RMS (экспоненциальное скользящее среднее)
    if (item->rms_count == 0) {
        item->item.rms_dbm = rms_dbm;
    } else {
        double alpha = 0.2; // Коэффициент сглаживания
        item->item.rms_dbm = item->item.rms_dbm * (1.0 - alpha) + rms_dbm * alpha;
    }
    
    item->rms_sum += rms_dbm;
    item->rms_count++;
    
    item->item.last_update_ts = get_time_ns();
    item->last_access_ns = item->item.last_update_ts;
    
    spinlock_release(&item->lock);
    
    pthread_rwlock_unlock(&wl->rwlock);
    
    return 0;
}

// ==================== Получение следующей цели ====================

int watchlist_get_next_target(Watchlist* wl, WatchItem* out_item) {
    if (!wl || !out_item) return -1;
    
    pthread_rwlock_rdlock(&wl->rwlock);
    
    int best_idx = -1;
    double best_priority = -1.0;
    uint64_t now_ns = get_time_ns();
    
    // Находим элемент с наивысшим приоритетом
    for (int i = 0; i < MAX_WATCHLIST; i++) {
        if (!wl->items[i].active) continue;
        
        WatchItemInternal* item = &wl->items[i];
        
        // Пропускаем недавно обработанные
        uint64_t time_since_access = now_ns - item->last_access_ns;
        if (time_since_access < 100000000ULL) { // 100 мс
            continue;
        }
        
        double priority = watchlist_calculate_priority(item);
        
        if (priority > best_priority) {
            best_priority = priority;
            best_idx = i;
        }
    }
    
    if (best_idx < 0) {
        pthread_rwlock_unlock(&wl->rwlock);
        return -1;
    }
    
    // Копируем данные
    WatchItemInternal* item = &wl->items[best_idx];
    
    spinlock_acquire(&item->lock);
    *out_item = item->item;
    item->last_access_ns = now_ns;
    spinlock_release(&item->lock);
    
    pthread_rwlock_unlock(&wl->rwlock);
    
    return 0;
}

// ==================== Снимок watchlist ====================

int watchlist_get_snapshot(Watchlist* wl, WatchItem* out_items, int max_items) {
    if (!wl || !out_items || max_items <= 0) return 0;
    
    pthread_rwlock_rdlock(&wl->rwlock);
    
    int copied = 0;
    
    for (int i = 0; i < MAX_WATCHLIST && copied < max_items; i++) {
        if (!wl->items[i].active) continue;
        
        WatchItemInternal* item = &wl->items[i];
        
        spinlock_acquire(&item->lock);
        out_items[copied] = item->item;
        spinlock_release(&item->lock);
        
        copied++;
    }
    
    pthread_rwlock_unlock(&wl->rwlock);
    
    return copied;
}

// ==================== Удаление элемента ====================

int watchlist_remove(Watchlist* wl, double freq_hz) {
    if (!wl) return -1;
    
    pthread_rwlock_wrlock(&wl->rwlock);
    
    int idx = watchlist_find_index(wl, freq_hz);
    
    if (idx < 0) {
        pthread_rwlock_unlock(&wl->rwlock);
        return -1;
    }
    
    wl->items[idx].active = false;
    atomic_fetch_sub(&wl->count, 1);
    atomic_fetch_add(&wl->total_removed, 1);
    
    pthread_rwlock_unlock(&wl->rwlock);
    
    return 0;
}

// ==================== Очистка watchlist ====================

void watchlist_clear(Watchlist* wl) {
    if (!wl) return;
    
    pthread_rwlock_wrlock(&wl->rwlock);
    
    for (int i = 0; i < MAX_WATCHLIST; i++) {
        wl->items[i].active = false;
    }
    
    atomic_store(&wl->count, 0);
    
    pthread_rwlock_unlock(&wl->rwlock);
}

// ==================== Вытеснение устаревших ====================

int watchlist_evict_stale(Watchlist* wl) {
    if (!wl) return 0;
    
    uint64_t now_ns = get_time_ns();
    int evicted = 0;
    
    // Находим самый старый/слабый элемент
    int worst_idx = -1;
    double worst_score = 1e9;
    
    for (int i = 0; i < MAX_WATCHLIST; i++) {
        if (!wl->items[i].active) continue;
        
        WatchItemInternal* item = &wl->items[i];
        
        // Проверяем таймаут
        uint64_t age = now_ns - item->item.last_update_ts;
        if (age > wl->eviction_timeout_ns) {
            item->active = false;
            atomic_fetch_sub(&wl->count, 1);
            atomic_fetch_add(&wl->total_removed, 1);
            evicted++;
            continue;
        }
        
        // Находим кандидата на вытеснение
        double score = watchlist_calculate_priority(item);
        if (score < worst_score) {
            worst_score = score;
            worst_idx = i;
        }
    }
    
    // Если не нашли по таймауту, вытесняем худший
    if (evicted == 0 && worst_idx >= 0) {
        wl->items[worst_idx].active = false;
        atomic_fetch_sub(&wl->count, 1);
        atomic_fetch_add(&wl->total_removed, 1);
        evicted = 1;
    }
    
    return evicted;
}

// ==================== Расчет приоритета ====================

double watchlist_calculate_priority(WatchItemInternal* item) {
    if (!item || !item->active) return 0.0;
    
    uint64_t now_ns = get_time_ns();
    
    // Факторы приоритета:
    // 1. Сила сигнала (пик)
    double strength_factor = (item->item.peak_dbm_last + 120.0) / 100.0; // Нормализация
    
    // 2. Свежесть
    double age_s = (now_ns - item->item.last_update_ts) / 1e9;
    double freshness_factor = exp(-age_s / 10.0); // Экспоненциальное затухание
    
    // 3. Стабильность (количество хитов)
    double stability_factor = log10(item->item.hit_count + 1);
    
    // 4. Активность (как давно обращались)
    double access_age_s = (now_ns - item->last_access_ns) / 1e9;
    double activity_factor = 1.0 + access_age_s / 10.0; // Чем дольше не трогали, тем выше приоритет
    
    // Комбинируем факторы
    double priority = strength_factor * freshness_factor * stability_factor * activity_factor;
    
    return priority;
}

// ==================== Слияние близких частот ====================

int watchlist_merge_nearby(Watchlist* wl) {
    if (!wl) return 0;
    
    int merged = 0;
    int count = atomic_load(&wl->count);
    
    // Простой O(n^2) алгоритм для малых размеров
    for (int i = 0; i < MAX_WATCHLIST && i < count; i++) {
        if (!wl->items[i].active) continue;
        
        for (int j = i + 1; j < MAX_WATCHLIST; j++) {
            if (!wl->items[j].active) continue;
            
            double freq_i = wl->items[i].item.center_freq_hz;
            double freq_j = wl->items[j].item.center_freq_hz;
            double diff = fabs(freq_i - freq_j);
            
            if (diff <= wl->grouping_eps_hz) {
                // Сливаем j в i
                WatchItemInternal* target = &wl->items[i];
                WatchItemInternal* source = &wl->items[j];
                
                spinlock_acquire(&target->lock);
                spinlock_acquire(&source->lock);
                
                // Обновляем центр (взвешенное среднее)
                int total_hits = target->item.hit_count + source->item.hit_count;
                target->item.center_freq_hz = 
                    (target->item.center_freq_hz * target->item.hit_count +
                     source->item.center_freq_hz * source->item.hit_count) / total_hits;
                
                // Обновляем счетчики
                target->item.hit_count = total_hits;
                target->item.group_size += source->item.group_size;
                
                // Берем максимальный пик
                if (source->item.peak_dbm_last > target->item.peak_dbm_last) {
                    target->item.peak_dbm_last = source->item.peak_dbm_last;
                }
                
                // Обновляем RMS
                if (source->rms_count > 0 && target->rms_count > 0) {
                    double total_rms = target->rms_sum + source->rms_sum;
                    int total_count = target->rms_count + source->rms_count;
                    target->item.rms_dbm = total_rms / total_count;
                    target->rms_sum = total_rms;
                    target->rms_count = total_count;
                } else if (source->rms_count > 0) {
                    target->item.rms_dbm = source->item.rms_dbm;
                    target->rms_sum = source->rms_sum;
                    target->rms_count = source->rms_count;
                }
                
                // Деактивируем источник
                source->active = false;
                
                spinlock_release(&source->lock);
                spinlock_release(&target->lock);
                
                atomic_fetch_sub(&wl->count, 1);
                merged++;
            }
        }
    }
    
    return merged;
}