// hq_watchlist.h - Потокобезопасный watchlist с LRU eviction
#ifndef HQ_WATCHLIST_H
#define HQ_WATCHLIST_H

#include "hq_api.h"
#include <pthread.h>
#include <stdatomic.h>

// Максимальный размер watchlist
#define MAX_WATCHLIST 100

// Внутренняя структура элемента watchlist с дополнительными полями
typedef struct {
    WatchItem item;              // Публичные данные
    
    // Внутренние поля для управления
    atomic_int lock;             // Спинлок для атомарного доступа
    double priority_score;       // Приоритет для планировщика
    uint64_t last_access_ns;    // Для LRU eviction
    bool active;                 // Активен ли элемент
    
    // Статистика
    double rms_sum;              // Сумма для скользящего среднего
    int rms_count;               // Количество измерений RMS
    double peak_history[10];     // История пиков
    int peak_history_idx;        // Индекс в кольцевом буфере
} WatchItemInternal;

// Глобальный watchlist
typedef struct {
    WatchItemInternal items[MAX_WATCHLIST];
    atomic_int count;
    pthread_rwlock_t rwlock;     // RW lock для массовых операций
    
    // Параметры
    int max_size;
    double grouping_eps_hz;      // Эпсилон для группировки
    double eviction_timeout_ns;  // Таймаут для вытеснения
    
    // Статистика
    atomic_ullong total_added;
    atomic_ullong total_removed;
    atomic_ullong total_hits;
} Watchlist;

// ==================== API функции ====================

// Инициализация watchlist
int watchlist_init(Watchlist* wl, int max_size);

// Очистка и освобождение ресурсов
void watchlist_destroy(Watchlist* wl);

// Добавление/обновление элемента (upsert)
// Если частота уже есть - обновляет, если нет - добавляет
// Автоматически группирует близкие частоты
int watchlist_upsert(Watchlist* wl, double freq_hz, double peak_dbm);

// Обновление RMS от slave
int watchlist_update_rms(Watchlist* wl, double freq_hz, double rms_dbm);

// Получение элемента для обработки slave'ом
// Возвращает элемент с наивысшим приоритетом
int watchlist_get_next_target(Watchlist* wl, WatchItem* out_item);

// Получение снимка всего watchlist
int watchlist_get_snapshot(Watchlist* wl, WatchItem* out_items, int max_items);

// Удаление элемента по частоте
int watchlist_remove(Watchlist* wl, double freq_hz);

// Очистка всего watchlist
void watchlist_clear(Watchlist* wl);

// Вытеснение устаревших элементов (LRU + timeout)
int watchlist_evict_stale(Watchlist* wl);

// ==================== Внутренние функции ====================

// Поиск элемента по частоте (с учетом группировки)
int watchlist_find_index(Watchlist* wl, double freq_hz);

// Расчет приоритета элемента
double watchlist_calculate_priority(WatchItemInternal* item);

// Слияние близких частот
int watchlist_merge_nearby(Watchlist* wl);

#endif // HQ_WATCHLIST_H