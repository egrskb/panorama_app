// hq_api.h - Публичный API для master/slave детектора с watchlist
#ifndef HQ_API_H
#define HQ_API_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Макросы для экспорта функций
#ifdef _WIN32
    #define HQ_EXPORT __declspec(dllexport)
#else
    #define HQ_EXPORT __attribute__((visibility("default")))
#endif

// ==================== Структуры данных ====================

// Элемент watchlist
typedef struct {
    double center_freq_hz;    // Центральная частота
    double span_hz;          // Ширина окна для slave (пользователь задает)
    double rms_dbm;          // RMS от slave в dBm
    double peak_dbm_last;    // Последний пик от master
    uint64_t last_update_ts; // Временная метка последнего обновления
    int hit_count;           // Количество обнаружений
    int group_size;          // Размер группы (если сгруппированы близкие)
} WatchItem;

// Пик от master
typedef struct {
    double freq_hz;
    double amp_dbm;
    uint64_t timestamp_ns;
} Peak;

// Статус системы
typedef struct {
    int master_running;
    int slaves_count;        // Количество активных slaves
    int watchlist_size;      // Текущий размер watchlist
    double master_coverage;  // Процент покрытия диапазона
} HqStatus;

// ==================== Callbacks ====================

// Callback при обнаружении пика master'ом
typedef void (*hq_on_peak_cb)(double freq_hz, double amp_db, double ts, void* user_data);

// Callback при обновлении watchlist (RMS от slave)
typedef void (*hq_on_watchlist_update_cb)(double center_freq_hz, double rms_dbm, 
                                          double ts, int hit_count, void* user_data);

// ==================== Жизненный цикл ====================

// Инициализация системы (перечисляет устройства, создает master и slaves)
HQ_EXPORT int  hq_init(void);

// Завершение работы системы
HQ_EXPORT void hq_shutdown(void);

// ==================== Конфигурация ====================

// Настройка диапазона master sweep
HQ_EXPORT int  hq_set_master_range(double f_start_hz, double f_stop_hz);
HQ_EXPORT int  hq_set_master_step(double step_hz);
HQ_EXPORT int  hq_set_master_rates(double sample_rate, double bw_hz);

// Параметры детектора пиков
HQ_EXPORT int  hq_set_peak_params(double threshold_db, double min_distance_hz);

// Параметры группировки близких пиков
HQ_EXPORT int  hq_set_grouping_eps(double eps_hz);

// Ширина окна для slave мониторинга (±span/2 от центра)
HQ_EXPORT int  hq_set_span_hz(double span_hz);

// Длительность окна для расчета RMS в миллисекундах
HQ_EXPORT int  hq_set_rms_window_ms(int ms);

// Максимальный размер watchlist
HQ_EXPORT int  hq_set_watchlist_limit(int max_items);

// Калибровочный оффсет для RMS
HQ_EXPORT int  hq_set_calibration_offset_db(double offset_db);

// ==================== Управление ====================

// Запуск системы (master начинает sweep, slaves ждут задач)
HQ_EXPORT int  hq_start(void);

// Остановка системы
HQ_EXPORT int  hq_stop(void);

// ==================== Callbacks ====================

// Установка callback для новых пиков от master
HQ_EXPORT int  hq_set_on_peak_callback(hq_on_peak_cb cb, void* user_data);

// Установка callback для обновлений watchlist (RMS от slaves)
HQ_EXPORT int  hq_set_on_watchlist_update_callback(hq_on_watchlist_update_cb cb, void* user_data);

// ==================== Диагностика/статус ====================

// Получение количества устройств
HQ_EXPORT int  hq_get_device_counts(int* out_master, int* out_slaves);

// Получение снимка watchlist
HQ_EXPORT int  hq_get_watchlist_snapshot(WatchItem* out_items, int max_items);

// Получение статуса системы
HQ_EXPORT int  hq_get_status(HqStatus* out_status);

// ==================== Ручное управление watchlist ====================

// Очистка watchlist
HQ_EXPORT int  hq_watchlist_clear(void);

// Добавление частоты в watchlist вручную
HQ_EXPORT int  hq_watchlist_add(double center_freq_hz);

// Удаление частоты из watchlist
HQ_EXPORT int  hq_watchlist_remove(double center_freq_hz);

#ifdef __cplusplus
}
#endif

#endif // HQ_API_H