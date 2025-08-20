#ifndef HACKRF_MASTER_H
#define HACKRF_MASTER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Структуры данных
typedef struct {
    double f_start;           // Начальная частота (Гц)
    double bin_hz;            // Ширина бина (Гц)
    int count;                // Количество бинов
    float* power;             // Массив мощностей (дБм)
    double t0;                // Временная метка
} sweep_tile_t;

typedef struct {
    double f_peak;            // Центральная частота пика (Гц)
    double snr_db;            // SNR в дБ
    double bin_hz;            // Ширина бина
    double t0;                // Время обнаружения
    int status;               // Статус пика
} detected_peak_t;

// Callback функции
typedef void (*sweep_tile_callback_t)(const sweep_tile_t* tile);
typedef void (*peak_detected_callback_t)(const detected_peak_t* peak);
typedef void (*error_callback_t)(const char* error_msg);

// Конфигурация sweep
typedef struct {
    double start_hz;          // Начальная частота (Гц)
    double stop_hz;           // Конечная частота (Гц)
    double bin_hz;            // Ширина бина (Гц)
    int dwell_ms;             // Время измерения (мс)
    double step_hz;           // Шаг sweep (Гц)
    int avg_count;            // Количество усреднений
    double min_snr_db;        // Минимальный SNR для детекции
    int min_peak_distance_bins; // Минимальное расстояние между пиками
} sweep_config_t;

// Основные функции
int hackrf_master_init(void);
void hackrf_master_cleanup(void);

// Управление sweep
int hackrf_master_start_sweep(const sweep_config_t* config);
int hackrf_master_stop_sweep(void);
bool hackrf_master_is_running(void);

// Настройка callbacks
void hackrf_master_set_sweep_callback(sweep_tile_callback_t callback);
void hackrf_master_set_peak_callback(peak_detected_callback_t callback);
void hackrf_master_set_error_callback(error_callback_t callback);

// Управление детекцией пиков
int hackrf_master_set_peak_detection_params(double min_snr_db, int min_peak_distance_bins);
int hackrf_master_get_peak_count(void);
int hackrf_master_get_peaks(detected_peak_t* peaks, int max_count);

// Статистика
typedef struct {
    int sweep_count;          // Количество выполненных sweep
    int peak_count;           // Количество обнаруженных пиков
    double last_sweep_time;   // Время последнего sweep
    double avg_sweep_time;    // Среднее время sweep
    int error_count;          // Количество ошибок
} master_stats_t;

int hackrf_master_get_stats(master_stats_t* stats);
void hackrf_master_reset_stats(void);

// Утилиты
double hackrf_master_get_frequency_range_min(void);
double hackrf_master_get_frequency_range_max(void);
int hackrf_master_get_max_bin_count(void);
double hackrf_master_get_max_bandwidth(void);

// Перечень устройств HackRF
typedef struct {
    char serial[64];
} hackrf_devinfo_t;

int hackrf_master_enumerate(hackrf_devinfo_t* out_list, int max_count);
void hackrf_master_set_serial(const char* serial);

// Проба инициализации (открыть/закрыть устройство для проверки доступности)
int hackrf_master_probe(void);

#ifdef __cplusplus
}
#endif

#endif // HACKRF_MASTER_H
