// hq_slave.h - Slave SDR для узкополосного мониторинга и расчета RMS
#ifndef HQ_SLAVE_H
#define HQ_SLAVE_H

#include <libhackrf/hackrf.h>
#include <pthread.h>
#include <stdbool.h>
#include "hq_watchlist.h"

// Конфигурация slave
typedef struct {
    uint32_t sample_rate;     // Sample rate для узкой полосы
    uint32_t bandwidth;       // Полоса пропускания
    double span_hz;          // Ширина окна мониторинга (задается пользователем)
    int rms_window_ms;       // Длительность окна для RMS
    double calibration_offset_db; // Калибровочный оффсет
} SlaveConfig;

// Текущая задача slave
typedef struct {
    double center_freq_hz;   // Центральная частота для мониторинга
    double span_hz;          // Ширина окна
    uint64_t start_time_ns;  // Время начала мониторинга
    int samples_collected;   // Собрано семплов
    int samples_needed;      // Требуется семплов для RMS
} SlaveTask;

// Состояние slave
typedef struct {
    hackrf_device* device;
    pthread_t thread;
    bool running;
    int slave_id;            // Идентификатор slave (0, 1, 2...)
    
    // Конфигурация
    SlaveConfig config;
    
    // Текущая задача
    SlaveTask current_task;
    pthread_mutex_t task_mutex;
    bool has_task;
    
    // Буфер для накопления IQ данных
    int8_t* iq_buffer;
    int buffer_size;
    int buffer_pos;
    
    // Watchlist для обновления RMS
    Watchlist* watchlist;
    
    // Callback для обновлений
    hq_on_watchlist_update_cb update_callback;
    void* update_callback_data;
    
    // Статистика
    uint64_t tasks_completed;
    uint64_t total_retunes;
    double avg_rms_dbm;
} SlaveState;

// API функции
int slave_init(SlaveState* state, hackrf_device* device, int slave_id, Watchlist* watchlist);
void slave_destroy(SlaveState* state);

int slave_configure(SlaveState* state, const SlaveConfig* config);
int slave_set_update_callback(SlaveState* state, hq_on_watchlist_update_cb cb, void* user_data);

// Назначение новой задачи slave'у
int slave_assign_task(SlaveState* state, double center_freq_hz, double span_hz);

// Проверка, свободен ли slave
bool slave_is_idle(SlaveState* state);

int slave_start(SlaveState* state);
int slave_stop(SlaveState* state);
bool slave_is_running(SlaveState* state);

// Внутренние функции
void slave_process_samples(SlaveState* state, const int8_t* buffer, int len);
double slave_calculate_rms_dbm(const int8_t* iq_data, int n_samples, double calibration_offset);
void slave_complete_task(SlaveState* state);

#endif // HQ_SLAVE_H