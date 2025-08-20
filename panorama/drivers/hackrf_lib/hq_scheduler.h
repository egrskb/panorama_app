// hq_scheduler.h - Планировщик распределения задач между slave SDR
#ifndef HQ_SCHEDULER_H
#define HQ_SCHEDULER_H

#include <pthread.h>
#include <stdbool.h>
#include "hq_watchlist.h"
#include "hq_slave.h"

#define MAX_SLAVES 8

// Состояние планировщика
typedef struct {
    SlaveState* slaves[MAX_SLAVES];
    int num_slaves;
    
    Watchlist* watchlist;
    
    pthread_t thread;
    bool running;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    
    // Параметры планирования
    double default_span_hz;       // Ширина окна по умолчанию
    int schedule_interval_ms;     // Интервал планирования
    int max_idle_time_ms;        // Максимальное время простоя slave
    
    // Статистика
    uint64_t total_scheduled;
    uint64_t total_skipped;
    double avg_utilization;
} SchedulerState;

// API функции
int scheduler_init(SchedulerState* sched, Watchlist* watchlist);
void scheduler_destroy(SchedulerState* sched);

// Добавление/удаление slave
int scheduler_add_slave(SchedulerState* sched, SlaveState* slave);
int scheduler_remove_slave(SchedulerState* sched, int slave_id);

// Настройка параметров
int scheduler_set_span(SchedulerState* sched, double span_hz);
int scheduler_set_interval(SchedulerState* sched, int interval_ms);

// Управление
int scheduler_start(SchedulerState* sched);
int scheduler_stop(SchedulerState* sched);
bool scheduler_is_running(SchedulerState* sched);

// Статистика
void scheduler_get_stats(SchedulerState* sched, uint64_t* scheduled, uint64_t* skipped, double* utilization);

// Внутренние функции
void* scheduler_thread_fn(void* arg);
void scheduler_assign_tasks(SchedulerState* sched);
int scheduler_find_idle_slave(SchedulerState* sched);
double scheduler_calculate_coverage(SchedulerState* sched);

#endif // HQ_SCHEDULER_H