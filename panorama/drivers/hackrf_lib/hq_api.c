// hq_api.c - Реализация основного API для master/slave системы
#include "hq_api.h"
#include "hq_watchlist.h"
#include "hq_master.h"
#include "hq_slave.h"
#include "hq_scheduler.h"
#include <libhackrf/hackrf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// ==================== Глобальное состояние ====================

typedef struct {
    // Устройства
    hackrf_device* master_device;
    hackrf_device* slave_devices[MAX_SLAVES];
    int num_slaves;
    
    // Компоненты системы
    Watchlist watchlist;
    MasterState master;
    SlaveState slaves[MAX_SLAVES];
    SchedulerState scheduler;
    
    // Конфигурация
    struct {
        double master_start_hz;
        double master_stop_hz;
        double master_step_hz;
        double sample_rate;
        double bandwidth;
        
        double peak_threshold_db;
        double peak_min_distance_hz;
        double grouping_eps_hz;
        
        double slave_span_hz;
        int rms_window_ms;
        int watchlist_limit;
        double calibration_offset_db;
    } config;
    
    // Callbacks
    hq_on_peak_cb peak_callback;
    void* peak_callback_data;
    hq_on_watchlist_update_cb watchlist_callback;
    void* watchlist_callback_data;
    
    // Состояние
    bool initialized;
    bool running;
} HqSystem;

static HqSystem g_system = {0};

// ==================== Вспомогательные функции ====================

static int enumerate_devices(void) {
    hackrf_device_list_t* list = hackrf_device_list();
    if (!list) {
        fprintf(stderr, "Failed to get device list\n");
        return 0;
    }
    
    int count = list->devicecount;
    fprintf(stdout, "Found %d HackRF devices\n", count);
    
    for (int i = 0; i < count; i++) {
        if (list->serial_numbers[i]) {
            fprintf(stdout, "  Device %d: %s\n", i, list->serial_numbers[i]);
        }
    }
    
    hackrf_device_list_free(list);
    return count;
}

static int open_devices(void) {
    hackrf_device_list_t* list = hackrf_device_list();
    if (!list) {
        return -1;
    }
    
    int count = list->devicecount;
    if (count < 1) {
        fprintf(stderr, "No HackRF devices found\n");
        hackrf_device_list_free(list);
        return -1;
    }
    
    // Открываем первое устройство как master
    int result = hackrf_open_by_serial(list->serial_numbers[0], &g_system.master_device);
    if (result != HACKRF_SUCCESS) {
        fprintf(stderr, "Failed to open master device: %s\n", hackrf_error_name(result));
        hackrf_device_list_free(list);
        return -1;
    }
    fprintf(stdout, "Opened master device: %s\n", list->serial_numbers[0]);
    
    // Открываем остальные как slaves
    g_system.num_slaves = 0;
    for (int i = 1; i < count && i <= MAX_SLAVES; i++) {
        result = hackrf_open_by_serial(list->serial_numbers[i], &g_system.slave_devices[g_system.num_slaves]);
        if (result == HACKRF_SUCCESS) {
            fprintf(stdout, "Opened slave %d: %s\n", g_system.num_slaves, list->serial_numbers[i]);
            g_system.num_slaves++;
        } else {
            fprintf(stderr, "Failed to open slave device %s: %s\n", 
                    list->serial_numbers[i], hackrf_error_name(result));
        }
    }
    
    hackrf_device_list_free(list);
    
    fprintf(stdout, "Successfully opened 1 master and %d slave devices\n", g_system.num_slaves);
    return 0;
}

static void close_devices(void) {
    if (g_system.master_device) {
        hackrf_close(g_system.master_device);
        g_system.master_device = NULL;
    }
    
    for (int i = 0; i < g_system.num_slaves; i++) {
        if (g_system.slave_devices[i]) {
            hackrf_close(g_system.slave_devices[i]);
            g_system.slave_devices[i] = NULL;
        }
    }
    
    g_system.num_slaves = 0;
}

// ==================== Инициализация/завершение ====================

int hq_init(void) {
    if (g_system.initialized) {
        fprintf(stderr, "System already initialized\n");
        return -1;
    }
    
    memset(&g_system, 0, sizeof(g_system));
    
    // Инициализация HackRF
    int result = hackrf_init();
    if (result != HACKRF_SUCCESS) {
        fprintf(stderr, "hackrf_init failed: %s\n", hackrf_error_name(result));
        return -1;
    }
    
    // Перечисление устройств
    int device_count = enumerate_devices();
    if (device_count < 1) {
        fprintf(stderr, "No devices available\n");
        hackrf_exit();
        return -1;
    }
    
    // Открытие устройств
    if (open_devices() != 0) {
        hackrf_exit();
        return -1;
    }
    
    // Инициализация watchlist
    if (watchlist_init(&g_system.watchlist, MAX_WATCHLIST) != 0) {
        fprintf(stderr, "Failed to initialize watchlist\n");
        close_devices();
        hackrf_exit();
        return -1;
    }
    
    // Инициализация master
    if (master_init(&g_system.master, g_system.master_device, &g_system.watchlist) != 0) {
        fprintf(stderr, "Failed to initialize master\n");
        watchlist_destroy(&g_system.watchlist);
        close_devices();
        hackrf_exit();
        return -1;
    }
    
    // Инициализация slaves
    for (int i = 0; i < g_system.num_slaves; i++) {
        if (slave_init(&g_system.slaves[i], g_system.slave_devices[i], i, &g_system.watchlist) != 0) {
            fprintf(stderr, "Failed to initialize slave %d\n", i);
            // Очистка уже инициализированных
            for (int j = 0; j < i; j++) {
                slave_destroy(&g_system.slaves[j]);
            }
            master_destroy(&g_system.master);
            watchlist_destroy(&g_system.watchlist);
            close_devices();
            hackrf_exit();
            return -1;
        }
    }
    
    // Инициализация планировщика
    if (scheduler_init(&g_system.scheduler, &g_system.watchlist) != 0) {
        fprintf(stderr, "Failed to initialize scheduler\n");
        for (int i = 0; i < g_system.num_slaves; i++) {
            slave_destroy(&g_system.slaves[i]);
        }
        master_destroy(&g_system.master);
        watchlist_destroy(&g_system.watchlist);
        close_devices();
        hackrf_exit();
        return -1;
    }
    
    // Добавляем slaves в планировщик
    for (int i = 0; i < g_system.num_slaves; i++) {
        scheduler_add_slave(&g_system.scheduler, &g_system.slaves[i]);
    }
    
    // Установка параметров по умолчанию
    g_system.config.master_start_hz = 50e6;     // 50 MHz
    g_system.config.master_stop_hz = 6000e6;    // 6 GHz
    g_system.config.master_step_hz = 1e6;       // 1 MHz
    g_system.config.sample_rate = 20e6;         // 20 MSPS
    g_system.config.bandwidth = 15e6;           // 15 MHz
    
    g_system.config.peak_threshold_db = 20.0;   // 20 dB над baseline
    g_system.config.peak_min_distance_hz = 250e3; // 250 kHz
    g_system.config.grouping_eps_hz = 250e3;    // 250 kHz
    
    g_system.config.slave_span_hz = 10e6;       // 10 MHz (±5 MHz)
    g_system.config.rms_window_ms = 100;        // 100 ms
    g_system.config.watchlist_limit = 50;       // 50 целей
    g_system.config.calibration_offset_db = 0.0;
    
    g_system.initialized = true;
    fprintf(stdout, "System initialized successfully\n");
    
    return 0;
}

void hq_shutdown(void) {
    if (!g_system.initialized) {
        return;
    }
    
    // Остановка если запущено
    if (g_system.running) {
        hq_stop();
    }
    
    // Очистка планировщика
    scheduler_destroy(&g_system.scheduler);
    
    // Очистка slaves
    for (int i = 0; i < g_system.num_slaves; i++) {
        slave_destroy(&g_system.slaves[i]);
    }
    
    // Очистка master
    master_destroy(&g_system.master);
    
    // Очистка watchlist
    watchlist_destroy(&g_system.watchlist);
    
    // Закрытие устройств
    close_devices();
    
    // Завершение HackRF
    hackrf_exit();
    
    memset(&g_system, 0, sizeof(g_system));
    fprintf(stdout, "System shut down\n");
}

// ==================== Конфигурация ====================

int hq_set_master_range(double f_start_hz, double f_stop_hz) {
    if (!g_system.initialized) return -1;
    if (g_system.running) return -2;
    
    g_system.config.master_start_hz = f_start_hz;
    g_system.config.master_stop_hz = f_stop_hz;
    
    return 0;
}

int hq_set_master_step(double step_hz) {
    if (!g_system.initialized) return -1;
    if (g_system.running) return -2;
    
    g_system.config.master_step_hz = step_hz;
    
    return 0;
}

int hq_set_master_rates(double sample_rate, double bw_hz) {
    if (!g_system.initialized) return -1;
    if (g_system.running) return -2;
    
    g_system.config.sample_rate = sample_rate;
    g_system.config.bandwidth = bw_hz;
    
    return 0;
}

int hq_set_peak_params(double threshold_db, double min_distance_hz) {
    if (!g_system.initialized) return -1;
    
    g_system.config.peak_threshold_db = threshold_db;
    g_system.config.peak_min_distance_hz = min_distance_hz;
    
    // Обновляем в master если запущен
    if (g_system.running) {
        PeakDetectorConfig detector = {
            .threshold_db = threshold_db,
            .min_distance_hz = min_distance_hz,
            .min_width_bins = 3,
            .baseline_alpha = 0.1
        };
        master_set_detector(&g_system.master, &detector);
    }
    
    return 0;
}

int hq_set_grouping_eps(double eps_hz) {
    if (!g_system.initialized) return -1;
    
    g_system.config.grouping_eps_hz = eps_hz;
    g_system.watchlist.grouping_eps_hz = eps_hz;
    
    return 0;
}

int hq_set_span_hz(double span_hz) {
    if (!g_system.initialized) return -1;
    
    g_system.config.slave_span_hz = span_hz;
    
    // Обновляем в планировщике
    scheduler_set_span(&g_system.scheduler, span_hz);
    
    // Обновляем в slaves
    for (int i = 0; i < g_system.num_slaves; i++) {
        g_system.slaves[i].config.span_hz = span_hz;
    }
    
    return 0;
}

int hq_set_rms_window_ms(int ms) {
    if (!g_system.initialized) return -1;
    
    g_system.config.rms_window_ms = ms;
    
    // Обновляем в slaves
    for (int i = 0; i < g_system.num_slaves; i++) {
        g_system.slaves[i].config.rms_window_ms = ms;
    }
    
    return 0;
}

int hq_set_watchlist_limit(int max_items) {
    if (!g_system.initialized) return -1;
    
    g_system.config.watchlist_limit = max_items;
    g_system.watchlist.max_size = max_items;
    
    return 0;
}

int hq_set_calibration_offset_db(double offset_db) {
    if (!g_system.initialized) return -1;
    
    g_system.config.calibration_offset_db = offset_db;
    
    // Обновляем в slaves
    for (int i = 0; i < g_system.num_slaves; i++) {
        g_system.slaves[i].config.calibration_offset_db = offset_db;
    }
    
    return 0;
}

// ==================== Управление ====================

int hq_start(void) {
    if (!g_system.initialized) return -1;
    if (g_system.running) return -2;
    
    // Конфигурация master
    MasterConfig master_cfg = {
        .start_hz = g_system.config.master_start_hz,
        .stop_hz = g_system.config.master_stop_hz,
        .step_hz = g_system.config.master_step_hz,
        .sample_rate = (uint32_t)g_system.config.sample_rate,
        .bandwidth = (uint32_t)g_system.config.bandwidth,
        .dwell_ms = 2
    };
    
    if (master_configure(&g_system.master, &master_cfg) != 0) {
        fprintf(stderr, "Failed to configure master\n");
        return -1;
    }
    
    // Конфигурация детектора
    PeakDetectorConfig detector_cfg = {
        .threshold_db = g_system.config.peak_threshold_db,
        .min_distance_hz = g_system.config.peak_min_distance_hz,
        .min_width_bins = 3,
        .baseline_alpha = 0.1
    };
    
    master_set_detector(&g_system.master, &detector_cfg);
    
    // Конфигурация slaves
    for (int i = 0; i < g_system.num_slaves; i++) {
        SlaveConfig slave_cfg = {
            .sample_rate = (uint32_t)g_system.config.sample_rate,
            .bandwidth = (uint32_t)g_system.config.bandwidth,
            .span_hz = g_system.config.slave_span_hz,
            .rms_window_ms = g_system.config.rms_window_ms,
            .calibration_offset_db = g_system.config.calibration_offset_db
        };
        
        if (slave_configure(&g_system.slaves[i], &slave_cfg) != 0) {
            fprintf(stderr, "Failed to configure slave %d\n", i);
            return -1;
        }
    }
    
    // Запуск master
    if (master_start(&g_system.master) != 0) {
        fprintf(stderr, "Failed to start master\n");
        return -1;
    }
    
    // Запуск slaves
    for (int i = 0; i < g_system.num_slaves; i++) {
        if (slave_start(&g_system.slaves[i]) != 0) {
            fprintf(stderr, "Failed to start slave %d\n", i);
            // Останавливаем уже запущенные
            for (int j = 0; j < i; j++) {
                slave_stop(&g_system.slaves[j]);
            }
            master_stop(&g_system.master);
            return -1;
        }
    }
    
    // Запуск планировщика
    if (scheduler_start(&g_system.scheduler) != 0) {
        fprintf(stderr, "Failed to start scheduler\n");
        for (int i = 0; i < g_system.num_slaves; i++) {
            slave_stop(&g_system.slaves[i]);
        }
        master_stop(&g_system.master);
        return -1;
    }
    
    g_system.running = true;
    fprintf(stdout, "System started: 1 master, %d slaves\n", g_system.num_slaves);
    
    return 0;
}

int hq_stop(void) {
    if (!g_system.initialized) return -1;
    if (!g_system.running) return -2;
    
    // Остановка планировщика
    scheduler_stop(&g_system.scheduler);
    
    // Остановка slaves
    for (int i = 0; i < g_system.num_slaves; i++) {
        slave_stop(&g_system.slaves[i]);
    }
    
    // Остановка master
    master_stop(&g_system.master);
    
    g_system.running = false;
    fprintf(stdout, "System stopped\n");
    
    return 0;
}

// ==================== Callbacks ====================

int hq_set_on_peak_callback(hq_on_peak_cb cb, void* user_data) {
    if (!g_system.initialized) return -1;
    
    g_system.peak_callback = cb;
    g_system.peak_callback_data = user_data;
    
    // Устанавливаем в master
    master_set_peak_callback(&g_system.master, cb, user_data);
    
    return 0;
}

int hq_set_on_watchlist_update_callback(hq_on_watchlist_update_cb cb, void* user_data) {
    if (!g_system.initialized) return -1;
    
    g_system.watchlist_callback = cb;
    g_system.watchlist_callback_data = user_data;
    
    // Устанавливаем в slaves
    for (int i = 0; i < g_system.num_slaves; i++) {
        slave_set_update_callback(&g_system.slaves[i], cb, user_data);
    }
    
    return 0;
}

// ==================== Диагностика ====================

int hq_get_device_counts(int* out_master, int* out_slaves) {
    if (!g_system.initialized) return -1;
    
    if (out_master) *out_master = g_system.master_device ? 1 : 0;
    if (out_slaves) *out_slaves = g_system.num_slaves;
    
    return 0;
}

int hq_get_watchlist_snapshot(WatchItem* out_items, int max_items) {
    if (!g_system.initialized) return 0;
    if (!out_items || max_items <= 0) return 0;
    
    return watchlist_get_snapshot(&g_system.watchlist, out_items, max_items);
}

int hq_get_status(HqStatus* out

    //  ДОДЕЛАТЬ