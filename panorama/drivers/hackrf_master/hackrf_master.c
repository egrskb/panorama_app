// Для корректных прототипов nanosleep и др.
#define _POSIX_C_SOURCE 199309L
#include "hackrf_master.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

// HackRF includes
#include <libhackrf/hackrf.h>

// Максимальные значения
#define MAX_BIN_COUNT 8192
#define MAX_PEAK_COUNT 100
#define MAX_SWEEP_BUFFER 100

// Константы/совместимость
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Глобальные переменные
static hackrf_device* device = NULL;
static bool lib_initialized = false;
static bool is_running = false;
static pthread_t sweep_thread;
static pthread_mutex_t data_mutex = PTHREAD_MUTEX_INITIALIZER;
static char selected_serial[64] = {0};

// Глобальные переменные для callback функций
static sweep_tile_callback_t sweep_callback = NULL;
static peak_detected_callback_t peak_callback = NULL;
static error_callback_t error_callback = NULL;

// Глобальные переменные для передачи данных между потоками
static sweep_tile_t last_sweep_tile;
static detected_peak_t last_detected_peak;
static char last_error_message[256];
static bool new_sweep_data_available = false;
static bool new_peak_data_available = false;
static bool new_error_message_available = false;
static pthread_mutex_t callback_mutex = PTHREAD_MUTEX_INITIALIZER;

// Конфигурация
static sweep_config_t current_config;

// Буферы - возвращаем static для правильной работы
static sweep_tile_t* sweep_buffer = NULL;
static detected_peak_t* peak_buffer = NULL;
static int peak_buffer_count = 0;

// Статистика
static master_stats_t stats = {0};

// Буферы для обработки - возвращаем static
static float* power_buffer = NULL;
static float* smoothed_power = NULL;
static int* peak_indices = NULL;

// Функции для работы с HackRF
static int hackrf_setup_device(void);
static void hackrf_cleanup_device(void);
static void* sweep_worker_thread(void* arg);
static int process_sweep_data(double f_start, double f_stop, int dwell_ms);
static int detect_peaks_in_sweep(const float* powers, int count, double f_start, double bin_hz);
static float estimate_noise_level(const float* powers, int count, int exclude_start, int exclude_end);
static void smooth_power_data(const float* input, float* output, int count);

// Функции для получения данных (альтернатива callback)
int hackrf_master_get_last_sweep_tile(sweep_tile_t* tile_out) {
    if (!tile_out) return -1;
    
    pthread_mutex_lock(&callback_mutex);
    if (new_sweep_data_available) {
        // Копируем данные вместо передачи указателей
        tile_out->f_start = last_sweep_tile.f_start;
        tile_out->bin_hz = last_sweep_tile.bin_hz;
        tile_out->count = last_sweep_tile.count;
        tile_out->t0 = last_sweep_tile.t0;
        
        // Копируем power данные
        if (last_sweep_tile.power && last_sweep_tile.count > 0) {
            // Выделяем новую память для power данных
            tile_out->power = malloc(last_sweep_tile.count * sizeof(float));
            if (tile_out->power) {
                memcpy(tile_out->power, last_sweep_tile.power, last_sweep_tile.count * sizeof(float));
                printf("DEBUG: Copied %d power values to output tile\n", last_sweep_tile.count);
            } else {
                printf("DEBUG: Failed to allocate memory for power data\n");
                pthread_mutex_unlock(&callback_mutex);
                return -1;
            }
        } else {
            tile_out->power = NULL;
        }
        
        new_sweep_data_available = false;
        pthread_mutex_unlock(&callback_mutex);
        printf("DEBUG: Successfully copied sweep tile data\n");
        return 0;
    }
    pthread_mutex_unlock(&callback_mutex);
    return -1; // Нет новых данных
}

int hackrf_master_get_last_peak(detected_peak_t* peak_out) {
    if (!peak_out) return -1;
    
    pthread_mutex_lock(&callback_mutex);
    if (new_peak_data_available) {
        memcpy(peak_out, &last_detected_peak, sizeof(detected_peak_t));
        new_peak_data_available = false;
        pthread_mutex_unlock(&callback_mutex);
        return 0;
    }
    pthread_mutex_unlock(&callback_mutex);
    return -1; // Нет новых данных
}

int hackrf_master_get_last_error_message(char* message_out, int max_len) {
    if (!message_out || max_len <= 0) return -1;
    
    pthread_mutex_lock(&callback_mutex);
    if (new_error_message_available) {
        strncpy(message_out, last_error_message, max_len - 1);
        message_out[max_len - 1] = '\0';
        new_error_message_available = false;
        pthread_mutex_unlock(&callback_mutex);
        return 0;
    }
    pthread_mutex_unlock(&callback_mutex);
    return -1; // Нет новых данных
}

// Инициализация библиотеки
int hackrf_master_init(void) {
    printf("DEBUG: hackrf_master_init called\n");
    
    if (lib_initialized) {
        printf("DEBUG: Library already initialized, returning success\n");
        return 0; // Уже инициализирован
    }
    
    printf("DEBUG: Initializing buffers...\n");
    
    // Выделяем память для буферов
    sweep_buffer = malloc(MAX_SWEEP_BUFFER * sizeof(sweep_tile_t));
    peak_buffer = malloc(MAX_PEAK_COUNT * sizeof(detected_peak_t));
    power_buffer = malloc(MAX_BIN_COUNT * sizeof(float));
    smoothed_power = malloc(MAX_BIN_COUNT * sizeof(float));
    peak_indices = malloc(MAX_BIN_COUNT * sizeof(int));
    
    if (!sweep_buffer || !peak_buffer || !power_buffer || !smoothed_power || !peak_indices) {
        printf("DEBUG: Failed to allocate buffers\n");
        hackrf_master_cleanup();
        return -1;
    }
    
    printf("DEBUG: Buffers allocated successfully\n");
    printf("DEBUG: power_buffer=%p, smoothed_power=%p\n", (void*)power_buffer, (void*)smoothed_power);
    
    // Сбрасываем статистику
    hackrf_master_reset_stats();
    
    lib_initialized = true;
    printf("DEBUG: Initialization completed successfully\n");
    
    return 0;
}

// Очистка библиотеки
void hackrf_master_cleanup(void) {
    // Останавливаем sweep если запущен
    if (is_running) {
        hackrf_master_stop_sweep();
    }
    
    // Очищаем HackRF
    hackrf_cleanup_device();
    
    // Освобождаем память
    if (sweep_buffer) {
        free(sweep_buffer);
        sweep_buffer = NULL;
    }
    if (peak_buffer) {
        free(peak_buffer);
        peak_buffer = NULL;
    }
    if (power_buffer) {
        free(power_buffer);
        power_buffer = NULL;
    }
    if (smoothed_power) {
        free(smoothed_power);
        smoothed_power = NULL;
    }
    if (peak_indices) {
        free(peak_indices);
        peak_indices = NULL;
    }
    // Сброс callback-ов и конфигурации
    sweep_callback = NULL;
    peak_callback = NULL;
    error_callback = NULL;
    memset(&current_config, 0, sizeof(current_config));
    memset(selected_serial, 0, sizeof(selected_serial));
}

// Запуск sweep
int hackrf_master_start_sweep(const sweep_config_t* config) {
    printf("DEBUG: hackrf_master_start_sweep called\n");
    printf("DEBUG: Current buffer status:\n");
    printf("DEBUG:   power_buffer=%p\n", (void*)power_buffer);
    printf("DEBUG:   smoothed_power=%p\n", (void*)smoothed_power);
    printf("DEBUG:   sweep_buffer=%p\n", (void*)sweep_buffer);
    printf("DEBUG:   peak_buffer=%p\n", (void*)peak_buffer);
    printf("DEBUG:   peak_indices=%p\n", (void*)peak_indices);
    printf("DEBUG:   lib_initialized=%d\n", lib_initialized);
    
    // Проверяем что буферы выделены
    if (!power_buffer || !smoothed_power || !sweep_buffer || !peak_buffer || !peak_indices) {
        printf("DEBUG: ERROR - Buffers not allocated! Library not properly initialized.\n");
        return -1;
    }
    
    printf("DEBUG: Buffers are properly allocated\n");
    
    // Открываем устройство при первом запуске
    if (!device) {
        if (hackrf_setup_device() != 0) {
            return -1;
        }
    }
    if (is_running) {
        return -1;
    }
    
    // Проверяем параметры
    if (config->start_hz < hackrf_master_get_frequency_range_min() ||
        config->stop_hz > hackrf_master_get_frequency_range_max() ||
        config->bin_hz <= 0 || config->dwell_ms <= 0) {
        return -1;
    }
    
    // Копируем конфигурацию
    memcpy(&current_config, config, sizeof(sweep_config_t));
    
    // Запускаем поток sweep
    is_running = true;
    if (pthread_create(&sweep_thread, NULL, sweep_worker_thread, NULL) != 0) {
        is_running = false;
        return -1;
    }
    
    printf("DEBUG: Sweep thread started successfully\n");
    return 0;
}

// Остановка sweep
int hackrf_master_stop_sweep(void) {
    if (!is_running) {
        return 0;
    }
    
    is_running = false;
    
    // Ждем завершения потока
    if (pthread_join(sweep_thread, NULL) != 0) {
        return -1;
    }
    
    // После остановки потока — закрываем устройство
    hackrf_cleanup_device();
    return 0;
}

// Проверка состояния
bool hackrf_master_is_running(void) {
    return is_running;
}

// Установка callbacks
void hackrf_master_set_sweep_callback(sweep_tile_callback_t callback) {
    printf("DEBUG: Setting sweep_callback to: %p\n", (void*)callback);
    
    pthread_mutex_lock(&callback_mutex);
    sweep_callback = callback;
    printf("DEBUG: sweep_callback now set to: %p\n", (void*)sweep_callback);
    pthread_mutex_unlock(&callback_mutex);
    
    // Проверяем что callback установлен
    if (sweep_callback) {
        printf("DEBUG: sweep_callback successfully set\n");
    } else {
        printf("DEBUG: WARNING: sweep_callback is still NULL after setting!\n");
    }
}

void hackrf_master_set_peak_callback(peak_detected_callback_t callback) {
    printf("DEBUG: Setting peak_callback to: %p\n", (void*)callback);
    
    pthread_mutex_lock(&callback_mutex);
    peak_callback = callback;
    printf("DEBUG: peak_callback now set to: %p\n", (void*)peak_callback);
    pthread_mutex_unlock(&callback_mutex);
    
    if (peak_callback) {
        printf("DEBUG: peak_callback successfully set\n");
    } else {
        printf("DEBUG: WARNING: peak_callback is still NULL after setting!\n");
    }
}

void hackrf_master_set_error_callback(error_callback_t callback) {
    printf("DEBUG: Setting error_callback to: %p\n", (void*)callback);
    
    pthread_mutex_lock(&callback_mutex);
    error_callback = callback;
    printf("DEBUG: error_callback now set to: %p\n", (void*)error_callback);
    pthread_mutex_unlock(&callback_mutex);
    
    if (error_callback) {
        printf("DEBUG: error_callback successfully set\n");
    } else {
        printf("DEBUG: WARNING: error_callback is still NULL after setting!\n");
    }
}

// Установка параметров детекции пиков
int hackrf_master_set_peak_detection_params(double min_snr_db, int min_peak_distance_bins) {
    if (min_snr_db < 0 || min_peak_distance_bins < 1) {
        return -1;
    }
    
    current_config.min_snr_db = min_snr_db;
    current_config.min_peak_distance_bins = min_peak_distance_bins;
    
    return 0;
}

// Получение количества пиков
int hackrf_master_get_peak_count(void) {
    pthread_mutex_lock(&data_mutex);
    int count = peak_buffer_count;
    pthread_mutex_unlock(&data_mutex);
    return count;
}

// Получение пиков
int hackrf_master_get_peaks(detected_peak_t* peaks, int max_count) {
    if (!peaks || max_count <= 0) {
        return -1;
    }
    
    pthread_mutex_lock(&data_mutex);
    int count = (peak_buffer_count < max_count) ? peak_buffer_count : max_count;
    memcpy(peaks, peak_buffer, count * sizeof(detected_peak_t));
    pthread_mutex_unlock(&data_mutex);
    
    return count;
}

// Получение статистики
int hackrf_master_get_stats(master_stats_t* stats_out) {
    if (!stats_out) {
        return -1;
    }
    
    pthread_mutex_lock(&data_mutex);
    memcpy(stats_out, &stats, sizeof(master_stats_t));
    pthread_mutex_unlock(&data_mutex);
    
    return 0;
}

// Сброс статистики
void hackrf_master_reset_stats(void) {
    pthread_mutex_lock(&data_mutex);
    memset(&stats, 0, sizeof(master_stats_t));
    pthread_mutex_unlock(&data_mutex);
}

// Утилиты
double hackrf_master_get_frequency_range_min(void) {
    return 24e6; // 24 МГц
}

double hackrf_master_get_frequency_range_max(void) {
    return 6e9;  // 6 ГГц
}

int hackrf_master_get_max_bin_count(void) {
    return MAX_BIN_COUNT;
}

double hackrf_master_get_max_bandwidth(void) {
    return 20e6; // 20 МГц
}

int hackrf_master_enumerate(hackrf_devinfo_t* out_list, int max_count) {
    printf("DEBUG: hackrf_master_enumerate called, max_count=%d\n", max_count);
    
    if (!out_list || max_count <= 0) {
        printf("DEBUG: Invalid parameters\n");
        return 0;
    }
    
    int found = 0;
    bool need_init = !lib_initialized;

    if (need_init) {
        printf("DEBUG: Initializing hackrf library\n");
        if (hackrf_init() != HACKRF_SUCCESS) {
            printf("DEBUG: hackrf_init failed\n");
            return 0;
        }
        lib_initialized = true;
    }

    // Получаем список устройств
    hackrf_device_list_t* list = hackrf_device_list();
    
    if (!list) {
        printf("DEBUG: hackrf_device_list returned NULL\n");
    } else {
        printf("DEBUG: hackrf_device_list found %d devices\n", list->devicecount);
        
        if (list->devicecount > 0) {
            for (int i = 0; i < list->devicecount && found < max_count; i++) {
                const char* serial = list->serial_numbers[i];
                if (serial && serial[0] != '\0') {
                    printf("DEBUG: Found device with serial: %s\n", serial);
                    strncpy(out_list[found].serial, serial, sizeof(out_list[found].serial) - 1);
                    out_list[found].serial[sizeof(out_list[found].serial) - 1] = '\0';
                    found++;
                } else {
                    printf("DEBUG: Device %d has empty serial\n", i);
                }
            }
        }
    }
    
    if (need_init) {
        hackrf_exit();
        lib_initialized = false;
    }
    
    printf("DEBUG: hackrf_master_enumerate returning %d devices\n", found);
    return found;  // Возвращаем реальное количество
}

void hackrf_master_set_serial(const char* serial) {
    if (serial) {
        strncpy(selected_serial, serial, sizeof(selected_serial) - 1);
        selected_serial[sizeof(selected_serial) - 1] = '\0';
    } else {
        selected_serial[0] = '\0';
    }
}

int hackrf_master_probe(void) {
    // Пытаемся открыть и закрыть устройство, не трогая глобальное device/is_running
    bool need_init = !lib_initialized;
    if (need_init) {
        if (hackrf_init() != HACKRF_SUCCESS) return -1;
        lib_initialized = true;
    }
    hackrf_device* probe = NULL;
    int result;
    if (selected_serial[0] != '\0') {
        result = hackrf_open_by_serial(selected_serial, &probe);
    } else {
        result = hackrf_open(&probe);
    }
    if (result != HACKRF_SUCCESS) {
        if (need_init) {
            hackrf_exit();
            lib_initialized = false;
        }
        return -1;
    }
    hackrf_close(probe);
    if (need_init) {
        hackrf_exit();
        lib_initialized = false;
    }
    return 0;
}

// Приватные функции

// Настройка HackRF устройства
static int hackrf_setup_device(void) {
    int result = HACKRF_SUCCESS;
    if (!lib_initialized) {
        result = hackrf_init();
    }
    if (result != HACKRF_SUCCESS) {
        if (error_callback) {
            error_callback("Failed to initialize HackRF library");
        }
        return -1;
    }
    lib_initialized = true;
    
    if (selected_serial[0] != '\0') {
        result = hackrf_open_by_serial(selected_serial, &device);
    } else {
        result = hackrf_open(&device);
    }
    if (result != HACKRF_SUCCESS) {
        if (error_callback) {
            error_callback("Failed to open HackRF device");
        }
        return -1;
    }
    
    // Настройка параметров по умолчанию
    result = hackrf_set_sample_rate(device, 20e6); // 20 Мс/с
    if (result != HACKRF_SUCCESS) {
        if (error_callback) {
            error_callback("Failed to set sample rate");
        }
        return -1;
    }
    
    result = hackrf_set_freq(device, 2.4e9); // 2.4 ГГц по умолчанию
    if (result != HACKRF_SUCCESS) {
        if (error_callback) {
            error_callback("Failed to set frequency");
        }
        return -1;
    }
    
    result = hackrf_set_amp_enable(device, 0); // Отключаем усилитель
    if (result != HACKRF_SUCCESS) {
        if (error_callback) {
            error_callback("Failed to set amp enable");
        }
        return -1;
    }
    
    return 0;
}

// Очистка HackRF устройства
static void hackrf_cleanup_device(void) {
    if (device) {
        hackrf_close(device);
        device = NULL;
    }
    if (lib_initialized) {
        hackrf_exit();
        lib_initialized = false;
    }
}

// Рабочий поток sweep
static void* sweep_worker_thread(void* arg) {
    (void)arg; // Неиспользуемый параметр
    
    printf("DEBUG: sweep_worker_thread started\n");
    
    while (is_running) {
        clock_t start_time = clock();
        
        printf("DEBUG: Starting sweep cycle\n");
        
        // Выполняем sweep по диапазону
        double current_freq = current_config.start_hz;
        while (current_freq < current_config.stop_hz && is_running) {
            // Проверяем состояние перед каждой итерацией
            if (!is_running) {
                printf("DEBUG: Sweep stopped, breaking loop\n");
                break;
            }
            
            // Устанавливаем частоту
            if (hackrf_set_freq(device, (uint64_t)current_freq) != HACKRF_SUCCESS) {
                printf("DEBUG: Failed to set frequency: %f\n", current_freq);
                if (error_callback) {
                    error_callback("Failed to set frequency");
                }
                break;
            }
            
            // Обрабатываем sweep данные
            // Вычисляем количество бинов в шаге, исходя из step_hz/bin_hz
            // Защита от деления на ноль/некорректных параметров
            if (current_config.bin_hz <= 0) {
                printf("DEBUG: Invalid bin_hz: %f\n", current_config.bin_hz);
                if (error_callback) {
                    error_callback("Invalid bin_hz (<=0)");
                }
                break;
            }
            int bins_per_step = (int)(current_config.step_hz / current_config.bin_hz);
            if (bins_per_step < 1) bins_per_step = 1;
            double f_stop = current_freq + current_config.bin_hz * bins_per_step;
            if (f_stop > current_config.stop_hz) {
                f_stop = current_config.stop_hz;
            }
            
            printf("DEBUG: Processing sweep data: %f - %f Hz\n", current_freq, f_stop);
            
            if (is_running && process_sweep_data(current_freq, f_stop, current_config.dwell_ms) == 0) {
                // Обновляем статистику
                pthread_mutex_lock(&data_mutex);
                stats.sweep_count++;
                pthread_mutex_unlock(&data_mutex);
                printf("DEBUG: Sweep data processed successfully\n");
            } else {
                printf("DEBUG: Failed to process sweep data\n");
            }
            
            // Переходим к следующей частоте
            current_freq += current_config.step_hz;
            if (!is_running) {
                printf("DEBUG: Sweep stopped during iteration\n");
                break;
            }
            
            // Небольшая задержка между sweep (1 мс)
            struct timespec ts;
            ts.tv_sec = 0;
            ts.tv_nsec = 1000000L;
            nanosleep(&ts, NULL);
        }
        
        // Обновляем время sweep
        clock_t end_time = clock();
        double sweep_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
        
        pthread_mutex_lock(&data_mutex);
        stats.last_sweep_time = sweep_time;
        stats.avg_sweep_time = (stats.avg_sweep_time * (stats.sweep_count - 1) + sweep_time) / stats.sweep_count;
        pthread_mutex_unlock(&data_mutex);
        
        printf("DEBUG: Sweep cycle completed, time: %.2f ms\n", sweep_time);
    }
    
    printf("DEBUG: sweep_worker_thread exiting\n");
    return NULL;
}

// Обработка sweep данных
static int process_sweep_data(double f_start, double f_stop, int dwell_ms) {
    printf("DEBUG: process_sweep_data called: f_start=%f, f_stop=%f, dwell_ms=%d\n", f_start, f_stop, dwell_ms);
    
    // Проверяем что буферы инициализированы
    if (!power_buffer || !smoothed_power) {
        printf("DEBUG: ERROR - Buffers not initialized!\n");
        printf("DEBUG: power_buffer=%p, smoothed_power=%p\n", (void*)power_buffer, (void*)smoothed_power);
        return -1;
    }
    
    // Симулируем получение данных (в реальной реализации здесь будет чтение от HackRF)
    int bin_count = (int)((f_stop - f_start) / current_config.bin_hz);
    if (bin_count > MAX_BIN_COUNT) {
        bin_count = MAX_BIN_COUNT;
    }
    
    printf("DEBUG: bin_count=%d, max_bins=%d\n", bin_count, MAX_BIN_COUNT);
    
    // Проверяем валидность параметров
    if (bin_count <= 0) {
        printf("DEBUG: Invalid bin_count: %d\n", bin_count);
        return -1;
    }
    
    printf("DEBUG: About to generate test data\n");
    
    // Генерируем тестовые данные (замените на реальное чтение от HackRF)
    for (int i = 0; i < bin_count; i++) {
        double freq = f_start + i * current_config.bin_hz;
        
        // Симулируем сигнал на определенных частотах
        if (freq >= 2.4e9 && freq <= 2.5e9) {
            power_buffer[i] = -45.0 + 10.0 * sin(2 * M_PI * freq / 1e9) + (rand() % 10 - 5);
        } else if (freq >= 5.0e9 && freq <= 5.1e9) {
            power_buffer[i] = -50.0 + 15.0 * sin(2 * M_PI * freq / 1e9) + (rand() % 10 - 5);
        } else {
            power_buffer[i] = -80.0 + (rand() % 20 - 10); // Шум
        }
    }
    
    printf("DEBUG: Generated test data for %d bins\n", bin_count);
    
    printf("DEBUG: About to smooth data\n");
    
    // Сглаживаем данные
    smooth_power_data(power_buffer, smoothed_power, bin_count);
    printf("DEBUG: Data smoothed\n");
    
    printf("DEBUG: About to detect peaks\n");
    
    // Детектируем пики
    int peaks_found = detect_peaks_in_sweep(smoothed_power, bin_count, f_start, current_config.bin_hz);
    printf("DEBUG: Detected %d peaks\n", peaks_found);
    
    // Создаем sweep tile с копированием данных
    sweep_tile_t tile;
    tile.f_start = f_start;
    tile.bin_hz = current_config.bin_hz;
    tile.count = bin_count;
    
    // Копируем данные мощности вместо передачи указателя
    if (tile.count > 0 && tile.count <= MAX_BIN_COUNT) {
        // Выделяем память для копии данных
        tile.power = malloc(tile.count * sizeof(float));
        if (tile.power) {
            memcpy(tile.power, smoothed_power, tile.count * sizeof(float));
            
            printf("DEBUG: Calling sweep_callback with %d bins\n", tile.count);
            
            pthread_mutex_lock(&callback_mutex);
            printf("DEBUG: sweep_callback pointer: %p\n", (void*)sweep_callback);
            
            // Сохраняем данные для получения через функции (альтернатива callback)
            pthread_mutex_lock(&callback_mutex);
            
            // Копируем основные данные
            last_sweep_tile.f_start = tile.f_start;
            last_sweep_tile.bin_hz = tile.bin_hz;
            last_sweep_tile.count = tile.count;
            last_sweep_tile.t0 = tile.t0;
            
            // Копируем power данные
            if (tile.power && tile.count > 0) {
                // Освобождаем старую память если есть
                if (last_sweep_tile.power) {
                    free(last_sweep_tile.power);
                }
                
                // Выделяем новую память и копируем данные
                last_sweep_tile.power = malloc(tile.count * sizeof(float));
                if (last_sweep_tile.power) {
                    memcpy(last_sweep_tile.power, tile.power, tile.count * sizeof(float));
                    printf("DEBUG: Saved %d power values for retrieval\n", tile.count);
                } else {
                    printf("DEBUG: Failed to allocate memory for saving power data\n");
                }
            } else {
                last_sweep_tile.power = NULL;
            }
            
            new_sweep_data_available = true;
            printf("DEBUG: Data saved for retrieval via get_last_sweep_tile\n");
            
            pthread_mutex_unlock(&callback_mutex);
            
            // Вызываем callback если установлен
            if (sweep_callback) {
                printf("DEBUG: About to call sweep_callback...\n");
                sweep_callback(&tile);
                printf("DEBUG: sweep_callback completed\n");
            } else {
                printf("DEBUG: sweep_callback is NULL, but data saved for polling\n");
            }
            pthread_mutex_unlock(&callback_mutex);
            
            // Освобождаем память после callback
            free(tile.power);
            printf("DEBUG: Memory freed\n");
        } else {
            printf("DEBUG: Failed to allocate memory for tile.power\n");
        }
    } else {
        printf("DEBUG: Invalid tile.count: %d\n", tile.count);
    }
    
    printf("DEBUG: process_sweep_data completed successfully\n");
    return 0;
}

// Детекция пиков в sweep
static int detect_peaks_in_sweep(const float* powers, int count, double f_start, double bin_hz) {
    int peaks_found = 0;
    
    for (int i = 1; i < count - 1; i++) {
        // Проверяем локальный максимум
        if (powers[i] > powers[i-1] && powers[i] > powers[i+1]) {
            // Оцениваем уровень шума
            float noise_level = estimate_noise_level(powers, count, i-5, i+6);
            float snr_db = powers[i] - noise_level;
            
            // Проверяем SNR
            if (snr_db >= current_config.min_snr_db) {
                // Проверяем расстояние до других пиков
                bool too_close = false;
                for (int j = 0; j < peaks_found; j++) {
                    if (abs(peak_indices[j] - i) < current_config.min_peak_distance_bins) {
                        too_close = true;
                        break;
                    }
                }
                
                if (!too_close && peaks_found < MAX_PEAK_COUNT) {
                    peak_indices[peaks_found] = i;
                    
                    // Создаем пик
                    detected_peak_t peak;
                    peak.f_peak = f_start + i * bin_hz;
                    peak.snr_db = snr_db;
                    peak.bin_hz = bin_hz;
                    peak.t0 = (double)time(NULL);
                    peak.status = 1; // ACTIVE
                    
                    // Добавляем в буфер
                    pthread_mutex_lock(&data_mutex);
                    if (peak_buffer_count < MAX_PEAK_COUNT) {
                        peak_buffer[peak_buffer_count] = peak;
                        peak_buffer_count++;
                    }
                    stats.peak_count++;
                    pthread_mutex_unlock(&data_mutex);
                    
                    // Вызываем callback с копией данных
                    if (peak_callback) {
                        detected_peak_t peak_copy = peak;  // Создаем копию
                        peak_callback(&peak_copy);
                    }
                    
                    peaks_found++;
                }
            }
        }
    }
    
    return peaks_found;
}

// Оценка уровня шума
static float estimate_noise_level(const float* powers, int count, int exclude_start, int exclude_end) {
    float sum = 0.0f;
    int valid_count = 0;
    
    for (int i = 0; i < count; i++) {
        if (i < exclude_start || i >= exclude_end) {
            sum += powers[i];
            valid_count++;
        }
    }
    
    if (valid_count > 0) {
        return sum / valid_count;
    }
    
    return -80.0f; // Значение по умолчанию
}

// Сглаживание данных мощности
static void smooth_power_data(const float* input, float* output, int count) {
    if (count < 3) {
        memcpy(output, input, count * sizeof(float));
        return;
    }
    
    // Простое сглаживание (3-точечное)
    output[0] = input[0];
    for (int i = 1; i < count - 1; i++) {
        output[i] = 0.25f * input[i-1] + 0.5f * input[i] + 0.25f * input[i+1];
    }
    output[count-1] = input[count-1];
}
