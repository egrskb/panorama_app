/**
 * hackrf_slave.c - Реализация нативной C библиотеки для слейв-устройств HackRF
 */

#include "hackrf_slave.h"
#include <libhackrf/hackrf.h>
#include <fftw3.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <stdarg.h>

// ================== Внутренние структуры ==================

/**
 * Структура для хранения захваченных данных
 */
typedef struct {
    int8_t* buffer;           // Буфер для IQ данных
    uint32_t buffer_size;     // Размер буфера
    uint32_t samples_captured; // Количество захваченных семплов
    uint32_t samples_needed;  // Необходимое количество семплов
    bool capture_complete;    // Флаг завершения захвата
    pthread_mutex_t mutex;    // Мутекс для синхронизации
    pthread_cond_t cond;      // Условная переменная
} capture_context_t;

/**
 * Структура устройства
 */
struct hackrf_slave_device {
    hackrf_device* dev;                    // Устройство HackRF
    char slave_id[64];                     // ID слейва
    hackrf_slave_config_t config;          // Текущая конфигурация
    bool is_configured;                    // Флаг конфигурации
    capture_context_t capture_ctx;         // Контекст захвата данных
    fftwf_plan fft_plan;                   // План FFT
    fftwf_complex* fft_in;                 // Входные данные FFT
    fftwf_complex* fft_out;                // Выходные данные FFT
    float* window;                         // Оконная функция
    uint32_t fft_size;                     // Размер FFT
};

// ================== Глобальные переменные ==================

static char g_last_error[512] = "No error";
static bool g_hackrf_initialized = false;

// ================== Внутренние функции ==================

/**
 * Установить ошибку
 */
static void set_error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(g_last_error, sizeof(g_last_error), format, args);
    va_end(args);
}

/**
 * Инициализация HackRF библиотеки
 */
static int ensure_hackrf_initialized(void) {
    if (!g_hackrf_initialized) {
        int result = hackrf_init();
        if (result != HACKRF_SUCCESS) {
            set_error("Failed to initialize HackRF library: %s", hackrf_error_name(result));
            return HACKRF_SLAVE_ERROR_DEVICE;
        }
        g_hackrf_initialized = true;
    }
    return HACKRF_SLAVE_SUCCESS;
}

/**
 * Колбэк для захвата данных
 */
static int rx_callback(hackrf_transfer* transfer) {
    capture_context_t* ctx = (capture_context_t*)transfer->rx_ctx;
    
    pthread_mutex_lock(&ctx->mutex);
    
    if (ctx->capture_complete) {
        pthread_mutex_unlock(&ctx->mutex);
        return 0;
    }
    
    uint32_t samples_in_transfer = transfer->valid_length / 2; // I и Q по 1 байту
    uint32_t samples_to_copy = samples_in_transfer;
    
    // Ограничиваем количество копируемых семплов
    if (ctx->samples_captured + samples_to_copy > ctx->samples_needed) {
        samples_to_copy = ctx->samples_needed - ctx->samples_captured;
    }
    
    if (samples_to_copy > 0) {
        uint32_t bytes_to_copy = samples_to_copy * 2;
        memcpy(ctx->buffer + ctx->samples_captured * 2, 
               transfer->buffer, 
               bytes_to_copy);
        ctx->samples_captured += samples_to_copy;
    }
    
    // Проверяем завершение захвата
    if (ctx->samples_captured >= ctx->samples_needed) {
        ctx->capture_complete = true;
        pthread_cond_signal(&ctx->cond);
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    return 0;
}

/**
 * Инициализация контекста захвата
 */
static int init_capture_context(capture_context_t* ctx, uint32_t max_samples) {
    memset(ctx, 0, sizeof(*ctx));
    
    ctx->buffer_size = max_samples * 2; // I и Q по 1 байту
    ctx->buffer = malloc(ctx->buffer_size);
    if (!ctx->buffer) {
        set_error("Failed to allocate capture buffer");
        return HACKRF_SLAVE_ERROR_DEVICE;
    }
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        free(ctx->buffer);
        set_error("Failed to initialize mutex");
        return HACKRF_SLAVE_ERROR_DEVICE;
    }
    
    if (pthread_cond_init(&ctx->cond, NULL) != 0) {
        pthread_mutex_destroy(&ctx->mutex);
        free(ctx->buffer);
        set_error("Failed to initialize condition variable");
        return HACKRF_SLAVE_ERROR_DEVICE;
    }
    
    return HACKRF_SLAVE_SUCCESS;
}

/**
 * Освобождение контекста захвата
 */
static void cleanup_capture_context(capture_context_t* ctx) {
    if (ctx->buffer) {
        free(ctx->buffer);
        ctx->buffer = NULL;
    }
    pthread_cond_destroy(&ctx->cond);
    pthread_mutex_destroy(&ctx->mutex);
}

/**
 * Создание различных оконных функций
 */
static void create_window(float* window, uint32_t size, uint32_t window_type, double beta) {
    switch (window_type) {
        case HACKRF_SLAVE_WINDOW_HAMMING:
            for (uint32_t i = 0; i < size; i++) {
                window[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (size - 1));
            }
            break;
            
        case HACKRF_SLAVE_WINDOW_HANN:
            for (uint32_t i = 0; i < size; i++) {
                window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (size - 1)));
            }
            break;
            
        case HACKRF_SLAVE_WINDOW_BLACKMAN:
            for (uint32_t i = 0; i < size; i++) {
                float n = (float)i / (size - 1);
                window[i] = 0.42f - 0.5f * cosf(2.0f * M_PI * n) + 0.08f * cosf(4.0f * M_PI * n);
            }
            break;
            
        case HACKRF_SLAVE_WINDOW_KAISER:
            // Упрощенная реализация окна Kaiser
            {
                double alpha = beta;
                double besseli0_alpha = 1.0; // Упрощенное значение для I0(alpha)
                for (uint32_t i = 0; i < size; i++) {
                    double n = (double)i / (size - 1) - 0.5;
                    // Упрощенная формула Kaiser window
                    double kaiser_val = 1.0 - 4.0 * n * n;
                    if (kaiser_val < 0) kaiser_val = 0;
                    window[i] = (float)(sqrt(kaiser_val) / besseli0_alpha);
                }
            }
            break;
            
        default:
            // По умолчанию используем Hamming
            for (uint32_t i = 0; i < size; i++) {
                window[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (size - 1));
            }
            break;
    }
}

/**
 * Создание оконной функции Хэмминга (для обратной совместимости)
 */
static void create_hamming_window(float* window, uint32_t size) {
    create_window(window, size, HACKRF_SLAVE_WINDOW_HAMMING, 0.0);
}

/**
 * Коррекция DC смещения в IQ данных
 */
static void apply_dc_offset_correction(int8_t* iq_data, uint32_t num_samples) {
    if (num_samples == 0) return;
    
    // Вычисляем среднее значение I и Q
    double i_sum = 0.0, q_sum = 0.0;
    for (uint32_t i = 0; i < num_samples; i++) {
        i_sum += iq_data[i * 2];
        q_sum += iq_data[i * 2 + 1];
    }
    
    double i_avg = i_sum / num_samples;
    double q_avg = q_sum / num_samples;
    
    // Вычитаем среднее из каждого семпла
    for (uint32_t i = 0; i < num_samples; i++) {
        iq_data[i * 2] = (int8_t)(iq_data[i * 2] - i_avg);
        iq_data[i * 2 + 1] = (int8_t)(iq_data[i * 2 + 1] - q_avg);
    }
}

/**
 * Коррекция I/Q баланса
 */
static void apply_iq_balance_correction(fftwf_complex* complex_data, uint32_t num_samples) {
    if (num_samples == 0) return;
    
    // Вычисляем среднеквадратичные значения I и Q
    double i_rms = 0.0, q_rms = 0.0;
    for (uint32_t i = 0; i < num_samples; i++) {
        i_rms += complex_data[i][0] * complex_data[i][0];
        q_rms += complex_data[i][1] * complex_data[i][1];
    }
    
    i_rms = sqrt(i_rms / num_samples);
    q_rms = sqrt(q_rms / num_samples);
    
    if (i_rms > 0.0 && q_rms > 0.0) {
        double balance_factor = i_rms / q_rms;
        
        // Корректируем Q канал для выравнивания с I каналом
        for (uint32_t i = 0; i < num_samples; i++) {
            complex_data[i][1] *= balance_factor;
        }
    }
}

/**
 * Применение частотного сдвига
 */
static void apply_frequency_offset(fftwf_complex* complex_data, uint32_t num_samples, 
                                  double frequency_offset_hz, double sample_rate) {
    if (frequency_offset_hz == 0.0) return;
    
    double phase_step = 2.0 * M_PI * frequency_offset_hz / sample_rate;
    
    for (uint32_t i = 0; i < num_samples; i++) {
        double phase = phase_step * i;
        float cos_phase = cosf(phase);
        float sin_phase = sinf(phase);
        
        // Комплексное умножение: (a + jb) * (c + jd) = (ac - bd) + j(ad + bc)
        float new_i = complex_data[i][0] * cos_phase - complex_data[i][1] * sin_phase;
        float new_q = complex_data[i][0] * sin_phase + complex_data[i][1] * cos_phase;
        
        complex_data[i][0] = new_i;
        complex_data[i][1] = new_q;
    }
}

/**
 * Спектральное сглаживание
 */
static void apply_spectral_smoothing(float* psd, uint32_t size, uint32_t smoothing_factor) {
    if (smoothing_factor <= 1) return;
    
    float* temp = malloc(size * sizeof(float));
    if (!temp) return;
    
    memcpy(temp, psd, size * sizeof(float));
    
    uint32_t half_window = smoothing_factor / 2;
    
    for (uint32_t i = 0; i < size; i++) {
        float sum = 0.0f;
        uint32_t count = 0;
        
        uint32_t start = (i >= half_window) ? i - half_window : 0;
        uint32_t end = (i + half_window < size) ? i + half_window : size - 1;
        
        for (uint32_t j = start; j <= end; j++) {
            sum += temp[j];
            count++;
        }
        
        psd[i] = sum / count;
    }
    
    free(temp);
}

/**
 * Преобразование IQ данных в комплексные числа с применением окна и фильтров
 */
static void convert_iq_to_complex(const int8_t* iq_data, 
                                 fftwf_complex* complex_data,
                                 const float* window,
                                 uint32_t num_samples,
                                 const hackrf_slave_config_t* config) {
    // Копируем IQ данные во временный буфер для модификации
    int8_t* temp_iq = malloc(num_samples * 2);
    if (temp_iq) {
        memcpy(temp_iq, iq_data, num_samples * 2);
        
        // Применяем коррекцию DC смещения если включена
        if (config && config->dc_offset_correction) {
            apply_dc_offset_correction(temp_iq, num_samples);
        }
        
        // Преобразуем в комплексные числа
        for (uint32_t i = 0; i < num_samples; i++) {
            // HackRF выдает signed 8-bit I/Q данные
            float i_val = (float)temp_iq[i * 2] / 128.0f;
            float q_val = (float)temp_iq[i * 2 + 1] / 128.0f;
            
            // Применяем оконную функцию
            float win_val = (window != NULL) ? window[i] : 1.0f;
            
            complex_data[i][0] = i_val * win_val; // Реальная часть
            complex_data[i][1] = q_val * win_val; // Мнимая часть
        }
        
        free(temp_iq);
    } else {
        // Fallback без коррекции DC
        for (uint32_t i = 0; i < num_samples; i++) {
            float i_val = (float)iq_data[i * 2] / 128.0f;
            float q_val = (float)iq_data[i * 2 + 1] / 128.0f;
            float win_val = (window != NULL) ? window[i] : 1.0f;
            
            complex_data[i][0] = i_val * win_val;
            complex_data[i][1] = q_val * win_val;
        }
    }
    
    // Применяем коррекцию I/Q баланса если включена
    if (config && config->iq_balance_correction) {
        apply_iq_balance_correction(complex_data, num_samples);
    }
    
    // Применяем частотный сдвиг если необходимо
    if (config && config->frequency_offset_hz != 0.0) {
        apply_frequency_offset(complex_data, num_samples, 
                              config->frequency_offset_hz, config->sample_rate);
    }
}

/**
 * Вычисление спектральной плотности мощности
 */
static void compute_psd(const fftwf_complex* fft_out,
                       float* psd,
                       const float* window,
                       uint32_t fft_size,
                       uint32_t sample_rate) {
    // Приводим расчёт к формуле master_hackrf.c (dbFS + поправки)
    // 1) Нормировка FFT
    const float fft_norm = 1.0f / (float)fft_size;

    // 2) Коррекции окна: потери мощности окна и ENBW
    //   Повторяем логику calculate_window_corrections() из master:
    double sum_linear = 0.0, sum_squared = 0.0;
    if (window) {
        for (uint32_t i = 0; i < fft_size; i++) {
            sum_linear  += window[i];
            sum_squared += (double)window[i] * (double)window[i];
        }
    } else {
        // rectangular window
        sum_linear = fft_size;
        sum_squared = fft_size;
    }
    double coherent_gain   = sum_linear / (double)fft_size;
    double processing_gain = sum_squared / (double)fft_size;
    float window_loss_db = -10.0f * log10f((float)processing_gain);
    double enbw = (double)fft_size * processing_gain / (coherent_gain * coherent_gain);
    float enbw_corr_db = 10.0f * log10f((float)enbw);

    // 3) Складываем поправки
    const float total_corr = window_loss_db + enbw_corr_db;

    for (uint32_t i = 0; i < fft_size; i++) {
        float re = fft_out[i][0] * fft_norm;
        float im = fft_out[i][1] * fft_norm;
        float mag2 = re * re + im * im;
        if (mag2 < 1e-20f) mag2 = 1e-20f;
        float dbfs = 10.0f * log10f(mag2);
        // Базовая поправка -10 дБ как в master + оконные/ENBW поправки;
        // абсолютную калибровку добавляем позже вне этой функции (через config.calibration_db)
        psd[i] = dbfs + total_corr - 10.0f;
    }
}

/**
 * Вычисление RSSI в заданной полосе частот
 */
static double compute_band_rssi(const double* freqs,
                               const float* psd,
                               uint32_t num_points,
                               double center_hz,
                               double span_hz) {
    double start_freq = center_hz - span_hz / 2.0;
    double stop_freq = center_hz + span_hz / 2.0;
    
    double power_sum = 0.0;
    int count = 0;
    
    for (uint32_t i = 0; i < num_points; i++) {
        if (freqs[i] >= start_freq && freqs[i] <= stop_freq) {
            // Преобразование из dBm в линейную шкалу для суммирования
            power_sum += pow(10.0, psd[i] / 10.0);
            count++;
        }
    }
    
    if (count == 0) {
        return -200.0; // Нет данных в полосе
    }
    
    // Преобразование обратно в dBm
    return 10.0 * log10(power_sum);
}

/**
 * Вычисление шумового пола (медиана нижних 30% значений)
 */
static double compute_noise_floor(const float* psd_band, uint32_t num_points) {
    if (num_points == 0) {
        return -200.0;
    }
    
    // Копируем и сортируем значения
    float* sorted = malloc(num_points * sizeof(float));
    memcpy(sorted, psd_band, num_points * sizeof(float));
    
    // Простая сортировка пузырьком
    for (uint32_t i = 0; i < num_points - 1; i++) {
        for (uint32_t j = 0; j < num_points - i - 1; j++) {
            if (sorted[j] > sorted[j + 1]) {
                float temp = sorted[j];
                sorted[j] = sorted[j + 1];
                sorted[j + 1] = temp;
            }
        }
    }
    
    // Берем медиану нижних 30%
    uint32_t count_30pct = num_points * 30 / 100;
    if (count_30pct == 0) count_30pct = 1;
    
    double noise_floor = sorted[count_30pct / 2];
    free(sorted);
    
    return noise_floor;
}

// ================== API функции ==================

const char* hackrf_slave_last_error(void) {
    return g_last_error;
}

int hackrf_slave_device_count(void) {
    if (ensure_hackrf_initialized() != HACKRF_SLAVE_SUCCESS) {
        return -1;
    }
    
    hackrf_device_list_t* list = hackrf_device_list();
    if (list == NULL) {
        set_error("Failed to get device list");
        return -1;
    }
    
    int count = list->devicecount;
    hackrf_device_list_free(list);
    
    return count;
}

int hackrf_slave_get_serial(int index, char* serial_out, int max_len) {
    if (ensure_hackrf_initialized() != HACKRF_SLAVE_SUCCESS) {
        return HACKRF_SLAVE_ERROR_DEVICE;
    }
    
    hackrf_device_list_t* list = hackrf_device_list();
    if (list == NULL || index >= list->devicecount) {
        set_error("Invalid device index");
        hackrf_device_list_free(list);
        return HACKRF_SLAVE_ERROR_DEVICE;
    }
    
    strncpy(serial_out, list->serial_numbers[index], max_len - 1);
    serial_out[max_len - 1] = '\0';
    
    hackrf_device_list_free(list);
    return HACKRF_SLAVE_SUCCESS;
}

hackrf_slave_device_t* hackrf_slave_open(const char* serial) {
    if (ensure_hackrf_initialized() != HACKRF_SLAVE_SUCCESS) {
        return NULL;
    }
    
    hackrf_slave_device_t* device = calloc(1, sizeof(hackrf_slave_device_t));
    if (!device) {
        set_error("Failed to allocate device structure");
        return NULL;
    }
    
    // Открываем устройство
    int result;
    if (serial && strlen(serial) > 0) {
        result = hackrf_open_by_serial(serial, &device->dev);
    } else {
        result = hackrf_open(&device->dev);
    }
    
    if (result != HACKRF_SUCCESS) {
        set_error("Failed to open HackRF device: %s", hackrf_error_name(result));
        free(device);
        return NULL;
    }
    
    // Инициализация контекста захвата
    if (init_capture_context(&device->capture_ctx, HACKRF_SLAVE_MAX_SAMPLES) != HACKRF_SLAVE_SUCCESS) {
        hackrf_close(device->dev);
        free(device);
        return NULL;
    }
    
    // Установка базовых параметров
    device->fft_size = HACKRF_SLAVE_DEFAULT_FFT_SIZE;
    strcpy(device->slave_id, "slave_unknown");
    
    // Инициализация FFTW
    device->fft_in = fftwf_malloc(device->fft_size * sizeof(fftwf_complex));
    device->fft_out = fftwf_malloc(device->fft_size * sizeof(fftwf_complex));
    device->window = malloc(device->fft_size * sizeof(float));
    
    if (!device->fft_in || !device->fft_out || !device->window) {
        set_error("Failed to allocate FFT buffers");
        hackrf_slave_close(device);
        return NULL;
    }
    
    device->fft_plan = fftwf_plan_dft_1d(device->fft_size,
                                        device->fft_in,
                                        device->fft_out,
                                        FFTW_FORWARD,
                                        FFTW_ESTIMATE);
    
    if (!device->fft_plan) {
        set_error("Failed to create FFT plan");
        hackrf_slave_close(device);
        return NULL;
    }
    
    create_window(device->window, device->fft_size, HACKRF_SLAVE_WINDOW_HAMMING, 0.0);
    
    return device;
}

void hackrf_slave_close(hackrf_slave_device_t* device) {
    if (!device) return;
    
    if (device->dev) {
        hackrf_stop_rx(device->dev);
        hackrf_close(device->dev);
    }
    
    cleanup_capture_context(&device->capture_ctx);
    
    if (device->fft_plan) {
        fftwf_destroy_plan(device->fft_plan);
    }
    
    if (device->fft_in) {
        fftwf_free(device->fft_in);
    }
    
    if (device->fft_out) {
        fftwf_free(device->fft_out);
    }
    
    if (device->window) {
        free(device->window);
    }
    
    free(device);
}

int hackrf_slave_configure(hackrf_slave_device_t* device,
                          const hackrf_slave_config_t* config) {
    if (!device || !config) {
        set_error("Invalid parameters");
        return HACKRF_SLAVE_ERROR_CONFIG;
    }
    
    int result;
    
    // Установка частоты дискретизации
    result = hackrf_set_sample_rate(device->dev, config->sample_rate);
    if (result != HACKRF_SUCCESS) {
        set_error("Failed to set sample rate: %s", hackrf_error_name(result));
        return HACKRF_SLAVE_ERROR_CONFIG;
    }
    
    // Установка полосы пропускания
    result = hackrf_set_baseband_filter_bandwidth(device->dev, config->bandwidth_hz);
    if (result != HACKRF_SUCCESS) {
        set_error("Failed to set bandwidth: %s", hackrf_error_name(result));
        return HACKRF_SLAVE_ERROR_CONFIG;
    }
    
    // Установка усилений
    result = hackrf_set_lna_gain(device->dev, config->lna_gain);
    if (result != HACKRF_SUCCESS) {
        set_error("Failed to set LNA gain: %s", hackrf_error_name(result));
        return HACKRF_SLAVE_ERROR_CONFIG;
    }
    
    result = hackrf_set_vga_gain(device->dev, config->vga_gain);
    if (result != HACKRF_SUCCESS) {
        set_error("Failed to set VGA gain: %s", hackrf_error_name(result));
        return HACKRF_SLAVE_ERROR_CONFIG;
    }
    
    // Установка RF усилителя
    result = hackrf_set_amp_enable(device->dev, config->amp_enable ? 1 : 0);
    if (result != HACKRF_SUCCESS) {
        set_error("Failed to set amp enable: %s", hackrf_error_name(result));
        return HACKRF_SLAVE_ERROR_CONFIG;
    }
    
    // Сохранение конфигурации
    device->config = *config;
    device->is_configured = true;
    
    // Пересоздание оконной функции с новым типом
    if (device->window) {
        create_window(device->window, device->fft_size, 
                     config->filter_window_type, config->filter_beta);
    }
    
    return HACKRF_SLAVE_SUCCESS;
}

int hackrf_slave_set_id(hackrf_slave_device_t* device, const char* slave_id) {
    if (!device || !slave_id) {
        set_error("Invalid parameters");
        return HACKRF_SLAVE_ERROR_CONFIG;
    }
    
    strncpy(device->slave_id, slave_id, sizeof(device->slave_id) - 1);
    device->slave_id[sizeof(device->slave_id) - 1] = '\0';
    
    return HACKRF_SLAVE_SUCCESS;
}

int hackrf_slave_measure_rssi(hackrf_slave_device_t* device,
                             double center_hz,
                             double span_hz,
                             uint32_t dwell_ms,
                             hackrf_slave_rssi_measurement_t* measurement) {
    if (!device || !measurement || !device->is_configured) {
        set_error("Invalid parameters or device not configured");
        return HACKRF_SLAVE_ERROR_CONFIG;
    }
    
    // Установка частоты
    int result = hackrf_set_freq(device->dev, (uint64_t)center_hz);
    if (result != HACKRF_SUCCESS) {
        set_error("Failed to set frequency: %s", hackrf_error_name(result));
        return HACKRF_SLAVE_ERROR_CONFIG;
    }
    
    // Вычисление количества необходимых семплов
    uint32_t samples_needed = (uint32_t)((double)device->config.sample_rate * dwell_ms / 1000.0);
    if (samples_needed > HACKRF_SLAVE_MAX_SAMPLES) {
        samples_needed = HACKRF_SLAVE_MAX_SAMPLES;
    }
    
    // Подготовка контекста захвата
    pthread_mutex_lock(&device->capture_ctx.mutex);
    device->capture_ctx.samples_captured = 0;
    device->capture_ctx.samples_needed = samples_needed;
    device->capture_ctx.capture_complete = false;
    pthread_mutex_unlock(&device->capture_ctx.mutex);
    
    // Запуск захвата
    result = hackrf_start_rx(device->dev, rx_callback, &device->capture_ctx);
    if (result != HACKRF_SUCCESS) {
        set_error("Failed to start RX: %s", hackrf_error_name(result));
        return HACKRF_SLAVE_ERROR_CAPTURE;
    }
    
    // Ожидание завершения захвата с таймаутом
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += (dwell_ms / 1000) + 2; // Таймаут = время измерения + 2 секунды
    
    pthread_mutex_lock(&device->capture_ctx.mutex);
    while (!device->capture_ctx.capture_complete) {
        int ret = pthread_cond_timedwait(&device->capture_ctx.cond, 
                                       &device->capture_ctx.mutex, 
                                       &timeout);
        if (ret != 0) {
            pthread_mutex_unlock(&device->capture_ctx.mutex);
            hackrf_stop_rx(device->dev);
            set_error("Capture timeout");
            return HACKRF_SLAVE_ERROR_TIMEOUT;
        }
    }
    uint32_t captured_samples = device->capture_ctx.samples_captured;
    pthread_mutex_unlock(&device->capture_ctx.mutex);
    
    // Остановка захвата
    hackrf_stop_rx(device->dev);
    
    if (captured_samples == 0) {
        set_error("No samples captured");
        return HACKRF_SLAVE_ERROR_CAPTURE;
    }
    
    // Обработка захваченных данных
    uint32_t fft_samples = (captured_samples / device->fft_size) * device->fft_size;
    uint32_t num_ffts = fft_samples / device->fft_size;
    
    if (num_ffts == 0) {
        set_error("Insufficient samples for FFT");
        return HACKRF_SLAVE_ERROR_PROCESSING;
    }
    
    // Усреднение спектров
    float* avg_psd = calloc(device->fft_size, sizeof(float));
    double* freqs = malloc(device->fft_size * sizeof(double));
    
    if (!avg_psd || !freqs) {
        free(avg_psd);
        free(freqs);
        set_error("Memory allocation failed");
        return HACKRF_SLAVE_ERROR_PROCESSING;
    }
    
    // Вычисление частотной оси
    for (uint32_t i = 0; i < device->fft_size; i++) {
        double freq_offset = ((double)i - device->fft_size / 2.0) * device->config.sample_rate / device->fft_size;
        freqs[i] = center_hz + freq_offset;
    }
    
    // Обработка всех FFT блоков
    for (uint32_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {
        uint32_t offset = fft_idx * device->fft_size;
        
        // Преобразование IQ данных в комплексные числа с фильтрами
        convert_iq_to_complex(&device->capture_ctx.buffer[offset * 2],
                             device->fft_in,
                             device->window,
                             device->fft_size,
                             &device->config);
        
        // Выполнение FFT
        fftwf_execute(device->fft_plan);
        
        // Вычисление PSD для этого блока
        float* block_psd = malloc(device->fft_size * sizeof(float));
        compute_psd(device->fft_out, block_psd, device->window, 
                   device->fft_size, device->config.sample_rate);
        
        // Сдвиг для центрирования спектра
        for (uint32_t i = 0; i < device->fft_size; i++) {
            uint32_t shifted_idx = (i + device->fft_size / 2) % device->fft_size;
            avg_psd[shifted_idx] += block_psd[i];
        }
        
        free(block_psd);
    }
    
    // Усреднение
    for (uint32_t i = 0; i < device->fft_size; i++) {
        avg_psd[i] /= num_ffts;
    }
    
    // Применение калибровки
    for (uint32_t i = 0; i < device->fft_size; i++) {
        avg_psd[i] += device->config.calibration_db;
    }
    
    // Применение спектрального сглаживания если включено
    if (device->config.spectral_smoothing && device->config.smoothing_factor > 1) {
        apply_spectral_smoothing(avg_psd, device->fft_size, device->config.smoothing_factor);
    }
    
    // Вычисление RSSI в полосе
    double band_rssi = compute_band_rssi(freqs, avg_psd, device->fft_size, 
                                        center_hz, span_hz);
    
    // Извлечение данных в полосе для вычисления шума
    uint32_t band_count = 0;
    for (uint32_t i = 0; i < device->fft_size; i++) {
        if (freqs[i] >= center_hz - span_hz/2 && freqs[i] <= center_hz + span_hz/2) {
            band_count++;
        }
    }
    
    float* band_psd = malloc(band_count * sizeof(float));
    uint32_t band_idx = 0;
    for (uint32_t i = 0; i < device->fft_size && band_idx < band_count; i++) {
        if (freqs[i] >= center_hz - span_hz/2 && freqs[i] <= center_hz + span_hz/2) {
            band_psd[band_idx++] = avg_psd[i];
        }
    }
    
    double noise_floor = compute_noise_floor(band_psd, band_count);
    double snr = band_rssi - noise_floor;
    
    // Заполнение результата
    strncpy(measurement->slave_id, device->slave_id, sizeof(measurement->slave_id) - 1);
    measurement->slave_id[sizeof(measurement->slave_id) - 1] = '\0';
    measurement->center_hz = center_hz;
    measurement->span_hz = span_hz;
    measurement->band_rssi_dbm = band_rssi;
    measurement->band_noise_dbm = noise_floor;
    measurement->snr_db = snr;
    measurement->n_samples = captured_samples;
    measurement->timestamp = (double)time(NULL);
    measurement->valid = true;
    
    free(avg_psd);
    free(freqs);
    free(band_psd);
    
    return HACKRF_SLAVE_SUCCESS;
}

int hackrf_slave_measure_target_rms(hackrf_slave_device_t* device,
                                   const char* target_id,
                                   double center_hz,
                                   double halfspan_hz,
                                   double guard_hz,
                                   uint32_t dwell_ms,
                                   hackrf_slave_rms_measurement_t* measurement) {
    if (!device || !target_id || !measurement || !device->is_configured) {
        set_error("Invalid parameters or device not configured");
        return HACKRF_SLAVE_ERROR_CONFIG;
    }
    
    // Вычисление рабочей полосы: 2 * (halfspan + guard)
    double working_span = 2.0 * (halfspan_hz + guard_hz);
    
    // Установка частоты
    int result = hackrf_set_freq(device->dev, (uint64_t)center_hz);
    if (result != HACKRF_SUCCESS) {
        set_error("Failed to set frequency: %s", hackrf_error_name(result));
        return HACKRF_SLAVE_ERROR_CONFIG;
    }
    
    // Выполняем измерение RSSI для рабочей полосы
    hackrf_slave_rssi_measurement_t rssi_measurement;
    result = hackrf_slave_measure_rssi(device, center_hz, working_span, 
                                      dwell_ms, &rssi_measurement);
    if (result != HACKRF_SLAVE_SUCCESS) {
        return result;
    }
    
    // Для RMS измерения используем полосу target (2 * halfspan)
    double target_span = 2.0 * halfspan_hz;
    double target_rssi = compute_band_rssi((double*)&center_hz, // Упрощенно
                                          &rssi_measurement.band_rssi_dbm,
                                          1, center_hz, target_span);
    
    // Заполнение результата RMS измерения
    strncpy(measurement->slave_id, device->slave_id, sizeof(measurement->slave_id) - 1);
    measurement->slave_id[sizeof(measurement->slave_id) - 1] = '\0';
    strncpy(measurement->target_id, target_id, sizeof(measurement->target_id) - 1);
    measurement->target_id[sizeof(measurement->target_id) - 1] = '\0';
    
    measurement->center_hz = center_hz;
    measurement->halfspan_hz = halfspan_hz;
    measurement->guard_hz = guard_hz;
    measurement->rssi_rms_dbm = rssi_measurement.band_rssi_dbm; // Используем как RMS
    measurement->noise_floor_dbm = rssi_measurement.band_noise_dbm;
    measurement->snr_db = rssi_measurement.snr_db;
    measurement->n_samples = rssi_measurement.n_samples;
    measurement->timestamp = rssi_measurement.timestamp;
    measurement->valid = true;
    
    return HACKRF_SLAVE_SUCCESS;
}

int hackrf_slave_get_spectrum(hackrf_slave_device_t* device,
                             double center_hz,
                             double span_hz,
                             uint32_t dwell_ms,
                             double* freqs_out,
                             float* powers_out,
                             int max_points) {
    // Эта функция использует тот же механизм что и measure_rssi
    // но возвращает весь спектр вместо агрегированных значений
    
    hackrf_slave_rssi_measurement_t dummy_measurement;
    int result = hackrf_slave_measure_rssi(device, center_hz, span_hz, 
                                          dwell_ms, &dummy_measurement);
    if (result != HACKRF_SLAVE_SUCCESS) {
        return -1;
    }
    
    // Возвращаем количество точек (упрощенная реализация)
    int points_to_return = (max_points < (int)device->fft_size) ? max_points : device->fft_size;
    
    for (int i = 0; i < points_to_return; i++) {
        double freq_offset = ((double)i - device->fft_size / 2.0) * device->config.sample_rate / device->fft_size;
        freqs_out[i] = center_hz + freq_offset;
        powers_out[i] = dummy_measurement.band_rssi_dbm; // Упрощенно
    }
    
    return points_to_return;
}

bool hackrf_slave_is_ready(hackrf_slave_device_t* device) {
    return (device != NULL) && (device->dev != NULL) && device->is_configured;
}

int hackrf_slave_get_config(hackrf_slave_device_t* device,
                           hackrf_slave_config_t* config_out) {
    if (!device || !config_out) {
        set_error("Invalid parameters");
        return HACKRF_SLAVE_ERROR_CONFIG;
    }
    
    *config_out = device->config;
    return HACKRF_SLAVE_SUCCESS;
}