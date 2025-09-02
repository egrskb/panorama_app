/**
 * hackrf_slave.h - Нативная C библиотека для работы со слейв-устройствами HackRF
 * 
 * Обеспечивает прямую работу с HackRF без использования SoapySDR.
 * Поддерживает измерение RSSI в заданных полосах частот и RMS измерения для целей.
 */

#ifndef HACKRF_SLAVE_H
#define HACKRF_SLAVE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ================== Константы ==================

// Константы HackRF
#define HACKRF_SLAVE_MIN_FREQ_HZ      1000000ULL      // 1 MHz
#define HACKRF_SLAVE_MAX_FREQ_HZ      6000000000ULL   // 6 GHz
#define HACKRF_SLAVE_MIN_SAMPLE_RATE  2000000         // 2 MHz
#define HACKRF_SLAVE_MAX_SAMPLE_RATE  20000000        // 20 MHz
#define HACKRF_SLAVE_MIN_GAIN         0
#define HACKRF_SLAVE_MAX_GAIN         62

// Размеры буферов
#define HACKRF_SLAVE_MAX_SAMPLES      262144          // 256K samples
#define HACKRF_SLAVE_DEFAULT_FFT_SIZE 8192

// Ошибки
#define HACKRF_SLAVE_SUCCESS          0
#define HACKRF_SLAVE_ERROR_DEVICE     -1
#define HACKRF_SLAVE_ERROR_CONFIG     -2
#define HACKRF_SLAVE_ERROR_CAPTURE    -3
#define HACKRF_SLAVE_ERROR_PROCESSING -4
#define HACKRF_SLAVE_ERROR_TIMEOUT    -5

// Типы оконных функций
#define HACKRF_SLAVE_WINDOW_HAMMING   0
#define HACKRF_SLAVE_WINDOW_HANN      1
#define HACKRF_SLAVE_WINDOW_BLACKMAN  2
#define HACKRF_SLAVE_WINDOW_KAISER    3

// ================== Типы данных ==================

/**
 * Структура для RSSI измерения
 */
typedef struct {
    char slave_id[64];          // ID слейва
    double center_hz;           // Центральная частота
    double span_hz;             // Полоса измерения
    double band_rssi_dbm;       // RSSI в полосе (dBm)
    double band_noise_dbm;      // Шумовой пол (dBm)
    double snr_db;              // SNR (dB)
    uint32_t n_samples;         // Количество обработанных семплов
    double timestamp;           // Время измерения (UNIX timestamp)
    bool valid;                 // Валидность измерения
} hackrf_slave_rssi_measurement_t;

/**
 * Структура для RMS измерения цели
 */
typedef struct {
    char slave_id[64];          // ID слейва  
    char target_id[64];         // ID цели
    double center_hz;           // Центральная частота цели
    double halfspan_hz;         // Половина полосы цели
    double guard_hz;            // Защитная полоса
    double rssi_rms_dbm;        // RMS RSSI (dBm)
    double noise_floor_dbm;     // Шумовой пол (dBm)
    double snr_db;              // SNR (dB)
    uint32_t n_samples;         // Количество семплов
    double timestamp;           // Время измерения
    bool valid;                 // Валидность измерения
} hackrf_slave_rms_measurement_t;

/**
 * Структура для конфигурации устройства
 */
typedef struct {
    uint32_t sample_rate;       // Частота дискретизации (Hz)
    uint32_t lna_gain;          // Усиление LNA (dB, 0-40, шаг 8)
    uint32_t vga_gain;          // Усиление VGA (dB, 0-62, шаг 2)
    bool amp_enable;            // Включение RF усилителя
    uint32_t bandwidth_hz;      // Полоса пропускания (Hz)
    double calibration_db;      // Калибровочная поправка (dB)
    
    // Дополнительные фильтры для IQ обработки
    bool dc_offset_correction;  // Коррекция DC смещения
    double frequency_offset_hz; // Коррекция частотного смещения (Hz)
    bool iq_balance_correction; // Коррекция I/Q баланса
    uint32_t filter_window_type; // Тип оконной функции (0=Hamming, 1=Hann, 2=Blackman, 3=Kaiser)
    double filter_beta;         // Параметр бета для окна Kaiser (если используется)
    bool spectral_smoothing;    // Спектральное сглаживание
    uint32_t smoothing_factor;  // Фактор сглаживания (количество соседних бинов)
} hackrf_slave_config_t;

/**
 * Обработчик устройства (непрозрачная структура)
 */
typedef struct hackrf_slave_device hackrf_slave_device_t;

// ================== API функции ==================

/**
 * Получить последнюю ошибку
 * @return Строка с описанием последней ошибки
 */
const char* hackrf_slave_last_error(void);

/**
 * Получить количество доступных HackRF устройств
 * @return Количество устройств или отрицательное значение при ошибке
 */
int hackrf_slave_device_count(void);

/**
 * Получить серийный номер устройства по индексу
 * @param index Индекс устройства (0 - первое устройство)
 * @param serial_out Буфер для серийного номера
 * @param max_len Размер буфера
 * @return HACKRF_SLAVE_SUCCESS при успехе
 */
int hackrf_slave_get_serial(int index, char* serial_out, int max_len);

/**
 * Открыть устройство по серийному номеру
 * @param serial Серийный номер устройства (NULL для первого доступного)
 * @return Указатель на устройство или NULL при ошибке
 */
hackrf_slave_device_t* hackrf_slave_open(const char* serial);

/**
 * Закрыть устройство
 * @param device Указатель на устройство
 */
void hackrf_slave_close(hackrf_slave_device_t* device);

/**
 * Настроить устройство
 * @param device Указатель на устройство
 * @param config Конфигурация устройства
 * @return HACKRF_SLAVE_SUCCESS при успехе
 */
int hackrf_slave_configure(hackrf_slave_device_t* device, 
                          const hackrf_slave_config_t* config);

/**
 * Установить ID слейва для измерений
 * @param device Указатель на устройство
 * @param slave_id ID слейва
 * @return HACKRF_SLAVE_SUCCESS при успехе
 */
int hackrf_slave_set_id(hackrf_slave_device_t* device, const char* slave_id);

/**
 * Измерить RSSI в заданной полосе частот
 * @param device Указатель на устройство
 * @param center_hz Центральная частота (Hz)
 * @param span_hz Полоса измерения (Hz)
 * @param dwell_ms Время измерения (мс)
 * @param measurement Результат измерения
 * @return HACKRF_SLAVE_SUCCESS при успехе
 */
int hackrf_slave_measure_rssi(hackrf_slave_device_t* device,
                             double center_hz,
                             double span_hz,
                             uint32_t dwell_ms,
                             hackrf_slave_rssi_measurement_t* measurement);

/**
 * Измерить RMS для конкретной цели
 * @param device Указатель на устройство
 * @param target_id ID цели
 * @param center_hz Центральная частота цели (Hz)
 * @param halfspan_hz Половина полосы цели (Hz)
 * @param guard_hz Защитная полоса (Hz)
 * @param dwell_ms Время измерения (мс)
 * @param measurement Результат измерения
 * @return HACKRF_SLAVE_SUCCESS при успехе
 */
int hackrf_slave_measure_target_rms(hackrf_slave_device_t* device,
                                   const char* target_id,
                                   double center_hz,
                                   double halfspan_hz,
                                   double guard_hz,
                                   uint32_t dwell_ms,
                                   hackrf_slave_rms_measurement_t* measurement);

/**
 * Получить спектр в заданной полосе (для отладки)
 * @param device Указатель на устройство
 * @param center_hz Центральная частота (Hz)
 * @param span_hz Полоса спектра (Hz)
 * @param dwell_ms Время накопления (мс)
 * @param freqs_out Массив частот (Hz) [выходной]
 * @param powers_out Массив мощностей (dBm) [выходной]
 * @param max_points Максимальное количество точек
 * @return Количество точек спектра или отрицательное значение при ошибке
 */
int hackrf_slave_get_spectrum(hackrf_slave_device_t* device,
                             double center_hz,
                             double span_hz,
                             uint32_t dwell_ms,
                             double* freqs_out,
                             float* powers_out,
                             int max_points);

/**
 * Проверить состояние устройства
 * @param device Указатель на устройство
 * @return true если устройство готово к работе
 */
bool hackrf_slave_is_ready(hackrf_slave_device_t* device);

/**
 * Получить текущую конфигурацию устройства
 * @param device Указатель на устройство
 * @param config_out Текущая конфигурация [выходная]
 * @return HACKRF_SLAVE_SUCCESS при успехе
 */
int hackrf_slave_get_config(hackrf_slave_device_t* device,
                           hackrf_slave_config_t* config_out);

#ifdef __cplusplus
}
#endif

#endif // HACKRF_SLAVE_H