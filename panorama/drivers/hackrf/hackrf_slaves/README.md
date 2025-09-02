# HackRF Slave Library

Нативная C библиотека для работы со слейв-устройствами HackRF без использования SoapySDR.
Обеспечивает прямой доступ к HackRF устройствам с высокой производительностью и стабильностью.

## Возможности

- **Прямая работа с HackRF**: Использование нативной libhackrf без промежуточных слоев
- **Высокая производительность**: Оптимизированная обработка сигналов на C с использованием FFTW
- **RSSI измерения**: Точные измерения мощности сигнала в заданных полосах частот  
- **RMS измерения для целей**: Специализированные измерения для триангуляции
- **Управление ошибками**: Автоматическое восстановление при сбоях и временное отключение проблемных устройств
- **Python интерфейс**: Удобные биндинги для интеграции с существующим кодом
- **Потокобезопасность**: Безопасная работа в многопоточной среде

## Архитектура

```
Python приложение
       ↓
hackrf_slave_wrapper.py (ctypes биндинги)
       ↓  
libhackrf_slave.so (C библиотека)
       ↓
libhackrf.so (драйвер HackRF)
       ↓
HackRF устройство
```

## Установка зависимостей

### Ubuntu/Debian

```bash
sudo apt-get install libhackrf-dev libfftw3-dev build-essential
```

### Fedora/CentOS

```bash
sudo yum install hackrf-devel fftw3-devel gcc make
```

## Сборка

1. **Автоматическая сборка** (рекомендуется):
   ```bash
   ./build_hackrf_slave.sh
   ```

2. **Ручная сборка**:
   ```bash
   cd panorama/drivers/hackrf_slave
   make clean
   make all
   ```

## Использование

### Python API

```python
from panorama.drivers.hackrf_slave.hackrf_slave_wrapper import HackRFSlaveDevice

# Открытие устройства
device = HackRFSlaveDevice()  # Первое доступное устройство
# или
device = HackRFSlaveDevice(serial="0000000000000000457863dc27e6381f")

# Конфигурация
device.configure(
    sample_rate=8000000,  # 8 МГц
    lna_gain=16,          # LNA усиление
    vga_gain=20,          # VGA усиление  
    amp_enable=False,     # RF усилитель
    bandwidth_hz=2500000, # Полоса пропускания
    calibration_db=0.0    # Калибровка
)

device.set_slave_id("slave_001")

# RSSI измерение
result = device.measure_rssi(
    center_hz=2450000000,  # 2.45 ГГц
    span_hz=20000000,      # 20 МГц  
    dwell_ms=400           # 400 мс
)

print(f"RSSI: {result['band_rssi_dbm']:.1f} dBm")
print(f"SNR: {result['snr_db']:.1f} dB")

# RMS измерение цели
rms_result = device.measure_target_rms(
    target_id="target_001",
    center_hz=2450000000,
    halfspan_hz=10000000,
    guard_hz=5000000,
    dwell_ms=400
)

print(f"RMS: {rms_result['rssi_rms_dbm']:.1f} dBm")

# Получение спектра (для отладки)
freqs, powers = device.get_spectrum(
    center_hz=2450000000,
    span_hz=20000000,
    dwell_ms=400
)

device.close()
```

### Интеграция с существующим кодом

Существующий код SlaveSDR автоматически использует новую библиотеку вместо SoapySDR:

```python
from panorama.features.slave_controller.slave import SlaveSDR

# URI может быть серийным номером HackRF
slave = SlaveSDR("slave_001", "serial=0000000000000000457863dc27e6381f", logger)

# Все существующие методы работают без изменений
measurement = slave.measure_band_rssi(2.45e9, 20e6, 400)
rms_measurement = slave.measure_target_rms(target_dict, 400)
```

## API Reference

### HackRFSlaveDevice

#### Конструктор
- `HackRFSlaveDevice(serial=None)` - Открывает устройство

#### Методы конфигурации
- `configure(**kwargs)` - Конфигурирует устройство
- `set_slave_id(slave_id)` - Устанавливает ID слейва
- `get_config()` - Получает текущую конфигурацию
- `is_ready()` - Проверяет готовность устройства

#### Измерения
- `measure_rssi(center_hz, span_hz, dwell_ms)` - RSSI измерение
- `measure_target_rms(target_id, center_hz, halfspan_hz, guard_hz, dwell_ms)` - RMS измерение
- `get_spectrum(center_hz, span_hz, dwell_ms, max_points)` - Получение спектра

#### Управление
- `close()` - Закрывает устройство

### Утилиты

- `get_device_count()` - Количество доступных устройств
- `get_device_serial(index)` - Серийный номер устройства по индексу  
- `list_devices()` - Список всех серийных номеров

## Структуры данных

### RSSI Measurement
```python
{
    'slave_id': str,        # ID слейва
    'center_hz': float,     # Центральная частота
    'span_hz': float,       # Полоса измерения
    'band_rssi_dbm': float, # RSSI в dBm
    'band_noise_dbm': float,# Шумовой пол в dBm
    'snr_db': float,        # SNR в dB
    'n_samples': int,       # Количество семплов
    'timestamp': float,     # UNIX timestamp
    'valid': bool           # Валидность измерения
}
```

### RMS Measurement
```python
{
    'slave_id': str,         # ID слейва
    'target_id': str,        # ID цели
    'center_hz': float,      # Центральная частота
    'halfspan_hz': float,    # Половина полосы цели
    'guard_hz': float,       # Защитная полоса
    'rssi_rms_dbm': float,   # RMS RSSI в dBm
    'noise_floor_dbm': float,# Шумовой пол в dBm
    'snr_db': float,         # SNR в dB
    'n_samples': int,        # Количество семплов
    'timestamp': float,      # UNIX timestamp
    'valid': bool            # Валидность измерения
}
```

## Обработка ошибок

Библиотека использует иерархию исключений:

- `HackRFSlaveError` - Базовое исключение
- `HackRFSlaveDeviceError` - Ошибки устройства
- `HackRFSlaveConfigError` - Ошибки конфигурации
- `HackRFSlaveCaptureError` - Ошибки захвата данных
- `HackRFSlaveProcessingError` - Ошибки обработки
- `HackRFSlaveTimeoutError` - Ошибки таймаута

```python
from panorama.drivers.hackrf_slave.hackrf_slave_wrapper import HackRFSlaveDeviceError

try:
    device = HackRFSlaveDevice("invalid_serial")
except HackRFSlaveDeviceError as e:
    print(f"Device error: {e}")
```

## Отличия от SoapySDR

| Параметр | SoapySDR | HackRF Slave |
|----------|----------|--------------|
| Зависимости | SoapySDR + SoapyHackRF | libhackrf + libfftw3f |
| Производительность | Средняя | Высокая |
| Стабильность | Периодические сбои | Высокая стабильность |
| Управление ошибками | Базовое | Продвинутое с восстановлением |
| FFT | NumPy | Оптимизированный FFTW |
| Настройка | Универсальная | Специализированная для HackRF |

## Производительность

- **Время измерения**: ~400-800 мс (в зависимости от dwell_ms)
- **Пропускная способность**: До 20 МГц полосы
- **Память**: Оптимизированное использование буферов
- **CPU**: Эффективное использование FFTW для FFT операций

## Диагностика

### Проверка зависимостей
```bash
# Проверка libhackrf
pkg-config --exists libhackrf && echo "OK" || echo "MISSING"

# Проверка libfftw3f  
pkg-config --exists fftw3f && echo "OK" || echo "MISSING"

# Проверка HackRF устройств
hackrf_info
```

### Отладка
```python
# Включение детального логирования
import logging
logging.basicConfig(level=logging.DEBUG)

# Проверка устройств
from panorama.drivers.hackrf_slave.hackrf_slave_wrapper import list_devices
print("Available devices:", list_devices())
```

## Совместимость

- **Python**: 3.7+
- **HackRF**: Все модели (One, Blue, One r9)
- **ОС**: Linux (Ubuntu 18.04+, CentOS 7+)
- **Архитектура**: x86_64, ARM64

## Известные ограничения

1. **Только Linux**: Windows и macOS не поддерживаются
2. **Одно устройство на процесс**: Нельзя одновременно использовать одно HackRF из разных процессов
3. **Частотный диапазон**: Ограничен возможностями HackRF (1 МГц - 6 ГГц)
4. **Полоса пропускания**: Максимум 20 МГц

## Миграция с SoapySDR

Для перехода с SoapySDR достаточно:

1. Собрать новую библиотеку: `./build_hackrf_slave.sh`
2. Обновить URI слейвов в конфигурации (если требуется указать конкретное устройство)
3. Перезапустить приложение

Весь существующий Python код работает без изменений благодаря совместимому API в `SlaveSDR`.

## Поддержка

При возникновении проблем:

1. Проверьте зависимости
2. Убедитесь что HackRF подключен и распознается (`hackrf_info`)
3. Проверьте логи приложения
4. Попробуйте пересобрать библиотеку

## Лицензия

Использует те же лицензии что и зависимости:
- libhackrf: GPLv2+
- libfftw3f: GPLv2+