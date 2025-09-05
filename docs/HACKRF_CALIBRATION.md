# Калибровка и синхронизация HackRF устройств

## Обзор

Система калибровки HackRF решает основные проблемы при работе с множественными SDR устройствами:

1. **Частотные смещения** - каждый HackRF имеет неточность тактового генератора ±20 ppm
2. **Амплитудные расхождения** - разные амплитудные характеристики устройств
3. **Временная десинхронизация** - отсутствие единого времени начала измерений

## Архитектура

### Master/Slave структура
- **Master устройство** - опорное устройство для калибровки (обычно первое в списке)
- **Slave устройства** - калибруются относительно master устройства

### Компоненты системы

1. **HackRFSyncCalibrator** - основной класс калибровки
2. **DeviceCalibration** - структура данных калибровки одного устройства
3. **CalibrationTarget** - опорные цели для калибровки (FM, GSM, WiFi сигналы)

## Процесс калибровки

### 1. Определение частотных смещений

```python
# Алгоритм определения частотного смещения:
# 1. Master измеряет спектр вокруг опорной частоты
# 2. Slave ищет максимум сигнала в той же области
# 3. Вычисляется разность частот пиков
frequency_offset = slave_peak_freq - master_peak_freq
```

### 2. Амплитудная калибровка

```python
# Синхронные измерения RSSI на одной частоте
amplitude_offset = slave_rssi - master_rssi
```

### 3. Опорные частоты для калибровки

По умолчанию используются следующие опорные сигналы:
- **FM Radio**: 100 МГц, -30 дБм
- **GSM 900**: 935 МГц, -45 дБм  
- **GSM 1800**: 1850 МГц, -50 дБм
- **WiFi 2.4G**: 2450 МГц, -40 дБм

## Использование

### CLI команда калибровки

```bash
# Показать статус калибровки
python -m panorama.cli.calibrate_hackrf --status

# Калибровать все устройства
python -m panorama.cli.calibrate_hackrf --calibrate-all

# Калибровать конкретное устройство
python -m panorama.cli.calibrate_hackrf --device 1234567890abcdef

# С подробным логированием
python -m panorama.cli.calibrate_hackrf --calibrate-all --log-level DEBUG
```

### Программное использование

```python
from panorama.features.calibration import HackRFSyncCalibrator
from panorama.drivers.hackrf.hackrf_slaves.hackrf_slave_wrapper import HackRFSlaveDevice

# Создание калибратора
calibrator = HackRFSyncCalibrator()

# Открытие устройств
master = HackRFSlaveDevice(serial="", logger=logger)  # Первое устройство
slave = HackRFSlaveDevice(serial="1234567890abcdef", logger=logger)

master.open()
slave.open()

# Калибровка
success = calibrator.calibrate_device_pair(master, slave, "1234567890abcdef")

# Применение калибровки
if success:
    calibrator.apply_calibration(slave, "1234567890abcdef")
```

### Интеграция с SlaveManager

```python
from panorama.features.slave_controller.slave import SlaveManager

# Создание менеджера slave устройств
slave_manager = SlaveManager(logger)

# Добавление устройств
slave_manager.add_slave("slave_1", "driver=hackrf,serial=1234567890abcdef")
slave_manager.add_slave("slave_2", "driver=hackrf,serial=fedcba0987654321")

# Калибровка всех slave относительно master
master_device = get_master_device()  # Получить master устройство
success = slave_manager.calibrate_all_slaves(master_device)

# Настройка синхронных измерений
slave_manager.setup_synchronized_measurements()
```

## Структура калибровочных данных

Калибровочные данные сохраняются в JSON файле `~/.panorama/hackrf_calibration.json`:

```json
{
  "calibrations": {
    "1234567890abcdef": {
      "serial": "1234567890abcdef",
      "frequency_offset_hz": -2350.0,
      "amplitude_offset_db": -1.2,
      "phase_offset_deg": 0.0,
      "temperature_coefficient": 0.0,
      "last_calibration_time": 1704067200.0,
      "reference_temperature": 25.0
    }
  },
  "last_update": 1704067200.0
}
```

## Синхронизация измерений

### Временная синхронизация

```python
# Настройка барьера синхронизации для N устройств
calibrator.setup_synchronous_measurement(device_count=3)

# В каждом потоке измерения
def measurement_thread():
    # Ожидание синхронного старта
    calibrator.wait_for_sync_start()
    
    # Выполнение измерения
    result = device.measure_rssi(...)
```

### Применение калибровочных поправок

После калибровки поправки автоматически применяются:

1. **Частотное смещение** - через параметр `freq_offset_hz` в конфигурации slave
2. **Амплитудная поправка** - через добавление `amplitude_offset_db` к результатам измерений

## Мониторинг калибровки

### Проверка актуальности

```python
# Проверка возраста калибровки (по умолчанию 24 часа)
validity = slave_manager.is_calibration_valid(max_age_hours=24.0)

# Результат: {"slave_1": True, "slave_2": False}
```

### Информация о калибровке

```python
# Получение подробной информации
info = slave_manager.get_calibration_info()

# Результат для каждого устройства:
# {
#   "serial": "1234567890abcdef",
#   "frequency_offset_hz": -2350.0,
#   "amplitude_offset_db": -1.2,
#   "last_calibration": "2024-01-01 12:00:00",
#   "age_hours": 2.5
# }
```

## Рекомендации

### Периодичность калибровки
- **Ежедневно** - для критически важных измерений
- **Еженедельно** - для обычных задач
- **При изменении температуры** - более 10°C от референсной

### Качество опорных сигналов
- Используйте стабильные сигналы (FM радио, сотовые базовые станции)
- Избегайте слабых или прерывистых сигналов
- Выбирайте сигналы в рабочем диапазоне частот

### Оптимизация точности
- Проводите калибровку при стабильной температуре
- Используйте несколько опорных частот
- Регулярно обновляйте калибровочные данные

## Устранение проблем

### Ошибка "Недостаточно данных для калибровки"
- Проверьте наличие опорных сигналов на целевых частотах
- Убедитесь в правильной работе антенн
- Увеличьте время измерения (`duration_sec`)

### Большие частотные смещения (>10 кГц)
- Проверьте температурную стабильность
- Рассмотрите замену кварцевого генератора на TCXO
- Проведите повторную калибровку

### Нестабильные результаты
- Уменьшите электромагнитные помехи
- Используйте экранирование
- Увеличьте количество опорных измерений

## Расширение функциональности

### Добавление новых опорных частот

```python
custom_targets = [
    CalibrationTarget("Custom_Signal", 1500e6, -40.0, 100e3, 2.0)
]

calibrator.reference_targets.extend(custom_targets)
```

### Температурная компенсация

```python
# В будущих версиях планируется автоматическая
# температурная компенсация на основе датчиков
```