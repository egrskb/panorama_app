# Сборка HackRF Master C библиотеки

## Описание

HackRF Master C библиотека предоставляет высокопроизводительный интерфейс для работы с HackRF в режиме sweep через Python CFFI. Библиотека реализует:

- Быстрое сканирование диапазона частот
- Детекцию пиков сигналов в реальном времени
- Callback систему для интеграции с Python
- Оптимизированную обработку данных

## Требования

### Системные зависимости

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential pkg-config libhackrf-dev

# CentOS/RHEL/Fedora
sudo yum groupinstall "Development Tools"
sudo yum install pkgconfig libhackrf-devel
# или для Fedora
sudo dnf groupinstall "Development Tools"
sudo dnf install pkgconfig libhackrf-devel
```

### Python зависимости

```bash
pip install cffi numpy scipy PyQt5
```

### HackRF устройство

- HackRF One или совместимое устройство
- USB 3.0 подключение (рекомендуется)
- Драйверы libhackrf

## Сборка

### Автоматическая сборка

```bash
# Делаем скрипт исполняемым
chmod +x build_hackrf_master.sh

# Запускаем сборку
./build_hackrf_master.sh
```

### Ручная сборка

```bash
# 1. Сборка C библиотеки
cd panorama/drivers/hackrf_master
make all

# 2. Создание Python CFFI интерфейса
python3 cffi_build.py

# 3. Проверка сборки
make test
```

### Опции сборки

```bash
# Отладочная версия
make debug

# Очистка
make clean

# Установка в систему
sudo make install

# Удаление из системы
sudo make uninstall

# Проверка зависимостей
make deps

# Информация о системе
make info

# Справка
make help
```

## Структура проекта

```
panorama/drivers/hackrf_master/
├── hackrf_master.h          # Заголовочный файл
├── hackrf_master.c          # C реализация
├── Makefile                 # Makefile для сборки
├── cffi_build.py           # CFFI build script
├── hackrf_master_wrapper.py # Python wrapper (автогенерируется)
└── build/                   # Директория сборки
    └── libhackrf_master.so # Скомпилированная библиотека
```

## Использование

### В Python коде

```python
from panorama.drivers.hackrf_master.hackrf_master_wrapper import HackRFMaster

# Создание экземпляра
master = HackRFMaster()

# Настройка callbacks
def on_sweep_tile(tile_data):
    print(f"Sweep tile: {tile_data['f_start']/1e6:.1f} MHz")

def on_peak_detected(peak_data):
    print(f"Peak: {peak_data['f_peak']/1e6:.1f} MHz, SNR: {peak_data['snr_db']:.1f} dB")

def on_error(error_msg):
    print(f"Error: {error_msg}")

master.set_sweep_callback(on_sweep_tile)
master.set_peak_callback(on_peak_detected)
master.set_error_callback(on_error)

# Запуск sweep
master.start_sweep(
    start_hz=24e6,      # 24 МГц
    stop_hz=6e9,        # 6 ГГц
    bin_hz=200e3,       # 200 кГц
    dwell_ms=100,       # 100 мс
    min_snr_db=10.0,    # Минимальный SNR
    min_peak_distance_bins=2
)

# Проверка состояния
while master.is_running():
    time.sleep(0.1)

# Остановка
master.stop_sweep()

# Получение статистики
stats = master.get_stats()
print(f"Sweep count: {stats['sweep_count']}")

# Очистка ресурсов
master.cleanup()
```

### Интеграция с Master контроллером

```python
from panorama.features.master_sweep.master import MasterSweepController

# Создание контроллера
controller = MasterSweepController(logger)

# Автоматическое подключение к C библиотеке
if controller.sweep_source:
    print("C библиотека доступна")
    
    # Запуск sweep
    controller.start_sweep(
        start_hz=24e6,
        stop_hz=6e9,
        bin_hz=200e3,
        dwell_ms=100
    )
else:
    print("C библиотека недоступна, используется Python fallback")
```

## API Reference

### Основные функции

- `hackrf_master_init()` - Инициализация библиотеки
- `hackrf_master_cleanup()` - Очистка ресурсов
- `hackrf_master_start_sweep(config)` - Запуск sweep
- `hackrf_master_stop_sweep()` - Остановка sweep
- `hackrf_master_is_running()` - Проверка состояния

### Callback функции

- `hackrf_master_set_sweep_callback(callback)` - Callback для sweep tiles
- `hackrf_master_set_peak_callback(callback)` - Callback для обнаруженных пиков
- `hackrf_master_set_error_callback(callback)` - Callback для ошибок

### Управление детекцией

- `hackrf_master_set_peak_detection_params(min_snr_db, min_peak_distance_bins)`
- `hackrf_master_get_peak_count()` - Количество обнаруженных пиков
- `hackrf_master_get_peaks(peaks, max_count)` - Получение пиков

### Статистика

- `hackrf_master_get_stats(stats)` - Получение статистики
- `hackrf_master_reset_stats()` - Сброс статистики

### Утилиты

- `hackrf_master_get_frequency_range_min()` - Минимальная частота
- `hackrf_master_get_frequency_range_max()` - Максимальная частота
- `hackrf_master_get_max_bin_count()` - Максимальное количество бинов
- `hackrf_master_get_max_bandwidth()` - Максимальная полоса

## Производительность

### Ограничения

- **Частотный диапазон**: 24 МГц - 6 ГГц
- **Максимальная полоса**: 20 МГц
- **Максимальное количество бинов**: 8192
- **Максимальное количество пиков**: 100

### Рекомендации

- Используйте USB 3.0 для максимальной производительности
- Оптимизируйте параметры sweep под ваши задачи
- Мониторьте статистику для выявления узких мест

## Устранение неполадок

### Частые проблемы

#### 1. Ошибка компиляции
```bash
# Проверьте зависимости
make deps

# Очистите и пересоберите
make clean
make all
```

#### 2. Ошибка линковки
```bash
# Проверьте наличие libhackrf
pkg-config --exists libhackrf

# Установите если отсутствует
sudo apt-get install libhackrf-dev
```

#### 3. Ошибка импорта Python
```bash
# Проверьте сборку
make test

# Пересоздайте CFFI интерфейс
python3 cffi_build.py
```

#### 4. HackRF не найден
```bash
# Проверьте подключение
lsusb | grep HackRF

# Проверьте права доступа
sudo usermod -a -G plugdev $USER
# Перезагрузитесь
```

### Логирование

Библиотека выводит подробные логи через callback функции. Настройте логирование в вашем приложении:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def on_error(error_msg):
    logger.error(f"HackRF error: {error_msg}")

master.set_error_callback(on_error)
```

## Разработка

### Добавление новых функций

1. Добавьте объявление в `hackrf_master.h`
2. Реализуйте функцию в `hackrf_master.c`
3. Обновите Python wrapper в `cffi_build.py`
4. Пересоберите библиотеку

### Отладка

```bash
# Сборка отладочной версии
make debug

# Запуск с отладчиком
gdb --args python3 your_script.py
```

### Тестирование

```bash
# Запуск тестов
make test

# Проверка покрытия (если настроено)
make coverage
```

## Лицензия

© 2024 ПАНОРАМА Team. Все права защищены.

## Поддержка

Для получения поддержки и сообщения об ошибках создайте issue в репозитории проекта.
