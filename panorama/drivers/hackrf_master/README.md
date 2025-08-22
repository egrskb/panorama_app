# HackRF QSA Library

Библиотека для работы с HackRF через Python CFFI с улучшенным покрытием спектра.

## Новые возможности

- **4 квартальных сегмента** для полного покрытия частотного диапазона
- **Модульная индексация FFT** для устранения "дырок" в данных
- **Поддержка калибровки** из CSV файлов
- **Оптимизированный сборщик** для стабильной работы
- **Встроенные FFI определения** - без генерации кода

## Зависимости

```bash
# Системные зависимости
sudo apt-get install build-essential pkg-config

# HackRF
sudo apt-get install libhackrf-dev

# FFTW3 (для FFT операций)
sudo apt-get install libfftw3-dev

# Python зависимости
pip3 install cffi
```

## Сборка

### Автоматическая сборка

```bash
# Из корня проекта
./build_hackrf_master.sh
```

### Ручная сборка

```bash
cd panorama/drivers/hackrf_master

# Очистка
make clean

# Сборка
make all
```

**Примечание:** Python интерфейс уже встроен в `panorama/drivers/hrf_backend.py` и не требует генерации.

## Использование

```python
from panorama.drivers.hrf_backend import HackRFQSABackend

# Создание экземпляра
hackrf = HackRFQSABackend()

# Открытие устройства
if hackrf.open():
    print("Устройство открыто")
    
    # Конфигурация
    hackrf.configure(
        f_start_mhz=24.0,    # 24 МГц
        f_stop_mhz=6000.0,   # 6 ГГц
        bin_hz=200000.0,     # 200 кГц
        lna_db=24,           # LNA усиление
        vga_db=20,           # VGA усиление
        amp_enable=False     # Усилитель выключен
    )
    
    # Callback для получения данных
    def on_segment(freqs, data, count, bin_width, hz_low, hz_high):
        print(f"Сегмент: {len(freqs)} точек, диапазон {hz_low/1e6:.1f}-{hz_high/1e6:.1f} МГц")
    
    hackrf.set_segment_callback(on_segment)
    
    # Запуск sweep
    if hackrf.start():
        print("Sweep запущен")
        
        # Ожидание...
        import time
        time.sleep(10)
        
        # Остановка
        hackrf.stop()
    
    # Закрытие
    hackrf.close()
```

## Архитектура сегментов

Библиотека эмитит 4 квартальных сегмента для каждого sweep:

- **Сегмент A:** `[f, f+Fs/4]` - бины `1+(5/8*N)` до `1+(5/8*N)+q-1`
- **Сегмент C:** `[f+Fs/4, f+Fs/2]` - бины `(1+(7/8*N)+i) % N` 
- **Сегмент B:** `[f+Fs/2, f+3/4*Fs]` - бины `1+(1/8*N)` до `1+(1/8*N)+q-1`
- **Сегмент D:** `[f+3/4*Fs, f+Fs]` - бины `(1+(3/8*N)+i) % N`

Где:
- `f` - центральная частота sweep
- `Fs` - частота дискретизации (20 МГц)
- `N` - размер FFT
- `q` - размер сегмента (N/4)

## Калибровка

```python
# Загрузка калибровки из CSV
hackrf.load_calibration("calibration.csv")

# Включение калибровки
hackrf.enable_calibration(True)

# Проверка статуса
if hackrf.is_calibration_loaded():
    print("Калибровка загружена")
```

Формат CSV файла: `freq_mhz,lna,vga,amp,offset_db`

## Структура файлов

```
panorama/drivers/
├── hrf_backend.py              # Основной Python backend
└── hackrf_master/
    ├── hackrf_master.c         # C исходный код
    ├── hackrf_master.h         # C заголовочный файл
    ├── Makefile                # Сборка C библиотеки
    ├── build/                  # Скомпилированные файлы
    │   └── libhackrf_qsa.so   # C библиотека
    └── README.md               # Этот файл
```

## Устранение неполадок

### Ошибка "Could not load libhackrf_qsa.so"
- Убедитесь что библиотека собрана: `make all`
- Проверьте зависимости: `make deps`

### Ошибка FFTW3
- Установите: `sudo apt-get install libfftw3-dev`

### Ошибка libhackrf
- Установите: `sudo apt-get install libhackrf-dev`

## Тестирование

```bash
# Запуск тестов
make test

# Проверка зависимостей
make deps

# Информация о системе
make info
```

## Преимущества новой архитектуры

1. **Упрощенная сборка** - нет генерации Python кода
2. **Лучшая производительность** - FFI определения встроены
3. **Проще отладка** - весь код в одном месте
4. **Меньше файлов** - проще поддерживать
