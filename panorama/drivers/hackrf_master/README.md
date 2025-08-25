# HackRF Master Driver для Panorama

## Обзор

`hackrf_master` - это C-библиотека и Python-обвязка для работы с HackRF One в режиме панорамного свипа. Библиотека обеспечивает:

- **Многосекционный режим**: 2 или 4 сегмента за проход для полного покрытия спектра
- **Нормализация мощности**: корректный перевод dBFS в dBm с учетом потерь окна и ENBW
- **Калибровка**: загрузка и применение калибровочных профилей
- **Высокое качество**: интерполяция для устранения ступенчатости

## Архитектура

### Многосекционный режим

Вместо передачи только 1/4 окна FFT, библиотека теперь передает 2 или 4 сегмента:

#### 4-сегментный режим (по умолчанию)
- **Сегмент A**: `[f-OFFSET, f-OFFSET+Fs/4]`
- **Сегмент C**: `[f+OFFSET+Fs/4, f+OFFSET+Fs/2]`
- **Сегмент B**: `[f+OFFSET+Fs/2, f+OFFSET+3Fs/4]`
- **Сегмент D**: `[f+OFFSET+3Fs/4, f+OFFSET+Fs]`

Где:
- `f` - центральная частота свипа
- `OFFSET` = 7.5 МГц (смещение LO)
- `Fs` = 20 МГц (частота дискретизации)

### Нормализация мощности

Библиотека применяет следующие поправки:

1. **Нормализация FFT**: `1/N` для каждого комплексного отсчета
2. **Потери окна**: -1.76 dB для окна Хэмминга
3. **ENBW коррекция**: +1.85 dB для эквивалентной ширины полосы
4. **Калибровочная поправка**: из загруженного CSV профиля

Итоговая формула:
```
power_dbm = power_dbfs + window_loss_db + enbw_corr_db + calibration_offset_db
```

## API

### C API

#### Инициализация
```c
int hq_open(const char* serial_suffix);
int hq_configure(double f_start_mhz, double f_stop_mhz, double bin_hz,
                 int lna_db, int vga_db, int amp_on);
```

#### Многосекционный режим
```c
int hq_start_multi_segment(hq_multi_segment_cb cb, void* user);
int hq_set_segment_mode(int mode);  // 2 или 4
int hq_get_segment_mode(void);
```

#### Калибровка
```c
int hq_load_calibration(const char* csv_path);
int hq_enable_calibration(int enable);
int hq_get_calibration_status(void);
```

#### Управление
```c
int hq_stop(void);
void hq_close(void);
const char* hq_last_error(void);
```

### Python API

#### Основные методы
```python
backend = HackRFQSABackend(serial_suffix="c483")
backend.start(config)
backend.stop()
```

#### Калибровка
```python
# Загрузка калибровочного профиля
backend.load_calibration("calibration.csv")

# Включение/выключение калибровки
backend.enable_calibration(True)

# Проверка статуса
status = backend.get_calibration_status()
```

#### Настройка режима
```python
# Установка 4-сегментного режима (по умолчанию)
backend.set_segment_mode(4)

# Получение текущего режима
mode = backend.get_segment_mode()
```

## Калибровка

### Формат CSV файла

Файл должен содержать заголовок и столбцы в указанном порядке:

```csv
freq_mhz,lna,vga,amp,offset_db
100,24,20,1,-4.2
500,24,20,1,-5.0
1000,24,20,1,-5.2
...
```

Где:
- `freq_mhz` - центральная частота в МГц
- `lna` - уровень предусилителя (0..40 дБ)
- `vga` - уровень усилителя (0..62 дБ)
- `amp` - бит включения усилителя (1 = включен, 0 = выключен)
- `offset_db` - поправка для перевода dBFS в dBm

### Процедура калибровки

#### Лабораторные условия

1. Подключите выход генератора сигналов напрямую к входу HackRF
2. Задайте частоту (например, 100 МГц, 500 МГц, 1 ГГц)
3. Установите уровень мощности (измеренный калиброванным анализатором)
4. Запишите полученный уровень в приложении
5. Повторите для комбинаций LNA/VGA/AMP
6. Вычислите разницу между эталонным и измеренным уровнем
7. Запишите данные в CSV

#### Полевые условия

1. Используйте генератор или известный источник (радиостанция, навигационный передатчик)
2. Сравните с эталонным анализатором
3. Запишите поправки в CSV

### Пример калибровочного файла

См. `calibration_example.csv` в корне проекта.

## Сборка

### Требования

- libhackrf-dev
- libfftw3-dev
- gcc/clang
- make

### Компиляция

```bash
cd panorama/drivers/hackrf_master
make
```

### Установка

```bash
sudo make install
```

## Использование

### Базовый пример

```python
from panorama.drivers.hrf_backend import HackRFQSABackend
from panorama.drivers.base import SweepConfig

# Создание конфигурации
config = SweepConfig(
    freq_start_hz=50e6,      # 50 МГц
    freq_end_hz=6000e6,      # 6 ГГц
    bin_hz=800e3,            # 800 кГц
    lna_db=24,
    vga_db=20,
    amp_on=False
)

# Создание и запуск бэкенда
backend = HackRFQSABackend(serial_suffix="c483")

# Загрузка калибровки
backend.load_calibration("calibration.csv")
backend.enable_calibration(True)

# Установка 4-сегментного режима
backend.set_segment_mode(4)

# Запуск
backend.start(config)
```

### Обработка данных

```python
def on_full_sweep(freqs, power):
    print(f"Получен спектр: {len(freqs)} точек")
    print(f"Диапазон: {freqs[0]/1e6:.1f} - {freqs[-1]/1e6:.1f} МГц")
    print(f"Мощность: {power.min():.1f} - {power.max():.1f} dBm")

backend.fullSweepReady.connect(on_full_sweep)
```

## Производительность

### Покрытие спектра

- **4 сегмента**: покрытие 20 МГц за проход
- **Порог покрытия**: настраивается через `coverage_threshold`

### Время прохода

- Зависит от ширины бина и диапазона частот
- При bin = 800 кГц: ~0.3-0.4 секунды на проход
- При bin = 200 кГц: ~1.2-1.6 секунды на проход

## Отладка

### Логирование

Библиотека выводит подробные логи:

```
[DEBUG] Loaded config: {...}
[SweepAssembler] Configured: 50.0-6000.0 MHz, 7438 bins
[HackRF Worker] Segment 0: 2048 bins, freq_range=[42.5, 47.5] MHz
[HackRF Worker] Segment 1: 2048 bins, freq_range=[67.5, 72.5] MHz
```

### Ошибки

Используйте `hq_last_error()` для получения описания ошибок:

```c
if (hq_open("c483") != 0) {
    printf("Ошибка: %s\n", hq_last_error());
}
```

## Совместимость

### Обратная совместимость

Старый API `hq_start()` продолжает работать, но использует только 1/4 окна FFT.

### Миграция

Для перехода на новый API:

1. Замените `hq_start()` на `hq_start_multi_segment()`
2. Обновите колбэк для работы с `hq_segment_data_t`
3. Настройте режим сегментов через `hq_set_segment_mode()`

## Лицензия

См. LICENSE файл в корне проекта.
