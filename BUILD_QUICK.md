# Быстрый старт: сборка libhackrf_multi

## 🚀 Один клик для сборки

```bash
./build_libhackrf.sh
```

## 📋 Что происходит автоматически

✅ **Проверка зависимостей** (gcc, libhackrf, fftw3)  
✅ **Умная пересборка** (только при изменении исходников)  
✅ **Сборка библиотеки** (make clean && make)  
✅ **Тестирование** (проверка символов + Python загрузка)  
✅ **Цветной вывод** с подробной информацией  

## 🔧 Если что-то пошло не так

### Установка зависимостей
```bash
sudo apt install build-essential hackrf-dev libfftw3-dev
```

### Ручная сборка
```bash
cd panorama/drivers/hackrf_lib
make clean && make
```

## 📁 Результат

Библиотека: `panorama/drivers/hackrf_lib/libhackrf_multi.so`

## 🐍 Использование в Python

```python
from panorama.drivers.hackrf_lib import HackRFLibSource
source = HackRFLibSource()  # Автоматически загрузит библиотеку
```

---
*Подробная документация: `README_build.md`*
