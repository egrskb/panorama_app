#!/bin/bash
# Скрипт компиляции libhackrf_qsa.so для Linux

echo "Компиляция libhackrf_qsa.so..."

# Проверяем наличие зависимостей
if ! pkg-config --exists libhackrf; then
    echo "Ошибка: libhackrf не найдена"
    echo "Установите: sudo apt install libhackrf-dev"
    exit 1
fi

if ! pkg-config --exists fftw3f; then
    echo "Ошибка: fftw3 не найдена"
    echo "Установите: sudo apt install libfftw3-dev"
    exit 1
fi

# Компилируем
gcc -shared -fPIC -O3 \
    $(pkg-config --cflags libhackrf) \
    $(pkg-config --cflags fftw3f) \
    -o libhackrf_qsa.so \
    panorama/drivers/hackrf_lib/hq_sweep.c \
    $(pkg-config --libs libhackrf) \
    $(pkg-config --libs fftw3f) \
    -lm -pthread

if [ $? -eq 0 ]; then
    echo "✓ Успешно скомпилировано: libhackrf_qsa.so"
    
    # Копируем в нужные места
    cp libhackrf_qsa.so panorama/drivers/hackrf_lib/
    echo "✓ Скопировано в panorama/drivers/hackrf_lib/"
else
    echo "✗ Ошибка компиляции"
    exit 1
fi