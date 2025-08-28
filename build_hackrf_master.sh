#!/bin/bash
# Скрипт для быстрой пересборки библиотеки HackRF Master с исправлениями

echo "=== Пересборка библиотеки HackRF Master ==="
echo "Убедитесь что установлены: libhackrf-dev, libfftw3-dev, build-essential"
echo ""

# Переходим в директорию с исходниками
cd panorama/drivers/hackrf_master || exit 1

# Очистка старых файлов
echo "Очистка старых файлов..."
rm -f *.o *.so

# Компиляция
echo "Компиляция hackrf_master.c..."
gcc -c -fPIC hackrf_master.c -o hackrf_master.o \
    -I/usr/include/hackrf \
    -I/usr/include \
    -Wall -O3 \
    -DDEFAULT_FFT_SIZE=4096 \
    -DMAX_FFT_SIZE=65536

if [ $? -ne 0 ]; then
    echo "Ошибка компиляции!"
    exit 1
fi

# Линковка
echo "Создание библиотеки libhackrf_master.so..."
gcc -shared -fPIC -o libhackrf_master.so hackrf_master.o \
    -lhackrf -lfftw3f -lm -pthread

if [ $? -ne 0 ]; then
    echo "Ошибка линковки!"
    exit 1
fi

# Проверка результата
if [ -f libhackrf_master.so ]; then
    echo ""
    echo "✓ Библиотека успешно создана: libhackrf_master.so"
    echo ""
    echo "Размер файла:"
    ls -lh libhackrf_master.so
    echo ""
    echo "Для установки в систему выполните:"
    echo "  sudo cp libhackrf_master.so /usr/local/lib/"
    echo "  sudo ldconfig"
else
    echo "✗ Ошибка: библиотека не создана"
    exit 1
fi