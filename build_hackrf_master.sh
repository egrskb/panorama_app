#!/bin/bash
# Скрипт сборки HackRF QSA C библиотеки для Python CFFI

set -euo pipefail

echo "🔧 Сборка HackRF QSA C библиотеки"
echo "=================================="

SRC_DIR="panorama/drivers/hackrf_master"
LIB_NAME="libhackrf_master.so"
SRC_FILES=("hackrf_master.c" "hackrf_master.h")

# === Проверка зависимостей ===
check_dependencies() {
    echo "Проверка зависимостей..."
    command -v gcc >/dev/null || { echo "❌ gcc не найден"; exit 1; }
    echo "✓ gcc найден: $(gcc --version | head -n1)"

    command -v make >/dev/null || { echo "❌ make не найден"; exit 1; }
    echo "✓ make найден: $(make --version | head -n1)"

    command -v pkg-config >/dev/null || { echo "❌ pkg-config не найден"; exit 1; }
    echo "✓ pkg-config найден: $(pkg-config --version)"

    pkg-config --exists libhackrf || { echo "❌ libhackrf-dev не найден"; exit 1; }
    echo "✓ libhackrf: $(pkg-config --modversion libhackrf)"

    pkg-config --exists fftw3f || { echo "❌ libfftw3-dev не найден"; exit 1; }
    echo "✓ fftw3f: $(pkg-config --modversion fftw3f)"

    python3 -c "import cffi" 2>/dev/null || { echo "❌ Python CFFI не найден (pip install cffi)"; exit 1; }
    echo "✓ Python CFFI найден"
}

# === Сборка C библиотеки ===
build_c_library() {
    echo ""
    echo "📦 Сборка C библиотеки..."

    cd "$SRC_DIR"

    for f in "${SRC_FILES[@]}"; do
        [ -f "$f" ] || { echo "❌ Нет файла $f"; exit 1; }
    done
    echo "✓ Исходники найдены"

    echo "  Очистка..."
    rm -f hackrf_master.o "$LIB_NAME"

    echo "  Компиляция с -fPIC..."
    gcc -O3 -Wall -Wextra -fPIC \
        -I. -I/usr/local/include -I/usr/local/include/libhackrf -I/usr/include/libusb-1.0 \
        -c hackrf_master.c -o hackrf_master.o

    echo "  Линковка..."
    gcc -shared -o "$LIB_NAME" hackrf_master.o \
        -L/usr/local/lib -lhackrf -lfftw3f -lm -lpthread

    echo "✓ Сборка завершена: $SRC_DIR/$LIB_NAME"
    ls -lh "$LIB_NAME"

    # Проверка экспорта символов
    echo "  Проверка экспорта символов..."
    if nm -D "$LIB_NAME" | grep -q " hq_open"; then
        echo "✓ Символ hq_open найден"
    else
        echo "❌ Символ hq_open не найден! Проверь hackrf_master.c"
        exit 1
    fi

    cd - >/dev/null
}

# === Тестирование ===
test_build() {
    echo ""
    echo "🧪 Тестирование..."

    [ -f "$SRC_DIR/$LIB_NAME" ] || { echo "❌ $LIB_NAME не найден"; exit 1; }

    if file "$SRC_DIR/$LIB_NAME" | grep -q "ELF"; then
        echo "✓ ELF библиотека корректна"
    else
        echo "❌ Библиотека повреждена"
        exit 1
    fi

    echo "  Тест Python..."
    if python3 -c "
import sys; sys.path.insert(0,'.')
from panorama.drivers.hrf_backend import HackRFQSABackend
print('✓ Python импорт успешен')
"; then
        echo "✓ Python backend работает"
    else
        echo "❌ Ошибка Python backend"
        exit 1
    fi
}

# === Установка в систему ===
install_library() {
    echo ""
    read -p "📥 Установить библиотеку в систему (/usr/local/lib)? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo cp -f "$SRC_DIR/$LIB_NAME" /usr/local/lib/
        sudo ldconfig
        echo "✓ Установлено в /usr/local/lib"
    else
        echo "Пропуск установки"
    fi
}

# === Основной запуск ===
main() {
    check_dependencies
    build_c_library
    test_build
    install_library

    echo ""
    echo "🎉 Готово! Используйте:"
    echo "  from panorama.drivers.hrf_backend import HackRFQSABackend"
    echo ""
}

trap 'echo "❌ Ошибка в строке $LINENO"; exit 1' ERR
main "$@"
