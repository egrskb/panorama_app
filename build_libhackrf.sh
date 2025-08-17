#!/bin/bash

# Скрипт для сборки библиотеки libhackrf_multi
# Версия: 2.0 - Исправленная

set -e  # Остановиться при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции для вывода
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверяем, что мы в корне проекта
if [ ! -f "panorama/drivers/hackrf_lib/Makefile" ]; then
    print_error "Makefile не найден в panorama/drivers/hackrf_lib/"
    print_error "Убедитесь, что скрипт запущен из корня проекта"
    exit 1
fi

# Переходим в папку с библиотекой
cd panorama/drivers/hackrf_lib

print_status "Переходим в папку: $(pwd)"

# Проверяем зависимости
print_status "Проверяем зависимости..."

# Проверяем gcc
if ! command -v gcc &> /dev/null; then
    print_error "gcc не найден. Установите build-essential:"
    print_error "  sudo apt install build-essential"
    exit 1
fi

# Проверяем libhackrf
if ! pkg-config --exists libhackrf; then
    print_error "libhackrf не найден. Установите:"
    print_error "  sudo apt install hackrf libhackrf-dev"
    exit 1
fi

# Проверяем fftw3
if ! pkg-config --exists fftw3f; then
    print_error "fftw3f не найден. Установите:"
    print_error "  sudo apt install libfftw3-dev"
    exit 1
fi

print_success "Все зависимости найдены"

# Проверяем, нужно ли пересобирать
LIB_PATH="libhackrf_multi.so"
NEED_REBUILD=false

if [ ! -f "$LIB_PATH" ]; then
    print_status "Библиотека не найдена, собираем..."
    NEED_REBUILD=true
else
    # Проверяем, новее ли исходники чем библиотека
    SOURCE_FILES=$(find . -name "*.c" -o -name "*.h" | grep -v __pycache__)
    LIB_TIME=$(stat -c %Y "$LIB_PATH" 2>/dev/null || echo 0)
    
    for source in $SOURCE_FILES; do
        if [ -f "$source" ] && [ $(stat -c %Y "$source" 2>/dev/null || echo 0) -gt $LIB_TIME ]; then
            print_status "Исходник $source новее библиотеки, пересобираем..."
            NEED_REBUILD=true
            break
        fi
    done
fi

if [ "$NEED_REBUILD" = true ]; then
    print_status "Очищаем предыдущую сборку..."
    make clean
    
    print_status "Собираем библиотеку..."
    if make; then
        print_success "Библиотека успешно собрана: $LIB_PATH"
        
        # Показываем информацию о библиотеке
        if [ -f "$LIB_PATH" ]; then
            LIB_SIZE=$(du -h "$LIB_PATH" | cut -f1)
            print_status "Размер библиотеки: $LIB_SIZE"
            
            # Проверяем основные символы
            print_status "Проверяем экспортированные функции..."
            
            # Список критичных функций
            REQUIRED_SYMBOLS=(
                "hq_open_all"
                "hq_close_all"
                "hq_start"
                "hq_stop"
                "hq_get_master_spectrum"
                "hq_get_watchlist_snapshot"
                "hq_config_set_rates"
                "hq_config_set_gains"
                "hq_set_detector_params"
            )
            
            MISSING_SYMBOLS=()
            for symbol in "${REQUIRED_SYMBOLS[@]}"; do
                if nm -D "$LIB_PATH" | grep -q " T $symbol"; then
                    print_success "✓ $symbol найден"
                else
                    print_warning "⚠ $symbol не найден"
                    MISSING_SYMBOLS+=($symbol)
                fi
            done
            
            if [ ${#MISSING_SYMBOLS[@]} -gt 0 ]; then
                print_warning "Некоторые символы отсутствуют, возможны проблемы"
            fi
        fi
    else
        print_error "Ошибка при сборке библиотеки"
        exit 1
    fi
else
    print_success "Библиотека уже актуальна, пересборка не требуется"
fi

# Тестируем загрузку библиотеки в Python
print_status "Тестируем загрузку библиотеки в Python..."

# Создаем временный тестовый скрипт
cat > test_lib.py << 'EOF'
#!/usr/bin/env python3
import sys
try:
    from cffi import FFI
    ffi = FFI()
    
    # Определяем минимальный интерфейс
    ffi.cdef("""
        int hq_open_all(int num_expected);
        void hq_close_all(void);
        int hq_start(void);
        void hq_stop(void);
    """)
    
    lib = ffi.dlopen("./libhackrf_multi.so")
    print("✓ Библиотека успешно загружена в Python")
    
    # Проверяем, что функции доступны
    assert hasattr(lib, 'hq_open_all'), "hq_open_all not found"
    assert hasattr(lib, 'hq_start'), "hq_start not found"
    print("✓ Основные функции доступны")
    
    sys.exit(0)
except Exception as e:
    print(f"✗ Ошибка загрузки: {e}")
    sys.exit(1)
EOF

if python3 test_lib.py; then
    print_success "Тест Python прошел успешно"
else
    print_warning "Тест Python не прошел, но библиотека собрана"
fi

# Удаляем временный файл
rm -f test_lib.py

print_status "Возвращаемся в корень проекта..."
cd - > /dev/null

print_success "Сборка завершена!"
echo ""
print_status "Библиотека находится в: panorama/drivers/hackrf_lib/libhackrf_multi.so"
print_status "Для запуска программы используйте: ./run.sh или python3 -m panorama.main"
echo ""
print_status "Для работы multi-SDR режима:"
print_status "  1. Подключите 3 HackRF устройства"
print_status "  2. Запустите программу"
print_status "  3. Источник → libhackrf_multi"
print_status "  4. Источник → Настройка SDR устройств"
print_status "  5. Источник → Multi-SDR режим"