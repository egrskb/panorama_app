#!/bin/bash

# Скрипт для сборки библиотеки libhackrf_multi
# Автор: AI Assistant
# Версия: 1.0

set -e  # Остановиться при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
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
    print_error "gcc не найден. Установите build-essential"
    exit 1
fi

# Проверяем libhackrf
if ! pkg-config --exists libhackrf; then
    print_error "libhackrf не найден. Установите hackrf development package"
    print_error "sudo apt install hackrf-dev"
    exit 1
fi

# Проверяем fftw3
if ! pkg-config --exists fftw3f; then
    print_error "fftw3f не найден. Установите libfftw3-dev"
    print_error "sudo apt install libfftw3-dev"
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
            
            # Проверяем символы
            print_status "Проверяем основные символы..."
            if nm -D "$LIB_PATH" | grep -q "hq_open_all"; then
                print_success "✓ hq_open_all найден"
            else
                print_warning "⚠ hq_open_all не найден"
            fi
            
            if nm -D "$LIB_PATH" | grep -q "hq_start"; then
                print_success "✓ hq_start найден"
            else
                print_warning "⚠ hq_start не найден"
            fi
        fi
    else
        print_error "Ошибка при сборке библиотеки"
        exit 1
    fi
else
    print_success "Библиотека уже актуальна, пересборка не требуется"
fi

# Проверяем, что библиотека загружается в Python
print_status "Тестируем загрузку библиотеки в Python..."

# Создаем временный тестовый скрипт
cat > test_lib.py << 'EOF'
#!/usr/bin/env python3
try:
    from cffi import FFI
    ffi = FFI()
    lib = ffi.dlopen("./libhackrf_multi.so")
    print("✓ Библиотека успешно загружена в Python")
    exit(0)
except Exception as e:
    print(f"✗ Ошибка загрузки: {e}")
    exit(1)
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
print_status "Библиотека находится в: panorama/drivers/hackrf_lib/libhackrf_multi.so"
print_status "Для использования в Python импортируйте: from panorama.drivers.hackrf_lib import HackRFLibSource"
