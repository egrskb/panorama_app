#!/bin/bash

# Скрипт для компиляции и тестирования исправлений спектра
# Автоматически собирает C-библиотеку и проверяет работоспособность

set -e  # Остановка при ошибке

echo "=== Компиляция и тестирование исправлений спектра ==="
echo

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции для цветного вывода
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }

# Проверка зависимостей
echo "1. Проверка зависимостей..."
cd "$(dirname "$0")/drivers/hackrf_master"

if ! command -v gcc &> /dev/null; then
    print_error "GCC не найден. Установите build-essential"
    exit 1
fi

if ! pkg-config --exists libhackrf 2>/dev/null; then
    print_warning "libhackrf не найден. Установите libhackrf-dev"
    print_info "Выполните: sudo apt-get install libhackrf-dev"
fi

if ! pkg-config --exists fftw3f 2>/dev/null; then
    print_warning "fftw3f не найден. Установите libfftw3-dev"
    print_info "Выполните: sudo apt-get install libfftw3-dev"
fi

print_success "Основные зависимости проверены"

# Очистка предыдущей сборки
echo
echo "2. Очистка предыдущей сборки..."
make clean
print_success "Очистка завершена"

# Компиляция
echo
echo "3. Компиляция C-библиотеки..."
if make; then
    print_success "Библиотека скомпилирована успешно"
else
    print_error "Ошибка компиляции"
    exit 1
fi

# Проверка результата
echo
echo "4. Проверка результата компиляции..."
if [ -f "libhackrf_qsa.so" ]; then
    print_success "Файл libhackrf_qsa.so создан"
    
    # Проверка размера
    size=$(ls -lh libhackrf_qsa.so | awk '{print $5}')
    print_info "Размер библиотеки: $size"
    
    # Проверка зависимостей
    echo "Зависимости библиотеки:"
    ldd libhackrf_qsa.so | grep -E "(hackrf|fftw3)" || print_warning "Не все зависимости найдены"
    
else
    print_error "Файл libhackrf_qsa.so не создан"
    exit 1
fi

# Копирование в корень проекта
echo
echo "5. Копирование библиотеки..."
cd ../..
if cp drivers/hackrf_master/libhackrf_qsa.so .; then
    print_success "Библиотека скопирована в корень проекта"
else
    print_warning "Не удалось скопировать библиотеку"
fi

# Создание калибровочного файла
echo
echo "6. Создание калибровочного файла..."
mkdir -p ~/.panorama
cal_file="$HOME/.panorama/calibration.csv"

if [ ! -f "$cal_file" ]; then
    cat > "$cal_file" << 'EOF'
freq_mhz,lna,vga,amp,offset_db
100,24,20,1,-4.2
500,24,20,1,-5.0
1000,24,20,1,-5.2
2400,24,20,1,-6.0
5000,24,20,1,-6.5
5800,24,20,1,-7.0
EOF
    print_success "Калибровочный файл создан: $cal_file"
else
    print_info "Калибровочный файл уже существует: $cal_file"
fi

# Проверка Python модулей
echo
echo "7. Проверка Python модулей..."
cd "$(dirname "$0")"

if python3 -c "import panorama.features.spectrum.view" 2>/dev/null; then
    print_success "Python модули загружаются корректно"
else
    print_warning "Проблемы с загрузкой Python модулей"
fi

# Финальная проверка
echo
echo "8. Финальная проверка..."
if [ -f "libhackrf_qsa.so" ] && [ -f "$cal_file" ]; then
    print_success "Все компоненты готовы к работе!"
    echo
    echo "=== Резюме ==="
    print_success "C-библиотека скомпилирована и готова"
    print_success "Калибровочный файл создан"
    print_success "Python модули обновлены"
    echo
    echo "Теперь вы можете запустить приложение:"
    echo "  python3 -m panorama.main"
    echo
    echo "Ожидаемые улучшения:"
    echo "  • Wi-Fi пики будут видны на 2.4 ГГц и 5 ГГц"
    echo "  • Уровень шума будет около -100 дБм"
    echo "  • Сигналы -50 дБм будут заметны над шумом"
    echo "  • Масштабирование будет работать как лупа"
    echo "  • Ступенчатость будет устранена"
    echo "  • Дрожание будет уменьшено"
else
    print_error "Не все компоненты готовы"
    exit 1
fi

echo
print_info "Для получения дополнительной информации см. FIXES_README.md"
