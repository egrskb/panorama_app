#!/bin/bash
# Скрипт сборки HackRF Master C библиотеки для Python CFFI

set -e  # Останавливаемся при ошибке

echo "🔧 Сборка HackRF Master C библиотеки"
echo "======================================"

# Проверяем наличие необходимых инструментов
check_dependencies() {
    echo "Проверка зависимостей..."
    
    # Проверяем gcc
    if ! command -v gcc &> /dev/null; then
        echo "❌ gcc не найден. Установите build-essential:"
        echo "   sudo apt-get install build-essential"
        exit 1
    fi
    echo "✓ gcc найден: $(gcc --version | head -n1)"
    
    # Проверяем make
    if ! command -v make &> /dev/null; then
        echo "❌ make не найден. Установите build-essential:"
        echo "   sudo apt-get install build-essential"
        exit 1
    fi
    echo "✓ make найден: $(make --version | head -n1)"
    
    # Проверяем pkg-config
    if ! command -v pkg-config &> /dev/null; then
        echo "❌ pkg-config не найден. Установите:"
        echo "   sudo apt-get install pkg-config"
        exit 1
    fi
    echo "✓ pkg-config найден: $(pkg-config --version)"
    
    # Проверяем libhackrf
    if ! pkg-config --exists libhackrf; then
        echo "❌ libhackrf не найден. Установите:"
        echo "   sudo apt-get install libhackrf-dev"
        echo "   или соберите из исходников: https://github.com/mossmann/hackrf"
        exit 1
    fi
    echo "✓ libhackrf найден: $(pkg-config --modversion libhackrf)"
    
    # Проверяем Python и CFFI
    if ! python3 -c "import cffi" 2>/dev/null; then
        echo "❌ Python CFFI не найден. Установите:"
        echo "   pip3 install cffi"
        exit 1
    fi
    echo "✓ Python CFFI найден"
}

# Сборка C библиотеки
build_c_library() {
    echo ""
    echo "📦 Сборка C библиотеки..."
    
    cd panorama/drivers/hackrf_master
    
    # Очищаем предыдущую сборку
    echo "  Очистка предыдущей сборки..."
    make clean 2>/dev/null || true
    
    # Собираем библиотеку
    echo "  Компиляция..."
    make all
    
    # Проверяем результат
    if [ -f "build/libhackrf_master.so" ]; then
        echo "✓ C библиотека успешно собрана"
        ls -la build/
    else
        echo "❌ Ошибка сборки C библиотеки"
        exit 1
    fi
    
    cd ../../..
}

# Создание Python CFFI интерфейса
build_python_interface() {
    echo ""
    echo "🐍 Создание Python CFFI интерфейса..."
    
    cd panorama/drivers/hackrf_master
    
    # Запускаем CFFI build script
    if python3 cffi_build.py; then
        echo "✓ Python CFFI интерфейс создан"
    else
        echo "❌ Ошибка создания Python интерфейса"
        exit 1
    fi
    
    cd ../../..
}

# Тестирование
test_build() {
    echo ""
    echo "🧪 Тестирование сборки..."
    
    # Проверяем наличие всех файлов
    required_files=(
        "panorama/drivers/hackrf_master/build/libhackrf_master.so"
        "panorama/drivers/hackrf_master/hackrf_master_wrapper.py"
    )
    
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            echo "✓ $file найден"
        else
            echo "❌ $file не найден"
            exit 1
        fi
    done
    
    # Тестируем Python импорт
    echo "  Тестирование Python импорта..."
    if python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from panorama.drivers.hackrf_master.hackrf_master_wrapper import HackRFMaster
    print('✓ Python импорт успешен')
except Exception as e:
    print(f'❌ Ошибка импорта: {e}')
    exit(1)
"; then
        echo "✓ Python модуль работает корректно"
    else
        echo "❌ Ошибка тестирования Python модуля"
        exit 1
    fi
}

# Установка (опционально)
install_library() {
    echo ""
    echo "📥 Установка библиотеки (опционально)..."
    
    read -p "Установить библиотеку в систему? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd panorama/drivers/hackrf_master
        sudo make install
        cd ../../..
        echo "✓ Библиотека установлена в систему"
    else
        echo "  Пропущено"
    fi
}

# Основная функция
main() {
    echo "Начинаем сборку HackRF Master C библиотеки..."
    echo ""
    
    # Проверяем зависимости
    check_dependencies
    
    # Собираем C библиотеку
    build_c_library
    
    # Создаем Python интерфейс
    build_python_interface
    
    # Тестируем сборку
    test_build
    
    # Предлагаем установку
    install_library
    
    echo ""
    echo "🎉 Сборка завершена успешно!"
    echo ""
    echo "Теперь вы можете использовать HackRF Master в Python:"
    echo "  from panorama.drivers.hackrf_master.hackrf_master_wrapper import HackRFMaster"
    echo ""
    echo "Файлы:"
    echo "  - C библиотека: panorama/drivers/hackrf_master/build/libhackrf_master.so"
    echo "  - Python wrapper: panorama/drivers/hackrf_master/hackrf_master_wrapper.py"
    echo ""
    echo "Для запуска приложения:"
    echo "  python3 run_rssi_panorama.py"
}

# Обработка ошибок
trap 'echo "❌ Ошибка в строке $LINENO"; exit 1' ERR

# Запуск
main "$@"
