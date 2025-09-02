#!/bin/bash

# Универсальный скрипт запуска Panorama App для macOS
# Объединяет: установку шрифтов, сборку библиотеки, запуск приложения

set -e  # Остановка при ошибке

echo "=== Panorama App для macOS ==="
echo

# Определяем путь к conda
CONDA_PATH="$HOME/radioconda"
if [ ! -d "$CONDA_PATH" ]; then
    echo "❌ Conda не найдена в $CONDA_PATH"
    echo "Установите conda или укажите правильный путь"
    exit 1
fi

# Функция для проверки и установки Homebrew пакетов
install_homebrew_deps() {
    echo "📦 Проверка и установка Homebrew пакетов..."
    
    # Список необходимых пакетов
    PACKAGES=("soapysdr" "soapyhackrf" "hackrf" "swig" "cmake" "pkg-config" "libusb" "fftw")
    
    for package in "${PACKAGES[@]}"; do
        if ! brew list | grep -q "^${package}$"; then
            echo "Устанавливаем $package..."
            brew install "$package"
        else
            echo "✓ $package уже установлен"
        fi
    done
    echo
}

# Функция для установки emoji шрифтов
install_emoji_fonts() {
    echo "🎨 Установка emoji шрифтов..."
    
    if [ -f "install_emoji_fonts_universal.sh" ]; then
        chmod +x install_emoji_fonts_universal.sh
        ./install_emoji_fonts_universal.sh
    else
        echo "⚠️  Скрипт install_emoji_fonts_universal.sh не найден"
    fi
    echo
}

# Функция для сборки библиотеки hackrf_master
build_hackrf_master() {
    echo "🔧 Сборка библиотеки hackrf_master..."
    
    cd panorama/drivers/hackrf_master
    
    # Используем macOS Makefile
    if [ -f "Makefile.macos" ]; then
        make -f Makefile.macos clean
        make -f Makefile.macos all
        make -f Makefile.macos install
        echo "✓ Библиотека hackrf_master собрана и установлена"
    else
        echo "❌ Makefile.macos не найден"
        exit 1
    fi
    
    cd ../..
    echo
}

# Функция для проверки conda окружения
check_conda_env() {
    echo "🐍 Проверка conda окружения..."
    
    # Активируем conda
    source "$CONDA_PATH/bin/activate" panorama_env
    
    # Проверяем Python
    if ! python --version > /dev/null 2>&1; then
        echo "❌ Python не найден в conda окружении"
        exit 1
    fi
    
    # Проверяем SoapySDR в активированном окружении
    if ! source "$CONDA_PATH/bin/activate" panorama_env && python -c "import SoapySDR" > /dev/null 2>&1; then
        echo "❌ SoapySDR не установлен"
        echo "Выполните: conda activate panorama_env && conda install -c conda-forge soapysdr=0.8.1"
        exit 1
    fi
    
    echo "✓ Conda окружение готово"
    echo
}

# Функция для запуска приложения
run_panorama() {
    echo "🚀 Запуск Panorama App..."
    
    # Активируем conda окружение
    source "$CONDA_PATH/bin/activate" panorama_env
    
    # Очищаем переменные окружения
    unset DYLD_LIBRARY_PATH
    unset SOAPY_SDR_PLUGIN_PATH
    
    # Запускаем приложение
    python run_rssi_panorama.py
}

# Основная логика
main() {
    # Проверяем аргументы
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        echo "Использование: $0 [опции]"
        echo
        echo "Опции:"
        echo "  --help, -h     Показать эту справку"
        echo "  --install      Установить зависимости и собрать библиотеку"
        echo "  --fonts        Только установить emoji шрифты"
        echo "  --build        Только собрать библиотеку hackrf_master"
        echo "  --run          Только запустить приложение"
        echo "  (без опций)    Полная установка и запуск"
        echo
        exit 0
    fi
    
    # Проверяем операционную систему
    if [[ "$OSTYPE" != "darwin"* ]]; then
        echo "❌ Этот скрипт предназначен только для macOS"
        echo "Для Linux используйте: ./run_linux.sh"
        exit 1
    fi
    
    case "$1" in
        "--install")
            install_homebrew_deps
            install_emoji_fonts
            build_hackrf_master
            check_conda_env
            echo "✅ Установка завершена!"
            ;;
        "--fonts")
            install_emoji_fonts
            ;;
        "--build")
            build_hackrf_master
            ;;
        "--run")
            check_conda_env
            run_panorama
            ;;
        "")
            # Полная установка и запуск
            install_homebrew_deps
            install_emoji_fonts
            build_hackrf_master
            check_conda_env
            run_panorama
            ;;
        *)
            echo "❌ Неизвестная опция: $1"
            echo "Используйте --help для справки"
            exit 1
            ;;
    esac
}

# Запускаем основную функцию
main "$@"
