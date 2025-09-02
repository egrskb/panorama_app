#!/bin/bash

# Универсальный скрипт запуска Panorama App для Linux
# Объединяет: установку шрифтов, сборку библиотеки, запуск приложения

set -e  # Остановка при ошибке

echo "=== Panorama App для Linux ==="
echo

# Определяем путь к conda
CONDA_PATH="$HOME/radioconda"
if [ ! -d "$CONDA_PATH" ]; then
    echo "❌ Conda не найдена в $CONDA_PATH"
    echo "Установите conda или укажите правильный путь"
    exit 1
fi

# Функция для определения дистрибутива Linux
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    elif [ -f /etc/debian_version ]; then
        echo "debian"
    elif [ -f /etc/redhat-release ]; then
        echo "redhat"
    else
        echo "unknown"
    fi
}

# Функция для установки системных зависимостей
install_system_deps() {
    echo "📦 Установка системных зависимостей..."
    
    DISTRO=$(detect_distro)
    
    case "$DISTRO" in
        "ubuntu"|"debian"|"linuxmint")
            echo "Обнаружен дистрибутив: $DISTRO"
            sudo apt-get update
            sudo apt-get install -y libhackrf-dev libfftw3-dev build-essential pkg-config \
                libusb-1.0-0-dev cmake swig python3-dev python3-pip
            ;;
        "fedora"|"rhel"|"centos"|"rocky"|"alma")
            echo "Обнаружен дистрибутив: $DISTRO"
            if command -v dnf >/dev/null 2>&1; then
                sudo dnf install -y hackrf-devel fftw3-devel gcc make pkgconfig \
                    libusb1-devel cmake swig python3-devel python3-pip
            else
                sudo yum install -y hackrf-devel fftw3-devel gcc make pkgconfig \
                    libusb1-devel cmake swig python3-devel python3-pip
            fi
            ;;
        "arch"|"manjaro")
            echo "Обнаружен дистрибутив: $DISTRO"
            sudo pacman -S --noconfirm hackrf fftw gcc make pkg-config \
                libusb cmake swig python python-pip
            ;;
        *)
            echo "⚠️  Неизвестный дистрибутив: $DISTRO"
            echo "Установите вручную: libhackrf-dev, libfftw3-dev, build-essential, pkg-config"
            ;;
    esac
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
    
    # Используем Linux Makefile
    if [ -f "Makefile.linux" ]; then
        make -f Makefile.linux clean
        make -f Makefile.linux all
        make -f Makefile.linux install
        echo "✓ Библиотека hackrf_master собрана и установлена"
    else
        echo "❌ Makefile.linux не найден"
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
    
    # Проверяем SoapySDR
    if ! python -c "import SoapySDR" > /dev/null 2>&1; then
        echo "❌ SoapySDR не установлен"
        echo "Выполните: conda activate panorama_env && conda install -c conda-forge soapysdr=0.8.1"
        exit 1
    fi
    
    echo "✓ Conda окружение готово"
    echo
}

# Функция для настройки прав доступа к USB
setup_usb_permissions() {
    echo "🔐 Настройка прав доступа к USB..."
    
    # Создаем правило для udev
    if [ ! -f "/etc/udev/rules.d/99-hackrf.rules" ]; then
        echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="6089", MODE="0666"' | sudo tee /etc/udev/rules.d/99-hackrf.rules > /dev/null
        sudo udevadm control --reload-rules
        sudo udevadm trigger
        echo "✓ Правила udev созданы"
    else
        echo "✓ Правила udev уже существуют"
    fi
    echo
}

# Функция для запуска приложения
run_panorama() {
    echo "🚀 Запуск Panorama App..."
    
    # Активируем conda окружение
    source "$CONDA_PATH/bin/activate" panorama_env
    
    # Очищаем переменные окружения
    unset LD_LIBRARY_PATH
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
        echo "  --usb          Только настроить права доступа к USB"
        echo "  --run          Только запустить приложение"
        echo "  (без опций)    Полная установка и запуск"
        echo
        exit 0
    fi
    
    # Проверяем операционную систему
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        echo "❌ Этот скрипт предназначен только для Linux"
        echo "Для macOS используйте: ./run_macos.sh"
        exit 1
    fi
    
    case "$1" in
        "--install")
            install_system_deps
            install_emoji_fonts
            build_hackrf_master
            setup_usb_permissions
            check_conda_env
            echo "✅ Установка завершена!"
            ;;
        "--fonts")
            install_emoji_fonts
            ;;
        "--build")
            build_hackrf_master
            ;;
        "--usb")
            setup_usb_permissions
            ;;
        "--run")
            check_conda_env
            run_panorama
            ;;
        "")
            # Полная установка и запуск
            install_system_deps
            install_emoji_fonts
            build_hackrf_master
            setup_usb_permissions
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
