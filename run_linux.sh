#!/bin/bash

# Универсальный скрипт запуска Panorama App для Linux
# Объединяет: установку шрифтов, сборку библиотеки, запуск приложения

set -e  # Остановка при ошибке

echo "=== Panorama App для Linux ==="
echo

# Путь к venv
VENV_DIR="$(pwd)/mvenv"

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
    
    local original_dir=$(pwd)
    cd panorama/drivers/hackrf/hackrf_master
    
    # Используем Linux Makefile
    if [ -f "Makefile.linux" ]; then
        make -f Makefile.linux clean
        make -f Makefile.linux all
        make -f Makefile.linux install
        echo "✓ Библиотека hackrf_master собрана и установлена"
    else
        echo "❌ Makefile.linux не найден"
        cd "$original_dir"
        exit 1
    fi
    
    cd "$original_dir"
    echo
}

# Функция для сборки библиотеки hackrf_slave
build_hackrf_slave() {
    echo "🔧 Сборка библиотеки hackrf_slave..."
    
    local original_dir=$(pwd)
    cd panorama/drivers/hackrf/hackrf_slaves
    
    # Используем Linux Makefile
    if [ -f "Makefile.linux" ]; then
        make -f Makefile.linux clean
        make -f Makefile.linux all
        echo "✓ Библиотека hackrf_slave собрана"
    elif [ -f "Makefile" ]; then
        make clean
        make all
        echo "✓ Библиотека hackrf_slave собрана"
    else
        echo "❌ Makefile для hackrf_slave не найден"
        cd "$original_dir"
        exit 1
    fi
    
    # Проверяем результат
    if [ -f "libhackrf_slave.so" ]; then
        echo "✓ libhackrf_slave.so создана успешно"
        
        # Показываем зависимости
        echo "Library dependencies:"
        ldd libhackrf_slave.so
        # Локальное использование: библиотека остается в каталоге hackrf_slaves
        # Код приложения загружает её напрямую из panorama/drivers/hackrf/hackrf_slaves
    else
        echo "❌ libhackrf_slave.so не найдена после сборки"
        cd "$original_dir"
        exit 1
    fi
    
    cd "$original_dir"
    echo
}

# Функция для проверки conda окружения
check_venv() {
    echo "🐍 Проверка python venv..."
    
    if [ ! -d "$VENV_DIR" ]; then
        echo "Создаю виртуальное окружение в $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
    
    # Активируем venv
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    
    python -m pip install --upgrade pip wheel setuptools >/dev/null
    
    # Установка зависимостей из requirements.txt
    if [ -f requirements.txt ]; then
        echo "📦 Установка python-зависимостей (requirements.txt)"
        pip install -r requirements.txt
    fi
    
    # SoapySDR больше не используется
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
    
    # Активируем venv
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    
    # Очищаем переменные окружения, чтобы не мешали системные плагины
    unset LD_LIBRARY_PATH || true
    unset SOAPY_SDR_PLUGIN_PATH || true
    
    # Запускаем приложение
    # Определяем путь к лаунчеру
    if [ -f "run_rssi_panorama.py" ]; then
        python run_rssi_panorama.py
    elif [ -f "panorama/run_rssi_panorama.py" ]; then
        python panorama/run_rssi_panorama.py
    else
        echo "❌ Не найден файл запуска run_rssi_panorama.py"
        exit 1
    fi
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
        echo "  --build        Только собрать библиотеки hackrf_master и hackrf_slave"
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
            build_hackrf_slave
            setup_usb_permissions
            check_venv
            echo "✅ Установка завершена!"
            ;;
        "--fonts")
            install_emoji_fonts
            ;;
        "--build")
            build_hackrf_master
            build_hackrf_slave
            ;;
        "--usb")
            setup_usb_permissions
            ;;
        "--run")
            check_venv
            run_panorama
            ;;
        "")
            # Полная установка и запуск
            install_system_deps
            install_emoji_fonts
            build_hackrf_master
            build_hackrf_slave
            setup_usb_permissions
            check_venv
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
