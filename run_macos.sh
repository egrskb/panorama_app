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
    
    # Проверяем наличие Homebrew
    if ! command -v brew >/dev/null 2>&1; then
        echo "❌ Homebrew не установлен. Установите его с https://brew.sh"
        return 1
    fi
    
    # Устанавливаем шрифты через Homebrew
    echo "📦 Установка emoji шрифтов через Homebrew..."
    
    # Проверяем наличие системных emoji шрифтов
    if [ -f "/System/Library/Fonts/Apple Color Emoji.ttc" ]; then
        echo "✓ Apple Color Emoji найден"
    fi
    
    # Устанавливаем шрифты с emoji поддержкой (новый способ)
    brew install --cask font-noto-emoji 2>/dev/null || {
        echo "⚠️  Font Noto Emoji уже установлен или недоступен"
    }
    
    # Устанавливаем дополнительные шрифты если нужно
    brew install --cask font-joypixels 2>/dev/null || {
        echo "⚠️  Font JoyPixels уже установлен или недоступен"
    }
    
    # Проверяем наличие других emoji шрифтов
    if [ -f "/System/Library/Fonts/Supplemental/Apple Color Emoji.ttc" ]; then
        echo "✓ Supplemental Apple Color Emoji найден"
    fi
    
    echo "✅ Emoji шрифты установлены для macOS"
    echo "💡 macOS имеет встроенную поддержку emoji шрифтов"
    echo
}

# Функция для сборки библиотеки hackrf_master
build_hackrf_master() {
    echo "🔧 Сборка библиотеки hackrf_master..."
    
    local original_dir=$(pwd)
    cd panorama/drivers/hackrf/hackrf_master
    
    # Определяем архитектуру
    ARCH=$(uname -m)
    echo "🏗️  Архитектура: $ARCH"
    
    # Используем macOS Makefile с поддержкой ARM
    if [ -f "Makefile.macos" ]; then
        make -f Makefile.macos clean
        make -f Makefile.macos all
        make -f Makefile.macos install
        echo "✓ Библиотека hackrf_master собрана и установлена"
    else
        echo "❌ Makefile.macos не найден"
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
    
    # Определяем архитектуру
    ARCH=$(uname -m)
    echo "🏗️  Архитектура: $ARCH"
    
    # Используем macOS Makefile с поддержкой ARM
    if [ -f "Makefile.macos" ]; then
        make -f Makefile.macos clean
        make -f Makefile.macos all
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
    if [ -f "libhackrf_slave.dylib" ] || [ -f "libhackrf_slave.so" ]; then
        LIBFILE=$(ls libhackrf_slave.* | head -1)
        echo "✓ $LIBFILE создана успешно"
        
        # Показываем зависимости
        echo "Library dependencies:"
        otool -L "$LIBFILE" || ldd "$LIBFILE" 2>/dev/null || true
        # Локальное использование: библиотека остается в каталоге hackrf_slaves
        # Приложение загружает её напрямую из panorama/drivers/hackrf/hackrf_slaves
    else
        echo "❌ libhackrf_slave не найдена после сборки"
        cd "$original_dir"
        exit 1
    fi
    
    cd "$original_dir"
    echo
}

# Функция для проверки conda окружения
check_conda_env() {
    echo "🐍 Проверка conda окружения..."
    
    # Активируем conda окружение panorama_env (поддержка разных установок)
    if command -v conda >/dev/null 2>&1; then
        # Если conda в PATH, используем стандартную активацию
        # Загружаем conda.sh, если доступен, чтобы гарантировать функцию 'conda'
        [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ] && . "$CONDA_PATH/etc/profile.d/conda.sh"
        conda activate panorama_env || {
            echo "❌ Не удалось активировать окружение panorama_env через conda"
            exit 1
        }
    else
        # Фолбэк: source activate из bin
        if [ -f "$CONDA_PATH/bin/activate" ]; then
            # shellcheck disable=SC1090
            . "$CONDA_PATH/bin/activate" panorama_env || {
                echo "❌ Не удалось активировать окружение panorama_env через $CONDA_PATH/bin/activate"
                exit 1
            }
        else
            echo "❌ Не найден conda и отсутствует $CONDA_PATH/bin/activate"
            exit 1
        fi
    fi
    
    # Проверяем Python
    if ! python --version > /dev/null 2>&1; then
        echo "❌ Python не найден в conda окружении"
        exit 1
    fi
    
    # SoapySDR больше не используется
    
    echo "✓ Conda окружение готово"
    echo
}

# Функция для запуска приложения
run_panorama() {
    echo "🚀 Запуск Panorama App..."
    
    # Активируем conda окружение
    if command -v conda >/dev/null 2>&1; then
        [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ] && . "$CONDA_PATH/etc/profile.d/conda.sh"
        conda activate panorama_env || {
            echo "❌ Не удалось активировать окружение panorama_env через conda"
            exit 1
        }
    else
        if [ -f "$CONDA_PATH/bin/activate" ]; then
            # shellcheck disable=SC1090
            . "$CONDA_PATH/bin/activate" panorama_env || {
                echo "❌ Не удалось активировать окружение panorama_env через $CONDA_PATH/bin/activate"
                exit 1
            }
        else
            echo "❌ Не найден conda и отсутствует $CONDA_PATH/bin/activate"
            exit 1
        fi
    fi
    
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
        echo "  --build        Только собрать библиотеки hackrf_master и hackrf_slave"
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
            build_hackrf_slave
            check_conda_env
            echo "✅ Установка завершена!"
            ;;
        "--fonts")
            install_emoji_fonts
            ;;
        "--build")
            build_hackrf_master
            build_hackrf_slave
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
            build_hackrf_slave
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
