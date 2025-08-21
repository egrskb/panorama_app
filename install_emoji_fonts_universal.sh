#!/bin/bash

# Универсальный скрипт установки шрифтов с поддержкой emoji для Panorama App
# Автор: Panorama App Team
# Версия: 1.0

set -e  # Останавливаем выполнение при ошибке

echo "🎨 Универсальная установка шрифтов с поддержкой emoji для Panorama App"
echo "=================================================================="

# Проверяем что мы в Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ Этот скрипт предназначен только для Linux"
    exit 1
fi

# Проверяем права root
if [[ $EUID -ne 0 ]]; then
    echo "⚠️  Для установки шрифтов требуются права администратора"
    echo "Запустите скрипт с sudo: sudo ./install_emoji_fonts_universal.sh"
    exit 1
fi

echo "🔍 Проверяем текущие emoji шрифты..."
CURRENT_EMOJI_FONTS=$(fc-list | grep -i emoji | wc -l)
echo "📊 Найдено emoji шрифтов: $CURRENT_EMOJI_FONTS"

if [ $CURRENT_EMOJI_FONTS -gt 0 ]; then
    echo "✅ Emoji шрифты уже установлены!"
    echo "📋 Доступные emoji шрифты:"
    fc-list | grep -i emoji | head -5
    echo ""
    echo "🔄 Перезапустите Panorama App для применения изменений"
    exit 0
fi

echo "📥 Emoji шрифты не найдены, начинаем установку..."
echo ""

# Определяем дистрибутив
if command -v apt-get &> /dev/null; then
    DISTRO="debian"
    echo "📦 Обнаружен Debian/Ubuntu (apt-get)"
elif command -v dnf &> /dev/null; then
    DISTRO="fedora"
    echo "📦 Обнаружен Fedora/RHEL (dnf)"
elif command -v pacman &> /dev/null; then
    DISTRO="arch"
    echo "📦 Обнаружен Arch Linux (pacman)"
elif command -v zypper &> /dev/null; then
    DISTRO="opensuse"
    echo "📦 Обнаружен openSUSE (zypper)"
else
    DISTRO="unknown"
    echo "⚠️  Неизвестный дистрибутив Linux"
fi

echo ""
echo "🚀 Способ 1: Установка через пакетный менеджер"
echo "=============================================="

# Пытаемся установить через пакетный менеджер
case $DISTRO in
    "debian")
        echo "🔄 Обновляем список пакетов..."
        apt-get update
        
        echo "📱 Устанавливаем Noto Color Emoji..."
        apt-get install -y fonts-noto-color-emoji
        
        echo "🐦 Устанавливаем Twitter Color Emoji..."
        apt-get install -y fonts-twemoji || echo "⚠️  Twitter Color Emoji недоступен"
        
        echo "😊 Устанавливаем JoyPixels..."
        apt-get install -y fonts-joypixels || echo "⚠️  JoyPixels недоступен"
        ;;
        
    "fedora")
        echo "🔄 Обновляем список пакетов..."
        dnf update -y
        
        echo "📱 Устанавливаем Noto Color Emoji..."
        dnf install -y google-noto-emoji-fonts
        
        echo "🐦 Устанавливаем Twitter Color Emoji..."
        dnf install -y twitter-twemoji-fonts || echo "⚠️  Twitter Color Emoji недоступен"
        ;;
        
    "arch")
        echo "🔄 Обновляем список пакетов..."
        pacman -Sy
        
        echo "📱 Устанавливаем Noto Color Emoji..."
        pacman -S --noconfirm noto-fonts-emoji
        
        echo "🐦 Устанавливаем Twitter Color Emoji..."
        pacman -S --noconfirm twitter-color-emoji-fonts || echo "⚠️  Twitter Color Emoji недоступен"
        ;;
        
    "opensuse")
        echo "🔄 Обновляем список пакетов..."
        zypper refresh
        
        echo "📱 Устанавливаем Noto Color Emoji..."
        zypper install -y google-noto-emoji-fonts
        
        echo "🐦 Устанавливаем Twitter Color Emoji..."
        zypper install -y twitter-twemoji-fonts || echo "⚠️  Twitter Color Emoji недоступен"
        ;;
        
    *)
        echo "⚠️  Пропускаем установку через пакетный менеджер"
        ;;
esac

echo ""
echo "🔍 Проверяем результат установки через пакетный менеджер..."
PACKAGE_EMOJI_FONTS=$(fc-cache -f && fc-list | grep -i emoji | wc -l)
echo "📊 Найдено emoji шрифтов после установки пакетов: $PACKAGE_EMOJI_FONTS"

if [ $PACKAGE_EMOJI_FONTS -gt 0 ]; then
    echo "✅ Установка через пакетный менеджер успешна!"
else
    echo "⚠️  Установка через пакетный менеджер не дала результатов"
    echo ""
    echo "🚀 Способ 2: Ручная установка шрифтов"
    echo "===================================="
    
    # Создаем директорию для шрифтов
    FONT_DIR="/usr/share/fonts/truetype/panorama-emoji"
    mkdir -p "$FONT_DIR"
    
    echo "📥 Скачиваем шрифты с поддержкой emoji..."
    
    # Скачиваем Noto Color Emoji (Google)
    echo "📱 Скачиваем Noto Color Emoji..."
    wget -q -O "$FONT_DIR/NotoColorEmoji.ttf" \
        "https://github.com/googlefonts/noto-emoji/raw/main/fonts/NotoColorEmoji.ttf" || \
        echo "⚠️  Не удалось скачать Noto Color Emoji"
    
    # Скачиваем Twitter Color Emoji
    echo "🐦 Скачиваем Twitter Color Emoji..."
    wget -q -O "$FONT_DIR/TwitterColorEmoji.ttf" \
        "https://github.com/eosrei/twemoji-color-font/raw/master/fonts/TwitterColorEmoji.ttf" || \
        echo "⚠️  Не удалось скачать Twitter Color Emoji"
    
    # Скачиваем JoyPixels
    echo "😊 Скачиваем JoyPixels..."
    wget -q -O "$FONT_DIR/JoyPixels.ttf" \
        "https://github.com/joypixels/emoji-font/raw/master/fonts/JoyPixels.ttf" || \
        echo "⚠️  Не удалось скачать JoyPixels"
    
    echo "🔧 Обновляем кэш шрифтов..."
    fc-cache -fv
    
    echo "🔍 Проверяем результат ручной установки..."
    MANUAL_EMOJI_FONTS=$(fc-list | grep -i emoji | wc -l)
    echo "📊 Найдено emoji шрифтов после ручной установки: $MANUAL_EMOJI_FONTS"
fi

echo ""
echo "🔍 Финальная проверка установленных шрифтов..."

TOTAL_EMOJI_FONTS=$(fc-list | grep -i emoji | wc -l)
echo "📊 Всего найдено emoji шрифтов: $TOTAL_EMOJI_FONTS"

if [ $TOTAL_EMOJI_FONTS -gt 0 ]; then
    echo "✅ Установка emoji шрифтов завершена успешно!"
    echo ""
    echo "📋 Доступные emoji шрифты:"
    fc-list | grep -i emoji | head -10
    
    echo ""
    echo "🔄 Перезапустите Panorama App для применения изменений"
    echo "💡 Если emoji все еще не отображаются, попробуйте перезагрузить систему"
else
    echo "❌ Не удалось установить emoji шрифты"
    echo ""
    echo "💡 Попробуйте:"
    echo "   1. Перезагрузить систему"
    echo "   2. Проверить интернет соединение"
    echo "   3. Установить шрифты вручную"
fi

echo ""
echo "🔍 Для проверки установки выполните:"
echo "   fc-list | grep -i emoji | wc -l"
echo "   (должно быть больше 0)"
