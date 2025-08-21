#!/bin/bash

# Скрипт установки шрифтов с поддержкой emoji через пакетный менеджер
# Автор: Panorama App Team
# Версия: 1.0

set -e  # Останавливаем выполнение при ошибке

echo "🎨 Установка шрифтов с поддержкой emoji для Panorama App"
echo "=================================================="

# Проверяем что мы в Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ Этот скрипт предназначен только для Linux"
    exit 1
fi

# Проверяем права root
if [[ $EUID -ne 0 ]]; then
    echo "⚠️  Для установки пакетов требуются права администратора"
    echo "Запустите скрипт с sudo: sudo ./install_emoji_fonts_package.sh"
    exit 1
fi

echo "🔍 Определяем дистрибутив Linux..."

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
    echo "❌ Неизвестный дистрибутив Linux"
    exit 1
fi

echo "📥 Устанавливаем шрифты с поддержкой emoji..."

case $DISTRO in
    "debian")
        echo "🔄 Обновляем список пакетов..."
        apt-get update
        
        echo "📱 Устанавливаем Noto Color Emoji..."
        apt-get install -y fonts-noto-color-emoji
        
        echo "🍎 Устанавливаем Apple Color Emoji..."
        apt-get install -y fonts-apple-emoji || echo "⚠️  Apple Color Emoji недоступен"
        
        echo "🐦 Устанавливаем Twitter Color Emoji..."
        apt-get install -y fonts-twemoji || echo "⚠️  Twitter Color Emoji недоступен"
        
        echo "😊 Устанавливаем JoyPixels..."
        apt-get install -y fonts-joypixels || echo "⚠️  JoyPixels недоступен"
        
        echo "🪟 Устанавливаем Segoe UI Emoji..."
        apt-get install -y fonts-segoe-ui-emoji || echo "⚠️  Segoe UI Emoji недоступен"
        ;;
        
    "fedora")
        echo "🔄 Обновляем список пакетов..."
        dnf update -y
        
        echo "📱 Устанавливаем Noto Color Emoji..."
        dnf install -y google-noto-emoji-fonts
        
        echo "🐦 Устанавливаем Twitter Color Emoji..."
        dnf install -y twitter-twemoji-fonts || echo "⚠️  Twitter Color Emoji недоступен"
        
        echo "😊 Устанавливаем JoyPixels..."
        dnf install -y joypixels-fonts || echo "⚠️  JoyPixels недоступен"
        ;;
        
    "arch")
        echo "🔄 Обновляем список пакетов..."
        pacman -Sy
        
        echo "📱 Устанавливаем Noto Color Emoji..."
        pacman -S --noconfirm noto-fonts-emoji
        
        echo "🐦 Устанавливаем Twitter Color Emoji..."
        pacman -S --noconfirm twitter-color-emoji-fonts || echo "⚠️  Twitter Color Emoji недоступен"
        
        echo "😊 Устанавливаем JoyPixels..."
        pacman -S --noconfirm joypixels-fonts || echo "⚠️  JoyPixels недоступен"
        ;;
        
    "opensuse")
        echo "🔄 Обновляем список пакетов..."
        zypper refresh
        
        echo "📱 Устанавливаем Noto Color Emoji..."
        zypper install -y google-noto-emoji-fonts
        
        echo "🐦 Устанавливаем Twitter Color Emoji..."
        zypper install -y twitter-twemoji-fonts || echo "⚠️  Twitter Color Emoji недоступен"
        ;;
esac

echo "🔧 Обновляем кэш шрифтов..."

# Обновляем кэш шрифтов
fc-cache -fv

echo "✅ Шрифты установлены!"
echo ""
echo "🔍 Проверяем установленные шрифты..."

# Показываем установленные emoji шрифты
echo "📋 Доступные emoji шрифты:"
fc-list | grep -i emoji | head -10

echo ""
echo "🔄 Перезапустите Panorama App для применения изменений"
echo "💡 Если emoji все еще не отображаются, попробуйте перезагрузить систему"
echo ""
echo "🔍 Для полной проверки выполните:"
echo "   fc-list | grep -i emoji | wc -l"
echo "   (должно быть больше 0)"
