#!/bin/bash

# Скрипт установки шрифтов с поддержкой emoji для Linux
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
    echo "⚠️  Для установки шрифтов требуются права администратора"
    echo "Запустите скрипт с sudo: sudo ./install_emoji_fonts.sh"
    exit 1
fi

echo "🔍 Проверяем текущие шрифты..."

# Создаем директорию для шрифтов если её нет
FONT_DIR="/usr/share/fonts/truetype/panorama-emoji"
mkdir -p "$FONT_DIR"

echo "📥 Скачиваем шрифты с поддержкой emoji..."

# Скачиваем Noto Color Emoji (Google)
echo "📱 Скачиваем Noto Color Emoji..."
wget -q -O "$FONT_DIR/NotoColorEmoji.ttf" \
    "https://github.com/googlefonts/noto-emoji/raw/main/fonts/NotoColorEmoji.ttf"

# Скачиваем Apple Color Emoji (если доступен)
echo "🍎 Скачиваем Apple Color Emoji..."
wget -q -O "$FONT_DIR/AppleColorEmoji.ttf" \
    "https://github.com/samuelngs/apple-emoji-linux/raw/master/AppleColorEmoji.ttf" || \
    echo "⚠️  Apple Color Emoji недоступен, пропускаем"

# Скачиваем Segoe UI Emoji (Microsoft)
echo "🪟 Скачиваем Segoe UI Emoji..."
wget -q -O "$FONT_DIR/seguiemj.ttf" \
    "https://github.com/microsoft/Windows/raw/master/Windows/System32/fonts/seguiemj.ttf" || \
    echo "⚠️  Segoe UI Emoji недоступен, пропускаем"

# Скачиваем Twitter Color Emoji
echo "🐦 Скачиваем Twitter Color Emoji..."
wget -q -O "$FONT_DIR/TwitterColorEmoji.ttf" \
    "https://github.com/eosrei/twemoji-color-font/raw/master/fonts/TwitterColorEmoji.ttf" || \
    echo "⚠️  Twitter Color Emoji недоступен, пропускаем"

# Скачиваем JoyPixels
echo "😊 Скачиваем JoyPixels..."
wget -q -O "$FONT_DIR/JoyPixels.ttf" \
    "https://github.com/joypixels/emoji-font/raw/master/fonts/JoyPixels.ttf" || \
    echo "⚠️  JoyPixels недоступен, пропускаем"

echo "🔧 Обновляем кэш шрифтов..."

# Обновляем кэш шрифтов
fc-cache -fv

echo "✅ Шрифты установлены!"
echo ""
echo "📋 Установленные шрифты:"
ls -la "$FONT_DIR"

echo ""
echo "🔄 Перезапустите Panorama App для применения изменений"
echo "💡 Если emoji все еще не отображаются, попробуйте перезагрузить систему"
echo ""
echo "🔍 Для проверки установки выполните:"
echo "   fc-list | grep -i emoji"
