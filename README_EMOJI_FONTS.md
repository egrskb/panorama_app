# 🎨 Установка Emoji шрифтов для Panorama App

Этот набор скриптов поможет установить шрифты с поддержкой emoji для корректного отображения в Panorama App.

## 📋 Проблема

В Linux системах часто отсутствуют шрифты с поддержкой emoji, что приводит к отображению квадратиков или пустых мест вместо emoji символов в интерфейсе Panorama App.

## 🚀 Решение

Предоставляются три скрипта для установки emoji шрифтов:

### 1. `install_emoji_fonts_universal.sh` (Рекомендуется)
**Универсальный скрипт** - автоматически определяет дистрибутив и использует лучший способ установки.

### 2. `install_emoji_fonts_package.sh`
**Установка через пакетный менеджер** - использует стандартные репозитории дистрибутива.

### 3. `install_emoji_fonts.sh`
**Ручная установка** - скачивает шрифты напрямую с GitHub.

## 📦 Поддерживаемые дистрибутивы

- **Debian/Ubuntu** (apt-get)
- **Fedora/RHEL** (dnf)
- **Arch Linux** (pacman)
- **openSUSE** (zypper)

## 🔧 Установка

### Быстрая установка (рекомендуется)

```bash
# Сделать скрипт исполняемым
chmod +x install_emoji_fonts_universal.sh

# Запустить с правами администратора
sudo ./install_emoji_fonts_universal.sh
```

### Альтернативные способы

```bash
# Через пакетный менеджер
sudo ./install_emoji_fonts_package.sh

# Ручная установка
sudo ./install_emoji_fonts.sh
```

## 📱 Устанавливаемые шрифты

- **Noto Color Emoji** (Google) - основной emoji шрифт
- **Twitter Color Emoji** - альтернативный emoji шрифт
- **JoyPixels** - профессиональный emoji шрифт
- **Apple Color Emoji** - для совместимости с macOS
- **Segoe UI Emoji** - для совместимости с Windows

## ✅ Проверка установки

После установки проверьте наличие emoji шрифтов:

```bash
# Количество установленных emoji шрифтов
fc-list | grep -i emoji | wc -l

# Список доступных emoji шрифтов
fc-list | grep -i emoji

# Обновление кэша шрифтов
fc-cache -fv
```

## 🔄 После установки

1. **Перезапустите Panorama App**
2. **Если emoji не отображаются** - перезагрузите систему
3. **Проверьте настройки шрифтов** в вашем дистрибутиве

## 🐛 Решение проблем

### Emoji все еще не отображаются

1. **Проверьте установку:**
   ```bash
   fc-list | grep -i emoji | wc -l
   ```

2. **Обновите кэш шрифтов:**
   ```bash
   sudo fc-cache -fv
   ```

3. **Перезагрузите систему**

4. **Проверьте настройки шрифтов** в вашем дистрибутиве

### Ошибки при установке

1. **Проверьте права администратора:**
   ```bash
   sudo ./install_emoji_fonts_universal.sh
   ```

2. **Проверьте интернет соединение**

3. **Попробуйте альтернативный скрипт**

## 📁 Структура файлов

```
panorama_app/
├── install_emoji_fonts_universal.sh    # Универсальный скрипт
├── install_emoji_fonts_package.sh      # Через пакетный менеджер
├── install_emoji_fonts.sh              # Ручная установка
└── README_EMOJI_FONTS.md               # Этот файл
```

## 🔗 Полезные ссылки

- [Noto Color Emoji](https://github.com/googlefonts/noto-emoji)
- [Twitter Color Emoji](https://github.com/eosrei/twemoji-color-font)
- [JoyPixels](https://github.com/joypixels/emoji-font)
- [Fontconfig](https://www.freedesktop.org/wiki/Software/fontconfig/)

## 📝 Лицензия

Скрипты распространяются под лицензией MIT. Шрифты имеют свои собственные лицензии.

## 🤝 Поддержка

Если у вас возникли проблемы с установкой emoji шрифтов:

1. Проверьте логи установки
2. Убедитесь что у вас есть права администратора
3. Проверьте совместимость с вашим дистрибутивом
4. Создайте issue в репозитории проекта

---

**Примечание:** После установки emoji шрифтов все emoji символы в Panorama App должны отображаться корректно! 🎉
