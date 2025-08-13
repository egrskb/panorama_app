#!/bin/bash
# Простой скрипт запуска PANORAMA

# Определяем директорию скрипта
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Переходим в корень проекта
cd "$SCRIPT_DIR"

# Активируем виртуальное окружение если есть
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Запускаем приложение
python -m panorama.main "$@"