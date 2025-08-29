#!/bin/bash
# Скрипт запуска ПАНОРАМА RSSI из виртуального окружения

echo "Запуск ПАНОРАМА RSSI..."
echo "Активация виртуального окружения..."

# Активируем виртуальное окружение
source mvenv/bin/activate

# Проверяем, что виртуальное окружение активировано
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Виртуальное окружение активировано: $VIRTUAL_ENV"
    echo "✓ Python: $(which python)"
    echo "✓ OpenGL ускорение: $(python -c "import OpenGL_accelerate; print('доступно')" 2>/dev/null || echo 'недоступно')"
    
    # Запускаем приложение
    echo "Запуск приложения..."
    python run_rssi_panorama.py
else
    echo "✗ Ошибка активации виртуального окружения"
    exit 1
fi
