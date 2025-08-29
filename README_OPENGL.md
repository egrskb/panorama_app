# Запуск ПАНОРАМА RSSI с OpenGL ускорением

## Проблема
При запуске приложения может появляться сообщение:
```
OpenGL.acceleratesupport - INFO - No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'
```

Это означает, что OpenGL ускорение не установлено, но приложение работает нормально.

## Решение

### 1. Автоматический запуск (рекомендуется)
```bash
./run_panorama.sh
```

### 2. Ручной запуск
```bash
# Активируем виртуальное окружение
source mvenv/bin/activate

# Запускаем приложение
python run_rssi_panorama.py
```

### 3. Проверка OpenGL ускорения
```bash
source mvenv/bin/activate
python -c "import OpenGL_accelerate; print('Ускорение доступно')"
```

## Преимущества OpenGL ускорения

- ✅ Более быстрая отрисовка графиков
- ✅ Плавная работа спектра и водопада
- ✅ Лучшая производительность при большом количестве точек
- ✅ Отсутствие предупреждений в логах

## Устранение неполадок

### Если виртуальное окружение не активируется:
```bash
# Пересоздаем виртуальное окружение
rm -rf mvenv
python3 -m venv mvenv
source mvenv/bin/activate
pip install -r requirements.txt
pip install PyOpenGL-accelerate
```

### Если PyOpenGL-accelerate не устанавливается:
```bash
# Обновляем pip
pip install --upgrade pip

# Устанавливаем ускорение
pip install PyOpenGL-accelerate
```

## Примечание
OpenGL ускорение **НЕ обязательно** для работы приложения. 
Приложение будет работать и без него, но графики могут отрисовываться медленнее.
