# Скрипт сборки libhackrf_multi

## Описание

`build_libhackrf.sh` - автоматический скрипт для сборки библиотеки `libhackrf_multi` в проекте ПАНОРАМА.

## Использование

### Запуск из корня проекта

```bash
./build_libhackrf.sh
```

### Что делает скрипт

1. **Проверяет зависимости:**
   - `gcc` (компилятор)
   - `libhackrf` (HackRF development package)
   - `fftw3f` (FFTW3 library)

2. **Анализирует необходимость пересборки:**
   - Если библиотека отсутствует → собирает
   - Если исходники новее библиотеки → пересобирает
   - Если всё актуально → пропускает сборку

3. **Собирает библиотеку:**
   - Очищает предыдущую сборку (`make clean`)
   - Компилирует все исходники
   - Линкует в `libhackrf_multi.so`

4. **Тестирует результат:**
   - Проверяет основные символы
   - Тестирует загрузку в Python
   - Показывает размер и статус

## Требования

### Системные пакеты

```bash
# Ubuntu/Debian
sudo apt install build-essential hackrf-dev libfftw3-dev

# Arch Linux
sudo pacman -S base-devel hackrf fftw

# macOS
brew install hackrf fftw
```

### Python зависимости

```bash
pip install cffi
```

## Структура файлов

```
panorama_app/
├── build_libhackrf.sh          # ← Этот скрипт
├── panorama/
│   └── drivers/
│       └── hackrf_lib/
│           ├── Makefile
│           ├── *.c, *.h        # Исходники
│           └── libhackrf_multi.so  # ← Собранная библиотека
```

## Примеры вывода

### Успешная сборка

```
[INFO] Переходим в папку: /path/to/panorama/drivers/hackrf_lib
[INFO] Проверяем зависимости...
[SUCCESS] Все зависимости найдены
[INFO] Исходник ./hq_master.c новее библиотеки, пересобираем...
[INFO] Очищаем предыдущую сборку...
[INFO] Собираем библиотеку...
[SUCCESS] Библиотека успешно собрана: libhackrf_multi.so
[INFO] Размер библиотеки: 128K
[SUCCESS] ✓ hq_open_all найден
[SUCCESS] ✓ hq_start найден
[SUCCESS] Тест Python прошел успешно
[SUCCESS] Сборка завершена!
```

### Библиотека актуальна

```
[INFO] Проверяем зависимости...
[SUCCESS] Все зависимости найдены
[SUCCESS] Библиотека уже актуальна, пересборка не требуется
[SUCCESS] Сборка завершена!
```

## Устранение проблем

### Ошибка: "gcc не найден"
```bash
sudo apt install build-essential
```

### Ошибка: "libhackrf не найден"
```bash
sudo apt install hackrf-dev
```

### Ошибка: "fftw3f не найден"
```bash
sudo apt install libfftw3-dev
```

### Ошибка сборки
1. Проверьте, что все зависимости установлены
2. Убедитесь, что исходники корректны
3. Запустите `make clean` вручную
4. Проверьте логи компиляции

## Интеграция с IDE

### VS Code
Добавьте в `.vscode/tasks.json`:

```json
{
    "label": "Build libhackrf",
    "type": "shell",
    "command": "./build_libhackrf.sh",
    "group": "build",
    "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
    }
}
```

### PyCharm
Создайте External Tool:
- Program: `./build_libhackrf.sh`
- Working directory: `$ProjectFileDir$`

## Автоматизация

### Pre-commit hook
Добавьте в `.git/hooks/pre-commit`:

```bash
#!/bin/bash
if git diff --cached --name-only | grep -q "panorama/drivers/hackrf_lib/.*\.c\|panorama/drivers/hackrf_lib/.*\.h"; then
    echo "Пересобираем libhackrf_multi..."
    ./build_libhackrf.sh
fi
```

### CI/CD
Добавьте в ваш CI pipeline:

```yaml
- name: Build libhackrf
  run: |
    chmod +x build_libhackrf.sh
    ./build_libhackrf.sh
```

## Лицензия

Скрипт создан для проекта ПАНОРАМА. Используйте свободно.
