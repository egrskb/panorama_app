# Установка ПАНОРАМА

## Системные требования

- **ОС**: Linux (Ubuntu 20.04+, Debian 11+), macOS, Windows 10+
- **Python**: 3.8 или выше
- **RAM**: минимум 2 ГБ
- **HackRF One**: с установленной прошивкой hackrf_sweep

## Быстрая установка (Linux/macOS)

```bash
# Клонируем репозиторий
git clone https://github.com/yourusername/panorama.git
cd panorama

# Используем Makefile
make all        # Полная установка
make run        # Запуск
```

## Подробная установка

### 1. Установка системных зависимостей

#### Ubuntu/Debian:
```bash
# Основные пакеты
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git

# HackRF tools
sudo apt install -y hackrf libhackrf-dev

# Для компиляции libhackrf_qsa (опционально)
sudo apt install -y gcc make libfftw3-dev pkg-config

# Qt5 (если PyQt5 не устанавливается через pip)
sudo apt install -y python3-pyqt5
```

#### Fedora/RHEL:
```bash
sudo dnf install -y python3 python3-pip git
sudo dnf install -y hackrf hackrf-devel
sudo dnf install -y gcc make fftw-devel pkgconfig
```

#### macOS:
```bash
# Установите Homebrew если нет
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Установка зависимостей
brew install python3 hackrf fftw pkg-config
```

#### Windows:
1. Установите Python 3.8+ с [python.org](https://python.org)
2. Установите драйверы HackRF с [greatscottgadgets.com](https://greatscottgadgets.com/hackrf/)
3. Скачайте hackrf_sweep.exe и добавьте в PATH

### 2. Создание виртуального окружения

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Установка Python-зависимостей

```bash
# Обновляем pip
pip install --upgrade pip setuptools wheel

# Устанавливаем зависимости
pip install -r requirements.txt
```

### 4. Компиляция libhackrf_qsa (опционально)

Для использования библиотечного режима через CFFI:

```bash
cd panorama/drivers/hackrf_lib

# Linux
gcc -shared -fPIC -O3 \
    $(pkg-config --cflags libhackrf) \
    $(pkg-config --cflags fftw3f) \
    -o libhackrf_qsa.so \
    hq_sweep.c \
    $(pkg-config --libs libhackrf) \
    $(pkg-config --libs fftw3f) \
    -lm -pthread

# macOS
gcc -shared -fPIC -O3 \
    $(pkg-config --cflags libhackrf) \
    $(pkg-config --cflags fftw3f) \
    -o libhackrf_qsa.dylib \
    hq_sweep.c \
    $(pkg-config --libs libhackrf) \
    $(pkg-config --libs fftw3f) \
    -lm -pthread

# Копируем в корень проекта
cp libhackrf_qsa.* ../../../

cd ../../../
```

### 5. Проверка установки

```bash
# Проверяем Python модули
python -c "import PyQt5, pyqtgraph, numpy, cffi; print('✓ Все модули установлены')"

# Проверяем hackrf_sweep
which hackrf_sweep || echo "⚠ hackrf_sweep не найден"

# Проверяем библиотеку (если скомпилирована)
ls -la libhackrf_qsa.* 2>/dev/null || echo "⚠ libhackrf_qsa не найдена"
```

## Запуск приложения

### Способ 1: Через скрипт
```bash
./run_panorama.sh
```

### Способ 2: Через Python
```bash
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate     # Windows

python -m panorama.main
```

### Способ 3: Через Makefile
```bash
make run
```

## Настройка калибровки

1. Создайте файл `hackrf_cal.csv` в корне проекта:
```csv
# freq_mhz,lna_db,vga_db,amp,offset_db
100.0,24,20,0,-2.5
500.0,24,20,0,-3.0
1000.0,24,20,0,-3.5
2400.0,24,20,0,-4.0
```

2. Или загрузите через меню: **Калибровка → Загрузить CSV**

## Решение проблем

### PyQt5 не устанавливается
```bash
# Linux: используйте системный пакет
sudo apt install python3-pyqt5
# Затем создайте venv с системными пакетами
python3 -m venv venv --system-site-packages
```

### hackrf_sweep не находит устройство
```bash
# Проверьте права доступа
sudo hackrf_info

# Добавьте пользователя в группу plugdev
sudo usermod -a -G plugdev $USER
# Перелогиньтесь
```

### Ошибка компиляции libhackrf_qsa
```bash
# Проверьте наличие всех библиотек
pkg-config --cflags --libs libhackrf
pkg-config --cflags --libs fftw3f

# Если не находит, установите dev-пакеты
sudo apt install libhackrf-dev libfftw3-dev
```

### Низкая производительность
- Увеличьте размер бина (bin_khz)
- Уменьшите диапазон сканирования
- Отключите сглаживание и эффекты
- Используйте libhackrf вместо hackrf_sweep

## Структура проекта

```
panorama/
├── panorama/
│   ├── main.py              # Главное окно
│   ├── drivers/             # Драйверы источников
│   │   ├── base.py
│   │   ├── hackrf_sweep.py
│   │   └── hackrf_lib/
│   │       ├── backend.py
│   │       ├── hq_sweep.c
│   │       └── hq_sweep.h
│   ├── features/            # Функциональные модули
│   │   ├── spectrum/        # Спектр и водопад
│   │   ├── peaks/           # Поиск пиков
│   │   ├── devices/         # Выбор устройств
│   │   └── map3d/           # Карта
│   └── shared/              # Общие утилиты
│       ├── calibration.py
│       ├── palettes.py
│       └── parsing.py
├── requirements.txt         # Зависимости Python
├── Makefile                # Автоматизация сборки
├── setup.sh               # Скрипт установки
└── README.md
```

## Дополнительные возможности

### Установка как системное приложение (Linux)
```bash
# Создание .desktop файла
cat > ~/.local/share/applications/panorama.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ПАНОРАМА
Comment=HackRF Sweep Analyzer
Exec=$(pwd)/run_panorama.sh
Icon=$(pwd)/panorama/assets/icon.png
Terminal=false
Categories=Network;HamRadio;
EOF

# Обновление базы приложений
update-desktop-database ~/.local/share/applications/
```

### Создание deb-пакета
```bash
make deb
sudo dpkg -i dist/panorama_0.1-1_amd64.deb
```

## Поддержка

- **Issues**: https://github.com/yourusername/panorama/issues
- **Wiki**: https://github.com/yourusername/panorama/wiki