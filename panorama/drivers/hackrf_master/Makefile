# Makefile для hackrf_master driver
# Компилирует C-библиотеку с исправлениями спектра

CC = gcc
CFLAGS = -shared -fPIC -O3 -Wall -Wextra
LIBS = -lhackrf -lfftw3f -lm -pthread

# Пути к заголовочным файлам
HACKRF_CFLAGS = $(shell pkg-config --cflags libhackrf 2>/dev/null || echo "-I/usr/include/hackrf")
FFTW_CFLAGS = $(shell pkg-config --cflags fftw3f 2>/dev/null || echo "-I/usr/include")

# Пути к библиотекам
HACKRF_LIBS = $(shell pkg-config --libs libhackrf 2>/dev/null || echo "-L/usr/lib -lhackrf")
FFTW_LIBS = $(shell pkg-config --libs fftw3f 2>/dev/null || echo "-L/usr/lib -lfftw3f")

# Целевые файлы
TARGET = libhackrf_qsa.so
SOURCE = hackrf_master.c
OBJECT = hackrf_master.o
HEADER = hackrf_master.h

# Основная цель
all: $(TARGET)

# Компиляция библиотеки
$(TARGET): $(OBJECT)
	@echo "Создание библиотеки $(TARGET)..."
	$(CC) $(CFLAGS) -o $@ $< $(HACKRF_LIBS) $(FFTW_LIBS) $(LIBS)
	@echo "Библиотека $(TARGET) создана успешно"

# Компиляция объектного файла
$(OBJECT): $(SOURCE) $(HEADER)
	@echo "Компиляция $(SOURCE)..."
	$(CC) -c $(SOURCE) -o $@ $(HACKRF_CFLAGS) $(FFTW_CFLAGS)

# Установка
install: $(TARGET)
	@echo "Установка библиотеки..."
	sudo cp $(TARGET) /usr/local/lib/
	sudo ldconfig
	@echo "Библиотека установлена в /usr/local/lib/"

# Очистка
clean:
	@echo "Очистка..."
	rm -f $(OBJECT) $(TARGET)
	@echo "Очистка завершена"

# Пересборка
rebuild: clean all

# Проверка зависимостей
check-deps:
	@echo "Проверка зависимостей..."
	@echo "HackRF: $(shell pkg-config --exists libhackrf && echo "✓ найдена" || echo "✗ не найдена")"
	@echo "FFTW3: $(shell pkg-config --exists fftw3f && echo "✓ найдена" || echo "✗ не найдена")"
	@echo "GCC: $(shell which gcc && echo "✓ найдена" || echo "✗ не найдена")"

# Помощь
help:
	@echo "Доступные команды:"
	@echo "  all        - компиляция библиотеки (по умолчанию)"
	@echo "  install    - установка в систему"
	@echo "  clean      - очистка временных файлов"
	@echo "  rebuild    - пересборка с нуля"
	@echo "  check-deps - проверка зависимостей"
	@echo "  help       - показать эту справку"

# Установка зависимостей (Ubuntu/Debian)
install-deps-ubuntu:
	@echo "Установка зависимостей для Ubuntu/Debian..."
	sudo apt-get update
	sudo apt-get install -y libhackrf-dev libfftw3-dev build-essential pkg-config

# Установка зависимостей (CentOS/RHEL/Fedora)
install-deps-centos:
	@echo "Установка зависимостей для CentOS/RHEL/Fedora..."
	sudo yum install -y hackrf-devel fftw3-devel gcc make pkgconfig
	# или для Fedora:
	# sudo dnf install -y hackrf-devel fftw3-devel gcc make pkgconfig

# Установка зависимостей (macOS)
install-deps-macos:
	@echo "Установка зависимостей для macOS..."
	brew install hackrf fftw pkg-config

.PHONY: all install clean rebuild check-deps help install-deps-ubuntu install-deps-centos install-deps-macos
