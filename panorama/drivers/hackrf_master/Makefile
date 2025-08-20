# Makefile для HackRF Master библиотеки
# Сборка C библиотеки для Python CFFI

CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -std=c99
LDFLAGS = -shared -lhackrf -lpthread -lm

# Директории
SRC_DIR = .
BUILD_DIR = build
INSTALL_DIR = /usr/local/lib

# Исходные файлы
SOURCES = hackrf_master.c
OBJECTS = $(SOURCES:%.c=$(BUILD_DIR)/%.o)

# Целевые файлы
TARGET = libhackrf_master.so
TARGET_DEBUG = libhackrf_master_debug.so

# Проверяем наличие HackRF
HACKRF_CHECK := $(shell pkg-config --exists libhackrf && echo "yes" || echo "no")

.PHONY: all clean install debug check-deps

all: check-deps $(BUILD_DIR)/$(TARGET)

debug: CFLAGS += -g -DDEBUG
debug: $(BUILD_DIR)/$(TARGET_DEBUG)

check-deps:
ifeq ($(HACKRF_CHECK),no)
	@echo "ОШИБКА: libhackrf не найден!"
	@echo "Установите libhackrf-dev: sudo apt-get install libhackrf-dev"
	@exit 1
else
	@echo "✓ libhackrf найден"
	@echo "Версия: $(shell pkg-config --modversion libhackrf)"
endif

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "✓ Собрана библиотека: $@"

$(BUILD_DIR)/$(TARGET_DEBUG): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "✓ Собрана отладочная библиотека: $@"

install: $(BUILD_DIR)/$(TARGET)
	sudo cp $(BUILD_DIR)/$(TARGET) $(INSTALL_DIR)/
	sudo ldconfig
	@echo "✓ Библиотека установлена в $(INSTALL_DIR)"

uninstall:
	sudo rm -f $(INSTALL_DIR)/$(TARGET)
	sudo ldconfig
	@echo "✓ Библиотека удалена"

clean:
	rm -rf $(BUILD_DIR)
	@echo "✓ Очищены файлы сборки"

# Тестирование
test: $(BUILD_DIR)/$(TARGET)
	@echo "Запуск тестов..."
	@echo "TODO: Добавить unit тесты"

# Проверка зависимостей
deps:
	@echo "Проверка зависимостей:"
	@echo "  - libhackrf: $(HACKRF_CHECK)"
	@echo "  - pthread: $(shell pkg-config --exists libpthread && echo "yes" || echo "no")"
	@echo "  - math: $(shell pkg-config --exists libm && echo "yes" || echo "no")"

# Информация о системе
info:
	@echo "Информация о системе:"
	@echo "  - Компилятор: $(CC)"
	@echo "  - Флаги: $(CFLAGS)"
	@echo "  - Целевая архитектура: $(shell uname -m)"
	@echo "  - ОС: $(shell uname -s) $(shell uname -r)"

# Помощь
help:
	@echo "Доступные цели:"
	@echo "  all        - Собрать библиотеку (по умолчанию)"
	@echo "  debug      - Собрать отладочную версию"
	@echo "  install    - Установить библиотеку"
	@echo "  uninstall  - Удалить библиотеку"
	@echo "  clean      - Очистить файлы сборки"
	@echo "  test       - Запустить тесты"
	@echo "  deps       - Проверить зависимости"
	@echo "  info       - Информация о системе"
	@echo "  help       - Показать эту справку"
