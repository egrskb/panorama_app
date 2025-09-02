# Makefile для сборки hackrf_slave библиотеки

CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -std=c99
INCLUDES = -I/usr/include -I/usr/local/include
LDFLAGS = -shared
LIBS = -lhackrf -lfftw3f -lm -lpthread

# Директории
SRC_DIR = .
BUILD_DIR = build
TARGET_DIR = .

# Файлы
SOURCES = hackrf_slave.c
OBJECTS = $(SOURCES:%.c=$(BUILD_DIR)/%.o)
TARGET = libhackrf_slave.so

# Основная цель
all: $(TARGET_DIR)/$(TARGET)

# Создание директории для сборки
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Сборка объектных файлов
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Сборка разделяемой библиотеки
$(TARGET_DIR)/$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

# Очистка
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET_DIR)/$(TARGET)

# Установка зависимостей (для разработчика)
install-deps:
	@echo "Установка зависимостей..."
	@echo "Убедитесь что установлены:"
	@echo "- libhackrf-dev"
	@echo "- libfftw3-dev" 
	@echo "Команды для Ubuntu/Debian:"
	@echo "sudo apt-get install libhackrf-dev libfftw3-dev"

# Тест сборки
test-build: all
	@echo "Библиотека $(TARGET) успешно собрана"
	@ldd $(TARGET_DIR)/$(TARGET)

.PHONY: all clean install-deps test-build