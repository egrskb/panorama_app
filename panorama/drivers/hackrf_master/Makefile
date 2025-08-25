# Makefile для hackrf_master библиотеки
# Сборка C-библиотеки для Panorama

CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -std=c99
LDFLAGS = -shared
LIBS = -lhackrf -lfftw3f -lm

TARGET = libhackrf_qsa.so
SOURCE = hackrf_master.c
HEADER = hackrf_master.h
OBJECT = hackrf_master.o

# Пути установки
PREFIX = /usr/local
LIBDIR = $(PREFIX)/lib
INCDIR = $(PREFIX)/include

# Цели по умолчанию
all: $(TARGET)

# Основная библиотека
$(TARGET): $(OBJECT)
	$(CC) $(LDFLAGS) -o $@ $< $(LIBS)

# Объектный файл
$(OBJECT): $(SOURCE) $(HEADER)
	$(CC) $(CFLAGS) -c $< -o $@

# Установка
install: $(TARGET)
	install -d $(LIBDIR)
	install -m 755 $(TARGET) $(LIBDIR)
	install -d $(INCDIR)
	install -m 644 $(HEADER) $(INCDIR)
	ldconfig

# Установка в пользовательскую директорию
install-user: $(TARGET)
	install -d $(HOME)/.local/lib
	install -m 755 $(TARGET) $(HOME)/.local/lib
	install -d $(HOME)/.local/include
	install -m 644 $(HEADER) $(HOME)/.local/include

# Очистка
clean:
	rm -f $(OBJECT) $(TARGET)

# Полная очистка
distclean: clean
	rm -f *.o *.so *.a

# Проверка зависимостей
check-deps:
	@echo "Проверка зависимостей..."
	@pkg-config --exists libhackrf && echo "✓ libhackrf найдена" || echo "✗ libhackrf не найдена"
	@pkg-config --exists fftw3 && echo "✓ fftw3 найдена" || echo "✗ fftw3 не найдена"
	@echo "Проверка заголовочных файлов..."
	@test -f /usr/include/hackrf.h && echo "✓ hackrf.h найден" || echo "✗ hackrf.h не найден"
	@test -f /usr/include/fftw3.h && echo "✓ fftw3.h найден" || echo "✗ fftw3.h не найден"

# Установка зависимостей (Ubuntu/Debian)
install-deps:
	@echo "Установка зависимостей..."
	sudo apt-get update
	sudo apt-get install -y build-essential pkg-config
	sudo apt-get install -y libhackrf-dev libfftw3-dev

# Тестирование
test: $(TARGET)
	@echo "Тестирование библиотеки..."
	@echo "✓ Библиотека собрана: $(TARGET)"
	@echo "✓ Размер: $(shell ls -lh $(TARGET) | awk '{print $$5}')"
	@echo "✓ Зависимости:"
	@ldd $(TARGET) | grep -E "(hackrf|fftw3)"

# Информация о системе
info:
	@echo "=== Информация о системе ==="
	@echo "Компилятор: $(CC)"
	@echo "Версия GCC: $(shell $(CC) --version | head -1)"
	@echo "Архитектура: $(shell uname -m)"
	@echo "ОС: $(shell uname -s) $(shell uname -r)"
	@echo "Python: $(shell python3 --version 2>/dev/null || echo "не найден")"
	@echo "CFFI: $(shell python3 -c "import cffi; print(cffi.__version__)" 2>/dev/null || echo "не найден")"

# Помощь
help:
	@echo "Доступные цели:"
	@echo "  all          - собрать библиотеку (по умолчанию)"
	@echo "  install      - установить в систему (требует sudo)"
	@echo "  install-user - установить в пользовательскую директорию"
	@echo "  clean        - удалить объектные файлы"
	@echo "  distclean    - полная очистка"
	@echo "  check-deps   - проверить зависимости"
	@echo "  install-deps - установить зависимости (Ubuntu/Debian)"
	@echo "  test         - протестировать сборку"
	@echo "  info         - информация о системе"
	@echo "  help         - показать эту справку"

# Файлы для отслеживания
.PHONY: all install install-user clean distclean check-deps install-deps test info help

# Правила для автоматических зависимостей
-include $(OBJECT:.o=.d)

%.d: %.c
	@set -e; rm -f $@; \
	$(CC) -MM $(CFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
