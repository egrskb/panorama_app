# Makefile для сборки libhackrf_multi
CC = gcc
CFLAGS = -fPIC -O3 -Wall -Wextra -pthread -g -fvisibility=hidden
LDFLAGS = -shared
LIBS = $(shell pkg-config --libs libhackrf) $(shell pkg-config --libs fftw3f) -lm -lpthread

# Проверка зависимостей
HACKRF_CHECK := $(shell pkg-config --exists libhackrf && echo yes)
FFTW_CHECK := $(shell pkg-config --exists fftw3f && echo yes)

ifneq ($(HACKRF_CHECK),yes)
    $(error libhackrf not found. Install: sudo apt install hackrf libhackrf-dev)
endif

ifneq ($(FFTW_CHECK),yes)
    $(error fftw3f not found. Install: sudo apt install libfftw3-dev)
endif

CFLAGS += $(shell pkg-config --cflags libhackrf) $(shell pkg-config --cflags fftw3f)

TARGET = libhackrf_multi.so
SOURCES = hq_init.c hq_master.c hq_slave.c hq_rssi.c \
          hq_grouping.c hq_scheduler.c hq_api.c
OBJECTS = $(SOURCES:.c=.o)
HEADERS = hq_api.h hq_init.h hq_master.h hq_slave.h hq_rssi.h \
          hq_grouping.h hq_scheduler.h

.PHONY: all clean install test

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "✓ Built $(TARGET)"
	@echo "  Size: $$(du -h $(TARGET) | cut -f1)"
	@echo "  Symbols: $$(nm -D $(TARGET) | grep ' T ' | wc -l) exported"

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Явно экспортируем нужные символы
hq_api.o: hq_api.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

hq_master.o: hq_master.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
	@echo "✓ Cleaned"

install: $(TARGET)
	@if [ -w ../../../ ]; then \
		cp $(TARGET) ../../../ && \
		echo "✓ Installed to project root"; \
	else \
		echo "✗ Cannot write to project root"; \
	fi

test: $(TARGET)
	@echo "Testing library loading..."
	@python3 -c "from cffi import FFI; ffi = FFI(); lib = ffi.dlopen('./$(TARGET)'); print('✓ Library loads in Python')" || echo "✗ Failed to load in Python"
	@echo "Testing symbols..."
	@nm -D $(TARGET) | grep -q "hq_open_all" && echo "✓ hq_open_all found" || echo "✗ hq_open_all missing"
	@nm -D $(TARGET) | grep -q "hq_start" && echo "✓ hq_start found" || echo "✗ hq_start missing"
	@nm -D $(TARGET) | grep -q "hq_get_master_spectrum" && echo "✓ hq_get_master_spectrum found" || echo "✗ hq_get_master_spectrum missing"