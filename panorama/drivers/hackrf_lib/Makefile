# Makefile - ОБНОВЛЕНИЕ
CC = gcc
CFLAGS = -fPIC -O3 -Wall -Wextra -pthread -g
LDFLAGS = -shared
LIBS = $(shell pkg-config --libs libhackrf) $(shell pkg-config --libs fftw3f) -lm -lpthread

# Check for libhackrf and fftw3
HACKRF_CHECK := $(shell pkg-config --exists libhackrf && echo yes)
FFTW_CHECK := $(shell pkg-config --exists fftw3f && echo yes)

ifneq ($(HACKRF_CHECK),yes)
    $(error libhackrf not found. Install hackrf development package)
endif

ifneq ($(FFTW_CHECK),yes)
    $(error fftw3f not found. Install libfftw3-dev package)
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

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
	@echo "✓ Cleaned"

# install: $(TARGET)
# 	cp $(TARGET) ../../../
# 	@echo "✓ Installed to project root"

test: $(TARGET)
	@echo "Testing library..."
	@gcc -o test_lib test_lib.c -L. -lhackrf_multi -Wl,-rpath,. $(LIBS)
	@./test_lib
	@rm -f test_lib
	@echo "✓ Test passed"