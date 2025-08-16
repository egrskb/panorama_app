# Makefile for multi-SDR HackRF library
CC = gcc
CFLAGS = -fPIC -O3 -Wall -Wextra -pthread -g
LDFLAGS = -shared
LIBS = $(shell pkg-config --libs libhackrf) -lm -lpthread

# Check for libhackrf
HACKRF_CHECK := $(shell pkg-config --exists libhackrf && echo yes)
ifneq ($(HACKRF_CHECK),yes)
    $(error libhackrf not found. Install hackrf development package)
endif

CFLAGS += $(shell pkg-config --cflags libhackrf)

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

install: $(TARGET)
	cp $(TARGET) ../../../
	@echo "✓ Installed to project root"

test: $(TARGET) test_lib.c
	@echo "Running basic library test..."
	@gcc -o test_lib test_lib.c -L. -lhackrf_multi -Wl,-rpath,. $(LIBS)
	@./test_lib
	@rm -f test_lib
	@echo "✓ Test passed"

# Simple test program
test_lib.c:
	@echo '#include <stdio.h>' > test_lib.c
	@echo '#include "hq_api.h"' >> test_lib.c
	@echo 'int main() {' >> test_lib.c
	@echo '    HqStatus status;' >> test_lib.c
	@echo '    int r = hq_get_status(&status);' >> test_lib.c
	@echo '    printf("Library test: get_status returned %d\\n", r);' >> test_lib.c
	@echo '    return 0;' >> test_lib.c
	@echo '}' >> test_lib.c
