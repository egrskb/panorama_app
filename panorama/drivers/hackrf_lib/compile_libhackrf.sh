#!/usr/bin/env bash
set -euo pipefail
echo "Компиляция libhackrf_qsa.so…"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/hq_sweep.c"
OUT="$SCRIPT_DIR/libhackrf_qsa.so"

for pc in libhackrf fftw3f; do
  if ! pkg-config --exists "$pc"; then
    echo "Ошибка: не найден $pc (sudo apt install libhackrf-dev libfftw3-dev)"
    exit 1
  fi
done

if [[ ! -f "$SRC" ]]; then
  echo "Ошибка: не найден исходник: $SRC"
  ls -la "$SCRIPT_DIR"
  exit 1
fi

gcc -shared -fPIC -O3 \
    $(pkg-config --cflags libhackrf fftw3f) \
    -o libhackrf_qsa.so \
    hq_sweep.c \
    $(pkg-config --libs libhackrf fftw3f) \
    -lm -pthread

echo "✓ Успешно: $OUT"
