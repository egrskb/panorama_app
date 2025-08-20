#!/usr/bin/env python3
"""
CFFI build script для HackRF Master библиотеки.
Создает Python интерфейс к C библиотеке.
"""

import os
import sys
import subprocess
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    import cffi
except ImportError:
    print("ОШИБКА: CFFI не установлен!")
    print("Установите: pip install cffi")
    sys.exit(1)

def check_library():
    """Проверяет наличие скомпилированной библиотеки."""
    lib_path = Path(__file__).parent / "build" / "libhackrf_master.so"
    if not lib_path.exists():
        print("ОШИБКА: C библиотека не найдена!")
        print("Сначала соберите библиотеку: make")
        return False
    return True

def build_cffi_interface():
    """Создает CFFI интерфейс к C библиотеке."""
    
    # Определяем CFFI интерфейс
    ffi = cffi.FFI()
    
    # Загружаем заголовочный файл
    header_path = Path(__file__).parent / "hackrf_master.h"
    with open(header_path, 'r') as f:
        header_content = f.read()
    
    # Убираем extern "C" для CFFI
    header_content = header_content.replace('extern "C" {', '').replace('}', '')
    
    # Определяем C типы и функции
    ffi.cdef(header_content)
    
    # Путь к библиотеке
    lib_path = str(Path(__file__).parent / "build" / "libhackrf_master.so")
    
    # Создаем интерфейс
    lib = ffi.dlopen(lib_path)
    
    return ffi, lib

def create_python_wrapper():
    """Создает Python wrapper для C библиотеки."""
    wrapper_code = r'''"""
Python wrapper для HackRF Master C библиотеки.
Автоматически сгенерирован CFFI.
"""

import os
from pathlib import Path
from cffi import FFI

HERE = Path(__file__).parent
LIB_PATH = HERE / "build" / "libhackrf_master.so"
HEADER_PATH = HERE / "hackrf_master.h"

if not LIB_PATH.exists():
    raise FileNotFoundError(f"Библиотека не найдена: {LIB_PATH}")
if not HEADER_PATH.exists():
    raise FileNotFoundError(f"Заголовок не найден: {HEADER_PATH}")

ffi = FFI()

# Загружаем заголовок и чистим extern "C"
header = HEADER_PATH.read_text(encoding='utf-8')

# Удаляем все препроцессорные директивы и внешние C-блоки, которые не понимает cdef
lines = []
for raw in header.splitlines():
    line = raw.strip()
    # Препроцессор: пропускаем любые строки, начинающиеся с '#'
    if line.startswith('#'):
        continue
    # extern "C" и отдельно стоящие скобки из него пропускаем
    if 'extern "C"' in line:
        continue
    if line == '{' or line == '}':
        continue
    lines.append(raw)
clean_header = "\n".join(lines)

ffi.cdef(clean_header)
lib = ffi.dlopen(str(LIB_PATH))

class HackRFMaster:
    """Python wrapper для HackRF Master C библиотеки."""

    def __init__(self):
        self.ffi = ffi
        self.lib = lib
        self._sweep_callback = None
        self._peak_callback = None
        self._error_callback = None
        rc = self.lib.hackrf_master_init()
        if rc != 0:
            raise RuntimeError("Failed to initialize HackRF Master")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def cleanup(self):
        self.lib.hackrf_master_cleanup()

    def start_sweep(self, start_hz, stop_hz, bin_hz, dwell_ms, step_hz=None, avg_count=1, min_snr_db=10.0, min_peak_distance_bins=2):
        if not self._sweep_callback:
            self._setup_callbacks()
        cfg = self.ffi.new("sweep_config_t*")
        cfg.start_hz = start_hz
        cfg.stop_hz = stop_hz
        cfg.bin_hz = bin_hz
        cfg.dwell_ms = int(dwell_ms)
        cfg.step_hz = step_hz if step_hz else bin_hz
        cfg.avg_count = int(avg_count)
        cfg.min_snr_db = float(min_snr_db)
        cfg.min_peak_distance_bins = int(min_peak_distance_bins)
        rc = self.lib.hackrf_master_start_sweep(cfg)
        if rc != 0:
            raise RuntimeError("Failed to start sweep")
        return True

    def stop_sweep(self):
        rc = self.lib.hackrf_master_stop_sweep()
        if rc != 0:
            raise RuntimeError("Failed to stop sweep")
        return True

    def is_running(self):
        return bool(self.lib.hackrf_master_is_running())

    def set_peak_detection_params(self, min_snr_db, min_peak_distance_bins):
        rc = self.lib.hackrf_master_set_peak_detection_params(float(min_snr_db), int(min_peak_distance_bins))
        if rc != 0:
            raise RuntimeError("Failed to set peak detection parameters")
        return True

    def enumerate_devices(self):
        # Возвращает список строк серийников (пустая строка = по умолчанию)
        maxn = 16
        arr = self.ffi.new("hackrf_devinfo_t[]", maxn)
        n = int(self.lib.hackrf_master_enumerate(arr, maxn))
        out = []
        for i in range(n):
            out.append(self.ffi.string(arr[i].serial).decode('utf-8'))
        return out

    def set_serial(self, serial: str | None):
        if serial is None:
            self.lib.hackrf_master_set_serial(self.ffi.NULL)
        else:
            self.lib.hackrf_master_set_serial(serial.encode('utf-8'))

    def probe(self) -> bool:
        return bool(self.lib.hackrf_master_probe() == 0)

    def get_peak_count(self):
        return int(self.lib.hackrf_master_get_peak_count())

    def get_peaks(self, max_count=100):
        arr = self.ffi.new("detected_peak_t[]", int(max_count))
        n = int(self.lib.hackrf_master_get_peaks(arr, int(max_count)))
        out = []
        for i in range(n):
            p = arr[i]
            out.append({'f_peak': p.f_peak, 'snr_db': p.snr_db, 'bin_hz': p.bin_hz, 't0': p.t0, 'status': p.status})
        return out

    def get_stats(self):
        s = self.ffi.new("master_stats_t*")
        rc = self.lib.hackrf_master_get_stats(s)
        if rc != 0:
            raise RuntimeError("Failed to get stats")
        return {
            'sweep_count': s.sweep_count,
            'peak_count': s.peak_count,
            'last_sweep_time': s.last_sweep_time,
            'avg_sweep_time': s.avg_sweep_time,
            'error_count': s.error_count,
        }

    def _setup_callbacks(self):
        @self.ffi.callback("void(const sweep_tile_t*)")
        def _sweep_cb(tile):
            if self._sweep_callback:
                data = {
                    'f_start': tile.f_start,
                    'bin_hz': tile.bin_hz,
                    'count': tile.count,
                    'power': [tile.power[i] for i in range(tile.count)],
                    't0': tile.t0,
                }
                self._sweep_callback(data)

        @self.ffi.callback("void(const detected_peak_t*)")
        def _peak_cb(peak):
            if self._peak_callback:
                self._peak_callback({'f_peak': peak.f_peak, 'snr_db': peak.snr_db, 'bin_hz': peak.bin_hz, 't0': peak.t0, 'status': peak.status})

        @self.ffi.callback("void(const char*)")
        def _err_cb(msg):
            if self._error_callback:
                self._error_callback(self.ffi.string(msg).decode('utf-8'))

        self.lib.hackrf_master_set_sweep_callback(_sweep_cb)
        self.lib.hackrf_master_set_peak_callback(_peak_cb)
        self.lib.hackrf_master_set_error_callback(_err_cb)
        self._sweep_callback_handle = _sweep_cb
        self._peak_callback_handle = _peak_cb
        self._error_callback_handle = _err_cb

    def set_sweep_callback(self, cb):
        self._sweep_callback = cb

    def set_peak_callback(self, cb):
        self._peak_callback = cb

    def set_error_callback(self, cb):
        self._error_callback = cb

__all__ = ['HackRFMaster']
'''

    wrapper_path = Path(__file__).parent / "hackrf_master_wrapper.py"
    with open(wrapper_path, 'w', encoding='utf-8') as f:
        f.write(wrapper_code)
    print(f"✓ Создан Python wrapper: {wrapper_path}")
    return wrapper_path

def main():
    """Главная функция."""
    print("CFFI build script для HackRF Master")
    print("=" * 40)
    
    # Проверяем библиотеку
    if not check_library():
        sys.exit(1)
    
    try:
        # Создаем Python wrapper
        wrapper_path = create_python_wrapper()
        
        print("\n✓ CFFI интерфейс успешно создан!")
        print(f"Python wrapper: {wrapper_path}")
        print("\nТеперь вы можете импортировать HackRFMaster:")
        print("from panorama.drivers.hackrf_master.hackrf_master_wrapper import HackRFMaster")
        
    except Exception as e:
        print(f"ОШИБКА: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
