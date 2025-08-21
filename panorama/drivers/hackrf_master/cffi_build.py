#!/usr/bin/env python3
"""
CFFI build script для HackRF Master
"""

from cffi import FFI
import os
from pathlib import Path

def main():
    print("CFFI build script для HackRF Master")
    print("=" * 40)
    
    # Пути
    current_dir = Path(__file__).parent
    lib_path = current_dir / "build" / "libhackrf_master.so"
    header_path = current_dir / "hackrf_master.h"
    output_path = current_dir / "hackrf_master_wrapper.py"
    
    # Проверяем что библиотека существует
    if not lib_path.exists():
        print(f"❌ Библиотека не найдена: {lib_path}")
        return False
    
    # Создаем FFI
    ffi = FFI()
    
    # Используем встроенные определения вместо парсинга заголовка
    ffi.cdef("""
        // Структуры данных
        typedef struct {
            double f_start;
            double bin_hz;
            int count;
            float* power;
            double t0;
        } sweep_tile_t;
        
        typedef struct {
            double f_peak;
            double snr_db;
            double bin_hz;
            double t0;
            int status;
        } detected_peak_t;
        
        typedef struct {
            double start_hz;
            double stop_hz;
            double bin_hz;
            int dwell_ms;
            double step_hz;
            int avg_count;
            double min_snr_db;
            int min_peak_distance_bins;
        } sweep_config_t;
        
        typedef struct {
            char serial[64];
        } hackrf_devinfo_t;
        
        // Функции
        int hackrf_master_init(void);
        void hackrf_master_cleanup(void);
        int hackrf_master_start_sweep(const sweep_config_t* config);
        int hackrf_master_stop_sweep(void);
        bool hackrf_master_is_running(void);
        int hackrf_master_enumerate(void* out_list, int max_count);
        void hackrf_master_set_serial(const char* serial);
        int hackrf_master_probe(void);
    """, override=True)
    
    # Создаем Python wrapper
    wrapper_code = f'''
"""
Python wrapper для libhackrf_master.so
"""

from cffi import FFI
from typing import Optional, Callable, List, Dict, Any
import os
from pathlib import Path

class HackRFMaster:
    """Wrapper for libhackrf_master.so via CFFI."""
    
    def __init__(self):
        # Путь к библиотеке
        current_dir = Path(__file__).parent
        lib_path = current_dir / "build" / "libhackrf_master.so"
        
        if not lib_path.exists():
            raise FileNotFoundError(f"Библиотека не найдена: {{lib_path}}")
        
        # Создаем FFI
        self.ffi = FFI()
        
        # Определяем функции
        self.ffi.cdef("""
            // Структуры данных
            typedef struct {{
                double f_start;
                double bin_hz;
                int count;
                float* power;
                double t0;
            }} sweep_tile_t;
            
            typedef struct {{
                double f_peak;
                double snr_db;
                double bin_hz;
                double t0;
                int status;
            }} detected_peak_t;
            
            typedef struct {{
                double start_hz;
                double stop_hz;
                double bin_hz;
                int dwell_ms;
                double step_hz;
                int avg_count;
                double min_snr_db;
                int min_peak_distance_bins;
            }} sweep_config_t;
            
            typedef struct {{
                char serial[64];
            }} hackrf_devinfo_t;
            
            // Функции
            int hackrf_master_init(void);
            void hackrf_master_cleanup(void);
            int hackrf_master_start_sweep(const sweep_config_t* config);
            int hackrf_master_stop_sweep(void);
            bool hackrf_master_is_running(void);
            int hackrf_master_enumerate(void* out_list, int max_count);
            void hackrf_master_set_serial(const char* serial);
            int hackrf_master_probe(void);
        """, override=True)
        
        # Загружаем библиотеку
        self.lib = self.ffi.dlopen(str(lib_path))
        
        # Callback функции
        self._sweep_callback: Optional[Callable] = None
        self._peak_callback: Optional[Callable] = None
        self._error_callback: Optional[Callable] = None
        self._is_initialized = False
        
        # Инициализируем
        rc = self.lib.hackrf_master_init()
        if rc != 0:
            raise RuntimeError(f"Failed to initialize HackRF Master, return code: {{rc}}")
        
        self._is_initialized = True
    
    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
    
    def cleanup(self):
        """Очищает ресурсы HackRF."""
        try:
            if self._is_initialized:
                if self.is_running():
                    self.stop_sweep()
                self.lib.hackrf_master_cleanup()
                self._is_initialized = False
        except Exception:
            pass
    
    def is_initialized(self) -> bool:
        """Проверяет инициализирован ли HackRF."""
        return self._is_initialized
    
    def start_sweep(self, start_hz, stop_hz, bin_hz, dwell_ms, step_hz=None, avg_count=1, min_snr_db=10.0, min_peak_distance_bins=2):
        """Запускает sweep с проверкой состояния."""
        if not self._is_initialized:
            raise RuntimeError("HackRF not initialized")
        
        if self.is_running():
            raise RuntimeError("Sweep already running. Stop it first.")
        
        if not self._sweep_callback:
            self._setup_callbacks()
        
        # Создаем структуру конфигурации
        cfg = self.ffi.new("sweep_config_t*")
        cfg.start_hz = float(start_hz)
        cfg.stop_hz = float(stop_hz)
        cfg.bin_hz = float(bin_hz)
        cfg.dwell_ms = int(dwell_ms)
        cfg.step_hz = float(step_hz if step_hz else bin_hz)
        cfg.avg_count = int(avg_count)
        cfg.min_snr_db = float(min_snr_db)
        cfg.min_peak_distance_bins = int(min_peak_distance_bins)
        
        rc = self.lib.hackrf_master_start_sweep(cfg)
        if rc != 0:
            raise RuntimeError(f"Failed to start sweep, return code: {{rc}}")
        return True
    
    def stop_sweep(self):
        """Останавливает sweep с проверкой состояния."""
        if not self._is_initialized:
            return True
        
        try:
            rc = self.lib.hackrf_master_stop_sweep()
            if rc != 0:
                raise RuntimeError(f"Failed to stop sweep, return code: {{rc}}")
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to stop sweep: {{e}}")
    
    def is_running(self):
        """Проверяет запущен ли sweep."""
        if not self._is_initialized:
            return False
        return bool(self.lib.hackrf_master_is_running())
    
    def _setup_callbacks(self):
        """Настраивает callback функции."""
        # Создаем C callback функции
        @self.ffi.callback("void(sweep_tile_t*)")
        def _sweep_cb(tile_ptr):
            if self._sweep_callback:
                # Преобразуем данные в Python dict
                tile = self.ffi.cast("sweep_tile_t*", tile_ptr)
                data = {{
                    'f_start': float(tile.f_start),
                    'bin_hz': float(tile.bin_hz),
                    'count': int(tile.count),
                    'power': [float(tile.power[i]) for i in range(tile.count)],
                    't0': float(tile.t0)
                }}
                self._sweep_callback(data)
        
        @self.ffi.callback("void(detected_peak_t*)")
        def _peak_cb(peak_ptr):
            if self._peak_callback:
                peak = self.ffi.cast("detected_peak_t*", peak_ptr)
                data = {{
                    'f_peak': float(peak.f_peak),
                    'snr_db': float(peak.snr_db),
                    'bin_hz': float(peak.bin_hz),
                    't0': float(peak.t0),
                    'status': int(peak.status)
                }}
                self._peak_callback(data)
        
        @self.ffi.callback("void(const char*)")
        def _error_cb(error_msg):
            if self._error_callback:
                msg = self.ffi.string(error_msg).decode('utf-8')
                self._error_callback(msg)
        
        # Сохраняем ссылки на callback'и
        self._sweep_callback_handle = _sweep_cb
        self._peak_callback_handle = _peak_cb
        self._error_callback_handle = _error_cb
    
    def set_sweep_callback(self, cb: Callable):
        """Устанавливает callback для sweep данных."""
        self._sweep_callback = cb
    
    def set_peak_callback(self, cb: Callable):
        """Устанавливает callback для обнаруженных пиков."""
        self._peak_callback = cb
    
    def set_error_callback(self, cb: Callable):
        """Устанавливает callback для ошибок."""
        self._error_callback = cb
    
    def enumerate_devices(self) -> List[str]:
        """Перечисляет доступные HackRF устройства."""
        try:
            # Создаем массив для результатов
            max_devices = 10
            devices = self.ffi.new("hackrf_devinfo_t[]", max_devices)
            
            count = self.lib.hackrf_master_enumerate(devices, max_devices)
            if count < 0:
                return []
            
            result = []
            for i in range(count):
                serial = self.ffi.string(devices[i].serial).decode('utf-8')
                result.append(serial)
            
            return result
        except Exception:
            return []
    
    def set_serial(self, serial: Optional[str]):
        """Устанавливает серийный номер устройства."""
        if serial:
            self.lib.hackrf_master_set_serial(serial.encode('utf-8'))
        else:
            self.lib.hackrf_master_set_serial(self.ffi.NULL)
    
    def probe(self) -> bool:
        """Проверяет доступность устройства."""
        try:
            return bool(self.lib.hackrf_master_probe())
        except Exception:
            return False

if __name__ == "__main__":
    print("HackRF Master wrapper создан успешно!")
'''
    
    # Записываем wrapper
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(wrapper_code)
    
    print(f"✓ Создан Python wrapper: {output_path}")
    return True

def _clean_header(header_content: str) -> str:
    """Очищает заголовок от проблемных частей."""
    lines = header_content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Убираем комментарии
        if line.strip().startswith('//'):
            continue
        # Убираем пустые строки
        if not line.strip():
            continue
        # Убираем extern "C"
        if 'extern "C"' in line:
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

if __name__ == "__main__":
    success = main()
    if success:
        print("✓ CFFI интерфейс успешно создан!")
        print(f"Python wrapper: {Path(__file__).parent / 'hackrf_master_wrapper.py'}")
    else:
        print("❌ Ошибка создания CFFI интерфейса")
