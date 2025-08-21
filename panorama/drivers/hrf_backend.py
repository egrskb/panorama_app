#!/usr/bin/env python3
"""
Python CFFI wrapper для libhackrf_master.so
"""

from __future__ import annotations
from typing import Optional, Callable, List, Dict, Any
import os
from pathlib import Path
from cffi import FFI

# Проверка доступности C библиотеки
try:
    current_dir = Path(__file__).parent
    lib_path = current_dir / "hackrf_master" / "build" / "libhackrf_master.so"
    C_LIB_AVAILABLE = lib_path.exists()
except Exception:
    C_LIB_AVAILABLE = False

class HackRFMasterDataReader:
    """Класс для получения данных от HackRF Master без использования callback."""
    
    def __init__(self, hackrf_master: "HackRFMaster"):
        self.hackrf_master = hackrf_master
        self.last_sweep_tile = None
        self.last_peak = None
        self.last_error_message = None
        
    def get_last_sweep_tile(self):
        """Получает последний sweep tile от C библиотеки."""
        try:
            # Создаем структуру для получения данных
            tile = self.hackrf_master.ffi.new("sweep_tile_t*")
            
            # Вызываем C функцию
            result = self.hackrf_master.lib.hackrf_master_get_last_sweep_tile(tile)
            
            if result == 0:
                # Данные получены успешно
                tile_data = {
                    'f_start': float(tile.f_start),
                    'bin_hz': float(tile.bin_hz),
                    'count': int(tile.count),
                    'power': [float(tile.power[i]) for i in range(tile.count)] if tile.power else [],
                    't0': float(tile.t0)
                }
                self.last_sweep_tile = tile_data
                return tile_data
            else:
                # Нет новых данных
                return None
                
        except Exception as e:
            print(f"DEBUG: Error getting sweep tile: {e}")
            return None
    
    def get_last_peak(self):
        """Получает последний обнаруженный пик от C библиотеки."""
        try:
            # Создаем структуру для получения данных
            peak = self.hackrf_master.ffi.new("detected_peak_t*")
            
            # Вызываем C функцию
            result = self.hackrf_master.lib.hackrf_master_get_last_peak(peak)
            
            if result == 0:
                # Данные получены успешно
                peak_data = {
                    'f_peak': float(peak.f_peak),
                    'snr_db': float(peak.snr_db),
                    'bin_hz': float(peak.bin_hz),
                    't0': float(peak.t0),
                    'status': int(peak.status)
                }
                self.last_peak = peak_data
                return peak_data
            else:
                # Нет новых данных
                return None
                
        except Exception as e:
            print(f"DEBUG: Error getting peak: {e}")
            return None
    
    def get_last_error_message(self):
        """Получает последнее сообщение об ошибке от C библиотеки."""
        try:
            # Создаем буфер для сообщения
            message_buffer = self.hackrf_master.ffi.new("char[]", 256)
            
            # Вызываем C функцию
            result = self.hackrf_master.lib.hackrf_master_get_last_error_message(message_buffer, 256)
            
            if result == 0:
                # Данные получены успешно
                message = self.hackrf_master.ffi.string(message_buffer).decode('utf-8')
                self.last_error_message = message
                return message
            else:
                # Нет новых данных
                return None
                
        except Exception as e:
            print(f"DEBUG: Error getting error message: {e}")
            return None


class HackRFMaster:
    """Wrapper for libhackrf_master.so via CFFI."""
    
    _enumerating = False  # Класс-переменная для предотвращения рекурсии
    
    def __init__(self):
        # Проверяем доступность библиотеки
        if not C_LIB_AVAILABLE:
            raise FileNotFoundError("C библиотека libhackrf_master.so недоступна")
        
        # Создаем FFI
        self.ffi = FFI()
        
        # Определяем функции
        self.ffi.cdef("""
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
            
            // Функции получения данных (вместо callback)
            int hackrf_master_get_last_sweep_tile(sweep_tile_t* tile_out);
            int hackrf_master_get_last_peak(detected_peak_t* peak_out);
            int hackrf_master_get_last_error_message(char* message_out, int max_len);
        """, override=True)
        
        # Загружаем библиотеку
        current_dir = Path(__file__).parent
        lib_path = current_dir / "hackrf_master" / "build" / "libhackrf_master.so"
        self.lib = self.ffi.dlopen(str(lib_path))
        
        # Callback функции (больше не используем)
        # self._sweep_callback: Optional[Callable] = None
        # self._peak_callback: Optional[Callable] = None
        # self._error_callback: Optional[Callable] = None
        self._is_initialized = False
        
        # Data reader для получения данных без callback
        self.data_reader = None
        
        # НЕ инициализируем SDR здесь - только загружаем библиотеку
        # self._is_initialized = False  # Уже установлено выше
                
    def initialize_sdr(self):
        """Инициализирует SDR устройство."""
        if self._is_initialized:
            return True
            
        try:
            rc = self.lib.hackrf_master_init()
            if rc != 0:
                raise RuntimeError(f"Failed to initialize HackRF Master, return code: {rc}")
            
            self._is_initialized = True
            
            # Инициализируем data reader
            self.data_reader = HackRFMasterDataReader(self)
            print("DEBUG: Data reader initialized successfully")
            
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HackRF Master: {e}")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def deinitialize_sdr(self):
        """Деинициализирует SDR устройство."""
        try:
            if self._is_initialized:
                # Сначала останавливаем sweep если он запущен
                if self.is_running():
                    self.stop_sweep()
                
                # Затем очищаем
                self.lib.hackrf_master_cleanup()
                self._is_initialized = False
        except Exception:
            pass

    def cleanup(self):
        """Очищает ресурсы HackRF."""
        self.deinitialize_sdr()

    def is_initialized(self) -> bool:
        """Проверяет инициализирован ли HackRF."""
        return self._is_initialized

    def start_sweep(self, start_hz, stop_hz, bin_hz, dwell_ms, step_hz=None, avg_count=1, min_snr_db=10.0, min_peak_distance_bins=2):
        """Запускает sweep с проверкой состояния."""
        if not self._is_initialized:
            # Автоматически инициализируем SDR если нужно
            if not self.initialize_sdr():
                raise RuntimeError("Failed to initialize SDR")
        
        # Проверяем не запущен ли уже sweep
        if self.is_running():
            raise RuntimeError("Sweep already running. Stop it first.")
        
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
            raise RuntimeError(f"Failed to start sweep, return code: {rc}")
        return True

    def stop_sweep(self):
        """Останавливает sweep с проверкой состояния."""
        if not self._is_initialized:
            return True
        
        try:
            rc = self.lib.hackrf_master_stop_sweep()
            if rc != 0:
                raise RuntimeError(f"Failed to stop sweep, return code: {rc}")
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to stop sweep: {e}")

    def is_running(self):
        """Проверяет запущен ли sweep."""
        if not self._is_initialized:
            return False
        return bool(self.lib.hackrf_master_is_running())

    def set_peak_detection_params(self, min_snr_db, min_peak_distance_bins):
        rc = self.lib.hackrf_master_set_peak_detection_params(float(min_snr_db), int(min_peak_distance_bins))
        if rc != 0:
            raise RuntimeError("Failed to set peak detection parameters")
        return True

    def enumerate_devices(self):
        """Перечисляет доступные HackRF устройства."""
        print(f"DEBUG: HackRFMaster.enumerate_devices called")
        
        if not C_LIB_AVAILABLE:
            print("DEBUG: C library not available")
            return []
        
        try:
            maxn = 16
            arr = self.ffi.new("hackrf_devinfo_t[]", maxn)
            n = int(self.lib.hackrf_master_enumerate(arr, maxn))
            
            print(f"DEBUG: C library returned {n} devices")
            
            if n <= 0:
                print("DEBUG: No devices found by C library")
                return []
            
            out = []
            seen_serials = set()
            
            for i in range(n):
                serial_bytes = arr[i].serial
                serial = self.ffi.string(serial_bytes).decode("utf-8").strip()
                
                print(f"DEBUG: Device {i}: serial='{serial}', len={len(serial)}")
                
                # Очень строгая проверка
                if not serial:
                    print(f"DEBUG: Skipping empty serial")
                    continue
                    
                if serial in ["", "default", "0", "none", "N/A"]:
                    print(f"DEBUG: Skipping invalid serial: {serial}")
                    continue
                
                # HackRF серийники - это 32 символа hex (16 байт)
                # Пример: 0000000000000000916463c829709f43
                if len(serial) != 32:
                    print(f"DEBUG: Skipping serial with wrong length: {len(serial)}")
                    continue
                    
                # Проверяем что все символы - hex
                try:
                    int(serial, 16)
                except ValueError:
                    print(f"DEBUG: Skipping non-hex serial: {serial}")
                    continue
                
                # Проверяем что не все нули
                if serial == "0" * 32:
                    print(f"DEBUG: Skipping all-zeros serial")
                    continue
                
                if serial not in seen_serials:
                    seen_serials.add(serial)
                    out.append(serial)
                    print(f"DEBUG: Added valid device: {serial}")
            
            print(f"DEBUG: Returning {len(out)} valid devices")
            return out
            
        except Exception as e:
            print(f"DEBUG: Exception in enumerate_devices: {e}")
            import traceback
            traceback.print_exc()
            return []

    def set_serial(self, serial: str | None):
        if serial is None:
            self.lib.hackrf_master_set_serial(self.ffi.NULL)
        else:
            self.lib.hackrf_master_set_serial(serial.encode("utf-8"))

    def probe(self) -> bool:
        """Проверяет доступность устройства."""
        try:
            # Если SDR не инициализирован, считаем что устройство доступно
            if not self._is_initialized:
                return True
            
            return bool(self.lib.hackrf_master_probe() == 0)
        except Exception:
            # В случае ошибки считаем что устройство доступно
            return True

    def get_peak_count(self):
        return int(self.lib.hackrf_master_get_peak_count())

    def get_peaks(self, max_count=100):
        arr = self.ffi.new("detected_peak_t[]", int(max_count))
        n = int(self.lib.hackrf_master_get_peaks(arr, int(max_count)))
        out: list[dict] = []
        for i in range(n):
            p = arr[i]
            out.append({"f_peak": p.f_peak, "snr_db": p.snr_db, "bin_hz": p.bin_hz, "t0": p.t0, "status": p.status})
        return out

    def get_stats(self):
        s = self.ffi.new("master_stats_t*")
        rc = self.lib.hackrf_master_get_stats(s)
        if rc != 0:
            raise RuntimeError("Failed to get stats")
        return {
            "sweep_count": s.sweep_count,
            "peak_count": s.peak_count,
            "last_sweep_time": s.last_sweep_time,
            "avg_sweep_time": s.avg_sweep_time,
            "error_count": s.error_count,
        }

    def _setup_callbacks(self):
        """Настраивает callback функции."""
        # Проверяем что SDR инициализирован
        if not self._is_initialized:
            raise RuntimeError("Cannot setup callbacks - SDR not initialized")
            
        @self.ffi.callback("void(const sweep_tile_t*)")
        def _sweep_cb(tile):
            if self.data_reader: # Use self.data_reader here
                data = {
                    "f_start": tile.f_start,
                    "bin_hz": tile.bin_hz,
                    "count": tile.count,
                    "power": [tile.power[i] for i in range(tile.count)],
                    "t0": tile.t0,
                }
                self.data_reader.last_sweep_tile = data # Update data reader

        @self.ffi.callback("void(const detected_peak_t*)")
        def _peak_cb(peak):
            if self.data_reader: # Use self.data_reader here
                self.data_reader.last_peak = {"f_peak": peak.f_peak, "snr_db": peak.snr_db, "bin_hz": peak.bin_hz, "t0": peak.t0, "status": peak.status} # Update data reader

        @self.ffi.callback("void(const char*)")
        def _err_cb(msg):
            if self.data_reader: # Use self.data_reader here
                self.data_reader.last_error_message = self.ffi.string(msg).decode("utf-8") # Update data reader

        self.lib.hackrf_master_set_sweep_callback(_sweep_cb)
        self.lib.hackrf_master_set_peak_callback(_peak_cb)
        self.lib.hackrf_master_set_error_callback(_err_cb)
        self._sweep_callback_handle = _sweep_cb
        self._peak_callback_handle = _peak_cb
        self._error_callback_handle = _err_cb

    def set_sweep_callback(self, callback):
        """Устанавливает callback для sweep данных."""
        print(f"DEBUG: Python set_sweep_callback called with: {callback}")
        print(f"DEBUG: Callback type: {type(callback)}")
        print(f"DEBUG: Callback callable: {callable(callback)}")
        
        try:
            # This method is no longer used for callbacks, but kept for compatibility
            # The actual data reading is handled by the data_reader
            print(f"DEBUG: set_sweep_callback: This method is deprecated. Use data_reader.get_last_sweep_tile() instead.")
        except Exception as e:
            print(f"DEBUG: Error setting sweep callback: {e}")
            import traceback
            traceback.print_exc()
            raise

    def set_peak_callback(self, cb: Callable):
        """Устанавливает callback для peak данных."""
        print(f"DEBUG: Python set_peak_callback called with: {cb}")
        print(f"DEBUG: Callback type: {type(cb)}")
        print(f"DEBUG: Callback callable: {callable(cb)}")
        
        try:
            # This method is no longer used for callbacks, but kept for compatibility
            # The actual data reading is handled by the data_reader
            print(f"DEBUG: set_peak_callback: This method is deprecated. Use data_reader.get_last_peak() instead.")
        except Exception as e:
            print(f"DEBUG: Error setting peak callback: {e}")
            import traceback
            traceback.print_exc()
            raise

    def set_error_callback(self, cb: Callable):
        """Устанавливает callback для error сообщений."""
        print(f"DEBUG: Python set_error_callback called with: {cb}")
        print(f"DEBUG: Callback type: {type(cb)}")
        print(f"DEBUG: Callback callable: {callable(cb)}")
        
        try:
            # This method is no longer used for callbacks, but kept for compatibility
            # The actual data reading is handled by the data_reader
            print(f"DEBUG: set_error_callback: This method is deprecated. Use data_reader.get_last_error_message() instead.")
        except Exception as e:
            print(f"DEBUG: Error setting error callback: {e}")
            import traceback
            traceback.print_exc()
            raise


__all__ = ["HackRFMaster"]
