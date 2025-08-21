#!/usr/bin/env python3
"""
Python CFFI wrapper для libhackrf_master.so
"""

from __future__ import annotations
from typing import Optional, Callable, List, Dict, Any
import os
from pathlib import Path
from cffi import FFI

class HackRFMaster:
    """Wrapper for libhackrf_master.so via CFFI."""
    
    def __init__(self):
        # Путь к библиотеке
        current_dir = Path(__file__).parent
        lib_path = current_dir / "hackrf_master" / "build" / "libhackrf_master.so"
        
        if not lib_path.exists():
            raise FileNotFoundError(f"Библиотека не найдена: {lib_path}")
        
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
        """, override=True)
        
        # Загружаем библиотеку
        self.lib = self.ffi.dlopen(str(lib_path))
        
        # Callback функции
        self._sweep_callback: Optional[Callable] = None
        self._peak_callback: Optional[Callable] = None
        self._error_callback: Optional[Callable] = None
        self._is_initialized = False
        
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
        try:
            # Если SDR не инициализирован, возвращаем базовый список
            if not self._is_initialized:
                return ["default"]
            
            maxn = 16
            arr = self.ffi.new("hackrf_devinfo_t[]", maxn)
            n = int(self.lib.hackrf_master_enumerate(arr, maxn))
            out: list[str] = []
            seen_serials = set()  # Для отслеживания уникальности
            
            for i in range(n):
                serial = self.ffi.string(arr[i].serial).decode("utf-8")
                # Добавляем только уникальные серийные номера
                if serial not in seen_serials:
                    seen_serials.add(serial)
                    out.append(serial)
            
            # Если ничего не найдено, возвращаем default
            return out if out else ["default"]
        except Exception:
            return ["default"]

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
            if self._sweep_callback:
                data = {
                    "f_start": tile.f_start,
                    "bin_hz": tile.bin_hz,
                    "count": tile.count,
                    "power": [tile.power[i] for i in range(tile.count)],
                    "t0": tile.t0,
                }
                self._sweep_callback(data)

        @self.ffi.callback("void(const detected_peak_t*)")
        def _peak_cb(peak):
            if self._peak_callback:
                self._peak_callback({"f_peak": peak.f_peak, "snr_db": peak.snr_db, "bin_hz": peak.bin_hz, "t0": peak.t0, "status": peak.status})

        @self.ffi.callback("void(const char*)")
        def _err_cb(msg):
            if self._error_callback:
                self._error_callback(self.ffi.string(msg).decode("utf-8"))

        self.lib.hackrf_master_set_sweep_callback(_sweep_cb)
        self.lib.hackrf_master_set_peak_callback(_peak_cb)
        self.lib.hackrf_master_set_error_callback(_err_cb)
        self._sweep_callback_handle = _sweep_cb
        self._peak_callback_handle = _peak_cb
        self._error_callback_handle = _err_cb

    def set_sweep_callback(self, cb: Callable):
        self._sweep_callback = cb

    def set_peak_callback(self, cb: Callable):
        self._peak_callback = cb

    def set_error_callback(self, cb: Callable):
        self._error_callback = cb


__all__ = ["HackRFMaster"]
