"""
backend.py - Обновленная интеграция с полноценным C-бэкендом
Теперь C-код выполняет все расчеты, Python только отображает
"""

from __future__ import annotations
import os, threading, time
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from PyQt5 import QtCore
import numpy as np
from cffi import FFI

from panorama.drivers.base import SourceBackend, SweepConfig


def _find_library() -> str:
    """Ищем библиотеку в разных местах."""
    here = os.path.abspath(os.path.dirname(__file__))
    
    names = ["libhackrf_master.so", "libhackrf_master.dylib", "hackrf_master.dll"]
    
    candidates = [
        os.path.join(here, "hackrf_master", name) for name in names
    ] + [
        os.path.join(here, "..", "drivers", "hackrf_master", name) for name in names
    ] + names
    
    for path in candidates:
        if os.path.exists(path):
            return path
    
    # Fallback
    return "libhackrf_master.so"


class HackRFQSABackend(SourceBackend):
    """
    Источник данных через полноценный C-бэкенд с расчетами спектра.
    C-код выполняет FFT, нормализацию, калибровку и выдает готовый спектр.
    """

    def __init__(self, serial_suffix: Optional[str] = None, logger=None, parent=None):
        super().__init__(parent)
        
        # Сохраняем logger
        self.log = logger
        
        self._ffi = FFI()
        self._define_interface()
        
        lib_path = _find_library()
        try:
            self._lib = self._ffi.dlopen(lib_path)
            self._emit_status(f"[HackRF Master] Loaded library: {lib_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load HackRF Master library: {e}")
        
        self._serial = serial_suffix
        self._worker: Optional[_MasterWorker] = None
        self._running = False
        self._configured = False
        
        # Параметры по умолчанию
        self._segment_mode = 4
        self._ema_alpha = 0.25
    
    def __del__(self):
        """Деструктор для автоматической очистки ресурсов."""
        try:
            if hasattr(self, '_running') and self._running:
                self.stop()
            if hasattr(self, '_lib') and self._lib:
                self._lib.hq_close()
        except:
            pass  # Игнорируем ошибки в деструкторе
        
    def _define_interface(self):
        """Определяем интерфейс C-библиотеки."""
        self._ffi.cdef("""
            // Типы колбэков
            typedef void (*hq_legacy_cb)(
                const double* freqs_hz,
                const float*  powers_dbm,
                int           n,
                uint64_t      center_hz,
                void*         user
            );
            
            typedef void (*hq_multi_segment_cb)(
                const void* segments,
                int         segment_count,
                double      fft_bin_width_hz,
                uint64_t    center_hz,
                void*       user
            );
            
            // Основные функции
            int  hq_open(const char* serial_suffix);
            void hq_close(void);
            int  hq_configure(double f_start_mhz, double f_stop_mhz, double bin_hz,
                              int lna_db, int vga_db, int amp_on);
            int  hq_start(hq_legacy_cb cb, void* user);
            int  hq_start_multi_segment(hq_multi_segment_cb cb, void* user);
            int  hq_stop(void);
            const char* hq_last_error(void);
            
            // Новые функции для полной интеграции
            int  hq_get_master_spectrum(double* freqs_hz, float* powers_dbm, int max_points);
            void hq_set_ema_alpha(float alpha);
            void hq_set_detector_params(float threshold_offset_db, int min_width_bins,
                                       int min_sweeps, float timeout_sec);
            int  hq_set_segment_mode(int mode);
            int  hq_get_segment_mode(void);
            int  hq_set_fft_size(int size);
            int  hq_get_fft_size(void);
            
            // Калибровка
            int  hq_load_calibration(const char* csv_path);
            int  hq_enable_calibration(int enable);
            int  hq_get_calibration_status(void);
            
            // Перечисление устройств
            int  hq_device_count(void);
            int  hq_get_device_serial(int idx, char* out, int cap);
        """)
        
        # Создаем колбэк для получения данных от C
        self._spectrum_callback = self._ffi.callback(
            "void(const double*, const float*, int, uint64_t, void*)",
            self._on_spectrum_data
        )
    
    def start(self, config: SweepConfig):
        """Запускает sweep с полными расчетами в C."""
        if self._running:
            self.status.emit("Already running")
            return
        
        # Используем серийник из конфига или из конструктора
        serial = config.serial or self._serial
        if not serial:
            # Пытаемся найти первое доступное устройство
            serials = self.list_serials()
            if serials:
                serial = serials[0]  # Берем полный серийник
                self.status.emit(f"Using device: {serial}")
            else:
                self.error.emit("No HackRF devices found")
                return
        
        # Открываем устройство
        try:
            # Передаем полный серийник или NULL для первого доступного
            serial_ptr = serial.encode('utf-8') if serial else self._ffi.NULL
            r = self._lib.hq_open(serial_ptr)
            if r != 0:
                error_msg = self._get_last_error()
                self.error.emit(f"Failed to open device: {error_msg}")
                return
            self.status.emit(f"Successfully opened HackRF device")
        except Exception as e:
            self.error.emit(f"Exception opening device: {e}")
            return
        
        # Конфигурируем
        try:
            r = self._lib.hq_configure(
                config.freq_start_hz / 1e6,
                config.freq_end_hz / 1e6,
                config.bin_hz,
                config.lna_db,
                config.vga_db,
                1 if config.amp_on else 0
            )
            if r != 0:
                error_msg = self._get_last_error()
                self.error.emit(f"Failed to configure: {error_msg}")
                self._lib.hq_close()
                return
            
            self._configured = True
            
            # Устанавливаем параметры обработки
            self._lib.hq_set_segment_mode(self._segment_mode)
            self._lib.hq_set_ema_alpha(self._ema_alpha)
            
            # Загружаем калибровку если доступна
            self._auto_load_calibration()
            
        except Exception as e:
            self.error.emit(f"Exception configuring: {e}")
            self._lib.hq_close()
            return
        
        # Запускаем worker для чтения спектра
        self._worker = _MasterWorker(self, config)
        self._worker.finished_sig.connect(self._on_worker_finished)
        self._worker.start()
        
        self._running = True
        self.started.emit()
        self.status.emit("HackRF Master started with C processing")
    
    def stop(self):
        """Останавливает sweep."""
        if not self._running:
            return
        
        self.status.emit("Stopping...")
        
        if self._worker:
            self._worker.stop()
            self._worker.wait(2000)
            self._worker = None
        
        try:
            self._lib.hq_stop()
            self.status.emit("C code stopped")
        except Exception as e:
            self.error.emit(f"Error stopping C code: {e}")
        
        try:
            self._lib.hq_close()
            self.status.emit("Device closed")
        except Exception as e:
            self.error.emit(f"Error closing device: {e}")
        
        # Сбрасываем состояние
        self._running = False
        self._configured = False
        self._serial = None
        
        self.finished.emit(0)
        self.status.emit("Stopped")
    
    def is_running(self) -> bool:
        return self._running
    
    def list_serials(self) -> List[str]:
        """Получает список серийных номеров устройств."""
        try:
            count = self._lib.hq_device_count()
            if count <= 0:
                return []
            
            serials = []
            for i in range(count):
                try:
                    buf = self._ffi.new("char[128]")
                    if self._lib.hq_get_device_serial(i, buf, 127) == 0:
                        serial = self._ffi.string(buf).decode('utf-8')
                        if serial and serial != "0000000000000000":
                            serials.append(serial)
                except Exception:
                    continue
                    
            return serials
            
        except Exception as e:
            self._emit_error(f"Error listing devices: {e}")
            return []
    
    def load_calibration(self, csv_path: str) -> bool:
        """Загружает калибровку из CSV файла."""
        try:
            path_bytes = csv_path.encode('utf-8')
            r = self._lib.hq_load_calibration(path_bytes)
            if r == 0:
                self._lib.hq_enable_calibration(1)
                self.status.emit(f"Calibration loaded from {csv_path}")
                return True
            else:
                error_msg = self._get_last_error()
                self.error.emit(f"Failed to load calibration: {error_msg}")
                return False
        except Exception as e:
            self.error.emit(f"Exception loading calibration: {e}")
            return False
    
    def set_calibration_enabled(self, enabled: bool):
        """Включает/выключает калибровку."""
        try:
            self._lib.hq_enable_calibration(1 if enabled else 0)
            status = "enabled" if enabled else "disabled"
            self.status.emit(f"Calibration {status}")
        except Exception:
            pass
    
    def calibration_loaded(self) -> bool:
        """Проверяет загружена ли калибровка."""
        try:
            return bool(self._lib.hq_get_calibration_status())
        except Exception:
            return False
    
    def set_segment_mode(self, mode: int):
        """Устанавливает режим сегментов (2 или 4)."""
        if mode not in [2, 4]:
            self.error.emit("Invalid segment mode (must be 2 or 4)")
            return
        
        self._segment_mode = mode
        if self._configured:
            try:
                self._lib.hq_set_segment_mode(mode)
                self.status.emit(f"Segment mode set to {mode}")
            except Exception:
                pass
    
    def set_ema_alpha(self, alpha: float):
        """Устанавливает коэффициент EMA фильтрации."""
        if alpha < 0.01 or alpha > 1.0:
            return
        
        self._ema_alpha = alpha
        if self._configured:
            try:
                self._lib.hq_set_ema_alpha(alpha)
            except Exception:
                pass
    
    def set_detector_params(self, threshold_db: float = -80.0, min_width: int = 3,
                            min_sweeps: int = 2, timeout: float = 5.0):
        """Устанавливает параметры детектора пиков."""
        if self._configured:
            try:
                self._lib.hq_set_detector_params(threshold_db, min_width, min_sweeps, timeout)
                self.status.emit(f"Detector params updated: threshold={threshold_db:.1f}dB")
            except Exception:
                pass
    
    def _get_last_error(self) -> str:
        """Получает последнюю ошибку из C-библиотеки."""
        try:
            err_ptr = self._lib.hq_last_error()
            if err_ptr != self._ffi.NULL:
                return self._ffi.string(err_ptr).decode('utf-8')
        except Exception:
            pass
        return "Unknown error"
    
    def _auto_load_calibration(self):
        """Автоматически ищет и загружает калибровку."""
        cal_paths = [
            Path.home() / ".panorama" / "calibration.csv",
            Path(__file__).parent / "hackrf_master" / "calibration.csv",
            Path.cwd() / "calibration.csv",
        ]
        
        for path in cal_paths:
            if path.exists():
                if self.load_calibration(str(path)):
                    self._emit_status(f"[HackRF Master] Auto-loaded calibration from {path}")
                    break
    
    @QtCore.pyqtSlot(int, str)
    def _on_worker_finished(self, code: int, msg: str):
        if code != 0 and msg:
            self.error.emit(msg)
        self._running = False
        self.finished.emit(code)
    
    def _emit_status(self, msg: str):
        """Эмитит статус и логирует."""
        if self.log:
            self.log.info(msg)
        print(msg)
        self.status.emit(msg)
    
    def _emit_error(self, msg: str):
        """Эмитит ошибку и логирует."""
        if self.log:
            self.log.error(msg)
        print(f"ERROR: {msg}")
        self.error.emit(msg)
    
    def _on_spectrum_data(self, freqs_ptr, powers_ptr, n, center_hz, user):
        """Колбэк от C кода для получения данных спектра."""
        try:
            # Копируем данные из C в numpy массивы
            freqs = np.frombuffer(
                self._ffi.buffer(freqs_ptr, n * 8),
                dtype=np.float64
            ).copy()
            
            powers = np.frombuffer(
                self._ffi.buffer(powers_ptr, n * 4),
                dtype=np.float32
            ).copy()
            
            # Отправляем данные в UI
            self.fullSweepReady.emit(freqs, powers)
            
            # Статус
            if n > 0:
                max_power = np.max(powers)
                self.status.emit(f"Spectrum: {n} points, max={max_power:.1f}dBm")
                
        except Exception as e:
            self.error.emit(f"Error in spectrum callback: {e}")


class _MasterWorker(QtCore.QThread):
    """Worker thread для чтения готового спектра из C."""
    
    finished_sig = QtCore.pyqtSignal(int, str)
    
    def __init__(self, backend: HackRFQSABackend, config: SweepConfig):
        super().__init__(backend)
        self._backend = backend
        self._config = config
        self._stop_flag = threading.Event()
        
        # Вычисляем количество точек спектра
        freq_range = config.freq_end_hz - config.freq_start_hz
        self._n_points = int(freq_range / config.bin_hz) + 1
        if self._n_points > 100000:  # Ограничение
            self._n_points = 100000
    
    def stop(self):
        self._stop_flag.set()
    
    def run(self):
        """Основной цикл чтения спектра из C."""
        code, msg = 0, ""
        
        try:
            # Запускаем C-код с колбэком для получения данных
            r = self._backend._lib.hq_start(self._backend._spectrum_callback, self._backend._ffi.NULL)
            if r != 0:
                msg = self._backend._get_last_error()
                raise RuntimeError(f"Failed to start: {msg}")
            
            # Ждем завершения или остановки
            while not self._stop_flag.is_set():
                self.msleep(100)  # Проверяем каждые 100 мс
        
        except Exception as e:
            code, msg = 1, str(e)
        
        finally:
            self.finished_sig.emit(code, msg)

# Экспортируем публичные классы
__all__ = ['HackRFQSABackend', 'SweepConfig']