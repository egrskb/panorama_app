"""
backend.py - Исправленная версия с правильной обработкой диапазонов
Сохранены все детали оригинального файла, исправлена только логика диапазонов
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
    Источник данных через C-бэкенд с правильным управлением ресурсами.
    
    ИСПРАВЛЕНО: Теперь использует пользовательский диапазон вместо жесткого 50-6000 МГц
    """

    def __init__(self, serial_suffix: Optional[str] = None, logger=None, parent=None):
        super().__init__(parent)
        
        self.log = logger
        self._serial = serial_suffix
        self._worker: Optional[_MasterWorker] = None
        self._running = False
        self._configured = False
        self._device_opened = False
        
        # ДОБАВЛЕНО: Храним пользовательские настройки диапазона
        self._user_freq_start_hz = None
        self._user_freq_stop_hz = None
        self._user_bin_hz = None
        
        # Загружаем CFFI интерфейс
        self._ffi = FFI()
        self._define_interface()
        
        # Загружаем библиотеку
        lib_path = _find_library()
        try:
            self._lib = self._ffi.dlopen(lib_path)
            self._emit_status(f"[HackRF Master] Библиотека загружена: {lib_path}")
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить библиотеку HackRF Master: {e}")
        

        
        # Загружаем конфигурацию из JSON
        self._load_config()
        
        # Параметры по умолчанию
        self._segment_mode = 4
        self._ema_alpha = 0.25
        
        # Поля для управления жизненным циклом
        self._closing = False
        self._last_close_ts = 0.0
    
    def __del__(self):
        """Деструктор - гарантирует закрытие устройства."""
        self._cleanup()
    
    def _cleanup(self):
        """Очистка ресурсов."""
        try:
            # Останавливаем воркер если запущен
            if hasattr(self, '_worker') and self._worker and self._worker.isRunning():
                try:
                    self._worker.stop()
                    self._worker.wait(2000)
                except Exception as e:
                    pass
                finally:
                    self._worker = None
            
            # Закрываем устройство только если открыто
            if self._device_opened and hasattr(self, '_lib'):
                try:
                    self._lib.hq_close()
                    self._device_opened = False
                    self._emit_status("[HackRF] Устройство закрыто")
                except Exception:
                    pass
            
            self._running = False
            self._configured = False
            
        except Exception as e:
            pass
    
    def _define_interface(self):
        """Определяем интерфейс C-библиотеки."""
        self._ffi.cdef("""
            // Multi-segment (на будущее)
            typedef struct {
                int     seg_count;
                double* freqs_hz[4];
                float*  pwr_dbm[4];
                int     lens[4];
                double  centers_hz[4];
            } hq_segment_data_t;
            typedef void (*hq_multi_segment_cb)(const hq_segment_data_t* data, void* user);
            
            // Основные функции
            int  hq_open(const char* serial_suffix);
            void hq_close(void);
            int  hq_configure(double f_start_mhz, double f_stop_mhz, double bin_hz,
                              int lna_db, int vga_db, int amp_on);
            int  hq_start_multi_segment(hq_multi_segment_cb cb, void* user);
            // Старт без колбэков
            int  hq_start_no_cb(void);
            int  hq_stop(void);
            const char* hq_last_error(void);
            
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
    
    def _load_config(self):
        """Загружает конфигурацию из JSON."""
        try:
            from panorama.features.settings.sdr_config import SDRConfigManager
            self.config_manager = SDRConfigManager()
            self._emit_status(f"[HackRF] Конфигурация загружена")
        except Exception as e:
            self._emit_error(f"Ошибка загрузки конфигурации: {e}")
            self.config_manager = None
    
    def start(self, config: SweepConfig):
        """Запускает sweep с параметрами из конфигурации."""
        if self._running or self._closing:
            self._emit_status("Старт отклонён: идёт закрытие/уже запущен")
            return

        # Анти-дребезг: не стартуем мгновенно после закрытия
        import time
        if time.time() - self._last_close_ts < 0.2:
            time.sleep(0.2)

        # открыть, если не открыто
        if not self._device_opened:
            if not self._open_device(config.serial or self._serial):
                self._emit_error("Не удалось открыть HackRF")
                return

        if not hasattr(self, "_lib") or not self._device_opened:
            self._emit_error("Устройство не открыто или закрывается")
            return

        # Обновляем конфигурацию из GUI
        if self.config_manager:
            self.config_manager.update_from_gui(
                freq_start_mhz=config.freq_start_hz / 1e6,
                freq_stop_mhz=config.freq_end_hz / 1e6,
                bin_khz=config.bin_hz / 1e3,
                lna_db=config.lna_db,
                vga_db=config.vga_db,
                amp_on=config.amp_on
            )
            self.config_manager.save()
        
        # ИСПРАВЛЕНИЕ: Используем пользовательские настройки диапазона
        self._user_freq_start_hz = config.freq_start_hz
        self._user_freq_stop_hz = config.freq_end_hz
        self._user_bin_hz = config.bin_hz
        
        # Для C библиотеки округляем до кратных 20 МГц (требование HackRF sweep)
        f_start_mhz = int(self._user_freq_start_hz / 1e6)
        f_stop_mhz = int(self._user_freq_stop_hz / 1e6)
        
        # Округляем вниз начало и вверх конец до кратных 20 МГц
        f_start_mhz = (f_start_mhz // 20) * 20
        f_stop_mhz = ((f_stop_mhz + 19) // 20) * 20
        
        if f_start_mhz < 24:  # Минимум для HackRF
            f_start_mhz = 24
        if f_stop_mhz > 6000:  # Максимум для HackRF
            f_stop_mhz = 6000
        
        # Сохраняем расширенный диапазон для C
        self._c_freq_start_hz = f_start_mhz * 1e6
        self._c_freq_stop_hz = f_stop_mhz * 1e6
        
        # Сохраняем параметры усиления для конфигурации
        self._lna_db = config.lna_db
        self._vga_db = config.vga_db
        self._amp_on = config.amp_on
        
        # Передаём в C-бэкенд расширенный диапазон
        if not self._configure_device_with(self._c_freq_start_hz, self._c_freq_stop_hz, self._user_bin_hz):
            self._cleanup()
            return
        
        # Запускаем worker для чтения спектра
        self._worker = _MasterWorker(self, config)
        self._worker.finished_sig.connect(self._on_worker_finished)
        self._worker.start()
        
        self._running = True
        self.started.emit()
        self._emit_status(f"HackRF Master started: user range {self._user_freq_start_hz/1e6:.1f}-{self._user_freq_stop_hz/1e6:.1f} MHz, C range {f_start_mhz}-{f_stop_mhz} MHz")
    
    def stop(self):
        """Останавливает sweep и закрывает устройство."""
        if not self._running and not self._device_opened:
            return
        self._closing = True
        self._emit_status("Остановка...")

        # Остановить воркер аккуратно
        if self._worker:
            try:
                self._worker.stop()
                self._worker.wait(3000)
            except Exception as e:
                self._emit_error(f"Ошибка остановки worker: {e}")
            finally:
                self._worker = None

        # Гасим и закрываем C-бэкенд
        try:
            if self._device_opened and hasattr(self, "_lib"):
                self._lib.hq_stop()
                self._lib.hq_close()
                self._device_opened = False
                self._emit_status("HackRF закрыт")
        except Exception as e:
            self._emit_error(f"Ошибка остановки: {e}")

        self._running = False
        self._closing = False
        import time
        self._last_close_ts = time.time()
        self.finished.emit(0)
        self._emit_status("Остановлен")
    
    def close(self):
        """Явное закрытие устройства."""
        self._cleanup()
    
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
    
    def _open_device(self, serial: Optional[str]) -> bool:
        """Открывает устройство HackRF."""
        try:
            # Определяем серийный номер
            device_serial = serial
            if not device_serial:
                serials = self.list_serials()
                if serials:
                    device_serial = serials[0]
                    self._emit_status(f"Используется устройство: {device_serial}")
                else:
                    self._emit_error("HackRF устройства не найдены")
                    return False
            
            # Открываем устройство через C библиотеку
            serial_bytes = device_serial.encode('utf-8') if device_serial else self._ffi.NULL
            r = self._lib.hq_open(serial_bytes)
            
            if r != 0:
                error_msg = self._get_last_error()
                self._emit_error(f"Не удалось открыть устройство: {error_msg}")
                return False
            
            self._device_opened = True
            self._emit_status(f"HackRF устройство открыто")
            return True
            
        except Exception as e:
            self._emit_error(f"Исключение при открытии устройства: {e}")
            return False
    
    def _configure_device(self) -> bool:
        """Не используем — конфигурируем через _configure_device_with()."""
        return self._configure_device_with(50e6, 6e9, 1e6)

    def _configure_device_with(self, f_start_hz: float, f_stop_hz: float, bin_hz: float) -> bool:
        """Конфигурирует C-бэкенд."""
        try:
            # Получаем параметры из config_manager или используем дефолтные
            lna_db = getattr(self, '_lna_db', 24)
            vga_db = getattr(self, '_vga_db', 20)
            amp_on = getattr(self, '_amp_on', False)
            
            ok = self._lib.hq_configure(
                float(f_start_hz / 1e6),
                float(f_stop_hz / 1e6),
                float(bin_hz),
                int(lna_db), int(vga_db), int(amp_on)
            )
            if ok != 0:
                self._emit_error("Ошибка конфигурации HackRF")
                return False
            self._emit_status(f"Устройство сконфигурировано: {f_start_hz/1e6:.1f}-{f_stop_hz/1e6:.1f} MHz")
            return True
        except Exception as e:
            self._emit_error(f"Ошибка конфигурации: {e}")
            return False
    
    def load_calibration(self, csv_path: str) -> bool:
        """Загружает калибровку из CSV файла."""
        try:
            path_bytes = csv_path.encode('utf-8')
            r = self._lib.hq_load_calibration(path_bytes)
            if r == 0:
                self._lib.hq_enable_calibration(1)
                self._emit_status(f"Calibration loaded from {csv_path}")
                return True
            else:
                error_msg = self._get_last_error()
                self._emit_error(f"Failed to load calibration: {error_msg}")
                return False
        except Exception as e:
            self._emit_error(f"Exception loading calibration: {e}")
            return False
    
    def set_calibration_enabled(self, enabled: bool):
        """Включает/выключает калибровку."""
        try:
            self._lib.hq_enable_calibration(1 if enabled else 0)
            status = "enabled" if enabled else "disabled"
            self._emit_status(f"Calibration {status}")
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
            self._emit_error("Invalid segment mode (must be 2 or 4)")
            return
        
        self._segment_mode = mode
        if self._configured:
            try:
                self._lib.hq_set_segment_mode(mode)
                self._emit_status(f"Segment mode set to {mode}")
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
                self._emit_status(f"Detector params updated: threshold={threshold_db:.1f}dB")
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
            self._emit_error(msg)
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
    
    # legacy C-колбэк был удалён — ничего тут не нужно


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
            # Запускаем C-код без колбэков
            r = self._backend._lib.hq_start_no_cb()
            if r != 0:
                msg = self._backend._get_last_error()
                raise RuntimeError(f"Failed to start: {msg}")
            
            # Ждем завершения или остановки
            while not self._stop_flag.is_set():
                self.msleep(100)  # Проверяем каждые 100 мс
            
        except Exception as e:
            code, msg = 1, str(e)
        
        finally:
            # Всегда останавливаем C код
            try:
                self._backend._lib.hq_stop()
            except Exception as e:
                pass  # Игнорируем ошибки при остановке
            self.finished_sig.emit(code, msg)

# Экспортируем публичные классы
__all__ = ['HackRFQSABackend', 'SweepConfig']