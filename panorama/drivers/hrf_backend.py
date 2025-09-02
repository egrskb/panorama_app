"""
HackRF QSA Backend - Fixed DC offset removal and spectrum processing
"""

from __future__ import annotations
import os, threading, time
from typing import Optional, List, Tuple
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
    
    return "libhackrf_master.so"


class HackRFQSABackend(SourceBackend):
    """
    Источник данных через C-бэкенд с чтением спектра.
    """

    def __init__(self, serial_suffix: Optional[str] = None, logger=None, parent=None):
        super().__init__(parent)
        
        self.log = logger
        self._serial = serial_suffix
        self._worker: Optional[_MasterWorker] = None
        self._running = False
        self._configured = False
        self._device_opened = False
        
        # Храним настройки диапазона
        self._user_freq_start_hz = None
        self._user_freq_stop_hz = None
        self._user_bin_hz = None
        
        # Параметры C библиотеки
        self._c_freq_start_hz = None
        self._c_freq_stop_hz = None
        self._spectrum_points = 0
        
        # Загружаем CFFI интерфейс
        self._ffi = FFI()
        self._define_interface()
        
        # Загружаем библиотеку
        lib_path = _find_library()
        try:
            self._lib = self._ffi.dlopen(lib_path)
            self._emit_status(f"[HackRF Master] Библиотека загружена: {lib_path}")
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить библиотеку: {e}")
        
        # Загружаем конфигурацию
        self._load_config()
        
        # Параметры по умолчанию
        self._segment_mode = 4
        self._ema_alpha = 0.25
        
        # Управление жизненным циклом
        self._closing = False
        self._last_close_ts = 0.0
    
    def __del__(self):
        """Деструктор - гарантирует закрытие устройства."""
        self._cleanup()
    
    def _cleanup(self):
        """Очистка ресурсов."""
        try:
            if hasattr(self, '_worker') and self._worker and self._worker.isRunning():
                try:
                    self._worker.stop()
                    self._worker.wait(2000)
                except:
                    pass
                finally:
                    self._worker = None
            
            if self._device_opened and hasattr(self, '_lib'):
                try:
                    self._lib.hq_close()
                    self._device_opened = False
                    self._emit_status("[HackRF] Устройство закрыто")
                except:
                    pass
            
            self._running = False
            self._configured = False
            
        except:
            pass
    
    def _define_interface(self):
        """Определяем интерфейс C-библиотеки."""
        self._ffi.cdef("""
            // Основные функции
            int  hq_open(const char* serial_suffix);
            void hq_close(void);
            const char* hq_last_error(void);
            int  hq_configure(double f_start_mhz, double f_stop_mhz, double bin_hz,
                              int lna_db, int vga_db, int amp_on);
            int  hq_start_no_cb(void);
            int  hq_stop(void);
            
            // Доступ к спектру
            int  hq_get_master_spectrum(double* freqs_hz, float* powers_dbm, int max_points);
            double hq_get_fft_bin_hz(void);
            
            // Настройки
            void hq_set_ema_alpha(float alpha);
            void hq_set_detector_params(float threshold_offset_db, int min_width_bins, int min_sweeps, float timeout_sec);
            int  hq_set_segment_mode(int mode);
            int  hq_get_segment_mode(void);
            int  hq_set_fft_size(int size);
            int  hq_get_fft_size(void);
            
            // Частотное сглаживание
            void hq_set_freq_smoothing(int enabled, int window_bins);
            int  hq_get_freq_smoothing_enabled(void);
            int  hq_get_freq_smoothing_window(void);
            
            // Калибровка
            int  hq_load_calibration(const char* csv_path);
            int  hq_enable_calibration(int enable);
            int  hq_get_calibration_status(void);
            
            // Диспетчер устройств
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

        # Анти-дребезг
        import time
        if time.time() - self._last_close_ts < 0.2:
            time.sleep(0.2)

        # Открыть устройство если не открыто
        if not self._device_opened:
            if not self._open_device(config.serial or self._serial):
                self._emit_error("Не удалось открыть HackRF")
                return

        if not hasattr(self, "_lib") or not self._device_opened:
            self._emit_error("Устройство не открыто")
            return

        # Обновляем конфигурацию
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
        
        # Сохраняем пользовательские настройки
        self._user_freq_start_hz = config.freq_start_hz
        self._user_freq_stop_hz = config.freq_end_hz
        self._user_bin_hz = config.bin_hz
        
        # Используем ровно пользовательские границы, только мягко клипим в допустимый диапазон HackRF
        f_start_hz = float(self._user_freq_start_hz)
        f_stop_hz = float(self._user_freq_stop_hz)
        # Клип 1–6000 МГц
        min_hz, max_hz = 1e6, 6000e6
        f_start_hz = max(min_hz, min(max_hz, f_start_hz))
        f_stop_hz = max(min_hz, min(max_hz, f_stop_hz))
        # Гарантируем, что правая граница > левой
        if f_stop_hz <= f_start_hz:
            f_stop_hz = min(max_hz, f_start_hz + max(1.0, float(self._user_bin_hz)))
        
        self._c_freq_start_hz = f_start_hz
        self._c_freq_stop_hz = f_stop_hz
        
        # Вычисляем количество точек спектра — без жёсткого капа
        freq_range = self._c_freq_stop_hz - self._c_freq_start_hz
        self._spectrum_points = int(freq_range / self._user_bin_hz) + 1
        
        # Сохраняем параметры усиления
        self._lna_db = config.lna_db
        self._vga_db = config.vga_db
        self._amp_on = config.amp_on
        
        # Конфигурируем C-бэкенд
        if not self._configure_device_with(self._c_freq_start_hz, self._c_freq_stop_hz, self._user_bin_hz):
            self._cleanup()
            return
        
        # Настраиваем частотное сглаживание
        try:
            if hasattr(self._lib, 'hq_set_freq_smoothing'):
                self._lib.hq_set_freq_smoothing(1, 5)  # Включено, окно 5 бинов
        except Exception as e:
            self._emit_status(f"hq_set_freq_smoothing unavailable: {e}")
        self._lib.hq_set_ema_alpha(self._ema_alpha)
        
        # Выравниваем размер буферов Python под фактический шаг бина из C (без капа)
        try:
            eff_step_hz = float(self._lib.hq_get_fft_bin_hz())
            if eff_step_hz > 0:
                freq_range = self._c_freq_stop_hz - self._c_freq_start_hz
                self._spectrum_points = int(freq_range / eff_step_hz) + 1
        except Exception:
            pass
        
        # Запускаем worker для чтения спектра
        self._worker = _MasterWorker(self, config)
        self._worker.spectrum_ready.connect(self._on_spectrum_from_worker)
        self._worker.finished_sig.connect(self._on_worker_finished)
        self._worker.start()
        
        self._running = True
        self.started.emit()
        
        # Логируем параметры
        try:
            eff_step_hz = float(self._lib.hq_get_fft_bin_hz())
        except Exception:
            eff_step_hz = 0.0
        if eff_step_hz > 0:
            est_points = int(((self._c_freq_stop_hz - self._c_freq_start_hz) / eff_step_hz) + 1)
            self._emit_status(
                f"HackRF Master started: {self._user_freq_start_hz/1e6:.1f}-{self._user_freq_stop_hz/1e6:.1f} MHz, "
                f"effective bin={eff_step_hz/1e3:.1f} kHz, ~points={est_points}"
            )
        else:
            self._emit_status(
                f"HackRF Master started: {self._user_freq_start_hz/1e6:.1f}-{self._user_freq_stop_hz/1e6:.1f} MHz"
            )
    
    def stop(self):
        """Останавливает sweep и закрывает устройство."""
        if not self._running and not self._device_opened:
            return
        self._closing = True
        self._emit_status("Остановка...")

        # Остановить воркер
        if self._worker:
            try:
                self._worker.stop()
                self._worker.wait(3000)
            except Exception as e:
                self._emit_error(f"Ошибка остановки worker: {e}")
            finally:
                self._worker = None

        # Остановить C-бэкенд
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
                except:
                    continue
                    
            return serials
            
        except Exception as e:
            self._emit_error(f"Error listing devices: {e}")
            return []
    
    def _open_device(self, serial: Optional[str]) -> bool:
        """Открывает устройство HackRF."""
        try:
            device_serial = serial
            if not device_serial:
                serials = self.list_serials()
                if serials:
                    device_serial = serials[0]
                    self._emit_status(f"Используется устройство: {device_serial}")
                else:
                    self._emit_error("HackRF устройства не найдены")
                    return False
            
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
    
    def _configure_device_with(self, f_start_hz: float, f_stop_hz: float, bin_hz: float) -> bool:
        """Конфигурирует C-бэкенд."""
        try:
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
                
            self._configured = True
            self._emit_status(f"Устройство сконфигурировано: {f_start_hz/1e6:.1f}-{f_stop_hz/1e6:.1f} MHz")
            return True
        except Exception as e:
            self._emit_error(f"Ошибка конфигурации: {e}")
            return False
    
    def _on_spectrum_from_worker(self, freqs_hz, power_dbm):
        """Обработка спектра от воркера - фильтрация по пользовательскому диапазону."""
        try:
            if freqs_hz is None or power_dbm is None:
                return
            
            # ВАЖНО: НЕ применяем DC offset removal здесь - это делается в C библиотеке!
            # C код уже применяет окно Hann и корректирует спектр
            
            # Просто фильтруем данные по пользовательскому диапазону
            mask = (freqs_hz >= self._user_freq_start_hz) & (freqs_hz <= self._user_freq_stop_hz)
            filtered_freqs = freqs_hz[mask]
            filtered_power = power_dbm[mask]
            
            if len(filtered_freqs) > 0:
                # Эмитим отфильтрованные данные
                self.fullSweepReady.emit(filtered_freqs, filtered_power)
                
        except Exception as e:
            self._emit_error(f"Ошибка обработки спектра: {e}")
    
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
        except:
            pass
    
    def calibration_loaded(self) -> bool:
        """Проверяет загружена ли калибровка."""
        try:
            return bool(self._lib.hq_get_calibration_status())
        except:
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
            except:
                pass
    
    def set_ema_alpha(self, alpha: float):
        """Устанавливает коэффициент EMA фильтрации."""
        if alpha < 0.01 or alpha > 1.0:
            return
        
        self._ema_alpha = alpha
        if self._configured:
            try:
                self._lib.hq_set_ema_alpha(alpha)
            except:
                pass
    
    def set_detector_params(self, threshold_db: float = -80.0, min_width: int = 3,
                            min_sweeps: int = 2, timeout: float = 5.0):
        """Устанавливает параметры детектора пиков."""
        if self._configured:
            try:
                self._lib.hq_set_detector_params(threshold_db, min_width, min_sweeps, timeout)
                self._emit_status(f"Detector params updated: threshold={threshold_db:.1f}dB")
            except:
                pass
    
    @QtCore.pyqtSlot(int, str)
    def _on_worker_finished(self, code: int, msg: str):
        if code != 0 and msg:
            self._emit_error(msg)
        self._running = False
        self.finished.emit(code)
    
    def _get_last_error(self) -> str:
        """Получает последнюю ошибку из C-библиотеки."""
        try:
            err_ptr = self._lib.hq_last_error()
            if err_ptr != self._ffi.NULL:
                return self._ffi.string(err_ptr).decode('utf-8')
        except:
            pass
        return "Unknown error"
    
    def _emit_status(self, msg: str):
        if self.log:
            self.log.info(msg)
        print(msg)
        self.status.emit(msg)
    
    def _emit_error(self, msg: str):
        if self.log:
            self.log.error(msg)
        print(f"ERROR: {msg}")
        self.error.emit(msg)


class _MasterWorker(QtCore.QThread):
    """Worker thread для чтения спектра из C библиотеки."""
    
    spectrum_ready = QtCore.pyqtSignal(object, object)
    finished_sig = QtCore.pyqtSignal(int, str)
    
    def __init__(self, backend: HackRFQSABackend, config: SweepConfig):
        super().__init__(backend)
        self._backend = backend
        self._config = config
        self._stop_flag = threading.Event()
        self._n_points = backend._spectrum_points
    
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
            
            # Аллоцируем буферы для чтения спектра
            freqs_buf = self._backend._ffi.new("double[]", self._n_points)
            power_buf = self._backend._ffi.new("float[]", self._n_points)
            
            # Читаем спектр периодически
            while not self._stop_flag.is_set():
                # Читаем накопленный спектр из C
                n = self._backend._lib.hq_get_master_spectrum(
                    freqs_buf, power_buf, self._n_points
                )
                
                if n > 0:
                    # Получаем векторизованные представления C-буферов без Python-циклов
                    freqs_view = self._backend._ffi.buffer(freqs_buf, n * self._backend._ffi.sizeof("double"))
                    power_view = self._backend._ffi.buffer(power_buf, n * self._backend._ffi.sizeof("float"))
                    # Создаём numpy-вью поверх буферов (копии, чтобы не зависеть от перезаписи C)
                    freqs_hz = np.frombuffer(freqs_view, dtype=np.float64, count=n).copy()
                    power_dbm = np.frombuffer(power_view, dtype=np.float32, count=n).copy()
                    
                    # Проверка и корректировка значений
                    # Убираем NaN/inf
                    valid_mask = np.isfinite(power_dbm)
                    if not np.all(valid_mask):
                        # Заменяем невалидные значения на медиану валидных
                        valid_values = power_dbm[valid_mask]
                        if len(valid_values) > 0:
                            median_val = np.median(valid_values)
                            power_dbm[~valid_mask] = median_val
                        else:
                            power_dbm[~valid_mask] = -120.0
                    
                    # Ограничиваем диапазон значений
                    power_dbm = np.clip(power_dbm, -150.0, 20.0)
                    
                    # Эмитим спектр
                    self.spectrum_ready.emit(freqs_hz, power_dbm)
                
                # Пауза между чтениями (100мс)
                self.msleep(100)
            
        except Exception as e:
            code, msg = 1, str(e)
        
        finally:
            # Останавливаем C код
            try:
                self._backend._lib.hq_stop()
            except:
                pass
            self.finished_sig.emit(code, msg)


# Экспортируем публичные классы
__all__ = ['HackRFQSABackend', 'SweepConfig']