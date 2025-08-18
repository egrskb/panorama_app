"""
Refactored HackRF backend with improved multi-device support, peak detection
and full sweep assembly.  Fixed MultiWorker crashes and hangs.
"""

from __future__ import annotations

import os
import threading
import time
import queue
from typing import Optional, List, Tuple, Dict, Any
from contextlib import contextmanager

import numpy as np
from PyQt5 import QtCore
from cffi import FFI

from panorama.drivers.base import SourceBackend, SweepConfig
from panorama.shared.parsing import SweepLine


def _find_library() -> List[str]:
    """Return a list of candidate library names for dlopen."""
    here = os.path.abspath(os.path.dirname(__file__))
    names = ["libhackrf_multi.so", "libhackrf_multi.dylib", "hackrf_multi.dll"]
    old = ["libhackrf_qsa.so", "libhackrf_qsa.dylib", "hackrf_qsa.dll"]
    candidates: List[str] = []
    for n in names + old:
        candidates.append(os.path.join(here, n))
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    for n in names + old:
        candidates.append(os.path.join(root, n))
    candidates.extend(names + old)
    return candidates


class HackRFLibSource(SourceBackend):
    """HackRF backend using CFFI with optional multi-SDR support."""

    def __init__(self, lib_path: Optional[str] = None, parent=None) -> None:
        super().__init__(parent)
        self._ffi = FFI()
        self._setup_cdefs()
        self._lib = self._load_library(lib_path)

        # Worker thread
        self._worker: Optional[_BaseWorker] = None

        # Sweep assembler
        self._assembler = _SweepAssembler()

        # Number of devices
        self._num_devices = 1

        # Detector parameters
        self._detector_params: Dict[str, float | int] = {
            'threshold_offset': 20.0,
            'min_width': 3,
            'min_sweeps': 3,
            'timeout': 2.0,
        }

        # Running flag
        self._running = False

        # Thread safety
        self._lock = threading.Lock()

    def _setup_cdefs(self) -> None:
        """Define the C API signatures."""
        self._ffi.cdef(
            r"""
            typedef unsigned int uint32_t;
            typedef unsigned long long uint64_t;
            typedef struct {
                double f_center_hz;
                double bw_hz;
                float rssi_ema;
                uint64_t last_ns;
                int hit_count;
            } WatchItem;
            typedef struct {
                double f_hz;
                float rssi_dbm;
                uint64_t last_ns;
            } Peak;
            typedef struct {
                int master_running;
                int slave_running[2];
                double retune_ms_avg;
                int watch_items;
            } HqStatus;
            int  hq_open_all(int num_expected);
            void hq_close_all(void);
            int  hq_config_set_rates(uint32_t samp_rate_hz, uint32_t bb_bw_hz);
            int  hq_config_set_gains(uint32_t lna_db, uint32_t vga_db, bool amp_on);
            int  hq_config_set_freq_range(double start_hz, double stop_hz, double step_hz);
            int  hq_config_set_dwell_time(uint32_t dwell_ms);
            void hq_set_detector_params(float threshold_offset_db, int min_width_bins,
                                       int min_sweeps, float timeout_sec);
            int  hq_start(void);
            void hq_stop(void);
            int  hq_get_watchlist_snapshot(WatchItem* out, int max_items);
            int  hq_get_recent_peaks(Peak* out, int max_items);
            int  hq_get_master_spectrum(double* freqs_hz, float* powers_dbm, int max_points);
            void hq_set_grouping_tolerance_hz(double delta_hz);
            void hq_set_ema_alpha(float alpha);
            int  hq_get_status(HqStatus* out);
            int  hq_list_devices(char* serials[], int max_count);
            int  hq_get_device_count(void);
        """,
            override=True
        )

    def _load_library(self, lib_path: Optional[str]) -> Any:
        """Attempt to load the HackRF library."""
        paths = [lib_path] if lib_path else _find_library()
        for p in paths:
            if not p:
                continue
            try:
                lib = self._ffi.dlopen(p)
                print(f"Loaded library: {p}")
                return lib
            except Exception:
                continue
        raise RuntimeError("Не удалось загрузить библиотеку HackRF")

    def set_num_devices(self, num: int) -> None:
        """Specify the number of devices to use."""
        if num < 1 or num > 3:
            raise ValueError("Number of devices must be between 1 and 3")

        with self._lock:
            restart = self._running and (self._num_devices != num)
            self._num_devices = num

        if restart:
            self.stop()
            time.sleep(0.1)

        mode = "Multi-SDR" if self._num_devices > 1 else "Single-SDR"
        print(f"📡 Режим: {mode} ({self._num_devices} устройств)")

    def set_detector_params(self, threshold_offset: float, min_width: int,
                            min_sweeps: int, timeout: float) -> None:
        """Update detector parameters."""
        with self._lock:
            self._detector_params = {
                'threshold_offset': threshold_offset,
                'min_width': min_width,
                'min_sweeps': min_sweeps,
                'timeout': timeout,
            }

        # Apply immediately if running
        if self._running and self._lib:
            try:
                self._lib.hq_set_detector_params(
                    float(threshold_offset), int(min_width), int(min_sweeps), float(timeout)
                )
                print(f"Detector params updated: +{threshold_offset:.1f} dB, width {min_width} bins")
            except Exception as e:
                print(f"Failed to set detector params: {e}")

    def list_serials(self) -> List[str]:
        """Return a list of detected HackRF serial numbers."""
        out: List[str] = []
        if self._lib is None:
            return out

        try:
            # Use multi-SDR API
            count = self._lib.hq_get_device_count()
            if count > 0:
                serials = self._ffi.new("char*[]", count)
                for i in range(count):
                    serials[i] = self._ffi.new("char[]", 128)
                actual = self._lib.hq_list_devices(serials, count)
                for i in range(actual):
                    if serials[i] != self._ffi.NULL:
                        s = self._ffi.string(serials[i]).decode(errors="ignore")
                        if s and s != "0000000000000000":
                            out.append(s)
                # Close devices after enumeration
                try:
                    self._lib.hq_close_all()
                except Exception:
                    pass
        except Exception:
            pass

        return out

    def is_running(self) -> bool:
        """Check if the worker thread is active."""
        with self._lock:
            return self._running and self._worker is not None and self._worker.is_alive()

    def start(self, config: SweepConfig) -> None:
        """Start data acquisition."""
        if self.is_running():
            self.status.emit("Уже запущено")
            return

        available = self.list_serials()
        required = self._num_devices if self._num_devices > 1 else 1

        print(f"🚀 Запуск в режиме: {'Multi-SDR' if required > 1 else 'Single-SDR'}")
        print(f"📡 Найдено устройств: {len(available)}")

        if len(available) < required:
            self.error.emit(f"Требуется {required} устройств, найдено {len(available)}")
            return

        # Configure assembler
        self._assembler.configure(config)

        # Select and start worker
        with self._lock:
            if required > 1:
                self._worker = _MultiWorker(self, self._ffi, self._lib, config, required, self._assembler)
            else:
                self._worker = _SingleWorker(self, self._ffi, self._lib, config, self._assembler)

            self._worker.finished_sig.connect(self._on_worker_finished)
            self._worker.start()
            self._running = True

        self.started.emit()
        self.status.emit(f"Запущен {'Multi' if required > 1 else 'Single'}-SDR")

    def stop(self) -> None:
        """Stop the current worker."""
        if not self.is_running():
            return

        print("⏹️ Остановка SDR...")
        self.status.emit("Остановка...")

        with self._lock:
            if self._worker:
                self._worker.stop()
                self._worker.join(timeout=2.0)
                self._worker = None
            self._running = False

        self.finished.emit(0)
        self.status.emit("Остановлено")

    def _on_worker_finished(self, code: int, msg: str) -> None:
        """Handle worker completion."""
        if code != 0:
            self.error.emit(msg or f"Завершился с кодом {code}")

        with self._lock:
            self._worker = None
            self._running = False

        self.finished.emit(code)

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Query the current status from the C library."""
        if not self.is_running() or self._num_devices <= 1:
            return None

        try:
            status = self._ffi.new("HqStatus*")
            if self._lib.hq_get_status(status) == 0:
                s = status[0]
                return {
                    'master_running': bool(s.master_running),
                    'slave1_running': bool(s.slave_running[0]),
                    'slave2_running': bool(s.slave_running[1]),
                    'retune_ms': s.retune_ms_avg,
                    'watch_items': s.watch_items,
                }
        except Exception:
            pass
        return None


class _BaseWorker(QtCore.QObject, threading.Thread):
    """Base class for worker threads."""
    finished_sig = QtCore.pyqtSignal(int, str)

    def __init__(self) -> None:
        QtCore.QObject.__init__(self)
        threading.Thread.__init__(self, daemon=True)
        self._running = True
        self._stop_ev = threading.Event()

    def stop(self) -> None:
        """Request the worker to stop."""
        self._running = False
        self._stop_ev.set()


class _SingleWorker(_BaseWorker):
    """Worker for single-SDR mode."""

    def __init__(self, parent: HackRFLibSource, ffi: FFI, lib: Any, config: SweepConfig,
                 assembler: _SweepAssembler) -> None:
        super().__init__()
        self._parent = parent
        self._ffi = ffi
        self._lib = lib
        self._config = config
        self._assembler = assembler
        self._spectrum_updates = 0

    def run(self) -> None:
        code = 0
        msg = ""
        try:
            # Open device
            if self._lib.hq_open_all(1) != 0:
                self.finished_sig.emit(1, "Не удалось открыть устройство")
                return

            cfg = self._config

            # Configure
            self._lib.hq_config_set_rates(12000000, 8000000)
            self._lib.hq_config_set_gains(cfg.lna_db, cfg.vga_db, cfg.amp_on)

            start_hz = max(cfg.freq_start_hz, 1_000_000)
            stop_hz = min(cfg.freq_end_hz, 6_000_000_000)
            step_hz = max(cfg.bin_hz, 100_000)

            self._lib.hq_config_set_freq_range(float(start_hz), float(stop_hz), float(step_hz))
            self._lib.hq_config_set_dwell_time(2)

            # Apply detector params
            p = self._parent._detector_params
            self._lib.hq_set_detector_params(
                float(p['threshold_offset']), int(p['min_width']),
                int(p['min_sweeps']), float(p['timeout'])
            )

            self._lib.hq_set_grouping_tolerance_hz(250_000.0)
            self._lib.hq_set_ema_alpha(0.25)

            # Start
            try:
                r = self._lib.hq_start()
            except TypeError:
                r = self._lib.hq_start(self._ffi.NULL, self._ffi.NULL)

            if r != 0:
                self._lib.hq_close_all()
                self.finished_sig.emit(3, "Не удалось запустить устройство")
                return

            print(f"✅ SingleWorker: Running {int((stop_hz - start_hz) / step_hz)} points")

            # Main loop
            n_bins = int((stop_hz - start_hz) / step_hz) + 1
            max_points = min(n_bins, 50_000)
            freqs_buf = self._ffi.new("double[]", max_points)
            powers_buf = self._ffi.new("float[]", max_points)
            last_log_time = time.time()

            while self._running and not self._stop_ev.is_set():
                n = self._lib.hq_get_master_spectrum(freqs_buf, powers_buf, max_points)
                if n > 0:
                    freqs = np.frombuffer(self._ffi.buffer(freqs_buf, n * 8), dtype=np.float64, count=n).copy()
                    power = np.frombuffer(self._ffi.buffer(powers_buf, n * 4), dtype=np.float32, count=n).copy()

                    if not (np.all(np.isnan(power)) or np.all(power == 0)):
                        self._parent.fullSweepReady.emit(freqs, power)
                        self._spectrum_updates += 1

                        result = self._assembler.add_segment(freqs, power, int(freqs[0]))
                        if result is not None:
                            full_freqs, full_power = result
                            self._parent.fullSweepReady.emit(full_freqs, full_power)

                        if time.time() - last_log_time > 5.0:
                            print(f"SingleWorker: {self._spectrum_updates} updates")
                            last_log_time = time.time()

                time.sleep(0.02)

        except Exception as e:
            code, msg = 99, str(e)
        finally:
            try:
                try:
                    self._lib.hq_stop()
                except TypeError:
                    self._lib.hq_stop(self._ffi.NULL, self._ffi.NULL)
                self._lib.hq_close_all()
            except Exception:
                pass
            print(f"SingleWorker stopped after {self._spectrum_updates} updates")
            self.finished_sig.emit(code, msg)


class _MultiWorker(_BaseWorker):
    """Worker для multi-SDR режима с улучшенной стабильностью."""

    def __init__(self, parent: HackRFLibSource, ffi: FFI, lib: Any, config: SweepConfig,
                 num_devices: int, assembler: _SweepAssembler) -> None:
        super().__init__()
        self._parent = parent
        self._ffi = ffi
        self._lib = lib
        self._config = config
        self._num_devices = num_devices
        self._assembler = assembler
        self._spectrum_updates = 0
        self._watchlist_updates = 0

        # Защита от переполнения/частых вызовов
        self._last_spectrum_time = 0.0
        self._last_watchlist_time = 0.0
        self._consecutive_errors = 0
        self._max_consecutive_errors = 10

    def run(self) -> None:
        code = 0
        msg = ""

        try:
            # Открываем устройства с повторными попытками
            if not self._open_devices_safe():
                self.finished_sig.emit(1, f"Не удалось открыть {self._num_devices} устройств")
                return

            # Конфигурируем устройства
            if not self._configure_devices_safe():
                self._lib.hq_close_all()
                self.finished_sig.emit(2, "Не удалось настроить устройства")
                return

            # Запускаем acquisition
            if not self._start_acquisition_safe():
                self._lib.hq_close_all()
                self.finished_sig.emit(3, "Не удалось запустить multi-SDR")
                return

            print(f"✅ MultiWorker: {self._num_devices} devices running")

            # Основной цикл с улучшенной обработкой ошибок
            self._run_main_loop_safe()

        except Exception as e:
            code, msg = 99, str(e)
            print(f"MultiWorker error: {e}")
        finally:
            self._cleanup_safe()
            print(f"MultiWorker stopped. Spectrum: {self._spectrum_updates}, Watchlist: {self._watchlist_updates}")
            self.finished_sig.emit(code, msg)

    def _open_devices_safe(self) -> bool:
        """Открывает устройства с повторными попытками."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                r = self._lib.hq_open_all(int(self._num_devices))
                if r == 0:
                    return True
                print(f"Попытка {attempt + 1}/{max_attempts} открытия устройств не удалась: код {r}")
                time.sleep(0.5)
            except Exception as e:
                print(f"Ошибка при открытии устройств: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(0.5)
        return False

    def _configure_devices_safe(self) -> bool:
        """Конфигурирует устройства с проверкой результатов."""
        try:
            cfg = self._config

            # Конфигурируем с проверкой каждого вызова
            if self._lib.hq_config_set_rates(12_000_000, 8_000_000) != 0:
                print("Ошибка установки sample rate")
                return False

            if self._lib.hq_config_set_gains(cfg.lna_db, cfg.vga_db, cfg.amp_on) != 0:
                print("Ошибка установки усилений")
                return False

            # Частотный диапазон
            start_hz = max(cfg.freq_start_hz, 1_000_000)
            stop_hz = min(cfg.freq_end_hz, 6_000_000_000)
            step_hz = max(cfg.bin_hz, 100_000)

            if self._lib.hq_config_set_freq_range(float(start_hz), float(stop_hz), float(step_hz)) != 0:
                print("Ошибка установки частотного диапазона")
                return False

            self._lib.hq_config_set_dwell_time(2)

            # Параметры группировки
            self._lib.hq_set_grouping_tolerance_hz(5_000_000.0)
            self._lib.hq_set_ema_alpha(0.25)

            # Параметры детектора
            p = self._parent._detector_params
            self._lib.hq_set_detector_params(
                float(p['threshold_offset']), int(p['min_width']),
                int(p['min_sweeps']), float(p['timeout'])
            )

            return True
        except Exception as e:
            print(f"Ошибка конфигурации: {e}")
            return False

    def _start_acquisition_safe(self) -> bool:
        """Запускает acquisition с проверкой."""
        try:
            try:
                r = self._lib.hq_start()
            except TypeError:
                r = self._lib.hq_start(self._ffi.NULL, self._ffi.NULL)

            if r == 0:
                time.sleep(0.1)  # Даем время на инициализацию
                return True

            print(f"hq_start вернул код ошибки: {r}")
            return False
        except Exception as e:
            print(f"Ошибка запуска: {e}")
            return False

    def _run_main_loop_safe(self) -> None:
        """Основной цикл с улучшенной стабильностью."""
        cfg = self._config
        start_hz = max(cfg.freq_start_hz, 1_000_000)
        stop_hz = min(cfg.freq_end_hz, 6_000_000_000)
        step_hz = max(cfg.bin_hz, 100_000)

        n_bins = int((stop_hz - start_hz) / step_hz) + 1
        max_points = min(n_bins, 50_000)

        # Выделяем буферы один раз
        freqs_buf = self._ffi.new("double[]", max_points)
        powers_buf = self._ffi.new("float[]", max_points)
        peaks_buf = self._ffi.new("Peak[10]")       # меньше для стабильности
        watch_buf = self._ffi.new("WatchItem[20]")  # меньше для стабильности

        last_log_time = time.time()
        last_spectrum_success = time.time()

        # Минимальные интервалы между операциями
        MIN_SPECTRUM_INTERVAL = 0.02  # 50 Hz максимум
        MIN_WATCHLIST_INTERVAL = 0.2  # 5 Hz максимум

        while self._running and not self._stop_ev.is_set():
            try:
                current_time = time.time()

                # 1) Спектр (throttling)
                if current_time - self._last_spectrum_time >= MIN_SPECTRUM_INTERVAL:
                    self._last_spectrum_time = current_time
                    try:
                        n = self._lib.hq_get_master_spectrum(freqs_buf, powers_buf, max_points)
                        if n > 0 and n <= max_points:
                            try:
                                freqs = np.frombuffer(self._ffi.buffer(freqs_buf, n * 8),
                                                      dtype=np.float64, count=n).copy()
                                power = np.frombuffer(self._ffi.buffer(powers_buf, n * 4),
                                                      dtype=np.float32, count=n).copy()

                                if not np.all(np.isnan(power)) and not np.all(power == 0):
                                    self._parent.fullSweepReady.emit(freqs, power)
                                    self._spectrum_updates += 1
                                    last_spectrum_success = current_time
                                    self._consecutive_errors = 0

                                    # Сборка полного свипа
                                    result = self._assembler.add_segment(freqs, power, int(freqs[0]))
                                    if result is not None:
                                        full_freqs, full_power = result
                                        self._parent.fullSweepReady.emit(full_freqs, full_power)

                            except Exception as e:
                                print(f"Ошибка обработки спектра: {e}")
                                self._consecutive_errors += 1
                    except Exception as e:
                        print(f"Ошибка получения спектра: {e}")
                        self._consecutive_errors += 1

                # 2) Пики и watchlist (throttling)
                if current_time - self._last_watchlist_time >= MIN_WATCHLIST_INTERVAL:
                    self._last_watchlist_time = current_time
                    try:
                        n_peaks = self._lib.hq_get_recent_peaks(peaks_buf, 5)
                        if n_peaks > 0 and n_peaks <= 10:
                            n_watch = self._lib.hq_get_watchlist_snapshot(watch_buf, 10)
                            if n_watch > 0 and n_watch <= 20:
                                self._watchlist_updates += 1
                                # Обрабатываем только значимые сигналы
                                for i in range(min(n_watch, 5)):
                                    w = watch_buf[i]
                                    if w.bw_hz > 1_000_000 and w.rssi_ema > -90:
                                        sw = SweepLine(
                                            ts=None,
                                            f_low_hz=int(w.f_center_hz - w.bw_hz / 2),
                                            f_high_hz=int(w.f_center_hz + w.bw_hz / 2),
                                            bin_hz=int(w.bw_hz / 10) if w.bw_hz > 0 else 200_000,
                                            power_dbm=np.array([w.rssi_ema], dtype=np.float32),
                                        )
                                        self._parent.sweepLine.emit(sw)
                    except Exception as e:
                        print(f"Ошибка обработки peaks/watchlist: {e}")

                # 3) Детект «тихого» зависания
                if current_time - last_spectrum_success > 5.0:
                    print("Предупреждение: нет данных спектра > 5 секунд")
                    self._consecutive_errors += 1
                    last_spectrum_success = current_time  # чтобы не спамить

                # 4) Критические ошибки
                if self._consecutive_errors >= self._max_consecutive_errors:
                    print(f"Слишком много ошибок ({self._consecutive_errors}), останавливаемся")
                    break

                # 5) Периодический лог статуса
                if current_time - last_log_time > 10.0:
                    try:
                        status = self._ffi.new("HqStatus*")
                        if self._lib.hq_get_status(status) == 0:
                            s = status[0]
                            active = s.slave_running[0] + s.slave_running[1]
                            if active > 0 or s.watch_items > 0:
                                print(f"MultiWorker: {self._spectrum_updates} spectra, "
                                      f"Slaves: {active}, Targets: {s.watch_items}")
                    except Exception:
                        pass
                    last_log_time = current_time

                # 6) Адаптивная задержка
                time.sleep(0.1 if self._consecutive_errors > 0 else 0.02)

            except Exception as e:
                print(f"Ошибка в основном цикле: {e}")
                self._consecutive_errors += 1
                time.sleep(0.1)
                if self._consecutive_errors >= self._max_consecutive_errors:
                    break

    def _cleanup_safe(self) -> None:
        """Безопасная очистка с защитой от зависания."""
        print("MultiWorker: начинаем очистку...")
        try:
            if self._lib:
                try:
                    self._lib.hq_stop()
                except Exception:
                    pass
                time.sleep(0.1)
                try:
                    self._lib.hq_close_all()
                except Exception:
                    pass
        except Exception as e:
            print(f"Ошибка при очистке: {e}")


class _SweepAssembler:
    """Collects partial spectrum segments and reconstructs full sweeps."""

    def __init__(self) -> None:
        self.f0_hz = 0
        self.f1_hz = 0
        self.bin_hz = 0
        self.grid: Optional[np.ndarray] = None  # centers of bins
        self.n_bins = 0
        self.sum: Optional[np.ndarray] = None
        self.cnt: Optional[np.ndarray] = None
        self.seen: Optional[np.ndarray] = None
        self.prev_low: Optional[int] = None

    def configure(self, cfg: SweepConfig) -> None:
        """Initialize the grid from sweep configuration."""
        self.f0_hz = int(cfg.freq_start_hz)
        self.f1_hz = int(cfg.freq_end_hz)
        self.bin_hz = int(cfg.bin_hz)
        self.grid = np.arange(
            self.f0_hz + self.bin_hz * 0.5,
            self.f1_hz + self.bin_hz * 0.5,
            self.bin_hz,
            dtype=np.float64,
        )
        self.n_bins = len(self.grid)
        self.reset()

    def reset(self) -> None:
        """Reset accumulators for a new sweep."""
        if self.n_bins == 0:
            return
        self.sum = np.zeros(self.n_bins, dtype=np.float64)
        self.cnt = np.zeros(self.n_bins, dtype=np.int32)
        self.seen = np.zeros(self.n_bins, dtype=bool)
        self.prev_low = None

    def add_segment(self, f_hz: np.ndarray, p_dbm: np.ndarray, hz_low: int
                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Add a partial segment and return (full_freqs, full_power) when the
        sweep is considered complete (wrap-around or full coverage).
        """
        if self.grid is None or self.n_bins == 0:
            return None

        # Detect wrap-around (частота начала сегмента «перепрыгнула» назад)
        if self.prev_low is not None and hz_low < self.prev_low - 10_000_000:
            result = self._finalize()
            self.reset()
            self.prev_low = hz_low
            self._add_to_grid(f_hz, p_dbm)
            return result

        # Обычная вставка
        self.prev_low = hz_low
        self._add_to_grid(f_hz, p_dbm)

        # Если покрыто почти всё — завершаем (можно настроить порог)
        coverage = float(self.seen.sum()) / float(self.n_bins) if self.n_bins else 0.0
        if coverage >= 0.98:  # 98% бинов увидено — считаем полный свип собран
            result = self._finalize()
            self.reset()
            return result

        return None

    def _add_to_grid(self, f_hz: np.ndarray, p_dbm: np.ndarray) -> None:
        """Accumulate segment points into the fixed grid using nearest bin."""
        if self.grid is None or self.n_bins == 0:
            return
        if f_hz.size == 0 or p_dbm.size == 0:
            return

        # Индексы ближайших центров бинов
        # grid centers: f_center = f0 + (k + 0.5)*bin_hz
        # => k ≈ round((f - (f0 + 0.5*bin)) / bin)
        k = np.rint((f_hz - (self.f0_hz + 0.5 * self.bin_hz)) / float(self.bin_hz)).astype(np.int64)
        valid = (k >= 0) & (k < self.n_bins) & np.isfinite(p_dbm)
        if not np.any(valid):
            return

        k = k[valid]
        v = p_dbm[valid].astype(np.float64)

        # Накопление суммы/счётчика
        np.add.at(self.sum, k, v)
        np.add.at(self.cnt, k, 1)
        self.seen[k] = True

    def _finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute averaged power for all bins; empty bins -> NaN."""
        if self.grid is None or self.n_bins == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float32)

        avg = np.full(self.n_bins, np.nan, dtype=np.float64)
        nz = self.cnt > 0
        avg[nz] = self.sum[nz] / self.cnt[nz]
        return self.grid.copy(), avg.astype(np.float32)
