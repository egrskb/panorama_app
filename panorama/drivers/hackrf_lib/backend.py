# panorama/drivers/hackrf_lib/backend.py
from __future__ import annotations
import os, threading, time
from typing import Optional, List, Tuple, Dict, Any
from PyQt5 import QtCore
import numpy as np
from cffi import FFI

from panorama.drivers.base import SourceBackend, SweepConfig
from panorama.shared.parsing import SweepLine


def _find_library() -> List[str]:
    """Ищем библиотеку в разных местах."""
    here = os.path.abspath(os.path.dirname(__file__))
    
    # Новые имена для multi-SDR версии
    names = ["libhackrf_multi.so", "libhackrf_multi.dylib", "hackrf_multi.dll"]
    
    # Старые имена для обратной совместимости
    old_names = ["libhackrf_qsa.so", "libhackrf_qsa.dylib", "hackrf_qsa.dll"]
    
    candidates = []
    
    # Сначала ищем рядом с этим файлом
    for n in names + old_names:
        candidates.append(os.path.join(here, n))
    
    # Потом в корне проекта
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    for n in names + old_names:
        candidates.append(os.path.join(root, n))
    
    # И наконец просто имена для dlopen
    candidates.extend(names + old_names)
    
    return candidates


class HackRFLibSource(SourceBackend):
    """Источник через CFFI с поддержкой multi-SDR режима."""
    
    def __init__(self, lib_path: Optional[str] = None, serial_suffix: Optional[str] = None, parent=None):
        super().__init__(parent)
        
        self._ffi = FFI()
        
        # Определяем интерфейс в зависимости от найденной библиотеки
        self._multi_mode = False
        self._lib = None
        
        # Пробуем загрузить библиотеку
        for p in ([lib_path] if lib_path else _find_library()):
            if not p:
                continue
            try:
                # Проверяем тип библиотеки по имени
                if 'multi' in p.lower():
                    self._setup_multi_interface()
                    self._multi_mode = True
                else:
                    self._setup_single_interface()
                    self._multi_mode = False
                
                self._lib = self._ffi.dlopen(p)
                print(f"Loaded library: {p} (multi-mode: {self._multi_mode})")
                break
            except Exception as e:
                continue
        
        if self._lib is None:
            raise RuntimeError(f"Не удалось загрузить библиотеку HackRF")

        self._serial_suffix = serial_suffix
        self._worker: Optional[_Worker] = None
        self._assembler = _SweepAssembler()
        
        # Multi-SDR параметры
        self._num_devices = 1
        self._multi_worker: Optional[_MultiWorker] = None
        self._running = False
        
    def _setup_multi_interface(self):
        """Настройка интерфейса для multi-SDR версии."""
        self._ffi.cdef(r"""
            // Структуры данных для multi-SDR
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
            
            // Multi-SDR API
            int  hq_open_all(int num_expected);
            void hq_close_all(void);
            
            int  hq_config_set_rates(uint32_t samp_rate_hz, uint32_t bb_bw_hz);
            int  hq_config_set_gains(uint32_t lna_db, uint32_t vga_db, bool amp_on);
            
            // Настройка диапазона частот
            int  hq_config_set_freq_range(double start_hz, double stop_hz, double step_hz);
            int  hq_config_set_dwell_time(uint32_t dwell_ms);
            
            int  hq_start(void);
            void hq_stop(void);
            
            int  hq_get_watchlist_snapshot(WatchItem* out, int max_items);
            int  hq_get_recent_peaks(Peak* out, int max_items);
            
            // Чтение непрерывного спектра от Master SDR
            int  hq_get_master_spectrum(double* freqs_hz, float* powers_dbm, int max_points);
            
            void hq_set_grouping_tolerance_hz(double delta_hz);
            void hq_set_ema_alpha(float alpha);
            
            int  hq_get_status(HqStatus* out);
            
            // Device enumeration
            int  hq_list_devices(char* serials[], int max_count);
            int  hq_get_device_count(void);
        """)
        
    def _setup_single_interface(self):
        """Настройка интерфейса для single-SDR версии (обратная совместимость)."""
        self._ffi.cdef(r"""
            typedef void (*hq_segment_cb)(const double*, const float*, int, double, uint64_t, uint64_t, void*);
            const char* hq_last_error(void);

            int  hq_open(const char* serial_suffix);
            int  hq_configure(double f_start_mhz, double f_stop_mhz,
                              double requested_bin_hz, int lna_db, int vga_db, int amp_enable);
            int  hq_start(hq_segment_cb cb, void* user);
            int  hq_stop(void);
            void hq_close(void);

            int  hq_device_count(void);
            int  hq_get_device_serial(int idx, char* buf, int buf_len);

            int  hq_load_calibration(const char* csv_path);
            void hq_enable_calibration(int enable);
            int  hq_calibration_loaded(void);
        """)

    # ---------- Общие методы ----------
    
    def set_num_devices(self, num: int):
        """Устанавливает количество устройств для multi-SDR режима."""
        if num < 1 or num > 3:
            raise ValueError("Поддерживается от 1 до 3 устройств")
        self._num_devices = num
        
    def is_multi_capable(self) -> bool:
        """Проверяет, поддерживает ли библиотека multi-SDR режим."""
        return self._multi_mode
        
    def list_serials(self) -> List[str]:
        """Список серийников устройств."""
        if self._multi_mode:
            # Multi-mode - используем новые функции
            out: List[str] = []
            try:
                count = self._lib.hq_get_device_count()
                if count > 0:
                    # Создаем массив указателей на строки
                    serials = self._ffi.new("char*[]", count)
                    actual_count = self._lib.hq_list_devices(serials, count)
                    
                    for i in range(actual_count):
                        if serials[i] != self._ffi.NULL:
                            s = self._ffi.string(serials[i]).decode(errors="ignore")
                            if s and s != "0000000000000000":
                                out.append(s)
                            # Освобождаем память
                            try:
                                import ctypes
                                ctypes.CDLL("libc.so.6").free(serials[i])
                            except Exception:
                                pass  # Игнорируем ошибки освобождения памяти
            except Exception as e:
                print(f"Error listing devices in multi-mode: {e}")
            return out
        else:
            # Single mode - используем старый метод
            out: List[str] = []
            try:
                n = int(self._lib.hq_device_count())
                for i in range(n):
                    b = self._ffi.new("char[128]")
                    if self._lib.hq_get_device_serial(i, b, 127) == 0:
                        s = self._ffi.string(b).decode(errors="ignore")
                        if s and s != "0000000000000000":
                            out.append(s)
            except Exception as e:
                print(f"Error listing devices: {e}")
            return out

    def set_serial_suffix(self, serial: Optional[str]):
        """Задать суффикс серийника (только для single mode)."""
        self._serial_suffix = serial or None

    def load_calibration(self, csv_path: str) -> bool:
        """Загружает калибровку (только для single mode)."""
        if not self._multi_mode:
            try:
                c = self._ffi.new("char[]", csv_path.encode("utf-8"))
                ok = (self._lib.hq_load_calibration(c) == 0)
                if ok:
                    self._lib.hq_enable_calibration(1)
                return bool(ok)
            except Exception:
                return False
        return False

    def set_calibration_enabled(self, enable: bool):
        """Включить/выключить калибровку (только для single mode)."""
        if not self._multi_mode:
            try:
                self._lib.hq_enable_calibration(1 if enable else 0)
            except Exception:
                pass

    def calibration_loaded(self) -> bool:
        """Проверка загруженной калибровки (только для single mode)."""
        if not self._multi_mode:
            try:
                return bool(self._lib.hq_calibration_loaded())
            except Exception:
                pass
        return False

    def last_error(self) -> str:
        """Получить последнее сообщение об ошибке (только для single mode)."""
        if not self._multi_mode:
            try:
                c = self._lib.hq_last_error()
                if c != self._ffi.NULL:
                    return self._ffi.string(c).decode(errors="ignore")
            except Exception:
                pass
        return ""

    # ---------- API SourceBackend ----------
    
    def is_running(self) -> bool:
        if self._multi_mode:
            return self._running and self._multi_worker is not None and self._multi_worker.is_alive()
        else:
            return self._worker is not None and self._worker.is_alive()

    def start(self, config: SweepConfig):
        if self.is_running():
            self.status.emit("Уже запущено")
            return
        
        if self._multi_mode:
            self._start_multi(config)
        else:
            self._start_single(config)
            
    def _start_multi(self, config: SweepConfig):
        """Запуск в multi-SDR режиме."""
        self.status.emit(f"Инициализация multi-SDR ({self._num_devices} устройств)...")
        
        # Открываем устройства
        r = self._lib.hq_open_all(self._num_devices)
        if r != 0:
            self.error.emit(f"Не удалось открыть {self._num_devices} устройств (код: {r})")
            return
        
        # Конфигурируем
        self._lib.hq_config_set_rates(12000000, 8000000)  # 12 MSPS, 8 MHz BB filter
        self._lib.hq_config_set_gains(config.lna_db, config.vga_db, config.amp_on)
        
        # Настраиваем диапазон частот
        start_hz = config.freq_start_hz
        stop_hz = config.freq_end_hz
        step_hz = config.bin_hz
        
        # Простое последовательное сканирование всего диапазона
        print(f"Configuring frequency range: {start_hz/1e6:.1f} - {stop_hz/1e6:.1f} MHz, step {step_hz/1e6:.3f} MHz")
        
        self._lib.hq_config_set_freq_range(start_hz, stop_hz, step_hz)
        self._lib.hq_config_set_dwell_time(2)  # 2 ms dwell time
        
        # Настраиваем параметры группировки
        self._lib.hq_set_grouping_tolerance_hz(250000.0)  # 250 kHz
        self._lib.hq_set_ema_alpha(0.25)
        
        # Запускаем
        r = self._lib.hq_start()
        if r != 0:
            self.error.emit(f"Не удалось запустить multi-SDR (код: {r})")
            self._lib.hq_close_all()
            return
        
        # Запускаем worker для чтения данных
        self._multi_worker = _MultiWorker(self, self._ffi, self._lib, config)
        self._multi_worker.finished_sig.connect(self._on_multi_worker_finished)
        self._multi_worker.start()
        
        self.started.emit()
        self.status.emit(f"Multi-SDR запущен ({self._num_devices} устройств)")
        
        # Устанавливаем флаг запуска
        self._running = True
        
    def _start_single(self, config: SweepConfig):
        """Запуск в single-SDR режиме (обратная совместимость)."""
        self.status.emit("Инициализация single-SDR...")
        self._assembler.configure(config)
        
        self._worker = _Worker(self, self._ffi, self._lib, config, self._serial_suffix, self._assembler)
        self._worker.finished_sig.connect(self._on_worker_finished)
        self._worker.start()
        self.started.emit()

    def stop(self):
        if not self.is_running():
            return
        
        self.status.emit("Остановка...")
        
        if self._multi_mode:
            # Останавливаем библиотеку
            try:
                self._lib.hq_stop()
            except Exception as e:
                print(f"Error stopping library: {e}")
            
            # Останавливаем worker
            if self._multi_worker:
                self._multi_worker.stop()
                self._multi_worker.join(timeout=2.0)
                self._multi_worker = None
            
            # Закрываем устройства
            try:
                self._lib.hq_close_all()
            except Exception as e:
                print(f"Error closing devices: {e}")
        else:
            # Single mode
            try:
                self._lib.hq_stop()
            except Exception:
                pass
            
            if self._worker:
                self._worker.ask_stop()
                self._worker.join(timeout=3.0)
                self._worker = None
        
        # Сбрасываем флаг запуска
        self._running = False
        
        self.finished.emit(0)
        self.status.emit("Остановлено")

    def _on_worker_finished(self, code: int, msg: str):
        """Обработчик завершения single-mode worker."""
        if code != 0:
            self.error.emit(msg or f"libhackrf завершился с кодом {code}")
        self._worker = None
        self.finished.emit(code)
        
    def _on_multi_worker_finished(self, code: int, msg: str):
        """Обработчик завершения multi-mode worker."""
        if code != 0:
            self.error.emit(msg or f"Multi-SDR завершился с кодом {code}")
        self._multi_worker = None
        self._running = False
        self.finished.emit(code)
        
    def get_status(self) -> Optional[Dict[str, Any]]:
        """Получить статус multi-SDR системы."""
        if self._multi_mode and self.is_running():
            try:
                status = self._ffi.new("HqStatus*")
                r = self._lib.hq_get_status(status)
                if r == 0:
                    return {
                        'master_running': bool(status.master_running),
                        'slave1_running': bool(status.slave_running[0]),
                        'slave2_running': bool(status.slave_running[1]),
                        'retune_ms': status.retune_ms_avg,
                        'watch_items': status.watch_items
                    }
            except Exception as e:
                print(f"Error getting status: {e}")
        return None


class _MultiWorker(QtCore.QObject, threading.Thread):
    """Worker для multi-SDR режима."""
    
    finished_sig = QtCore.pyqtSignal(int, str)
    
    def __init__(self, parent: HackRFLibSource, ffi, lib, config: SweepConfig):
        QtCore.QObject.__init__(self)
        threading.Thread.__init__(self, daemon=True)
        
        self._parent = parent
        self._ffi = ffi
        self._lib = lib
        self._config = config
        self._running = True
        
        # Настраиваем диапазон для мастера
        self._freq_start = config.freq_start_hz
        self._freq_end = config.freq_end_hz
        self._bin_hz = config.bin_hz
        
    def stop(self):
        self._running = False
        
    def run(self):
        """Worker для multi-SDR режима с правильной обработкой sweep."""
        # Буферы для чтения данных
        peaks_buf = self._ffi.new("Peak[100]")
        status_buf = self._ffi.new("HqStatus*")

        # Параметры для непрерывного спектра
        last_emit_time = time.time()
        emit_interval = 0.05  # 20 Hz для плавного обновления

        # Создаем сетку частот для полного диапазона
        freq_start = self._freq_start
        freq_end = self._freq_end
        bin_hz = self._bin_hz

        # Вычисляем количество точек
        n_points = int((freq_end - freq_start) / bin_hz) + 1

        # Ограничиваем для стабильности
        if n_points > 50000:
            print(f"Limiting frequency points from {n_points} to 50000")
            n_points = 50000
            bin_hz = (freq_end - freq_start) / (n_points - 1)

        # Буферы для спектра
        freqs_buf = self._ffi.new("double[]", n_points)
        powers_buf = self._ffi.new("float[]", n_points)

        # Инициализируем частоты
        for i in range(n_points):
            freqs_buf[i] = freq_start + i * bin_hz

        print(
            f"Multi-SDR worker started: {n_points} points, "
            f"{freq_start/1e6:.1f}-{freq_end/1e6:.1f} MHz"
        )

        # Счетчик для диагностики
        update_count = 0
        last_status_time = time.time()

        while self._running:
            try:
                current_time = time.time()

                # Читаем спектр от Master SDR
                n_read = self._lib.hq_get_master_spectrum(freqs_buf, powers_buf, n_points)

                if n_read > 0:
                    # Копируем данные
                    freqs_array = np.frombuffer(
                        self._ffi.buffer(freqs_buf, n_read * 8),
                        dtype=np.float64,
                    )
                    powers_array = np.frombuffer(
                        self._ffi.buffer(powers_buf, n_read * 4),
                        dtype=np.float32,
                    )

                    # Эмитим полный спектр
                    if (current_time - last_emit_time) >= emit_interval:
                        self._parent.fullSweepReady.emit(
                            freqs_array.copy(), powers_array.copy()
                        )
                        last_emit_time = current_time
                        update_count += 1

                        # Диагностика каждые 2 секунды
                        if current_time - last_status_time > 2.0:
                            active_points = np.sum(powers_array > -110)
                            print(
                                f"Spectrum update #{update_count}: "
                                f"{active_points}/{n_read} active points"
                            )
                            last_status_time = current_time

                # Читаем последние пики для отображения целей
                n_peaks = self._lib.hq_get_recent_peaks(peaks_buf, 100)

                if n_peaks > 0:
                    for i in range(n_peaks):
                        peak = peaks_buf[i]

                        # Создаем SweepLine для совместимости с детектором
                        sw = SweepLine(
                            ts=None,
                            f_low_hz=int(peak.f_hz - 100000),
                            f_high_hz=int(peak.f_hz + 100000),
                            bin_hz=200000,
                            power_dbm=np.array([peak.rssi_dbm], dtype=np.float32),
                        )
                        self._parent.sweepLine.emit(sw)

                # Читаем статус системы
                self._lib.hq_get_status(status_buf)
                status = status_buf[0]

                # Формируем текст статуса
                status_parts = []
                if status.master_running:
                    status_parts.append("Master: SWEEP")
                else:
                    status_parts.append("Master: OFF")

                slave_count = 0
                if status.slave_running[0]:
                    status_parts.append("S1: TRACK")
                    slave_count += 1
                if status.slave_running[1]:
                    status_parts.append("S2: TRACK")
                    slave_count += 1

                if status.watch_items > 0:
                    status_parts.append(f"Targets: {status.watch_items}")

                self._parent.status.emit(" | ".join(status_parts))

                # Небольшая задержка
                time.sleep(0.02)  # 50 Hz polling

            except Exception as e:
                if self._running:
                    print(f"Multi-worker error: {e}")
                    import traceback
                    traceback.print_exc()
                break

        print("Multi-SDR worker stopped")
        self.finished_sig.emit(0, "Multi-worker stopped")



class _SweepAssembler:
    """Собирает полный проход из сегментов (для single-mode)."""
    
    def __init__(self):
        self.f0_hz = 0
        self.f1_hz = 0
        self.bin_hz = 0
        self.grid = None
        self.n_bins = 0
        self.sum = None
        self.cnt = None
        self.seen = None
        self.prev_low = None
        
    def configure(self, cfg: SweepConfig):
        self.f0_hz = cfg.freq_start_hz
        self.f1_hz = cfg.freq_end_hz
        self.bin_hz = cfg.bin_hz
        
        # Создаем сетку частот
        self.grid = np.arange(
            self.f0_hz + self.bin_hz * 0.5,
            self.f1_hz + self.bin_hz * 0.5,
            self.bin_hz,
            dtype=np.float64
        )
        self.n_bins = len(self.grid)
        self.reset()
    
    def reset(self):
        if self.n_bins == 0:
            return
        self.sum = np.zeros(self.n_bins, np.float64)
        self.cnt = np.zeros(self.n_bins, np.int32)
        self.seen = np.zeros(self.n_bins, bool)
        self.prev_low = None
    
    def add_segment(self, f_hz: np.ndarray, p_dbm: np.ndarray, hz_low: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Добавляет сегмент и возвращает полный проход если готов."""
        if self.grid is None or self.n_bins == 0:
            return None
        
        # Детекция обмотки
        if self.prev_low is not None and hz_low < self.prev_low - 10e6:
            result = self._finalize()
            self.reset()
            self.prev_low = hz_low
            self._add_to_grid(f_hz, p_dbm)
            return result
        
        self.prev_low = hz_low
        self._add_to_grid(f_hz, p_dbm)
        
        # Проверяем покрытие
        coverage = float(self.seen.sum()) / float(self.n_bins) if self.n_bins else 0
        if coverage >= 0.95:
            result = self._finalize()
            self.reset()
            return result
        
        return None
    
    def _add_to_grid(self, f_hz: np.ndarray, p_dbm: np.ndarray):
        """Раскладывает сегмент в сетку."""
        idx = np.rint((f_hz - self.grid[0]) / self.bin_hz).astype(np.int32)
        mask = (idx >= 0) & (idx < self.n_bins)
        if not np.any(mask):
            return
        
        idx = idx[mask]
        p = p_dbm[mask].astype(np.float64)
        
        np.add.at(self.sum, idx, p)
        np.add.at(self.cnt, idx, 1)
        self.seen[idx] = True
    
    def _finalize(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Финализирует текущий проход."""
        if self.n_bins == 0:
            return None
        
        coverage = float(self.seen.sum()) / float(self.n_bins)
        if coverage < 0.5:
            return None
        
        p = np.full(self.n_bins, np.nan, np.float32)
        valid = self.cnt > 0
        p[valid] = (self.sum[valid] / self.cnt[valid]).astype(np.float32)
        
        # Интерполируем пропуски
        if np.isnan(p).any():
            vmask = ~np.isnan(p)
            if vmask.any():
                p = np.interp(np.arange(self.n_bins), np.flatnonzero(vmask), p[vmask]).astype(np.float32)
            p[np.isnan(p)] = -120.0
        
        return self.grid.copy(), p


class _Worker(QtCore.QObject, threading.Thread):
    """Worker для single-SDR режима (обратная совместимость)."""
    
    finished_sig = QtCore.pyqtSignal(int, str)

    def __init__(self, parent_obj: HackRFLibSource, ffi, lib, cfg: SweepConfig, 
                 serial_suffix: Optional[str], assembler: _SweepAssembler):
        QtCore.QObject.__init__(self)
        threading.Thread.__init__(self, daemon=True)
        
        self._parent = parent_obj
        self._ffi = ffi
        self._lib = lib
        self._cfg = cfg
        self._serial = serial_suffix
        self._assembler = assembler
        self._stop_ev = threading.Event()

        @self._ffi.callback("void(const double*, const float*, int, double, uint64_t, uint64_t, void*)")
        def _cb(freqs_ptr, pwr_ptr, n_bins, fft_bin_width_hz, f_low_hz, f_high_hz, user):
            if self._stop_ev.is_set():
                return
            
            try:
                n = int(n_bins)
                if n <= 0:
                    return
                
                # Копируем данные
                freqs = np.frombuffer(self._ffi.buffer(freqs_ptr, n * 8), dtype=np.float64, count=n).copy()
                power = np.frombuffer(self._ffi.buffer(pwr_ptr, n * 4), dtype=np.float32, count=n).copy()
                
                # Проверяем валидность
                if np.all(np.isnan(power)) or np.all(power == 0):
                    return
                
                # Отправляем сегмент
                bin_hz = int(round(float(fft_bin_width_hz))) or int(self._cfg.bin_hz)
                sw = SweepLine(
                    ts=None,
                    f_low_hz=int(f_low_hz),
                    f_high_hz=int(f_high_hz),
                    bin_hz=bin_hz,
                    power_dbm=power
                )
                self._parent.sweepLine.emit(sw)
                
                # Собираем полный проход
                result = self._assembler.add_segment(freqs, power, int(f_low_hz))
                if result is not None:
                    full_freqs, full_power = result
                    self._parent.fullSweepReady.emit(full_freqs, full_power)
                    
            except Exception as e:
                self._parent.status.emit(f"Callback error: {e}")

        self._cb = _cb

    def run(self):
        code = 0
        msg = ""
        
        try:
            # Открываем устройство
            c_ser = self._ffi.NULL if not self._serial else self._ffi.new("char[]", self._serial.encode("utf-8"))
            r = self._lib.hq_open(c_ser)
            if r != 0:
                msg = self._last_err(f"hq_open() failed ({r})")
                self.finished_sig.emit(1, msg)
                return

            # Конфигурируем
            cfg = self._cfg
            r = self._lib.hq_configure(
                float(cfg.freq_start_hz) / 1e6,
                float(cfg.freq_end_hz) / 1e6,
                float(cfg.bin_hz),
                int(cfg.lna_db),
                int(cfg.vga_db),
                int(1 if cfg.amp_on else 0)
            )
            if r != 0:
                msg = self._last_err(f"hq_configure() failed ({r})")
                self._lib.hq_close()
                self.finished_sig.emit(2, msg)
                return

            # Запускаем
            r = self._lib.hq_start(self._cb, self._ffi.NULL)
            if r != 0:
                msg = self._last_err(f"hq_start() failed ({r})")
                self._lib.hq_close()
                self.finished_sig.emit(3, msg)
                return

            # Ждем остановки
            while not self._stop_ev.wait(0.05):
                pass

            self._lib.hq_stop()
            self._lib.hq_close()

        except Exception as e:
            code, msg = 99, f"Worker exception: {e}"
        finally:
            self.finished_sig.emit(code, msg)

    def ask_stop(self):
        self._stop_ev.set()

    def _last_err(self, prefix: str) -> str:
        try:
            c = self._lib.hq_last_error()
            if c != self._ffi.NULL:
                s = self._ffi.string(c).decode(errors="ignore")
                return f"{prefix}: {s}" if prefix else s
        except Exception:
            pass
        return prefix or ""