from __future__ import annotations
import os, threading, time
from typing import Optional, List, Tuple
from PyQt5 import QtCore
import numpy as np
from cffi import FFI

from panorama.drivers.base import SourceBackend, SweepConfig
from panorama.shared.parsing import SweepLine


def _candidates() -> List[str]:
    """Ищем библиотеку в разных местах."""
    here = os.path.abspath(os.path.dirname(__file__))
    names = ["libhackrf_qsa.so", "libhackrf_qsa.dylib", "hackrf_qsa.dll"]
    candidates = []
    
    # Сначала ищем рядом с этим файлом
    for n in names:
        candidates.append(os.path.join(here, n))
    
    # Потом в корне проекта
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    for n in names:
        candidates.append(os.path.join(root, n))
    
    # И наконец просто имена для dlopen
    candidates.extend(names)
    
    return candidates


class HackRFLibSource(SourceBackend):
    """Источник через CFFI + dlopen libhackrf_qsa с полной поддержкой калибровки."""
    
    def __init__(self, lib_path: Optional[str] = None, serial_suffix: Optional[str] = None, parent=None):
        super().__init__(parent)
        
        self._ffi = FFI()
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

        self._lib = None
        last_err = None
        
        for p in ([lib_path] if lib_path else _candidates()):
            if not p:
                continue
            try:
                self._lib = self._ffi.dlopen(p)
                break
            except Exception as e:
                last_err = e
        
        if self._lib is None:
            raise RuntimeError(f"Не удалось загрузить libhackrf_qsa: {last_err}")

        self._serial_suffix = serial_suffix
        self._worker: Optional[_Worker] = None
        self._assembler = _SweepAssembler()  # Для сборки полных проходов

    # ---------- утилиты ----------
    def list_serials(self) -> List[str]:
        """Список серийников, которые видит библиотека."""
        out: List[str] = []
        try:
            # Сначала инициализируем hackrf если не инициализировано
            # (делается автоматически внутри hq_device_count)
            n = int(self._lib.hq_device_count())
            print(f"Found {n} HackRF devices")  # Для отладки
            
            for i in range(n):
                b = self._ffi.new("char[128]")
                if self._lib.hq_get_device_serial(i, b, 127) == 0:
                    s = self._ffi.string(b).decode(errors="ignore")
                    if s and s != "0000000000000000":  # Фильтруем пустые серийники
                        out.append(s)
                        print(f"Device {i}: {s}")  # Для отладки
        except Exception as e:
            print(f"Error listing devices: {e}")
        return out

    def set_serial_suffix(self, serial: Optional[str]):
        """Задать суффикс серийника для следующего запуска."""
        self._serial_suffix = serial or None

    def load_calibration(self, csv_path: str) -> bool:
        """Загружает калибровку из CSV файла формата: freq_mhz,lna,vga,amp,offset_db"""
        try:
            c = self._ffi.new("char[]", csv_path.encode("utf-8"))
            ok = (self._lib.hq_load_calibration(c) == 0)
            if ok:
                self._lib.hq_enable_calibration(1)
            return bool(ok)
        except Exception:
            return False

    def set_calibration_enabled(self, enable: bool):
        """Включить/выключить применение калибровки."""
        try:
            self._lib.hq_enable_calibration(1 if enable else 0)
        except Exception:
            pass

    def calibration_loaded(self) -> bool:
        """Проверка, загружена ли калибровка."""
        try:
            return bool(self._lib.hq_calibration_loaded())
        except Exception:
            return False

    def last_error(self) -> str:
        """Получить последнее сообщение об ошибке из библиотеки."""
        try:
            c = self._lib.hq_last_error()
            if c != self._ffi.NULL:
                return self._ffi.string(c).decode(errors="ignore")
        except Exception:
            pass
        return ""

    # ---------- API SourceBackend ----------
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self, config: SweepConfig):
        if self.is_running():
            self.status.emit("Уже запущено")
            return
        
        self.status.emit("Инициализация libhackrf_qsa…")
        self._assembler.configure(config)
        
        self._worker = _Worker(self, self._ffi, self._lib, config, self._serial_suffix, self._assembler)
        self._worker.finished_sig.connect(self._on_worker_finished)
        self._worker.start()
        self.started.emit()

    def stop(self):
        if not self.is_running():
            return
        
        # Останавливаем через библиотеку
        try:
            self._lib.hq_stop()
        except Exception:
            pass
        
        self._worker.ask_stop()
        self._worker.join(timeout=3.0)
        self._worker = None
        self.finished.emit(0)

    def _on_worker_finished(self, code: int, msg: str):
        if code != 0:
            self.error.emit(msg or f"libhackrf завершился с кодом {code}")
        self._worker = None
        self.finished.emit(code)


class _SweepAssembler:
    """Собирает полный проход из сегментов, детектирует обмотку."""
    
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
        """
        Добавляет сегмент. Если детектирована обмотка и покрытие достаточное,
        возвращает (freqs, power) полного прохода.
        """
        if self.grid is None or self.n_bins == 0:
            return None
        
        # Детекция обмотки (новый проход)
        if self.prev_low is not None and hz_low < self.prev_low - 10e6:
            # Началcя новый проход, финализируем старый
            result = self._finalize()
            self.reset()
            self.prev_low = hz_low
            # Добавляем текущий сегмент в новый проход
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
        if coverage < 0.5:  # Слишком мало данных
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
    finished_sig = QtCore.pyqtSignal(int, str)  # (code, message)

    def __init__(self, parent_obj: HackRFLibSource, ffi: FFI, lib, cfg: SweepConfig, 
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
                
                # Отладка для высоких частот
                if f_low_hz > 5900000000 and f_low_hz < 6100000000:
                    print(f"CB: {f_low_hz/1e6:.1f}-{f_high_hz/1e6:.1f} MHz, {n} bins")
                
                # Копируем данные из C-памяти
                freqs = np.frombuffer(self._ffi.buffer(freqs_ptr, n * 8), dtype=np.float64, count=n).copy()
                power = np.frombuffer(self._ffi.buffer(pwr_ptr, n * 4), dtype=np.float32, count=n).copy()
                
                # Проверяем валидность данных
                if np.all(np.isnan(power)) or np.all(power == 0):
                    print(f"WARNING: Invalid power data at {f_low_hz/1e6:.1f} MHz")
                    return
                
                # Отправляем сегмент как есть
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
                    # Эмитим полный проход через специальный сигнал
                    self._parent.fullSweepReady.emit(full_freqs, full_power)
                    
            except Exception as e:
                self._parent.status.emit(f"Callback error: {e}")

        self._cb = _cb  # Сохраняем ссылку, чтобы GC не удалил

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