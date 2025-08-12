from __future__ import annotations
import os, threading
from typing import Optional, List
from PyQt5 import QtCore
import numpy as np
from cffi import FFI

from panorama.drivers.base import SourceBackend, SweepConfig
from panorama.shared.parsing import SweepLine


def _candidates() -> List[str]:
    here = os.path.abspath(os.path.dirname(__file__))
    names = ["libhackrf_qsa.so", "libhackrf_qsa.dylib", "hackrf_qsa.dll"]
    return [os.path.join(here, n) for n in names] + names


class HackRFLibSource(SourceBackend):
    """Источник через CFFI + dlopen libhackrf_qsa, API см. hq_sweep.h."""
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

            int  hq_load_calibration(const char* csv_path); // freq_mhz,lna,vga,amp,offset_db
            void hq_enable_calibration(int enable);
            int  hq_calibration_loaded(void);
        """)

        self._lib = None
        last_err = None
        for p in ([lib_path] if lib_path else _candidates()):
            if not p: continue
            try:
                self._lib = self._ffi.dlopen(p)
                break
            except Exception as e:
                last_err = e
        if self._lib is None:
            raise RuntimeError(f"Не удалось загрузить libhackrf_qsa: {last_err}")

        self._serial_suffix = serial_suffix
        self._worker: Optional[_Worker] = None

    # ---------- утилиты ----------
    def list_serials(self) -> List[str]:
        """Список серийников, которые видит библиотека."""
        out: List[str] = []
        try:
            n = int(self._lib.hq_device_count())
            for i in range(n):
                b = self._ffi.new("char[128]")
                if self._lib.hq_get_device_serial(i, b, 127) == 0:
                    s = self._ffi.string(b).decode(errors="ignore")
                    if s:
                        out.append(s)
        except Exception:
            pass
        return out

    def set_serial_suffix(self, serial: Optional[str]):
        """Задать суффикс серийника для следующего запуска."""
        self._serial_suffix = serial or None

    def load_calibration(self, csv_path: str) -> bool:
        try:
            c = self._ffi.new("char[]", csv_path.encode("utf-8"))
            ok = (self._lib.hq_load_calibration(c) == 0)
            if ok:
                self._lib.hq_enable_calibration(1)
            return bool(ok)
        except Exception:
            return False

    def set_calibration_enabled(self, enable: bool):
        try:
            self._lib.hq_enable_calibration(1 if enable else 0)
        except Exception:
            pass

    def calibration_loaded(self) -> bool:
        try:
            return bool(self._lib.hq_calibration_loaded())
        except Exception:
            return False

    def last_error(self) -> str:
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
        self._worker = _Worker(self, self._ffi, self._lib, config, self._serial_suffix)
        self._worker.finished_sig.connect(self._on_worker_finished)
        self._worker.start()
        self.started.emit()

    def stop(self):
        if not self.is_running():
            return
        self._worker.ask_stop()
        self._worker.join(timeout=3.0)
        self._worker = None
        self.finished.emit(0)

    def _on_worker_finished(self, code: int, msg: str):
        if code != 0:
            self.error.emit(msg or f"libhackrf завершился с кодом {code}")
        self._worker = None
        self.finished.emit(code)


class _Worker(QtCore.QObject, threading.Thread):
    finished_sig = QtCore.pyqtSignal(int, str)  # (code, message)

    def __init__(self, parent_obj: HackRFLibSource, ffi: FFI, lib, cfg: SweepConfig, serial_suffix: Optional[str]):
        QtCore.QObject.__init__(self)
        threading.Thread.__init__(self, daemon=True)
        self._parent = parent_obj
        self._ffi = ffi
        self._lib = lib
        self._cfg = cfg
        self._serial = serial_suffix
        self._stop_ev = threading.Event()

        @self._ffi.callback("void(const double*, const float*, int, double, uint64_t, uint64_t, void*)")
        def _cb(freqs_ptr, pwr_ptr, n_bins, fft_bin_width_hz, f_low_hz, f_high_hz, user):
            try:
                n = int(n_bins)
                if n <= 0: return
                power = np.frombuffer(self._ffi.buffer(pwr_ptr, n * 4), dtype=np.float32, count=n).copy()
                bin_hz = int(round(float(fft_bin_width_hz))) or int(self._cfg.bin_hz)
                sw = SweepLine(ts=None,
                               f_low_hz=int(f_low_hz),
                               f_high_hz=int(f_high_hz),
                               bin_hz=bin_hz,
                               power_dbm=power)
                self._parent.sweepLine.emit(sw)
            except Exception as e:
                self._parent.status.emit(f"cffi callback error: {e}")

        self._cb = _cb  # не даём GC его убрать

    def run(self):
        code = 0
        msg = ""
        try:
            # open
            c_ser = self._ffi.NULL if not self._serial else self._ffi.new("char[]", self._serial.encode("utf-8"))
            r = self._lib.hq_open(c_ser)
            if r != 0:
                msg = self._last_err("hq_open() failed")
                self.finished_sig.emit(1, msg); return

            # configure (MHz + bin in Hz)
            cfg = self._cfg
            r = self._lib.hq_configure(float(cfg.freq_start_hz)/1e6,
                                       float(cfg.freq_end_hz)/1e6,
                                       float(cfg.bin_hz),
                                       int(cfg.lna_db), int(cfg.vga_db), int(1 if cfg.amp_on else 0))
            if r != 0:
                msg = self._last_err(f"hq_configure() failed ({r})")
                self._lib.hq_close()
                self.finished_sig.emit(2, msg); return

            # start
            r = self._lib.hq_start(self._cb, self._ffi.NULL)
            if r != 0:
                msg = self._last_err(f"hq_start() failed ({r})")
                self._lib.hq_close()
                self.finished_sig.emit(3, msg); return

            while not self._stop_ev.wait(0.05):
                pass

            self._lib.hq_stop()
            self._lib.hq_close()

        except Exception as e:
            code, msg = 99, f"worker exception: {e}"
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
