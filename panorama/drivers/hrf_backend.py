#!/usr/bin/env python3
"""
HackRF sweep backend (CFFI) для Panorama.
Работает через libhackrf_qsa.so (наш C-бэкенд с RX_SWEEP).
ЖЁСТКО: серийник обязателен — без него не стартуем.
"""

from __future__ import annotations
import numpy as np, threading
from pathlib import Path
from cffi import FFI
from PyQt5 import QtCore

from panorama.drivers.base import SourceBackend, SweepConfig
from panorama.features.spectrum.service import SweepAssembler


def _qt_parent(parent):
    """Вернёт parent, только если он QObject; иначе None (защита от Logger и пр.)."""
    return parent if isinstance(parent, QtCore.QObject) else None


def _load_lib():
    ffi = FFI()
    ffi.cdef(r"""
        typedef void (*hq_segment_cb)(
            const double* freqs_hz,
            const float*  data_dbm,
            int count,
            double fft_bin_width_hz,
            uint64_t hz_low, uint64_t hz_high,
            void* user
        );

        int  hq_open(const char* serial_suffix_or_null);
        void hq_close(void);

        int  hq_configure(double f_start_mhz, double f_stop_mhz, double bin_hz,
                          int lna_db, int vga_db, int amp_on);

        int  hq_start(hq_segment_cb cb, void* user);
        int  hq_stop(void);

        const char* hq_last_error(void);

        int  hq_device_count(void);
        int  hq_get_device_serial(int idx, char* out, int cap);
    """)
    libname = "libhackrf_qsa.so"
    search_paths = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path("/usr/local/lib"),
        Path("/usr/lib"),
        Path.cwd(),
    ]
    libpath = None
    for p in search_paths:
        f = p / libname
        if f.exists():
            libpath = str(f)
            break
    if libpath is None:
        libpath = libname
    lib = ffi.dlopen(libpath)
    return ffi, lib


def _list_serials_via_lib() -> list[str]:
    out: list[str] = []
    try:
        ffi = FFI()
        ffi.cdef("int hq_device_count(void); int hq_get_device_serial(int idx, char* out, int cap);")
        lib = ffi.dlopen("libhackrf_qsa.so")
        n = int(lib.hq_device_count())
        for i in range(n):
            buf = ffi.new("char[128]")
            if int(lib.hq_get_device_serial(i, buf, 127)) == 0:
                s = ffi.string(buf).decode(errors="ignore").strip()
                if s:
                    out.append(s)
    except Exception:
        pass
    return out


class HackRFQSABackend(SourceBackend):
    """
    Источник sweep-сегментов через наш C-бэкенд.
    Каждый сегмент = четверть окна sweep (как в hackrf_sweep).
    SweepAssembler собирает полную строку спектра.
    """

    def __init__(self, *args, **kwargs):
        # молча проглатываем любые неожиданные kwargs
        self.log = kwargs.pop("logger", None)
        serial_suffix = kwargs.pop("serial_suffix", None)
        parent = kwargs.pop("parent", None)
        
        super().__init__(_qt_parent(parent))
        self._ffi, self._lib = _load_lib()
        self._serial = (serial_suffix or "").strip()
        self._worker: _Worker | None = None
        self._assembler: SweepAssembler | None = None

    def start(self, cfg, on_segment=None, on_full=None, on_full_pass=None,
              on_status=None, on_error=None):
        # совместимость с разными именами callback'ов
        if on_full is None and on_full_pass is not None:
            on_full = on_full_pass
        self._cb_segment = on_segment
        self._cb_full = on_full
        self._cb_status = on_status
        self._cb_error = on_error
        
        if self.is_running():
            return

        # Серийник обязателен: берём из конфига либо из конструктора.
        serial = (cfg.serial or self._serial or "").strip()
        if not serial:
            avail = _list_serials_via_lib()
            hint = f"Доступные серийники: {', '.join(avail)}" if avail else "Устройства HackRF не найдены"
            self.error.emit(f"Серийный номер обязателен (serial). {hint}")
            self.finished.emit(1)
            return

        self._assembler = SweepAssembler()
        self._assembler.configure(cfg.freq_start_hz,
                                  cfg.freq_end_hz,
                                  cfg.bin_hz)

        self._worker = _Worker(self, self._ffi, self._lib,
                               cfg, serial, self._assembler)
        self._worker.finished_sig.connect(self._on_finished)
        self._worker.start()
        self.started.emit()

    def stop(self):
        if self._worker:
            self._worker.stop()
            self._worker.wait(2000)
            self._worker = None

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    @QtCore.pyqtSlot(int, str)
    def _on_finished(self, code: int, msg: str):
        if code != 0 and msg:
            self.error.emit(msg)
        self.finished.emit(code)

    @staticmethod
    def enumerate_devices() -> list[str]:
        return _list_serials_via_lib()


class _Worker(QtCore.QThread):
    finished_sig = QtCore.pyqtSignal(int, str)

    def __init__(self, backend: HackRFQSABackend, ffi, lib, config, serial: str,
                 assembler: SweepAssembler):
        super().__init__(_qt_parent(backend))
        self._backend = backend
        self._ffi = ffi
        self._lib = lib
        self._config = config
        self._serial = serial
        self._assembler = assembler
        self._stop = threading.Event()

        @ffi.callback("void(const double*,const float*,int,double,uint64_t,uint64_t,void*)")
        def _cb(freqs_ptr, pwr_ptr, n_bins, bin_w, low, high, user):
            if self._stop.is_set():
                return
            try:
                freqs = np.frombuffer(ffi.buffer(freqs_ptr, n_bins*8),
                                      dtype=np.float64, count=n_bins).copy()
                power = np.frombuffer(ffi.buffer(pwr_ptr, n_bins*4),
                                      dtype=np.float32, count=n_bins).copy()
                row, coverage = self._assembler.feed({
                    "freqs_hz": freqs,
                    "data_dbm": power,
                    "hz_low": float(low),
                })
                if row is not None:
                    print(f"[HackRF Worker] Assembler returned row: size={row.size}, coverage={coverage:.3f}")
                    freqs = self._assembler.freq_grid()
                    print(f"[HackRF Worker] Freq grid: size={freqs.size}, range=[{freqs[0]/1e6:.1f}, {freqs[-1]/1e6:.1f}] MHz")
                    self._backend.fullSweepReady.emit(freqs, row)
                    self._backend.status.emit(f"Spectrum coverage: {coverage*100:.1f}%")
                    print(f"[HackRF Worker] fullSweepReady signal emitted")
            except Exception as e:
                self._backend.error.emit(f"backend callback error: {e}")

        self._cb = _cb

    def run(self):
        code, msg = 0, ""
        try:
            sfx = self._ffi.new("char[]", self._serial.encode())
            if self._lib.hq_open(sfx) != 0:
                err = self._ffi.string(self._lib.hq_last_error()).decode(errors="ignore") if hasattr(self._lib, "hq_last_error") else "hq_open failed"
                raise RuntimeError(err or "hq_open failed")

            if self._lib.hq_configure(
                self._config.freq_start_hz / 1e6,
                self._config.freq_end_hz / 1e6,
                self._config.bin_hz,
                self._config.lna_db,
                self._config.vga_db,
                1 if self._config.amp_on else 0
            ) != 0:
                err = self._ffi.string(self._lib.hq_last_error()).decode(errors="ignore") if hasattr(self._lib, "hq_last_error") else "hq_configure failed"
                raise RuntimeError(err or "hq_configure failed")

            if self._lib.hq_start(self._cb, self._ffi.NULL) != 0:
                err = self._ffi.string(self._lib.hq_last_error()).decode(errors="ignore") if hasattr(self._lib, "hq_last_error") else "hq_start failed"
                raise RuntimeError(err or "hq_start failed")

            while not self._stop.is_set():
                self.msleep(20)

            self._lib.hq_stop()

        except Exception as e:
            code, msg = 1, str(e)
        finally:
            try:
                self._lib.hq_close()
            except Exception:
                pass
            self.finished_sig.emit(code, msg)

    def stop(self):
        self._stop.set()


# Для совместимости со старой проверкой окружения:
HackRFMaster = HackRFQSABackend
__all__ = ["HackRFQSABackend", "HackRFMaster"]
