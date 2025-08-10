import threading
import time

import numpy as np

from .base import SDRWorker

HAVE_HQ = None
HQ_IMPORT_ERR = ""


def _try_import_hq():  # pragma: no cover - optional dependency
    global HAVE_HQ, HQ_IMPORT_ERR
    if HAVE_HQ is not None:
        return HAVE_HQ
    try:
        import hq_cffi  # noqa: F401
        HAVE_HQ = True
    except Exception as e:  # pragma: no cover
        HAVE_HQ = False
        HQ_IMPORT_ERR = str(e)
    return HAVE_HQ


class LibWorker(SDRWorker):
    def __init__(self, f_start_mhz, f_stop_mhz, bin_hz, lna, vga, serial_suffix=""):
        super().__init__()
        self.f0_mhz = float(f_start_mhz)
        self.f1_mhz = float(f_stop_mhz)
        self.bin_hz_req = float(bin_hz)
        self.lna = int(lna)
        self.vga = int(vga)
        self.serial = serial_suffix
        self._stop = threading.Event()
        self._prev_low = None
        self._grid = None
        self._sum = None
        self._cnt = None
        self._seen = None
        self._n = 0

    def stop(self):  # pragma: no cover - requires hardware
        self._stop.set()
        try:
            from hq_cffi import lib

            lib.hq_stop()
        except Exception:
            pass

    def _reset_grid(self):
        f0 = int(round(self.f0_mhz * 1e6))
        f1 = int(round(self.f1_mhz * 1e6))
        bw = float(self.bin_hz_req)
        grid = np.arange(f0 + bw * 0.5, f1 + bw * 0.5, bw, dtype=np.float64)
        self._grid = grid
        self._n = len(grid)
        self._sum = np.zeros(self._n, np.float64)
        self._cnt = np.zeros(self._n, np.int32)
        self._seen = np.zeros(self._n, bool)

    def _add_segment(self, f_hz, p_dbm):
        if self._grid is None or self._n == 0:
            return
        idx = np.rint((f_hz - self._grid[0]) / self.bin_hz_req).astype(np.int32)
        m = (idx >= 0) & (idx < self._n)
        if not np.any(m):
            return
        idx = idx[m]
        p = p_dbm[m].astype(np.float64)
        np.add.at(self._sum, idx, p)
        np.add.at(self._cnt, idx, 1)
        self._seen[idx] = True

    def _finish_sweep(self):
        if self._n == 0:
            return
        coverage = float(self._seen.sum()) / float(self._n)
        if coverage < 0.95:
            self._reset_grid()
            return
        p = np.full(self._n, np.nan, np.float32)
        valid = self._cnt > 0
        p[valid] = (self._sum[valid] / self._cnt[valid]).astype(np.float32)
        if np.isnan(p).any():
            vmask = ~np.isnan(p)
            if vmask.any():
                p = np.interp(np.arange(self._n), np.flatnonzero(vmask), p[vmask]).astype(np.float32)
            p[np.isnan(p)] = -120.0
        self.spectrumReady.emit(self._grid.copy(), p)
        self._reset_grid()

    def run(self):  # pragma: no cover - requires hardware
        if not _try_import_hq():
            self.error.emit(f"Библиотечный режим недоступен: {HQ_IMPORT_ERR}")
            return
        from hq_cffi import ffi, lib

        self._reset_grid()
        self.status.emit("Открытие HackRF (library)…")
        if lib.hq_open(self.serial.encode("utf-8") if self.serial else ffi.NULL) != 0:
            self.error.emit(ffi.string(lib.hq_last_error()).decode("utf-8"))
            return
        if lib.hq_configure(
            self.f0_mhz, self.f1_mhz, self.bin_hz_req, self.lna, self.vga, 0
        ) != 0:
            self.error.emit(ffi.string(lib.hq_last_error()).decode("utf-8"))
            lib.hq_close()
            return
        self.status.emit("Старт свипа (library)…")

        @ffi.callback("void(const double*, const float*, int, double, uint64_t, uint64_t, void*)")
        def on_segment(freqs_ptr, pwr_ptr, count, bin_hz, hz_low, hz_high, user):
            if self._stop.is_set():
                return
            f = np.frombuffer(ffi.buffer(freqs_ptr, count * 8), dtype=np.float64).copy()
            p = np.frombuffer(ffi.buffer(pwr_ptr, count * 4), dtype=np.float32).copy()
            if self._prev_low is not None and hz_low < (self._prev_low - self.bin_hz_req * 10):
                self._finish_sweep()
            self._prev_low = float(hz_low)
            self._add_segment(f, p)

        if lib.hq_start(on_segment, ffi.NULL) != 0:
            self.error.emit(ffi.string(lib.hq_last_error()).decode("utf-8"))
            lib.hq_close()
            return
        try:
            while not self._stop.is_set():
                time.sleep(0.05)
        finally:
            lib.hq_stop()
            self._finish_sweep()
            lib.hq_close()
