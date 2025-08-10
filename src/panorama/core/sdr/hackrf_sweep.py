import shutil
import subprocess
import threading
import time

import numpy as np

from .base import SDRWorker
from ..parsing import parse_sweep_line


class SweepWorker(SDRWorker):
    def __init__(self, f_start_mhz=2400.0, f_stop_mhz=2483.0, bin_hz=1_000_000,
                 lna=24, vga=20, serial_suffix=""):
        super().__init__()
        self.f0_mhz = float(f_start_mhz)
        self.f1_mhz = float(f_stop_mhz)
        self.bin_hz = int(bin_hz)
        self.lna = int(lna)
        self.vga = int(vga)
        self._stop = threading.Event()
        self._proc = None
        self.serial = serial_suffix
        self._grid = None
        self._sum = None
        self._cnt = None
        self._seen = None
        self._n = 0
        self._prev_low = None

    def stop(self):  # pragma: no cover - requires hardware
        self._stop.set()
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                for _ in range(10):
                    if self._proc.poll() is not None:
                        break
                    time.sleep(0.05)
                if self._proc.poll() is None:
                    self._proc.kill()
        except Exception:
            pass

    def _reset_grid(self):
        f0 = int(round(self.f0_mhz * 1e6))
        f1 = int(round(self.f1_mhz * 1e6))
        bw = float(self.bin_hz)
        grid = np.arange(f0 + bw * 0.5, f1 + bw * 0.5, bw, dtype=np.float64)
        self._grid = grid
        self._n = len(grid)
        self._sum = np.zeros(self._n, np.float64)
        self._cnt = np.zeros(self._n, np.int32)
        self._seen = np.zeros(self._n, bool)

    def _add_segment(self, freqs, power):
        if self._grid is None or self._n == 0:
            return
        idx = np.rint((freqs - self._grid[0]) / self.bin_hz).astype(np.int32)
        m = (idx >= 0) & (idx < self._n)
        if not np.any(m):
            return
        idx = idx[m]
        p = power[m].astype(np.float64)
        np.add.at(self._sum, idx, p)
        np.add.at(self._cnt, idx, 1)
        self._seen[idx] = True

    def _finish_sweep(self):
        coverage = float(self._seen.sum()) / float(self._n) if self._n else 0.0
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
        if not shutil.which("hackrf_sweep"):
            self.error.emit("Не найден 'hackrf_sweep' в PATH.")
            return
        f0 = int(round(self.f0_mhz))
        f1 = int(round(self.f1_mhz))
        if f1 <= f0:
            self.error.emit("Fstop (МГц) должен быть > Fstart (МГц).")
            return
        self._reset_grid()
        cmd = [
            "hackrf_sweep",
            "-f",
            f"{f0}:{f1}",
            "-w",
            str(self.bin_hz),
            "-l",
            str(self.lna),
            "-g",
            str(self.vga),
        ]
        if self.serial:
            cmd.extend(["-d", self.serial])
        self.status.emit(" ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
        except Exception as e:
            self.error.emit(f"Не удалось запустить hackrf_sweep: {e}")
            return
        try:
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                if not line or line.startswith("Exiting") or "hackrf_" in line:
                    continue
                parsed = parse_sweep_line(line)
                if parsed is None:
                    continue
                fseg, pseg, _, f_low, _ = parsed
                if self._prev_low is not None and f_low < self._prev_low - self.bin_hz * 10:
                    self._finish_sweep()
                self._add_segment(fseg, pseg)
                self._prev_low = f_low
            self._finish_sweep()
        finally:
            try:
                if self._proc and self._proc.poll() is None:
                    self._proc.terminate()
            except Exception:
                pass
