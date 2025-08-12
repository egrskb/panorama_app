from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from panorama.shared.parsing import SweepLine, bins_to_freqs
from panorama.shared.calibration import apply_lut


@dataclass
class GridSpec:
    f_start: int
    f_end: int
    bin_hz: int

    @property
    def n_bins(self) -> int:
        width = int(self.f_end - self.f_start)
        return int(np.floor(width / self.bin_hz))

    def centers(self) -> np.ndarray:
        return bins_to_freqs(self.f_start, self.bin_hz, self.n_bins, centers=True).astype(np.float64)


class SweepAssembler:
    """
    Собирает из отдельных строк (сегментов) полную строку спектра по заданной сетке.
    Когда покрытие >= coverage_threshold — считает проход завершённым и отдаёт готовую строку.
    """
    def __init__(self, coverage_threshold: float = 0.95):
        self.grid: Optional[GridSpec] = None
        self.coverage_threshold = float(coverage_threshold)
        self._row: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None  # True — заполнено
        self._freq_centers: Optional[np.ndarray] = None
        self._lut = None  # (f_lut, off_lut)

    def configure(self, f_start_hz: int, f_end_hz: int, bin_hz: int, lut=None):
        self.grid = GridSpec(f_start_hz, f_end_hz, bin_hz)
        n = self.grid.n_bins
        self._row = np.full(n, -120.0, dtype=np.float32)
        self._mask = np.zeros(n, dtype=np.bool_)
        self._freq_centers = self.grid.centers()
        self._lut = lut

    def reset_pass(self):
        if not self.grid:
            return
        n = self.grid.n_bins
        self._row[:] = -120.0
        self._mask[:] = False

    def feed(self, sw: SweepLine) -> Tuple[Optional[np.ndarray], float]:
        """
        Кладёт сегмент свипа. Возвращает (готовая_строка | None, покрытие_0..1).
        """
        assert self.grid is not None, "Assembler not configured"

        # Частоты текущего сегмента (центры бинов)
        seg_f = bins_to_freqs(sw.f_low_hz, sw.bin_hz, sw.n_bins, centers=True).astype(np.float64)
        seg_y = sw.power_dbm.astype(np.float32)

        # Применим калибровку, если задана
        seg_y = apply_lut(seg_f, seg_y, self._lut)

        # Если bin_hz отличается — ресэмплим сегмент в глобальную сетку
        if abs(sw.bin_hz - self.grid.bin_hz) > max(1, self.grid.bin_hz * 0.01):
            # интерполируем на пересечение диапазонов
            f0, f1 = seg_f[0], seg_f[-1]
            g = self._freq_centers
            lo = np.searchsorted(g, f0, side="left")
            hi = np.searchsorted(g, f1, side="right")
            if hi > lo:
                self._row[lo:hi] = np.interp(g[lo:hi], seg_f, seg_y).astype(np.float32)
                self._mask[lo:hi] = True
        else:
            # Прямая укладка в глобальный буфер
            i0 = int(np.floor((sw.f_low_hz - self.grid.f_start) / self.grid.bin_hz))
            if i0 < 0:
                # обрезаем левый хвост
                seg_skip = -i0
                if seg_skip >= sw.n_bins:
                    return None, float(self._mask.mean())
                i0 = 0
            else:
                seg_skip = 0
            i1 = i0 + (sw.n_bins - seg_skip)
            if i1 > self.grid.n_bins:
                # обрезаем правый хвост
                cut = i1 - self.grid.n_bins
                i1 = self.grid.n_bins
            else:
                cut = 0
            if i1 > i0:
                self._row[i0:i1] = seg_y[seg_skip: sw.n_bins - cut]
                self._mask[i0:i1] = True

        cov = float(self._mask.mean())
        if cov >= self.coverage_threshold:
            full = self._row.copy()
            self.reset_pass()
            return full, cov
        return None, cov
