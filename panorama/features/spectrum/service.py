# panorama/features/spectrum/service.py
from __future__ import annotations

from typing import Optional, Any, Tuple
import numpy as np


class SweepAssembler:
    """
    Собирает ОДНУ полную строку спектра из приходящих сегментов sweep.
    См. комментарии внутри — логика «wrap» и «coverage threshold».
    """

    def __init__(self, coverage_threshold: float = 0.995, wrap_guard_hz: float = 15e6):
        self.threshold = float(coverage_threshold)
        self.wrap_guard_hz = float(wrap_guard_hz)

        # конфигурация сетки
        self.f0: float = 0.0
        self.f1: float = 0.0
        self.bin_hz: float = 0.0
        self.nbins: int = 0

        # накопители текущего прохода
        self.row: Optional[np.ndarray] = None  # float32, len=nbins
        self.mask: Optional[np.ndarray] = None  # bool, len=nbins

        # служебное для wrap-детектора
        self._last_low: Optional[float] = None

    # ---------------- конфигурация/сброс ----------------

    def configure(self, f0_hz: int, f1_hz: int, bin_hz: int, lut: Any = None) -> None:
        f0 = float(f0_hz)
        f1 = float(f1_hz)
        b  = float(bin_hz)
        if f1 <= f0 or b <= 0.0:
            raise ValueError("Bad grid: f1<=f0 or bin<=0")

        # Используем round для согласования с мастером (Fs/N)
        nb = int(round((f1 - f0) / b))
        if nb <= 0:
            raise ValueError("nbins <= 0")

        self.f0, self.f1, self.bin_hz, self.nbins = f0, f1, b, nb
        self.row  = np.full(nb, np.nan, dtype=np.float32)
        self.mask = np.zeros(nb, dtype=bool)
        self._last_low = None

    def reset_pass(self) -> None:
        if self.row is not None:
            self.row.fill(np.nan)
        if self.mask is not None:
            self.mask[:] = False
        self._last_low = None

    # ---------------- основная укладка ----------------

    def feed(self, sw: Any) -> Tuple[Optional[np.ndarray], float]:
        """Кладёт сегмент. При достаточном покрытии возвращает (row, coverage)."""
        if self.row is None or self.mask is None:
            return None, 0.0

        # извлекаем поля сегмента
        freqs = _get(sw, "freqs_hz", None)
        data  = _get(sw, "data_dbm", None)
        if data is None:
            return None, float(self.mask.mean())

        low   = float(_get(sw, "hz_low", _infer_low(freqs)))
        # wrap: если «прыгнули» к началу — начинаем новый проход
        if self._last_low is not None and low < (self._last_low - self.wrap_guard_hz):
            self.reset_pass()
        self._last_low = low

        y = np.asarray(data, dtype=np.float32).ravel()
        if y.size == 0:
            return None, float(self.mask.mean())

        if freqs is not None:
            # укладка по индексам глобальной сетки
            f = np.asarray(freqs, dtype=np.float64).ravel()
            idx = np.round((f - self.f0) / self.bin_hz).astype(np.int64)
            valid = (idx >= 0) & (idx < self.nbins)
            if not np.any(valid):
                return None, float(self.mask.mean())
            self.row[idx[valid]] = y[valid]
            self.mask[idx[valid]] = True
        else:
            # быстрый непрерывный сегмент
            i0 = int(round((low - self.f0) / self.bin_hz))
            if i0 < 0:
                y = y[-i0:]
                i0 = 0
            if i0 >= self.nbins or y.size <= 0:
                return None, float(self.mask.mean())
            span = min(y.size, self.nbins - i0)
            if span > 0:
                self.row[i0:i0 + span] = y[:span]
                self.mask[i0:i0 + span] = True

        cov = float(self.mask.mean())
        if cov >= self.threshold:
            out = np.where(np.isnan(self.row), -120.0, self.row).astype(np.float32, copy=False)
            self.reset_pass()
            return out.copy(), cov
        return None, cov


# -------------------- утилиты --------------------

def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _infer_low(freqs: Optional[np.ndarray]) -> float:
    if freqs is None or len(freqs) == 0:
        return 0.0
    return float(np.asarray(freqs, dtype=np.float64).ravel()[0])
