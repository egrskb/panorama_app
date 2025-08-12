# panorama/features/spectrum/service.py
from __future__ import annotations

from typing import Optional, Any, Tuple
import numpy as np


class SweepAssembler:
    """
    Собирает ОДНУ полную строку спектра из приходящих сегментов sweep.

    Идея:
      - При configure() задаём единую сетку [f0..f1) с шагом bin_hz.
      - feed(sw) аккуратно раскладывает сегмент в эту сетку
        (по массиву freqs_hz, либо по hz_low/len(data) с равным шагом).
      - Как только покрытие >= threshold → возвращаем копию строки и СРАЗУ
        начинаем новый проход (reset_pass), чтобы не было «швов» между проходами.
      - Детектируем «обмотку» (wrap): если hz_low резко уменьшился — это начало
        нового прохода → сбрасываем неполную строку.

    Ожидаемые поля у `sw` (объект, namedtuple или dict):
        sw.freqs_hz : np.ndarray (N)      – частоты, Гц (если есть — используем их)
        sw.data_dbm : np.ndarray (N)      – мощности, dBm
        sw.hz_low   : float/int           – нижняя частота сегмента, Гц
        sw.hz_high  : float/int           – верхняя частота сегмента, Гц (необязательно)
        sw.bin_hz   : float/int           – ширина бина в сегменте, Гц (опционально)

    Возврат feed(sw): tuple[np.ndarray|None, float]
        full_row : np.ndarray(float32, len=nbins) или None
        coverage : float в [0..1]
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
        # нормализуем границы
        f0 = float(f0_hz)
        f1 = float(f1_hz)
        b  = float(bin_hz)
        if f1 <= f0 or b <= 0.0:
            raise ValueError("Bad grid: f1<=f0 or bin<=0")

        nb = int(round((f1 - f0) / b))
        if nb <= 0:
            raise ValueError("nbins <= 0")

        self.f0, self.f1, self.bin_hz, self.nbins = f0, f1, b, nb
        self.row  = np.full(nb, np.nan, dtype=np.float32)
        self.mask = np.zeros(nb, dtype=bool)
        self._last_low = None
        # LUT (калибровка) можно завести здесь позже; сейчас не используется

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
        low   = float(_get(sw, "hz_low", _infer_low(freqs)))
        high  = float(_get(sw, "hz_high", _infer_high(freqs)))
        # для диагностики wrap сравниваем low с прошлым
        if self._last_low is not None and low < (self._last_low - self.wrap_guard_hz):
            # начался НОВЫЙ проход — не домешиваем к старому
            self.reset_pass()
        self._last_low = low

        y = np.asarray(data, dtype=np.float32)
        if y.ndim != 1:
            y = np.ravel(y).astype(np.float32)

        # Укладка:
        # 1) если есть массив частот — кладём поиндексно через округление в глобальную сетку
        # 2) иначе считаем, что сегмент непрерывный с шагом self.bin_hz (или sw.bin_hz)
        if freqs is not None:
            f = np.asarray(freqs, dtype=np.float64)
            idx = np.round((f - self.f0) / self.bin_hz).astype(np.int64)
            valid = (idx >= 0) & (idx < self.nbins)
            if not np.any(valid):
                return None, float(self.mask.mean())

            idx = idx[valid]
            yv  = y[valid]

            # одинаковые индексы могут повторяться -> берём последнее значение
            # (можно сделать max/mean по дубликатам при желании)
            self.row[idx] = yv
            self.mask[idx] = True

        else:
            # непрерывный отрезок (быстрее)
            seg_bin = float(_get(sw, "bin_hz", self.bin_hz))
            if seg_bin <= 0.0:
                seg_bin = self.bin_hz
            i0 = int(round((low - self.f0) / self.bin_hz))
            if i0 < 0:
                # подрезаем левую часть
                skip = -i0
                y = y[skip:]
                i0 = 0
            if i0 >= self.nbins or y.size == 0:
                return None, float(self.mask.mean())
            span = min(y.size, self.nbins - i0)
            if span <= 0:
                return None, float(self.mask.mean())
            self.row[i0:i0 + span] = y[:span]
            self.mask[i0:i0 + span] = True

        cov = float(self.mask.mean())
        if cov >= self.threshold:
            out = self.row.copy()
            self.reset_pass()  # ← критично: новый проход начинаем «с нуля»
            return out, cov
        return None, cov


# -------------------- утилиты --------------------

def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Достаёт поле `name` из объекта/namedtuple/словаря."""
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
    return float(np.asarray(freqs, dtype=np.float64)[0])


def _infer_high(freqs: Optional[np.ndarray]) -> float:
    if freqs is None or len(freqs) == 0:
        return 0.0
    f = np.asarray(freqs, dtype=np.float64)
    return float(f[-1])
