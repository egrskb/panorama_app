# panorama/features/spectrum/service.py
from __future__ import annotations

from typing import Optional, Any, Tuple
import numpy as np


class SweepAssembler:
    """
    Собирает ОДНУ полную строку спектра из приходящих сегментов sweep.
    Логика:
      - глобальная сетка [f0, f1) с шагом bin_hz (центры бинов: f0 + (i+0.5)*bin_hz)
      - кладём сегменты либо по точным freq-х, либо «непрерывно» от hz_low
      - отслеживаем wrap: если низ частоты внезапно «перескочил» далеко назад — начинаем новую строку
      - как только покрытие >= threshold — возвращаем готовую строку и сбрасываемся
    """

    def __init__(self, coverage_threshold: float = 0.95, wrap_guard_hz: float = 15e6):
        # Порог покрытия чуть ниже 1.0, чтобы микропотери не блокировали кадры
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

        # нижняя частота сегмента (если нет — прикидываем из freqs)
        low   = float(_get(sw, "hz_low", _infer_low(freqs)))

        # wrap: если «прыгнули» к началу — начинаем новый проход
        if self._last_low is not None and low < (self._last_low - self.wrap_guard_hz):
            self.reset_pass()
        self._last_low = low

        y = np.asarray(data, dtype=np.float32).ravel()
        if y.size == 0:
            return None, float(self.mask.mean())

        if freqs is not None:
            # укладка по индексам глобальной сетки (через freqs)
            f = np.asarray(freqs, dtype=np.float64).ravel()
            # округление к ближайшему бину глобальной сетки:
            idx = np.round((f - self.f0) / self.bin_hz).astype(np.int64)

            # допускаем небольшую неточность (Fs/N != bin_hz): фильтруем валидные
            valid = (idx >= 0) & (idx < self.nbins)
            if np.any(valid):
                self.row[idx[valid]] = y[valid]
                self.mask[idx[valid]] = True

            # если вообще ничего не легло — попробуем непрерывной укладкой
            elif low == low:  # low is finite
                i0 = int(round((low - self.f0) / self.bin_hz))
                if i0 < 0:
                    y = y[-i0:]
                    i0 = 0
                if i0 < self.nbins and y.size > 0:
                    span = min(y.size, self.nbins - i0)
                    if span > 0:
                        self.row[i0:i0 + span] = y[:span]
                        self.mask[i0:i0 + span] = True
        else:
            # быстрый непрерывный сегмент
            i0 = int(round((low - self.f0) / self.bin_hz))
            if i0 < 0:
                y = y[-i0:]
                i0 = 0
            if i0 < self.nbins and y.size > 0:
                span = min(y.size, self.nbins - i0)
                if span > 0:
                    self.row[i0:i0 + span] = y[:span]
                    self.mask[i0:i0 + span] = True

        # покрытие
        cov = float(self.mask.mean())

        # микро-заплатка: если не хватает считанные бины на стыках — заполняем ближайшими
        if cov < self.threshold:
            # добиваем одиночные «дырочки» интерполяцией ближайшего соседа
            holes = np.where(~self.mask)[0]
            if holes.size:
                filled = 0
                for k in holes:
                    # ищем ближайший установленный бином слева/справа
                    left = k - 1
                    right = k + 1
                    v = None
                    if left >= 0 and self.mask[left]:
                        v = self.row[left]
                    if v is None and right < self.nbins and self.mask[right]:
                        v = self.row[right]
                    if v is not None:
                        self.row[k] = v
                        self.mask[k] = True
                        filled += 1
                if filled:
                    cov = float(self.mask.mean())

        if cov >= self.threshold:
            out = np.where(np.isnan(self.row), -120.0, self.row).astype(np.float32, copy=False)
            self.reset_pass()
            # ВАЖНО: вернуть массив, а не метод
            return out.copy(), cov

        return None, cov


# ---------------- утилиты ----------------

def _get(d: Any, key: str, default=None):
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _infer_low(freqs: Optional[np.ndarray]) -> float:
    if freqs is None:
        return float("nan")
    f = np.asarray(freqs, dtype=np.float64)
    return float(f.min()) if f.size else float("nan")
