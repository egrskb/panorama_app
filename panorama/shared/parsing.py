from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import numpy as np


@dataclass
class SweepLine:
    ts: Optional[datetime]
    f_low_hz: int
    f_high_hz: int
    bin_hz: int
    power_dbm: np.ndarray  # shape (n_bins,)

    @property
    def n_bins(self) -> int:
        return self.power_dbm.size


def _safe_int(x: str) -> int:
    return int(float(x.strip()))  # иногда встречаются "числа" с .0


def parse_sweep_line(line: str) -> SweepLine:
    """
    Парсит одну CSV-строку из hackrf_sweep:
    date, time, hz_low, hz_high, hz_bin_width, num_samples, dB, dB, ...

    Возвращает SweepLine, где power_dbm — np.ndarray длиной n_bins.
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        raise ValueError(f"Bad sweep line (too few columns): {line[:120]}")

    # первые два поля — дата/время; могут быть пустыми, если источник не такой
    ts: Optional[datetime] = None
    try:
        d, t = parts[0], parts[1]
        if d and t:
            ts = datetime.fromisoformat(f"{d} {t}")
    except Exception:
        ts = None  # безопасно игнорируем странные форматы времени

    f_low = _safe_int(parts[2])
    f_high = _safe_int(parts[3])
    bin_hz = _safe_int(parts[4])

    # parts[5] — num_samples, но реальные точки = len(оставшихся значений)
    power_vals: List[float] = []
    for s in parts[6:]:
        if not s:
            continue
        try:
            power_vals.append(float(s))
        except ValueError:
            # на случай мусора в конце строки
            break

    power = np.asarray(power_vals, dtype=np.float32)
    if power.size == 0:
        raise ValueError(f"No power bins parsed from line: {line[:120]}")

    return SweepLine(
        ts=ts,
        f_low_hz=f_low,
        f_high_hz=f_high,
        bin_hz=bin_hz,
        power_dbm=power,
    )


def bins_to_freqs(f_low_hz: int, bin_hz: int, n_bins: int, centers: bool = True) -> np.ndarray:
    """
    Возвращает массив частот для бинов.
    По умолчанию — центры бинов: f = f_low + (i + 0.5) * bin_hz
    Если centers=False — левые края бинов: f = f_low + i * bin_hz
    """
    idx = np.arange(n_bins, dtype=np.float64)
    if centers:
        return f_low_hz + (idx + 0.5) * bin_hz
    else:
        return f_low_hz + idx * bin_hz
