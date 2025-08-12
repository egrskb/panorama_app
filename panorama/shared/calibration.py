from __future__ import annotations
from functools import lru_cache
import numpy as np
from typing import Optional, Tuple


# Формат CSV: "freq_hz,offset_db" (без заголовка) — пример для LUT
# Если у тебя другой формат — позже адаптируем парсер.

@lru_cache(maxsize=8)
def _load_raw_lut(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                freq = float(parts[0])
                offs = float(parts[1])
                data.append((freq, offs))
            except ValueError:
                continue
    if not data:
        raise ValueError(f"Empty or bad LUT file: {path}")
    arr = np.asarray(data, dtype=np.float64)
    freq_hz = arr[:, 0]
    off_db = arr[:, 1]
    # гарантируем сортировку по частоте
    idx = np.argsort(freq_hz)
    return freq_hz[idx], off_db[idx]


def load_lut(path: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path:
        return None
    return _load_raw_lut(path)


def apply_lut(freqs_hz: np.ndarray, power_dbm: np.ndarray,
              lut: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Применяет калибровку (прибавляет оффсет) к power_dbm по частотам freqs_hz.
    Если lut=None — вернёт исходные значения.
    """
    if lut is None:
        return power_dbm
    f_lut, off_lut = lut
    offs = np.interp(freqs_hz.astype(np.float64), f_lut, off_lut).astype(np.float32)
    return (power_dbm.astype(np.float32) + offs)
