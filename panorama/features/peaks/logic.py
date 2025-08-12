from __future__ import annotations
import numpy as np

def find_peak_indices(y: np.ndarray, min_distance_bins: int = 3, threshold_dbm: float = -80.0) -> np.ndarray:
    """Простой локальный максимум без scipy: устойчиво и быстро."""
    if y is None or y.size < 3:
        return np.array([], dtype=np.int32)

    y1 = y[1:-1]
    mask = (y1 > y[:-2]) & (y1 >= y[2:]) & (y1 >= threshold_dbm)
    idx = np.nonzero(mask)[0] + 1  # сдвиг из-за обрезки краёв

    if idx.size == 0 or min_distance_bins <= 1:
        return idx.astype(np.int32)

    # Жадное прореживание по амплитуде
    order = np.argsort(y[idx])[::-1]
    taken = np.zeros(y.size, dtype=bool)
    keep = []
    for j in order:
        i = idx[j]
        if taken[i]:
            continue
        keep.append(i)
        lo = max(0, i - min_distance_bins)
        hi = min(y.size, i + min_distance_bins + 1)
        taken[lo:hi] = True
    keep.sort()
    return np.array(keep, dtype=np.int32)

def summarize(freqs_hz: np.ndarray, y: np.ndarray, idx: np.ndarray):
    rows = []
    for i in idx:
        rows.append({
            "freq_hz": float(freqs_hz[i]),
            "freq_mhz": float(freqs_hz[i] / 1e6),
            "level_dbm": float(y[i]),
        })
    return rows
