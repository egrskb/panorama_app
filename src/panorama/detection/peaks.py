from typing import List, Tuple

import numpy as np


Peak = Tuple[float, float, float, int]


def _find_peaks_simple(x_mhz: np.ndarray, y_dbm: np.ndarray,
                        min_dbm: float, min_sep_khz: float) -> List[Peak]:
    if x_mhz.size < 3:
        return []
    peaks: List[Peak] = []
    for i in range(1, len(y_dbm) - 1):
        if y_dbm[i] > y_dbm[i - 1] and y_dbm[i] > y_dbm[i + 1] and y_dbm[i] >= min_dbm:
            th = y_dbm[i] - 3.0
            l = i
            while l > 0 and y_dbm[l] > th:
                l -= 1
            r = i
            while r < len(y_dbm) - 1 and y_dbm[r] > th:
                r += 1
            width_khz = (x_mhz[max(r, i)] - x_mhz[min(l, i)]) * 1000.0
            peaks.append((x_mhz[i], y_dbm[i], width_khz, i))
    peaks.sort(key=lambda t: -t[1])
    filtered: List[Peak] = []
    sep = min_sep_khz / 1000.0
    for p in peaks:
        if not filtered or all(abs(p[0] - q[0]) >= sep for q in filtered):
            filtered.append(p)
    return filtered


def find_peaks(x_mhz: np.ndarray, y_dbm: np.ndarray,
               min_dbm: float = -80.0, min_sep_khz: float = 30.0) -> List[Peak]:
    """Public API to search for peaks in spectrum."""
    return _find_peaks_simple(x_mhz, y_dbm, min_dbm, min_sep_khz)
