from __future__ import annotations
import os
import numpy as np

def write_row_csv(path: str, freqs_hz: np.ndarray, row_dbm: np.ndarray) -> None:
    """Сохраняет текущую полную строку: freq_hz, freq_mhz, dbm."""
    if freqs_hz is None or row_dbm is None:
        raise ValueError("Нет данных для экспорта")
    if freqs_hz.shape[0] != row_dbm.shape[0]:
        raise ValueError("Размеры частот и амплитуд не совпадают")
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        f.write("freq_hz,freq_mhz,dbm\n")
        for fz, y in zip(freqs_hz, row_dbm):
            f.write(f"{float(fz):.3f},{float(fz)/1e6:.6f},{float(y):.2f}\n")
