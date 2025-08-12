from __future__ import annotations
import numpy as np


class SpectrumModel:
    """
    Модель с глобальной сеткой частот: держит последнюю полную строку и водопад.
    """
    def __init__(self, rows: int = 300):
        self.rows = int(rows)
        self.freqs_hz: np.ndarray | None = None
        self.last_row: np.ndarray | None = None
        self.water: np.ndarray | None = None  # (rows, n_bins)

    def set_grid(self, f_start_hz: int, f_end_hz: int, bin_hz: int):
        width = int(f_end_hz - f_start_hz)
        n_bins = int(np.floor(width / bin_hz))
        self.freqs_hz = (f_start_hz + (np.arange(n_bins, dtype=np.float64) + 0.5) * bin_hz)
        self.last_row = np.full(n_bins, -120.0, dtype=np.float32)
        self.water = np.full((self.rows, n_bins), -120.0, dtype=np.float32)

    def append_row(self, row_dbm: np.ndarray):
        assert self.water is not None, "Grid not set"
        assert row_dbm.shape[0] == self.water.shape[1], "Row size mismatch"
        self.last_row = row_dbm.astype(np.float32, copy=False)
        self.water[:-1, :] = self.water[1:, :]
        self.water[-1, :] = self.last_row
