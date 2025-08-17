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
        # Используем round, чтобы сетка совпадала с покрытием sweep'а
        width = float(f_end_hz - f_start_hz)
        bw = float(bin_hz)
        n_bins = int(np.round(width / bw))
        n_bins = max(1, n_bins)

        # Центрируем бины (как у мастера: (i+0.5)*bin_w)
        self.freqs_hz = (f_start_hz + (np.arange(n_bins, dtype=np.float64) + 0.5) * bw)

        self.last_row = np.full(n_bins, -120.0, dtype=np.float32)
        self.water = np.full((self.rows, n_bins), -120.0, dtype=np.float32)

    def append_row(self, row_dbm: np.ndarray):
        assert self.water is not None, "Grid not set"
        # Если пришла строка не того размера — игнорируем (UI сам инициирует reset при смене)
        if row_dbm.shape[0] != self.water.shape[1]:
            return
        self.last_row = row_dbm.astype(np.float32, copy=False)
        # Сдвиг вверх, свежая строка вниз (индекс -1)
        self.water[:-1, :] = self.water[1:, :]
        self.water[-1, :] = self.last_row
