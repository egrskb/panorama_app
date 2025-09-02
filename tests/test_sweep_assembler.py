import importlib.util
import numpy as np
import pytest

# Ensure new backend module is importable
spec = importlib.util.find_spec("panorama.drivers.hackrf.hrf_backend")
if spec is None:
    pytest.skip("hrf_backend not importable", allow_module_level=True)

# Minimal local assembler (to decouple from internal non-exported types)
class _Cfg:
    def __init__(self, freq_start_hz: int, freq_end_hz: int, bin_hz: int):
        self.freq_start_hz = freq_start_hz
        self.freq_end_hz = freq_end_hz
        self.bin_hz = bin_hz

class _SweepAssembler:
    def __init__(self):
        self.f0_hz = 0
        self.f1_hz = 0
        self.bin_hz = 0
        self.grid = None
        self.n_bins = 0
        self.sum = None
        self.cnt = None
        self.seen = None
        self.prev_low = None

    def configure(self, cfg: _Cfg):
        self.f0_hz = cfg.freq_start_hz
        self.f1_hz = cfg.freq_end_hz
        self.bin_hz = cfg.bin_hz
        self.grid = np.arange(
            self.f0_hz + self.bin_hz * 0.5,
            self.f1_hz + self.bin_hz * 0.5,
            self.bin_hz,
            dtype=np.float64,
        )
        self.n_bins = len(self.grid)
        self.reset()

    def reset(self):
        if self.n_bins == 0:
            return
        self.sum = np.zeros(self.n_bins, np.float64)
        self.cnt = np.zeros(self.n_bins, np.int32)
        self.seen = np.zeros(self.n_bins, bool)
        self.prev_low = None

    def add_segment(self, f_hz: np.ndarray, p_dbm: np.ndarray, hz_low: int):
        if self.grid is None or self.n_bins == 0:
            return None
        if self.prev_low is not None and hz_low < self.prev_low - 10e6:
            result = self._finalize()
            self.reset()
            self.prev_low = hz_low
            self._add_to_grid(f_hz, p_dbm)
            return result
        self.prev_low = hz_low
        self._add_to_grid(f_hz, p_dbm)
        coverage = float(self.seen.sum()) / float(self.n_bins) if self.n_bins else 0
        if coverage >= 0.95:
            result = self._finalize()
            self.reset()
            return result
        return None

    def _add_to_grid(self, f_hz: np.ndarray, p_dbm: np.ndarray):
        idx = np.rint((f_hz - self.grid[0]) / self.bin_hz).astype(np.int32)
        mask = (idx >= 0) & (idx < self.n_bins)
        if not np.any(mask):
            return
        idx = idx[mask]
        p = p_dbm[mask].astype(np.float64)
        np.add.at(self.sum, idx, p)
        np.add.at(self.cnt, idx, 1)
        self.seen[idx] = True

    def _finalize(self):
        if self.n_bins == 0:
            return None
        coverage = float(self.seen.sum()) / float(self.n_bins)
        if coverage < 0.5:
            return None
        p = np.full(self.n_bins, np.nan, np.float32)
        valid = self.cnt > 0
        p[valid] = (self.sum[valid] / self.cnt[valid]).astype(np.float32)
        if np.isnan(p).any():
            vmask = ~np.isnan(p)
            if vmask.any():
                p = np.interp(np.arange(self.n_bins), np.flatnonzero(vmask), p[vmask]).astype(np.float32)
            p[np.isnan(p)] = -120.0
        return self.grid.copy(), p


def test_sweep_assembler_full_pass():
    cfg = _Cfg(freq_start_hz=2400_000_000, freq_end_hz=2401_000_000, bin_hz=100_000)
    asm = _SweepAssembler()
    asm.configure(cfg)

    grid = np.arange(cfg.freq_start_hz + cfg.bin_hz * 0.5, cfg.freq_end_hz + cfg.bin_hz * 0.5, cfg.bin_hz, dtype=np.float64)
    power = np.linspace(-80.0, -40.0, grid.size).astype(np.float32)

    seg_size = 10
    result = None
    for i in range(0, grid.size, seg_size):
        f = grid[i:i+seg_size]
        p = power[i:i+seg_size]
        result = asm.add_segment(f, p, int(f[0] - cfg.bin_hz * 0.5))

    assert result is not None
    full_f, full_p = result
    assert full_f.size == grid.size
    assert np.isfinite(full_p).all()

