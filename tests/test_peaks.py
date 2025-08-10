import numpy as np

from panorama.detection.peaks import find_peaks


def test_isolated_peak():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([-100.0, -10.0, -100.0])
    peaks = find_peaks(x, y, min_dbm=-20.0, min_sep_khz=30.0)
    assert len(peaks) == 1
    assert abs(peaks[0][0] - 1.0) < 1e-6


def test_min_separation():
    x = np.linspace(0, 2, 201)
    y = np.full_like(x, -100.0)
    y[50] = -10.0  # ~0.5 MHz
    y[52] = -9.0   # close peak ~0.52 MHz
    y[150] = -8.0  # ~1.5 MHz
    peaks = find_peaks(x, y, min_dbm=-20.0, min_sep_khz=30.0)
    assert len(peaks) == 2
    freqs = sorted(p[0] for p in peaks)
    assert freqs[0] < 1.0 and freqs[1] > 1.0
