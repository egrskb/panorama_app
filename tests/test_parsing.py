import numpy as np

from panorama.core.parsing import parse_sweep_line


def test_parse_valid_line():
    line = "0,0,2400000000,2403000000,1000000,0,-50,-40,-30"
    freqs, power, bin_hz, f_low, f_high = parse_sweep_line(line)
    assert freqs.size == 3
    assert power.tolist() == [-50.0, -40.0, -30.0]
    assert bin_hz == 1000000.0
    assert f_low == 2400000000.0
    assert f_high == 2403000000.0


def test_parse_invalid_line():
    assert parse_sweep_line("bad line") is None
