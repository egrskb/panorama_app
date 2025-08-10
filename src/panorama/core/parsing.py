import numpy as np


def parse_sweep_line(line: str):
    """Parse a single line produced by ``hackrf_sweep``.

    Parameters
    ----------
    line: str
        Input line.

    Returns
    -------
    tuple | None
        ``(freqs, power, bin_hz, f_low, f_high)`` or ``None`` on parse
        failure.
    """
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 7:
        return None
    try:
        f_low = float(parts[2])
        f_high = float(parts[3])
        bin_hz = float(parts[4])
        vals = [float(x) for x in parts[6:]]
        if not vals:
            return None
        freqs = f_low + (np.arange(len(vals), dtype=np.float64) + 0.5) * bin_hz
        power = np.array(vals, dtype=np.float32)
        return freqs, power, bin_hz, f_low, f_high
    except Exception:
        return None
