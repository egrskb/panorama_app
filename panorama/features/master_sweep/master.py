# panorama/features/master_sweep/master.py
# Совместимость со старым кодом: здесь есть и DetectedPeak, и реэкспорт MasterSweepController.

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import time

@dataclass
class DetectedPeak:
    """
    Совместимая структура "пик" для оркестратора.
    Поля ожидаются старым кодом: id, f_peak, snr_db, bin_hz, t0, last_seen, span_user, status
    """
    id: str
    f_peak: float
    snr_db: float
    bin_hz: float
    t0: float
    last_seen: float
    span_user: float
    status: str = field(default="ACTIVE")

    @property
    def freq_hz(self) -> float: return self.f_peak
    @property
    def center_freq(self) -> float: return self.f_peak
    @property
    def span_hz(self) -> float: return self.span_user

    def touch(self) -> None: self.last_seen = time.time()
    def to_dict(self) -> Dict[str, Any]: return asdict(self)

    @classmethod
    def from_peak(cls, freq_hz: float, snr_db: float, span_hz: float,
                  peak_id: Optional[str] = None, status: str = "ACTIVE",
                  t0: Optional[float] = None) -> "DetectedPeak":
        now = time.time()
        pid = peak_id or f"peak_{int(round(freq_hz))}"
        t0v = t0 if t0 is not None else now
        return cls(
            id=pid, f_peak=float(freq_hz), snr_db=float(snr_db),
            bin_hz=float(span_hz), t0=float(t0v), last_seen=float(now),
            span_user=float(span_hz), status=str(status),
        )

# Реэкспорт реального контроллера
from panorama.features.spectrum.master import MasterSweepController  # noqa: E402
