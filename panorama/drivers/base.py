from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from PyQt5 import QtCore


@dataclass
class SweepConfig:
    freq_start_hz: int
    freq_end_hz: int
    bin_hz: int = 200_000
    lna_db: int = 24
    vga_db: int = 20
    amp_on: bool = False
    extra: Optional[list[str]] = None

    def to_args(self) -> list[str]:
        # hackrf_sweep принимает -f в МЕГАгерцах!
        f_start_mhz = int(round(self.freq_start_hz / 1e6))
        f_end_mhz   = int(round(self.freq_end_hz   / 1e6))

        args = [
            "-f", f"{f_start_mhz}:{f_end_mhz}",
            "-w", str(self.bin_hz),          # это в Гц (как у тебя в терминале)
            "-l", str(self.lna_db),
            "-g", str(self.vga_db),
            "-a", "1" if self.amp_on else "0",
        ]
        if self.extra:
            args.extend(self.extra)
        return args


class SourceBackend(QtCore.QObject):
    """
    Abstract-ish base for data sources.
    Emits one SweepLine at a time; feature layer занимается «склейкой» в грид.
    """
    sweepLine = QtCore.pyqtSignal(object)   # payload: panorama.shared.parsing.SweepLine
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal(int)       # exit code or 0 on graceful stop

    def __init__(self, parent=None):
        super().__init__(parent)

    # Implement in subclass
    def start(self, config: SweepConfig):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def is_running(self) -> bool:
        raise NotImplementedError
