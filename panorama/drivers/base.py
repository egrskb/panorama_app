from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from PyQt5 import QtCore


@dataclass
class SweepConfig:
    """Конфигурация для свипа."""
    freq_start_hz: int
    freq_end_hz: int
    bin_hz: int = 800_000  # ОПТИМИЗАЦИЯ: крупнее бин — быстрее UI
    lna_db: int = 24
    vga_db: int = 20
    amp_on: bool = False
    serial: Optional[str] = None  # Серийник устройства обязателен для HackRF QSA
    extra: Optional[list[str]] = None

    def to_args(self) -> list[str]:
        """Аргументы в стиле hackrf_sweep (если пригодятся)."""
        f_start_mhz = int(round(self.freq_start_hz / 1e6))
        f_end_mhz   = int(round(self.freq_end_hz   / 1e6))

        args = [
            "-f", f"{f_start_mhz}:{f_end_mhz}",
            "-w", str(self.bin_hz),          # в Гц
            "-l", str(self.lna_db),
            "-g", str(self.vga_db),
            "-a", "1" if self.amp_on else "0",
        ]
        
        if self.serial:
            args.extend(["-d", self.serial])
        
        if self.extra:
            args.extend(self.extra)
        
        return args


class SourceBackend(QtCore.QObject):
    """
    Базовый класс для источников данных.
    Эмитит как отдельные сегменты (sweepLine), так и полные проходы (fullSweepReady).
    """
    sweepLine = QtCore.pyqtSignal(object)       # payload: panorama.shared.parsing.SweepLine
    fullSweepReady = QtCore.pyqtSignal(object, object)  # (freqs_hz, power_dbm) - полный проход
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal(int)       # exit code или 0 при graceful stop

    def __init__(self, parent=None):
        super().__init__(parent)

    def start(self, config: SweepConfig):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def is_running(self) -> bool:
        raise NotImplementedError
