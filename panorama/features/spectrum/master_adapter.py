from __future__ import annotations
from typing import Optional
import numpy as np
from PyQt5 import QtCore

from panorama.drivers.base import SourceBackend, SweepConfig
from panorama.features.master_sweep.master import MasterSweepController


class MasterSourceAdapter(SourceBackend):
    """Адаптер: трансформирует события Master (плитки sweep) в строки для SpectrumView.

    fullSweepReady эмитится на каждую плитку как на «полную строку» — SpectrumView корректно перерисует
    водопад и график даже при изменяющемся размере.
    """

    def __init__(self, master: MasterSweepController):
        super().__init__()
        self._master = master
        self._running = False
        # Подключаемся к коллбэкам мастера
        master.sweep_tile_received.connect(self._on_tile)
        master.sweep_error.connect(self._on_error)

    def start(self, config: SweepConfig):
        if self._running:
            return
        # Передаём параметры в мастер
        try:
            self._master.start_sweep(
                start_hz=config.freq_start_hz,
                stop_hz=config.freq_end_hz,
                bin_hz=config.bin_hz,
                dwell_ms=100,
            )
            self._running = True
            self.started.emit()
            self.status.emit("Master sweep running")
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        if not self._running:
            return
        try:
            self._master.stop_sweep()
        except Exception:
            pass
        self._running = False
        self.finished.emit(0)

    def is_running(self) -> bool:
        return self._running

    # Master events → SpectrumView
    @QtCore.pyqtSlot(object)
    def _on_tile(self, tile):
        try:
            f0 = float(tile.f_start)
            bin_hz = float(tile.bin_hz)
            count = int(tile.count)
            if count <= 0 or bin_hz <= 0:
                return
            # Частоты и мощность
            freqs = f0 + np.arange(count, dtype=np.float64) * bin_hz
            # tile.power — list[float]
            power = np.asarray(tile.power, dtype=np.float32)
            if power.size != count:
                return
            self.fullSweepReady.emit(freqs, power)
        except Exception as e:
            self.error.emit(f"Adapter tile error: {e}")

    @QtCore.pyqtSlot(str)
    def _on_error(self, msg: str):
        self.error.emit(msg)


