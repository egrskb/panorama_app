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
            print(f"[MasterSourceAdapter] Starting sweep: {config.freq_start_hz/1e6:.1f}-{config.freq_end_hz/1e6:.1f} MHz, bin={config.bin_hz/1e3:.0f} kHz")
            
            self._master.start_sweep(
                start_hz=config.freq_start_hz,
                stop_hz=config.freq_end_hz,
                bin_hz=config.bin_hz,
                dwell_ms=100,
            )
            self._running = True
            self.started.emit()
            self.status.emit("Master sweep running")
            print("[MasterSourceAdapter] Sweep started successfully")
            
        except Exception as e:
            print(f"[MasterSourceAdapter] Error starting sweep: {e}")
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
            print(f"[MasterSourceAdapter] Received tile: type={type(tile)}")
            print(f"[MasterSourceAdapter] Tile attributes: {dir(tile)}")
            
            # Проверяем что это SweepTile
            if not hasattr(tile, 'f_start') or not hasattr(tile, 'power'):
                print(f"[MasterSourceAdapter] Invalid tile object: missing required attributes")
                return
                
            print(f"[MasterSourceAdapter] Tile data: f_start={tile.f_start/1e6:.1f} MHz, "
                  f"count={tile.count}, power_range=[{min(tile.power):.1f}, {max(tile.power):.1f}] dBm")
            
            f0 = float(tile.f_start)
            bin_hz = float(tile.bin_hz)
            count = int(tile.count)
            if count <= 0 or bin_hz <= 0:
                print(f"[MasterSourceAdapter] Invalid tile data: count={count}, bin_hz={bin_hz}")
                return
            # Частоты и мощность
            freqs = f0 + np.arange(count, dtype=np.float64) * bin_hz
            # tile.power — list[float]
            power = np.asarray(tile.power, dtype=np.float32)
            if power.size != count:
                print(f"[MasterSourceAdapter] Power size mismatch: {power.size} != {count}")
                return
            
            print(f"[MasterSourceAdapter] Emitting fullSweepReady: freqs={freqs.size}, power={power.size}")
            self.fullSweepReady.emit(freqs, power)
            
        except Exception as e:
            print(f"[MasterSourceAdapter] Error processing tile: {e}")
            import traceback
            print(f"[MasterSourceAdapter] Traceback: {traceback.format_exc()}")
            self.error.emit(f"Adapter tile error: {e}")

    @QtCore.pyqtSlot(str)
    def _on_error(self, msg: str):
        self.error.emit(msg)


