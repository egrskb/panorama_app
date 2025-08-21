from __future__ import annotations
from typing import Optional
import numpy as np
from PyQt5 import QtCore

from panorama.drivers.base import SourceBackend, SweepConfig
from panorama.features.master_sweep.master import MasterSweepController


class MasterSourceAdapter(SourceBackend):
    def __init__(self, master: MasterSweepController):
        super().__init__()
        self._master = master
        self._running = False
        
        # Буфер для накопления плиток
        self._sweep_accumulator = {}  # freq -> power
        self._expected_range = None
        self._last_freq = None
        
        master.sweep_tile_received.connect(self._on_tile)
        master.sweep_error.connect(self._on_error)

    def start(self, config: SweepConfig):
        if self._running:
            return
        
        self._expected_range = (config.freq_start_hz, config.freq_end_hz)
        self._sweep_accumulator.clear()
        self._last_freq = None
        
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

    @QtCore.pyqtSlot(object)
    def _on_tile(self, tile):
        """Накапливаем плитки и эмитим полный sweep при завершении прохода."""
        try:
            print(f"[MasterSourceAdapter] Received tile: type={type(tile)}")
            print(f"[MasterSourceAdapter] Tile attributes: {dir(tile)}")
            
            # Проверяем что это SweepTile
            if not hasattr(tile, 'f_start') or not hasattr(tile, 'power'):
                print(f"[MasterSourceAdapter] Invalid tile object: missing required attributes")
                return
                
            print(f"[MasterSourceAdapter] Tile data: f_start={tile.f_start/1e6:.1f} MHz, "
                  f"count={tile.count}, power_range=[{min(tile.power):.1f}, {max(tile.power):.1f}] dBm")
            
            f_start = float(tile.f_start)
            bin_hz = float(tile.bin_hz)
            count = int(tile.count)
            
            if count <= 0 or bin_hz <= 0:
                print(f"[MasterSourceAdapter] Invalid tile data: count={count}, bin_hz={bin_hz}")
                return
            
            # Детектируем wrap (начало нового прохода)
            if self._last_freq is not None and f_start < self._last_freq - 10e6:
                print(f"[MasterSourceAdapter] Detected wrap: f_start={f_start/1e6:.1f} < last_freq={self._last_freq/1e6:.1f}")
                # Эмитим накопленный sweep
                self._emit_accumulated_sweep()
                self._sweep_accumulator.clear()
            
            # Добавляем текущую плитку
            for i in range(count):
                freq = f_start + i * bin_hz
                self._sweep_accumulator[freq] = tile.power[i]
            
            self._last_freq = f_start
            
            # Проверяем покрытие
            if self._check_coverage():
                print(f"[MasterSourceAdapter] Coverage complete, emitting accumulated sweep")
                self._emit_accumulated_sweep()
                self._sweep_accumulator.clear()
                
        except Exception as e:
            print(f"[MasterSourceAdapter] Error processing tile: {e}")
            import traceback
            print(f"[MasterSourceAdapter] Traceback: {traceback.format_exc()}")
            self.error.emit(f"Adapter tile error: {e}")

    def _check_coverage(self):
        """Проверяет, покрыт ли весь ожидаемый диапазон."""
        if not self._expected_range or not self._sweep_accumulator:
            return False
        
        freqs = sorted(self._sweep_accumulator.keys())
        if not freqs:
            return False
            
        coverage = (freqs[-1] - freqs[0]) / (self._expected_range[1] - self._expected_range[0])
        return coverage >= 0.95

    def _emit_accumulated_sweep(self):
        """Эмитит накопленный полный sweep."""
        if not self._sweep_accumulator:
            return
            
        freqs = sorted(self._sweep_accumulator.keys())
        powers = [self._sweep_accumulator[f] for f in freqs]
        
        freqs_array = np.array(freqs, dtype=np.float64)
        powers_array = np.array(powers, dtype=np.float32)
        
        print(f"[MasterAdapter] Emitting full sweep: {len(freqs)} points, "
              f"range {freqs[0]/1e6:.1f}-{freqs[-1]/1e6:.1f} MHz")
        
        self.fullSweepReady.emit(freqs_array, powers_array)

    @QtCore.pyqtSlot(str)
    def _on_error(self, msg: str):
        self.error.emit(msg)


