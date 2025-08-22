from __future__ import annotations
from typing import Optional
import numpy as np
from PyQt5 import QtCore

from panorama.drivers.base import SourceBackend, SweepConfig
from panorama.features.master_sweep.master import MasterSweepController


class MasterSourceAdapter(SourceBackend):
    """Адаптер для использования MasterSweepController как источника данных для спектра."""
    
    def __init__(self, master: MasterSweepController):
        super().__init__()
        self._master = master
        self._running = False
        
        # Подключаемся к новому сигналу full_sweep_ready
        if hasattr(master, 'full_sweep_ready'):
            master.full_sweep_ready.connect(self._on_full_sweep)
            print("[MasterSourceAdapter] Using new full_sweep_ready signal")
        else:
            # Fallback на старый способ через tile
            master.sweep_tile_received.connect(self._on_tile)
            print("[MasterSourceAdapter] Using fallback sweep_tile_received signal")
        
        master.sweep_error.connect(self._on_error)

    def start(self, config: SweepConfig):
        """Запуск источника."""
        if self._running:
            return
        
        try:
            print(f"[MasterSourceAdapter] Starting sweep: {config.freq_start_hz/1e6:.1f}-{config.freq_end_hz/1e6:.1f} MHz")
            
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
            import traceback
            print(f"[MasterSourceAdapter] Traceback: {traceback.format_exc()}")
            self.error.emit(str(e))

    def stop(self):
        """Остановка источника."""
        if not self._running:
            return
        try:
            self._master.stop_sweep()
        except Exception:
            pass
        self._running = False
        self.finished.emit(0)

    def is_running(self) -> bool:
        """Проверка состояния."""
        return self._running

    @QtCore.pyqtSlot(object, object)
    def _on_full_sweep(self, freqs_hz, power_dbm):
        """Обрабатываем полный sweep напрямую."""
        try:
            # Преобразуем в numpy массивы если нужно
            if not isinstance(freqs_hz, np.ndarray):
                freqs_hz = np.array(freqs_hz, dtype=np.float64)
            if not isinstance(power_dbm, np.ndarray):
                power_dbm = np.array(power_dbm, dtype=np.float32)
            
            print(f"[MasterSourceAdapter] Received full sweep: {len(freqs_hz)} points")
            
            # Эмитим для спектра
            self.fullSweepReady.emit(freqs_hz, power_dbm)
            print(f"[MasterSourceAdapter] Full sweep emitted to spectrum")
            
        except Exception as e:
            print(f"[MasterSourceAdapter] Error processing full sweep: {e}")
            import traceback
            print(f"[MasterSourceAdapter] Traceback: {traceback.format_exc()}")
            self.error.emit(f"Full sweep processing error: {e}")

    @QtCore.pyqtSlot(object)
    def _on_tile(self, tile):
        """Fallback обработчик для старого формата через tiles."""
        try:
            print(f"[MasterSourceAdapter] Received tile (fallback mode)")
            
            if hasattr(tile, 'power') and hasattr(tile, 'f_start'):
                # Создаем массивы частот и мощностей из tile
                f_start = float(tile.f_start)
                bin_hz = float(tile.bin_hz)
                count = int(tile.count)
                
                freqs_hz = np.arange(count, dtype=np.float64) * bin_hz + f_start
                power_dbm = np.array(tile.power, dtype=np.float32)
                
                self.fullSweepReady.emit(freqs_hz, power_dbm)
                print(f"[MasterSourceAdapter] Tile converted and emitted")
            else:
                print(f"[MasterSourceAdapter] Invalid tile object: missing required attributes")
                
        except Exception as e:
            print(f"[MasterSourceAdapter] Error processing tile: {e}")
            import traceback
            print(f"[MasterSourceAdapter] Traceback: {traceback.format_exc()}")
            self.error.emit(f"Tile processing error: {e}")

    @QtCore.pyqtSlot(str)
    def _on_error(self, msg: str):
        """Обработка ошибок."""
        self.error.emit(msg)