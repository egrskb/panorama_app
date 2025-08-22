from __future__ import annotations
from typing import Optional
import numpy as np
from PyQt5 import QtCore
import logging

from panorama.drivers.base import SourceBackend, SweepConfig
from panorama.features.master_sweep.master import MasterSweepController

logger = logging.getLogger(__name__)


class MasterSourceAdapter(SourceBackend):
    """
    Адаптер для использования MasterSweepController как источника данных для спектра.
    Теперь корректно работает с полным диапазоном спектра.
    """
    
    def __init__(self, master: MasterSweepController):
        super().__init__()
        self._master = master
        self._running = False
        self._config = None
        
        # Подключаемся к сигналу full_sweep_ready
        master.full_sweep_ready.connect(self._on_full_sweep)
        master.sweep_error.connect(self._on_error)
        master.sweep_progress.connect(self._on_progress)
        
        logger.info("[MasterSourceAdapter] Initialized with full spectrum support")

    def start(self, config: SweepConfig):
        """Запуск источника."""
        if self._running:
            logger.warning("[MasterSourceAdapter] Already running")
            return
        
        try:
            self._config = config
            
            logger.info(f"[MasterSourceAdapter] Starting sweep: "
                       f"{config.freq_start_hz/1e6:.1f}-{config.freq_end_hz/1e6:.1f} MHz, "
                       f"bin: {config.bin_hz/1e3:.1f} kHz")
            
            # Запускаем Master sweep с параметрами
            self._master.start_sweep(
                start_hz=config.freq_start_hz,
                stop_hz=config.freq_end_hz,
                bin_hz=config.bin_hz
            )
            
            self._running = True
            self.started.emit()
            self.status.emit("Master sweep running - full spectrum scan")
            
        except Exception as e:
            logger.error(f"[MasterSourceAdapter] Error starting sweep: {e}")
            import traceback
            logger.error(f"[MasterSourceAdapter] Traceback: {traceback.format_exc()}")
            self.error.emit(str(e))

    def stop(self):
        """Остановка источника."""
        if not self._running:
            return
            
        try:
            self._master.stop_sweep()
            logger.info("[MasterSourceAdapter] Sweep stopped")
        except Exception as e:
            logger.error(f"[MasterSourceAdapter] Error stopping sweep: {e}")
            
        self._running = False
        self.finished.emit(0)

    def is_running(self) -> bool:
        """Проверка состояния."""
        return self._running

    @QtCore.pyqtSlot(object, object)
    def _on_full_sweep(self, freqs_hz, power_dbm):
        """
        Обрабатываем полный спектр.
        Фильтруем только нужный диапазон если конфигурация задает подмножество.
        """
        try:
            # Преобразуем в numpy массивы если нужно
            if not isinstance(freqs_hz, np.ndarray):
                freqs_hz = np.array(freqs_hz, dtype=np.float64)
            if not isinstance(power_dbm, np.ndarray):
                power_dbm = np.array(power_dbm, dtype=np.float32)
            
            # Если есть конфигурация с ограниченным диапазоном - фильтруем
            if self._config:
                f_start = self._config.freq_start_hz
                f_stop = self._config.freq_end_hz
                
                # Находим индексы в пределах нужного диапазона
                mask = (freqs_hz >= f_start) & (freqs_hz <= f_stop)
                
                if np.any(mask):
                    freqs_hz = freqs_hz[mask]
                    power_dbm = power_dbm[mask]
                    
                    logger.debug(f"[MasterSourceAdapter] Filtered spectrum: "
                               f"{len(freqs_hz)} points in range "
                               f"{f_start/1e6:.1f}-{f_stop/1e6:.1f} MHz")
            
            # Проверяем валидность данных
            if len(freqs_hz) == 0 or len(power_dbm) == 0:
                logger.warning("[MasterSourceAdapter] Empty spectrum after filtering")
                return
            
            # Эмитим для спектра
            self.fullSweepReady.emit(freqs_hz, power_dbm)
            
            # Логируем только каждый 10-й sweep для уменьшения спама
            if not hasattr(self, '_emit_count'):
                self._emit_count = 0
            self._emit_count += 1
            
            if self._emit_count % 10 == 0:
                logger.info(f"[MasterSourceAdapter] Emitted sweep #{self._emit_count}: "
                           f"{len(freqs_hz)} points, "
                           f"range={freqs_hz[0]/1e6:.1f}-{freqs_hz[-1]/1e6:.1f} MHz, "
                           f"power={power_dbm.min():.1f} to {power_dbm.max():.1f} dBm")
            
        except Exception as e:
            logger.error(f"[MasterSourceAdapter] Error processing full sweep: {e}")
            import traceback
            logger.error(f"[MasterSourceAdapter] Traceback: {traceback.format_exc()}")
            self.error.emit(f"Full sweep processing error: {e}")

    @QtCore.pyqtSlot(float)
    def _on_progress(self, coverage: float):
        """Обработка прогресса покрытия спектра."""
        self.status.emit(f"Spectrum coverage: {coverage:.1f}%")

    @QtCore.pyqtSlot(str)
    def _on_error(self, msg: str):
        """Обработка ошибок."""
        logger.error(f"[MasterSourceAdapter] Master error: {msg}")
        self.error.emit(msg)