# Исправленный panorama/features/spectrum/master_adapter.py

# -*- coding: utf-8 -*-
"""
MasterSourceAdapter: glue between SpectrumView and Master controller + HackRF QSA backend.
Starts/stops backend, pushes sweeps to MasterSweepController, emits status to UI.
"""

from __future__ import annotations
from typing import Optional, Callable, Dict, Any
import logging
import numpy as np
from PyQt5 import QtCore

# наш backend
from panorama.drivers.hrf_backend import HackRFQSABackend, SweepConfig
# контроллер поиска пиков
from panorama.features.spectrum.master import MasterSweepController


class MasterSourceAdapter(QtCore.QObject):
    """Адаптер между SpectrumView и Master контроллером + HackRF backend."""
    
    # события для view
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    started = QtCore.pyqtSignal()
    stopped = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal(int)  # Добавляем для совместимости
    
    # прокидываем спектр в виджет
    spectrumReady = QtCore.pyqtSignal(object, object)  # (freqs, dbm)
    fullSweepReady = QtCore.pyqtSignal(object, object)  # Для совместимости с SpectrumView
    
    # прокидываем пики
    peak_detected = QtCore.pyqtSignal(object)  # DetectedPeak
    
    def __init__(self, logger=None):
        super().__init__()
        if logger and hasattr(logger, 'info') and callable(logger.info):
            self.log = logger
        else:
            self.log = None
        self.backend: Optional[HackRFQSABackend] = None
        self.master: Optional[MasterSweepController] = None
        self.running: bool = False
        
        # Загружаем настройки SDR
        try:
            from panorama.features.settings.storage import load_sdr_settings
            self.sdr_settings = load_sdr_settings()
        except Exception as e:
            if self.log:
                self.log.warning(f"Failed to load SDR settings: {e}")
            self.sdr_settings = {}
    
    def is_running(self) -> bool:
        return self.running
    
    def start(self, cfg: SweepConfig) -> bool:
        """Запускает backend и master controller."""
        try:
            if self.running:
                return True
            
            # Создаем backend
            self.backend = HackRFQSABackend(logger=self.log)
            
            # Создаем master controller
            self.master = MasterSweepController(self)
            
            # ВАЖНО: Правильно подключаем сигналы backend'а
            self.backend.fullSweepReady.connect(self._on_full_sweep_from_backend)
            self.backend.error.connect(self._relay_error)
            self.backend.status.connect(self._relay_status)
            self.backend.started.connect(self._on_backend_started)
            self.backend.finished.connect(self._on_backend_stopped)
            
            # Подключаем сигналы master controller'а
            self.master.spectrumReady.connect(self._on_master_spectrum)
            self.master.peak_detected.connect(self._on_peak_detected)
            self.master.status.connect(self._relay_status)
            self.master.error.connect(self._relay_error)
            
            # Запускаем backend
            self.backend.start(cfg)
            
            self.running = True
            self.started.emit()
            self._relay_status("MasterSourceAdapter started")
            return True
            
        except Exception as e:
            self._relay_error(f"Master start failed: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Останавливает backend и master controller."""
        try:
            if self.backend:
                self.backend.stop()
            self.running = False
            if self.master:
                self.master.stop()
            self.stopped.emit()
            self.finished.emit(0)  # Для совместимости
            self._relay_status("MasterSourceAdapter stopped")
        except Exception as e:
            self._relay_error(f"Master stop error: {e}")
    
    @QtCore.pyqtSlot(object, object)
    def _on_full_sweep_from_backend(self, freqs, dbm):
        """Обрабатывает полный sweep от backend'а."""
        try:
            print(f"[MasterSourceAdapter] _on_full_sweep_from_backend: freqs_size={freqs.size if hasattr(freqs, 'size') else len(freqs)}, dbm_size={dbm.size if hasattr(dbm, 'size') else len(dbm)}")
            
            # Конвертируем в numpy arrays если нужно
            if not isinstance(freqs, np.ndarray):
                freqs = np.array(freqs, dtype=np.float64)
            if not isinstance(dbm, np.ndarray):
                dbm = np.array(dbm, dtype=np.float32)
            
            # Обновляем master controller
            if self.master:
                self.master.handle_full_sweep(freqs, dbm)
            
            # Эмитим для SpectrumView
            self.spectrumReady.emit(freqs, dbm)
            self.fullSweepReady.emit(freqs, dbm)  # Для совместимости
            
            print(f"[MasterSourceAdapter] Signals emitted successfully")
            
        except Exception as e:
            self._relay_error(f"Error in _on_full_sweep_from_backend: {e}")
            import traceback
            traceback.print_exc()
    
    @QtCore.pyqtSlot(object, object)
    def _on_master_spectrum(self, freqs, dbm):
        """Прокидывает спектр от master controller'а."""
        try:
            self.spectrumReady.emit(freqs, dbm)
            self.fullSweepReady.emit(freqs, dbm)
        except Exception as e:
            self._relay_error(f"Error in _on_master_spectrum: {e}")
    
    @QtCore.pyqtSlot(object)
    def _on_peak_detected(self, peak_obj):
        """Прокидывает обнаруженный пик."""
        self.peak_detected.emit(peak_obj)
    
    def _on_backend_started(self):
        """Обрабатывает запуск backend'а."""
        if self.master:
            self.master.start()
        self._relay_status("HackRF backend started")
    
    def _on_backend_stopped(self, code=0):
        """Обрабатывает остановку backend'а."""
        self._relay_status(f"HackRF backend stopped (code={code})")
        self.finished.emit(code)
    
    def _relay_status(self, msg: str):
        """Передает статусное сообщение."""
        if self.log and hasattr(self.log, 'info') and callable(self.log.info):
            try:
                self.log.info(f"[Spectrum] status: {msg}")
            except Exception:
                pass
        self.status.emit(msg)
    
    def _relay_error(self, msg: str):
        """Передает сообщение об ошибке."""
        if self.log and hasattr(self.log, 'error') and callable(self.log.error):
            try:
                self.log.error(f"[Spectrum] ERROR: {msg}")
            except Exception:
                pass
        self.error.emit(msg)