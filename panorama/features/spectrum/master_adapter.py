# -*- coding: utf-8 -*-
"""
MasterSourceAdapter: glue between SpectrumView and Master controller + HackRF QSA backend.
Starts/stops backend, pushes sweeps to MasterSweepController, emits status to UI.
"""

from __future__ import annotations
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
import logging
import numpy as np
from PyQt5 import QtCore

# наш backend
from panorama.drivers.hrf_backend import HackRFQSABackend
# контроллер поиска пиков (совместимый DetectedPeak уже реэкспортирован)
from panorama.features.spectrum.master import MasterSweepController


@dataclass
class SweepConfig:
    freq_start_hz: int
    freq_end_hz: int
    bin_hz: int
    lna_db: int
    vga_db: int
    amp_on: bool
    serial: Optional[str] = None  # Добавляем серийный номер
    threshold_dbm: Optional[float] = None
    peak_span_hz: Optional[float] = None
    dwell_ms: Optional[int] = None


class MasterSourceAdapter(QtCore.QObject):
    # события, которые слушает view
    status  = QtCore.pyqtSignal(str)
    error   = QtCore.pyqtSignal(str)
    started = QtCore.pyqtSignal()
    stopped = QtCore.pyqtSignal()

    # прокидываем спектр в виджет
    spectrumReady = QtCore.pyqtSignal(object, object)  # (freqs, dbm)

    # прокидываем пики (старый контракт оркестратора)
    peak_detected = QtCore.pyqtSignal(object)  # DetectedPeak

    def __init__(self, logger=None):
        super().__init__()
        # logger может быть None или не логгером - добавляем защиту
        if logger and hasattr(logger, 'info') and callable(logger.info):
            self.log = logger
        else:
            self.log = None
        self.backend: Optional[HackRFQSABackend] = None
        self.master: Optional[MasterSweepController] = None
        self.running: bool = False
        
        # Загружаем настройки SDR для получения серийного номера master
        try:
            from panorama.features.settings.storage import load_sdr_settings
            self.sdr_settings = load_sdr_settings()
        except Exception as e:
            if self.log:
                self.log.warning(f"Failed to load SDR settings: {e}")
            self.sdr_settings = {}

    # совместимость с проверками в UI
    def is_running(self) -> bool:
        return self.running

    # --- lifecycle ---
    def start(self, cfg: SweepConfig) -> bool:
        try:
            if self.running:
                return True

            # backend - серийный номер передается через SweepConfig
            self.backend = HackRFQSABackend(logger=self.log)

            # master controller (порог по SNR = 10 дБ, окно по умолчанию 5 МГц — корректируется UI через orchestrator)
            self.master = MasterSweepController(self)
            # подключения сигналов
            self.master.spectrumReady.connect(self._on_master_spectrum)
            self.master.peak_detected.connect(self._on_peak_detected)
            self.master.status.connect(self._relay_status)
            self.master.error.connect(self._relay_error)

            # backend callbacks -> обновление master
            self.backend.fullSweepReady.connect(self._on_full_sweep_from_backend)
            self.backend.error.connect(self._relay_error)
            self.backend.started.connect(self._on_backend_started)
            self.backend.finished.connect(self._on_backend_stopped)

            # запустить железо
            self.backend.start(
                cfg,
                on_full=self._on_full_sweep_from_backend,
                on_error=self._relay_error,
                on_status=self._relay_status
            )
            self.running = True
            self.started.emit()
            self._relay_status("MasterSourceAdapter started")
            return True
        except Exception as e:
            self._relay_error(f"Master start failed: {e}")
            self.stop()
            return False

    def stop(self):
        try:
            if self.backend:
                self.backend.stop()
            self.running = False
            if self.master:
                self.master.stop()
            self.stopped.emit()
            self._relay_status("MasterSourceAdapter stopped")
        except Exception as e:
            self._relay_error(f"Master stop error: {e}")

    # --- internal wiring ---
    @QtCore.pyqtSlot(object, object)
    def _on_full_sweep_from_backend(self, freqs: np.ndarray, dbm: np.ndarray):
        # прокидываем в Master для детекции пиков + в UI
        print(f"[MasterSourceAdapter] Received full sweep from backend: freqs={freqs.size}, dbm={dbm.size}")
        print(f"[MasterSourceAdapter] Emitting spectrumReady signal...")
        
        if self.master:
            self.master.last_freqs = freqs
            self.master.last_spectrum = dbm
            # Master сам триггерит детекцию по таймеру; но разок дадим сразу
            self.master.spectrumReady.emit(freqs, dbm)
        
        self.spectrumReady.emit(freqs, dbm)
        print(f"[MasterSourceAdapter] spectrumReady signal emitted")

    @QtCore.pyqtSlot(object)
    def _on_master_spectrum(self, freqs: np.ndarray, dbm: np.ndarray):
        # дублирующая прослойка на будущее
        self.spectrumReady.emit(freqs, dbm)

    @QtCore.pyqtSlot(object)
    def _on_peak_detected(self, peak_obj):
        # Прокинуть в оркестратор по старому контракту
        self.peak_detected.emit(peak_obj)

    def _on_backend_started(self):
        if self.master:
            self.master.start()
        self._relay_status("HackRF backend created")

    def _on_backend_stopped(self):
        self._relay_status("HackRF backend stopped")

    def _relay_status(self, msg: str):
        # Проверяем, что self.log - это действительно логгер, а не Qt сигнал
        if self.log and hasattr(self.log, 'info') and callable(self.log.info):
            try:
                self.log.info(f"[Spectrum] status: {msg}")
            except Exception:
                pass  # Игнорируем ошибки логирования
        self.status.emit(msg)

    def _relay_error(self, msg: str):
        # Проверяем, что self.log - это действительно логгер, а не Qt сигнал
        if self.log and hasattr(self.log, 'error') and callable(self.log.error):
            try:
                self.log.error(f"[Spectrum] ERROR: {msg}")
            except Exception:
                pass  # Игнорируем ошибки логирования
        self.error.emit(msg)
