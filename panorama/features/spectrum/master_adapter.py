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
from panorama.drivers.hackrf.hrf_backend import HackRFQSABackend, SweepConfig


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
        self.master = None
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
            
            # ВАЖНО: Правильно подключаем сигналы backend'а
            self.backend.fullSweepReady.connect(self._on_full_sweep_from_backend)
            self.backend.error.connect(self._relay_error)
            self.backend.status.connect(self._relay_status)
            self.backend.started.connect(self._on_backend_started)
            self.backend.finished.connect(self._on_backend_stopped)
            
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
            # master контроллер больше не используется
            self.stopped.emit()
            self.finished.emit(0)  # Для совместимости
            self._relay_status("MasterSourceAdapter stopped")
        except Exception as e:
            self._relay_error(f"Master stop error: {e}")
    
    @QtCore.pyqtSlot(object, object)
    def _on_full_sweep_from_backend(self, freqs, dbm):
        """Обрабатывает полный sweep от backend'а."""
        try:
            # Конвертируем в numpy arrays если нужно
            if not isinstance(freqs, np.ndarray):
                freqs = np.array(freqs, dtype=np.float64)
            if not isinstance(dbm, np.ndarray):
                dbm = np.array(dbm, dtype=np.float32)
            
            # Эмитим для SpectrumView
            self.spectrumReady.emit(freqs, dbm)
            self.fullSweepReady.emit(freqs, dbm)  # Для совместимости
            
        except Exception as e:
            self._relay_error(f"Error in _on_full_sweep_from_backend: {e}")
            import traceback
            traceback.print_exc()
    
    # Пики детектируются только в detector/, адаптер их не формирует
    
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

    # ───────────── runtime processing settings from GUI ─────────────
    def set_freq_smoothing(self, enabled: bool, window_bins: int):
        """Обновляет частотное сглаживание в C-бэкенде по настройкам GUI."""
        try:
            if self.backend and hasattr(self.backend, "_lib"):
                w = int(window_bins)
                if w < 1:
                    w = 1
                self.backend._lib.hq_set_freq_smoothing(1 if enabled else 0, w)
                self._relay_status(f"Freq smoothing: {'on' if enabled else 'off'} (W={w})")
        except Exception as e:
            self._relay_error(f"Failed to set freq smoothing: {e}")

    def set_ema_alpha(self, alpha: float):
        """Обновляет коэффициент EMA в C-бэкенде по настройкам GUI."""
        try:
            if self.backend and hasattr(self.backend, "_lib"):
                a = float(alpha)
                if a < 0.01:
                    a = 0.01
                if a > 1.0:
                    a = 1.0
                self.backend._lib.hq_set_ema_alpha(a)
                self._relay_status(f"EMA alpha set to {a:.2f}")
        except Exception as e:
            self._relay_error(f"Failed to set EMA alpha: {e}")

    def set_video_detector_params(self, settings):
        """Пробрасывает расширенные параметры видео‑детектора в C-бэкенд."""
        try:
            if not self.backend:
                return
            hi = float(getattr(settings, 'hysteresis_high_db', 10.0))
            lo = float(getattr(settings, 'hysteresis_low_db', 5.0))
            bridge = int(getattr(settings, 'bridge_gap_bins', 4))
            end_delta = float(getattr(settings, 'boundary_end_delta_db', 1.2))
            edge_run = int(getattr(settings, 'edge_min_run_bins', 3))
            med = int(getattr(settings, 'prefilter_median_bins', 5))
            occ = float(getattr(settings, 'min_region_occupancy', 0.4))
            area = float(getattr(settings, 'min_region_area', 3.0))
            min_width = int(getattr(settings, 'min_peak_width_bins', 3))
            merge_gap_hz = float(getattr(settings, 'rms_halfspan_mhz', 2.5)) * 1e6 * 0.25
            self.backend.set_video_params(hi, lo, bridge, end_delta, edge_run, med, occ, area, min_width, merge_gap_hz)
            self._relay_status("Video detector params updated (C)")
        except Exception as e:
            self._relay_error(f"Failed to set video detector params: {e}")

    def detect_video_bands(self, max_bands: int = 16):
        try:
            if not self.backend:
                return []
            return self.backend.detect_video_bands(max_bands=max_bands)
        except Exception as e:
            self._relay_error(f"detect_video_bands failed: {e}")
            return []