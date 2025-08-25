from __future__ import annotations
from typing import Optional
import numpy as np
import time
from PyQt5 import QtCore

from panorama.features.spectrum.model import PeakDetector
from panorama.features.master_sweep.master import DetectedPeak


def _qt_parent(parent):
    return parent if isinstance(parent, QtCore.QObject) else None


class PeakWorker(QtCore.QRunnable):
    def __init__(self, controller: "MasterSweepController"):
        super().__init__()
        self.controller = controller

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.controller._detect_and_emit()
        except Exception as e:
            msg = f"PeakWorker error: {e}"
            self.controller.error.emit(msg)
            self.controller.sweep_error.emit(msg)


class MasterSweepController(QtCore.QObject):
    # данные спектра
    spectrumReady   = QtCore.pyqtSignal(object, object)  # (freqs, row_dbm)

    # сигналы пиков
    peak_detected   = QtCore.pyqtSignal(object)          # DetectedPeak (старый контракт)
    peakTaskReady   = QtCore.pyqtSignal(dict)            # новая watchlist-задача для slaves

    # статусы/ошибки
    status          = QtCore.pyqtSignal(str)
    error           = QtCore.pyqtSignal(str)
    sweep_error     = QtCore.pyqtSignal(str)             # совместимость со старым кодом

    def __init__(self, parent=None, threshold_dbm: float = -10.0, span_hz: float = 5e6):
        super().__init__(_qt_parent(parent))
        self.last_freqs: Optional[np.ndarray] = None
        self.last_spectrum: Optional[np.ndarray] = None
        self.peak_detector = PeakDetector()
        self.threshold_dbm = float(threshold_dbm)
        self.span_hz = float(span_hz)

        self.peak_detection_timer = QtCore.QTimer(self)
        self.peak_detection_timer.setInterval(2000)
        self.peak_detection_timer.timeout.connect(self._on_timer)

        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self._last_emit_ts = 0.0

        # дублируем error -> sweep_error для совместимости
        self.error.connect(self.sweep_error)

    # --- stringify/encode (безопасность) ---
    def __str__(self) -> str:
        return f"<MasterSweepController thr={self.threshold_dbm} dBm, span={self.span_hz/1e6:.2f} MHz>"

    def __bytes__(self) -> bytes:
        return str(self).encode("utf-8", errors="ignore")

    def encode(self, encoding: str = "utf-8", errors: str = "ignore") -> bytes:
        return str(self).encode(encoding, errors=errors)

    # --- совместимость со старым API ---
    def start_sweep(self):
        self.start()

    def stop_sweep(self):
        self.stop()

    def cleanup(self):
        """Вызывается при закрытии приложения: мягко останавливаем таймеры/ссылки."""
        try:
            self.stop()
            # обнулим большие буферы
            self.last_freqs = None
            self.last_spectrum = None
        except Exception as e:
            self.error.emit(f"cleanup error: {e}")
            self.sweep_error.emit(f"cleanup error: {e}")

    # ---- управление ----
    def start(self):
        self.peak_detection_timer.start()
        self.status.emit("Master sweep controller started")

    def stop(self):
        self.peak_detection_timer.stop()
        self.status.emit("Master sweep controller stopped")

    # ---- данные спектра ----
    def handle_full_sweep(self, freqs: np.ndarray, row_dbm: np.ndarray):
        self.last_freqs = freqs
        self.last_spectrum = row_dbm
        self.spectrumReady.emit(freqs, row_dbm)

    # ---- фон ----
    def _on_timer(self):
        if self.last_freqs is None or self.last_spectrum is None:
            return
        self.thread_pool.start(PeakWorker(self))

    def _detect_and_emit(self):
        try:
            if self.last_freqs is None or self.last_spectrum is None:
                return
            freqs = self.last_freqs
            power = self.last_spectrum

            # грубая маска по абсолютному уровню
            if not np.any(power > self.threshold_dbm):
                return

            # детектор пиков -> список DetectedPeak
            peaks = self.peak_detector.detect_peaks(freqs, power)
            if not peaks:
                return

            # берём наибольший по SNR
            best_peak = max(peaks, key=lambda p: p.snr_db)
            peak_freq = best_peak.freq_hz
            peak_snr = best_peak.snr_db
            now = time.time()
            if now - self._last_emit_ts < 1.0:  # антиспам
                return
            self._last_emit_ts = now

            # старый контракт: DetectedPeak
            dp = DetectedPeak(
                id=f"peak_{int(round(peak_freq))}",
                f_peak=float(peak_freq),
                snr_db=float(peak_snr),
                bin_hz=float(self.span_hz),  # исторически здесь bin; кладём span для совместимости
                t0=now,
                last_seen=now,
                span_user=float(self.span_hz),
                status="ACTIVE",
            )
            self.peak_detected.emit(dp)

            # новая watchlist-задача для slaves
            f0 = peak_freq - self.span_hz/2.0
            f1 = peak_freq + self.span_hz/2.0
            task = {
                "center_freq": float(peak_freq),
                "span_hz": float(self.span_hz),
                "range": (float(f0), float(f1)),
                "timestamp": now,
                "peak_power": float(power[np.argmin(np.abs(freqs - peak_freq))])
            }
            self.peakTaskReady.emit(task)

            self.status.emit(
                f"Peak {peak_freq/1e6:.3f} MHz, SNR={peak_snr:.1f} dB, span={self.span_hz/1e6:.2f} MHz"
            )
        except Exception as e:
            msg = f"detect_and_emit error: {e}"
            self.error.emit(msg)
            self.sweep_error.emit(msg)
