# -*- coding: utf-8 -*-
"""
Slave SDR controller: измерение RMS RSSI в заданном окне частот.
"""

from __future__ import annotations
import os
os.environ.setdefault("SOAPY_SDR_DISABLE_AVAHI", "1")  # без Avahi
import time
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QMutex

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
    SOAPY_AVAILABLE = True
except Exception:
    SOAPY_AVAILABLE = False
    SoapySDR = None


@dataclass
class RSSIMeasurement:
    slave_id: str
    center_hz: float
    span_hz: float
    band_rssi_dbm: float
    band_noise_dbm: float
    snr_db: float
    n_samples: int
    ts: float
    flags: Dict


@dataclass
class MeasurementWindow:
    center: float
    span: float
    dwell_ms: int
    epoch: float


class SlaveSDR(QObject):
    measurement_complete = pyqtSignal(object)  # RSSIMeasurement
    measurement_error = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(self, slave_id: str, uri: str, logger: logging.Logger):
        super().__init__()
        self.slave_id = slave_id
        self.uri = uri
        self.log = logger

        self.sdr = None
        self.rx_stream = None
        self.is_initialized = False

        self.sample_rate = 8e6
        self.gain = 20.0
        self.frequency = 2.4e9
        self.bandwidth = 2.5e6

        self.k_cal_db = 0.0

        self.is_measuring = False
        self.current_window: Optional[MeasurementWindow] = None

        self._mutex = QMutex()
        self._init_sdr()

    # --- SDR init/close ---
    def _init_sdr(self):
        if not SOAPY_AVAILABLE:
            self.log.error("SoapySDR not available")
            return
        try:
            self.sdr = SoapySDR.Device(self.uri)
            self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
            self.sdr.setBandwidth(SOAPY_SDR_RX, 0, self.bandwidth)
            self.sdr.setGain(SOAPY_SDR_RX, 0, self.gain)
            self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.frequency)
            self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
            self.sdr.activateStream(self.rx_stream)
            self.is_initialized = True
            self.status_changed.emit("READY")
            self.log.info(f"Slave {self.slave_id} initialized: {self.uri}")
        except Exception as e:
            self.log.error(f"Failed to initialize slave {self.slave_id}: {e}")
            self.status_changed.emit("ERROR")

    def close(self):
        try:
            if self.rx_stream:
                self.sdr.deactivateStream(self.rx_stream)
                self.sdr.closeStream(self.rx_stream)
            self.sdr = None
            self.is_initialized = False
            self.log.info(f"Slave {self.slave_id} closed")
        except Exception as e:
            self.log.error(f"Error closing slave {self.slave_id}: {e}")

    # --- band measurement ---
    def measure_band_rssi(self, center_hz: float, span_hz: float, dwell_ms: int,
                          k_cal_db: Optional[float] = None) -> RSSIMeasurement:
        if not self.is_initialized:
            raise RuntimeError(f"Slave {self.slave_id} not initialized")
        if self.is_measuring:
            raise RuntimeError(f"Slave {self.slave_id} busy")

        try:
            self.is_measuring = True
            self.current_window = MeasurementWindow(center=center_hz, span=span_hz,
                                                    dwell_ms=dwell_ms, epoch=time.time())

            # Настройка тюнера
            self.sdr.setFrequency(SOAPY_SDR_RX, 0, center_hz)
            # чтобы захватить всю полосу, выставим bandwidth ≈ span
            bw = max(200e3, min(span_hz, self.sample_rate * 0.8))
            self.sdr.setBandwidth(SOAPY_SDR_RX, 0, bw)

            # Сбор выборки
            n_samp = int(max(1, round(self.sample_rate * (dwell_ms / 1000.0))))
            buf = np.empty(n_samp, dtype=np.complex64)
            rc = self.sdr.readStream(self.rx_stream, [buf], len(buf))
            if isinstance(rc, tuple):
                got = int(rc[0])
            else:
                got = int(rc)
            if got <= 0:
                raise RuntimeError("readStream returned no data")

            sig = buf[:got]

            # спектр
            # окно Хэмминга для RMS оценки
            win = np.hamming(len(sig)).astype(np.float32)
            sig_w = sig * win
            # FFT
            sp = np.fft.rfft(sig_w)
            psd = (np.abs(sp) ** 2) / np.sum(win ** 2)
            psd_dbm = 10.0 * np.log10(psd + 1e-20) - 174.0 + 10.0 * np.log10(self.sample_rate / len(sig_w))

            # частотная ось этого рида
            freqs = np.fft.rfftfreq(len(sig_w), d=1.0 / self.sample_rate)
            # центрируем вокруг center_hz
            freqs = freqs + (center_hz - self.sample_rate / 2.0)

            # извлекаем окно по span
            half = span_hz / 2.0
            mask = (freqs >= center_hz - half) & (freqs <= center_hz + half)
            if not np.any(mask):
                raise RuntimeError("no bins in band window")

            band_vals = psd_dbm[mask]

            # шумовой пол — медиана нижних 30%
            sorted_vals = np.sort(band_vals)
            k = max(1, int(len(sorted_vals) * 0.3))
            noise_dbm = float(np.median(sorted_vals[:k]))

            # RMS в линейной шкале в полосе
            band_lin = 10.0 ** (band_vals / 10.0)
            rms_lin = float(np.sqrt(np.mean(band_lin ** 2)))
            band_rssi_dbm = 10.0 * np.log10(rms_lin + 1e-20)

            # калибровка
            k_corr = float(self.k_cal_db if k_cal_db is None else k_cal_db)
            band_rssi_dbm += k_corr
            noise_dbm += k_corr

            snr_db = float(band_rssi_dbm - noise_dbm)

            return RSSIMeasurement(
                slave_id=self.slave_id,
                center_hz=float(center_hz),
                span_hz=float(span_hz),
                band_rssi_dbm=band_rssi_dbm,
                band_noise_dbm=noise_dbm,
                snr_db=snr_db,
                n_samples=int(got),
                ts=time.time(),
                flags={"valid": True},
            )
        finally:
            self.is_measuring = False

    # --- manager glue helpers ---
    def set_k_cal(self, k_db: float):
        self.k_cal_db = float(k_db)


class SlaveManager(QObject):
    all_measurements_complete = pyqtSignal(list)  # List[RSSIMeasurement]
    measurement_error = pyqtSignal(str)

    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.log = logger
        self.slaves: Dict[str, SlaveSDR] = {}
        self._lock = threading.Lock()

    def add_slave(self, slave_id: str, uri: str) -> bool:
        try:
            sl = SlaveSDR(slave_id, uri, self.log)
            if not sl.is_initialized:
                return False
            sl.measurement_complete.connect(self._on_measurement_complete)
            sl.measurement_error.connect(self._on_measurement_error)
            self.slaves[slave_id] = sl
            self.log.info(f"Added slave: {slave_id} ({uri})")
            return True
        except Exception as e:
            self.log.error(f"Add slave failed: {e}")
            return False

    def remove_slave(self, slave_id: str):
        try:
            if slave_id in self.slaves:
                self.slaves[slave_id].close()
                del self.slaves[slave_id]
        except Exception as e:
            self.log.error(f"Remove slave error: {e}")

    def measure_all_bands(self, windows: List[MeasurementWindow],
                          k_cal_db: Dict[str, float]) -> bool:
        if not self.slaves:
            self.log.warning("No slaves available")
            return False

        try:
            threads: List[threading.Thread] = []
            results: List[RSSIMeasurement] = []
            results_lock = threading.Lock()

            def worker(sl: SlaveSDR, w: MeasurementWindow):
                try:
                    k = k_cal_db.get(sl.slave_id, 0.0)
                    r = sl.measure_band_rssi(w.center, w.span, w.dwell_ms, k_cal_db=k)
                    with results_lock:
                        results.append(r)
                except Exception as e:
                    self.measurement_error.emit(f"{sl.slave_id}: {e}")

            for sl in list(self.slaves.values()):
                if not sl.is_initialized:
                    continue
                for w in windows:
                    t = threading.Thread(target=worker, args=(sl, w), daemon=True)
                    t.start()
                    threads.append(t)

            # дождаться всех
            for t in threads:
                t.join()

            self.all_measurements_complete.emit(results)
            return True

        except Exception as e:
            self.measurement_error.emit(str(e))
            return False

    def close_all(self):
        """Совместимость со старым кодом UI."""
        try:
            self.close()
        except AttributeError:
            try:
                self.shutdown()
            except Exception:
                pass
