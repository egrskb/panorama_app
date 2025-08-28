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
            # Небольшая задержка на установление ГКЧ/фильтров
            time.sleep(0.01)

            # Сбор выборки
            n_samp = int(max(1, round(self.sample_rate * (dwell_ms / 1000.0))))
            # Читаем потоковыми чанками с ретраями
            out = np.empty(n_samp, dtype=np.complex64)
            offset = 0
            max_attempts = 10
            attempts = 0
            # 20мс чанки или меньше, чтобы быстрее получать данные
            chunk = max(1024, min(int(self.sample_rate * 0.02), n_samp))
            while offset < n_samp and attempts < max_attempts:
                tmp = np.empty(chunk, dtype=np.complex64)
                rc = self.sdr.readStream(self.rx_stream, [tmp], len(tmp))
                if isinstance(rc, tuple):
                    got = int(rc[0])
                else:
                    try:
                        got = int(getattr(rc, 'ret', rc))
                    except Exception:
                        got = -1
                if got and got > 0:
                    take = min(got, n_samp - offset)
                    out[offset:offset+take] = tmp[:take]
                    offset += take
                    attempts = 0
                else:
                    attempts += 1
                    time.sleep(0.005)
            if offset == 0:
                raise RuntimeError("readStream returned no data")

            sig = out[:offset]

            # спектр
            # окно Хэмминга для RMS оценки
            win = np.hamming(len(sig)).astype(np.float32)
            sig_w = sig * win
            # FFT для комплексного потока CF32
            sp = np.fft.fft(sig_w)
            sp = np.fft.fftshift(sp)
            psd = (np.abs(sp) ** 2) / np.sum(win ** 2)
            psd_dbm = 10.0 * np.log10(psd + 1e-20) - 174.0 + 10.0 * np.log10(self.sample_rate / len(sig_w))

            # частотная ось этого рида (центрированная вокруг 0), затем к абсолютной частоте
            freqs = np.fft.fftfreq(len(sig_w), d=1.0 / self.sample_rate)
            freqs = np.fft.fftshift(freqs)
            freqs = freqs + 0.0  # базовая полоса вокруг 0 Гц
            freqs_hz = freqs + center_hz

            # ВАЖНО: Вычисляем RMS по всей полосе span_hz
            # Фильтруем только нужный диапазон
            freq_mask = (freqs_hz >= center_hz - span_hz/2) & (freqs_hz <= center_hz + span_hz/2)
            band_powers = psd_dbm[freq_mask]
            
            # Вычисляем RMS в линейной шкале
            band_linear = 10.0 ** (band_powers / 10.0)
            rms_linear = np.sqrt(np.mean(band_linear))
            band_rssi_dbm = 10.0 * np.log10(rms_linear + 1e-20)
            
            # шумовой пол — медиана нижних 30%
            sorted_vals = np.sort(band_powers)
            k = max(1, int(len(sorted_vals) * 0.3))
            noise_dbm = float(np.median(sorted_vals[:k]))

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
    slaves_updated = pyqtSignal(dict)  # {slave_id: {"uri": str, "is_initialized": bool}}

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
            # Обработчики могут использоваться в одиночном режиме измерений
            # Добавляем no-op/лог-обработчики, чтобы не падать при connect
            try:
                sl.measurement_complete.connect(self._on_measurement_complete)
                sl.measurement_error.connect(self._on_measurement_error)
            except Exception:
                pass
            self.slaves[slave_id] = sl
            self.log.info(f"Added slave: {slave_id} ({uri})")
            self._emit_slaves_updated()
            return True
        except Exception as e:
            self.log.error(f"Add slave failed: {e}")
            return False

    def remove_slave(self, slave_id: str):
        try:
            if slave_id in self.slaves:
                self.slaves[slave_id].close()
                del self.slaves[slave_id]
                self._emit_slaves_updated()
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

    def get_slave_status(self) -> Dict[str, Dict[str, object]]:
        """Snapshot for UI/map: {id: {uri, is_initialized}}"""
        out: Dict[str, Dict[str, object]] = {}
        for sid, sl in self.slaves.items():
            out[sid] = {"uri": sl.uri, "is_initialized": bool(sl.is_initialized)}
        return out

    def _emit_slaves_updated(self):
        try:
            self.slaves_updated.emit(self.get_slave_status())
        except Exception:
            pass

    # --- optional signal handlers for single-shot flows ---
    def _on_measurement_complete(self, measurement: RSSIMeasurement):
        """Опциональный обработчик завершения измерения одного слейва (для совместимости)."""
        try:
            # Передаём как список для совместимости с сигналом all_measurements_complete
            self.all_measurements_complete.emit([measurement])
        except Exception as e:
            self.log.error(f"_on_measurement_complete error: {e}")

    def _on_measurement_error(self, msg: str):
        """Опциональный обработчик ошибки измерения."""
        try:
            self.measurement_error.emit(str(msg))
        except Exception:
            pass
