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
from PyQt5.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker

from panorama.shared.rms_utils import compute_band_rssi_dbm, RMSCalculator, RMSMeasurement

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

        # RMS calculator for trilateration
        self.rms_calculator = RMSCalculator()

        # Отслеживание ошибок и автоматическое отключение
        self.error_count = 0
        self.max_errors = 20  # Увеличили порог до 20 ошибок
        self.error_timeout = 180  # Уменьшили время отключения до 3 минут
        self.disabled_until = 0  # Время до которого устройство отключено
        self.last_error_time = 0
        self.success_count = 0  # Счетчик успехов для восстановления

        self._mutex = QMutex()
        self._init_sdr()

    # --- Error management ---
    def _record_error(self):
        """Записывает ошибку и проверяет необходимость отключения устройства."""
        current_time = time.time()
        self.error_count += 1
        self.success_count = 0  # Сброс счетчика успехов при ошибке
        self.last_error_time = current_time
        
        if self.error_count >= self.max_errors:
            self.disabled_until = current_time + self.error_timeout
            self.log.error(f"Slave {self.slave_id} disabled due to {self.error_count} consecutive errors. "
                          f"Will retry after {self.error_timeout} seconds.")
            self.status_changed.emit("DISABLED")
            # Сброс счетчика для следующего периода
            self.error_count = 0

    def _record_success(self):
        """Записывает успешное измерение и уменьшает счетчик ошибок."""
        self.success_count += 1
        
        # Постепенно "прощаем" ошибки при успешных измерениях
        if self.success_count >= 3 and self.error_count > 0:
            self.error_count = max(0, self.error_count - 1)
            self.success_count = 0
            
        if self.error_count == 0 and self.success_count == 1:
            self.log.debug(f"Slave {self.slave_id} fully recovered")

    def is_disabled(self) -> bool:
        """Проверяет, отключено ли устройство временно."""
        current_time = time.time()
        if self.disabled_until > current_time:
            return True
        elif self.disabled_until > 0:
            # Время отключения прошло, сбрасываем статус
            self.disabled_until = 0
            self.log.info(f"Slave {self.slave_id} re-enabled after timeout")
            if self.is_initialized:
                self.status_changed.emit("READY")
        return False

    def get_status_info(self) -> dict:
        """Возвращает подробную информацию о состоянии устройства."""
        current_time = time.time()
        return {
            'slave_id': self.slave_id,
            'uri': self.uri,
            'is_initialized': self.is_initialized,
            'is_measuring': self.is_measuring,
            'is_disabled': self.is_disabled(),
            'error_count': self.error_count,
            'disabled_until': self.disabled_until,
            'time_until_enable': max(0, self.disabled_until - current_time),
            'last_error_time': self.last_error_time
        }

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
        # Проверяем, не отключено ли устройство
        if self.is_disabled():
            current_time = time.time()
            remaining = self.disabled_until - current_time
            raise RuntimeError(f"Slave {self.slave_id} temporarily disabled. Retry in {remaining:.0f} seconds")
            
        if not self.is_initialized:
            raise RuntimeError(f"Slave {self.slave_id} not initialized")
        if self.is_measuring:
            raise RuntimeError(f"Slave {self.slave_id} busy")

        with QMutexLocker(self._mutex):
            if self.is_measuring:
                raise RuntimeError(f"Slave {self.slave_id} already measuring (thread safety)")
            self.is_measuring = True
        
        try:
            self.current_window = MeasurementWindow(center=center_hz, span=span_hz,
                                                    dwell_ms=dwell_ms, epoch=time.time())

            # Настройка тюнера
            self.sdr.setFrequency(SOAPY_SDR_RX, 0, center_hz)
            # чтобы захватить всю полосу, выставим bandwidth ≈ span
            bw = max(200e3, min(span_hz, self.sample_rate * 0.8))
            self.sdr.setBandwidth(SOAPY_SDR_RX, 0, bw)
            # Небольшая задержка на установление ГКЧ/фильтров
            time.sleep(0.01)

            # Сбор выборки с улучшенной обработкой ошибок
            n_samp = int(max(1, round(self.sample_rate * (dwell_ms / 1000.0))))
            
            # Повторяем попытку сбора данных несколько раз
            max_collection_attempts = 3
            for collection_attempt in range(max_collection_attempts):
                out = np.empty(n_samp, dtype=np.complex64)
                offset = 0
                max_read_attempts = 15  # Увеличили количество попыток
                read_attempts = 0
                # 20мс чанки или меньше, чтобы быстрее получать данные
                chunk = max(1024, min(int(self.sample_rate * 0.02), n_samp))
                
                while offset < n_samp and read_attempts < max_read_attempts:
                    try:
                        tmp = np.empty(chunk, dtype=np.complex64)
                        rc = self.sdr.readStream(self.rx_stream, [tmp], len(tmp))
                        
                        # Обработка возвращаемого значения
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
                            read_attempts = 0  # Сброс счетчика при успехе
                        else:
                            read_attempts += 1
                            time.sleep(0.01)  # Увеличили задержку между попытками
                            
                    except Exception as e:
                        read_attempts += 1
                        self.log.debug(f"readStream exception in {self.slave_id}: {e}")
                        time.sleep(0.02)
                
                # Если получили данные, выходим из цикла повторений
                if offset > 0:
                    break
                    
                # Логируем неудачную попытку
                self.log.warning(f"Collection attempt {collection_attempt + 1}/{max_collection_attempts} failed for {self.slave_id}, offset={offset}")
                
                # Между попытками делаем более длительную паузу вместо перезапуска потока
                if collection_attempt < max_collection_attempts - 1:
                    time.sleep(0.2)  # Увеличенная пауза между попытками
            
            if offset == 0:
                raise RuntimeError(f"readStream returned no data after {max_collection_attempts} attempts")

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

            # Используем централизованную функцию для RMS расчета
            band_rssi_dbm = compute_band_rssi_dbm(freqs_hz, psd_dbm, center_hz, span_hz/2)
            
            if band_rssi_dbm is None:
                raise RuntimeError("Failed to compute band RSSI - insufficient frequency bins")
            
            # Для расчета шумового пола извлекаем нужную полосу
            freq_mask = (freqs_hz >= center_hz - span_hz/2) & (freqs_hz <= center_hz + span_hz/2)
            band_powers = psd_dbm[freq_mask]
            
            # шумовой пол — медиана нижних 30%
            sorted_vals = np.sort(band_powers)
            k = max(1, int(len(sorted_vals) * 0.3))
            noise_dbm = float(np.median(sorted_vals[:k]))

            # калибровка
            k_corr = float(self.k_cal_db if k_cal_db is None else k_cal_db)
            band_rssi_dbm += k_corr
            noise_dbm += k_corr

            snr_db = float(band_rssi_dbm - noise_dbm)

            measurement = RSSIMeasurement(
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
            
            # Записываем успех
            self._record_success()
            return measurement
            
        except Exception as e:
            # Записываем ошибку
            self._record_error()
            raise e
        finally:
            self.is_measuring = False

    def measure_target_rms(self, target: Dict, dwell_ms: int = 400) -> Optional[RMSMeasurement]:
        """
        Measures RMS for a specific target using halfspan approach.
        
        Args:
            target: Target dict with {id, center_hz, halfspan_hz, guard_hz, ...}
            dwell_ms: Dwell time in milliseconds
            
        Returns:
            RMSMeasurement or None if failed
        """
        if not self.is_initialized:
            self.log.error(f"Slave {self.slave_id} not initialized")
            return None
            
        if self.is_measuring:
            self.log.warning(f"Slave {self.slave_id} busy")
            return None

        try:
            self.is_measuring = True
            
            center_hz = float(target.get('center_hz', 0))
            halfspan_hz = float(target.get('halfspan_hz', 2.5e6))
            guard_hz = float(target.get('guard_hz', 1e6))
            
            # Calculate working span: 2 * (halfspan + guard)
            working_span = 2 * (halfspan_hz + guard_hz)
            
            # Tune to center frequency
            self.sdr.setFrequency(SOAPY_SDR_RX, 0, center_hz)
            
            # Set bandwidth to cover working span
            bw = max(200e3, min(working_span, self.sample_rate * 0.8))
            self.sdr.setBandwidth(SOAPY_SDR_RX, 0, bw)
            
            # Brief settling time
            time.sleep(0.01)
            
            # Collect samples
            n_samp = int(max(1, round(self.sample_rate * (dwell_ms / 1000.0))))
            out = np.empty(n_samp, dtype=np.complex64)
            offset = 0
            max_attempts = 10
            attempts = 0
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
                self.log.error(f"Slave {self.slave_id}: no data received")
                return None

            sig = out[:offset]
            
            # Compute spectrum
            win = np.hamming(len(sig)).astype(np.float32)
            sig_w = sig * win
            sp = np.fft.fft(sig_w)
            sp = np.fft.fftshift(sp)
            psd = (np.abs(sp) ** 2) / np.sum(win ** 2)
            
            # Convert to dBm
            psd_dbm = 10.0 * np.log10(psd + 1e-20) - 174.0 + 10.0 * np.log10(self.sample_rate / len(sig_w))
            
            # Frequency axis
            freqs = np.fft.fftfreq(len(sig_w), d=1.0 / self.sample_rate)
            freqs = np.fft.fftshift(freqs)
            freqs_hz = freqs + center_hz
            
            # Apply calibration
            psd_dbm += self.k_cal_db
            
            # Use RMS calculator to get measurement
            measurement = self.rms_calculator.measure_target_rms(
                freqs_hz, psd_dbm, target, self.slave_id
            )
            
            if measurement:
                self.log.debug(f"Slave {self.slave_id}: RMS measurement for target {target.get('id')} = {measurement.rssi_rms_dbm:.1f} dBm")
            
            return measurement
            
        except Exception as e:
            self.log.error(f"Slave {self.slave_id}: target RMS measurement failed: {e}")
            return None
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

    def measure_target_rms_all(self, targets: List[Dict], 
                              dwell_ms: int = 400) -> List[RMSMeasurement]:
        """
        Measures RMS for targets using all available slaves.
        
        Args:
            targets: List of target dicts with {id, center_hz, halfspan_hz, ...}
            dwell_ms: Dwell time in milliseconds
            
        Returns:
            List of RMS measurements from all slaves
        """
        if not self.slaves:
            self.log.warning("No slaves available for target RMS measurement")
            return []

        results: List[RMSMeasurement] = []
        threads: List[threading.Thread] = []
        results_lock = threading.Lock()

        def worker(slave: SlaveSDR, target: Dict):
            try:
                measurement = slave.measure_target_rms(target, dwell_ms)
                if measurement:
                    with results_lock:
                        results.append(measurement)
            except Exception as e:
                self.log.error(f"Target RMS measurement error for {slave.slave_id}: {e}")

        # Start measurement threads for each slave-target combination
        for slave in self.slaves.values():
            if not slave.is_initialized or slave.is_disabled():
                continue
            for target in targets:
                thread = threading.Thread(target=worker, args=(slave, target), daemon=True)
                thread.start()
                threads.append(thread)

        # Wait for all measurements to complete
        for thread in threads:
            thread.join()

        self.log.info(f"Completed {len(results)} RMS measurements for {len(targets)} targets")
        return results

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

            active_slaves = 0
            total_slaves = len(self.slaves)
            
            for sl in list(self.slaves.values()):
                if not sl.is_initialized:
                    self.log.debug(f"Skipping {sl.slave_id}: not initialized")
                    continue
                if sl.is_disabled():
                    self.log.debug(f"Skipping {sl.slave_id}: temporarily disabled")
                    continue
                    
                active_slaves += 1
                for w in windows:
                    t = threading.Thread(target=worker, args=(sl, w), daemon=True)
                    t.start()
                    threads.append(t)
            
            self.log.info(f"Starting measurements with {active_slaves}/{total_slaves} active slaves")
            if active_slaves == 0:
                self.log.warning("No active slaves available for measurement!")

            # дождаться всех
            for t in threads:
                t.join()

            self.log.info(f"Measurements completed: {len(results)} results from {active_slaves} slaves")
            for result in results:
                self.log.debug(f"Result: {result.slave_id} -> {result.band_rssi_dbm:.1f} dBm at {result.center_hz/1e6:.1f} MHz")
            
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
        """Snapshot for UI/map: {id: {uri, is_initialized, is_disabled, error_info}}"""
        out: Dict[str, Dict[str, object]] = {}
        for sid, sl in self.slaves.items():
            status_info = sl.get_status_info()
            out[sid] = {
                "uri": sl.uri, 
                "is_initialized": bool(sl.is_initialized),
                "is_disabled": status_info['is_disabled'],
                "error_count": status_info['error_count'],
                "time_until_enable": status_info['time_until_enable']
            }
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
