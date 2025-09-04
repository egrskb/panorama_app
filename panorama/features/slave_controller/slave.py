# -*- coding: utf-8 -*-
"""
Slave SDR controller: измерение RMS RSSI в заданном окне частот.
Использует нативную C библиотеку hackrf_slave вместо SoapySDR.
"""

from __future__ import annotations
import os
import time
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker

from panorama.shared.rms_utils import compute_band_rssi_dbm, RMSCalculator, RMSMeasurement

# Импорт нашей нативной C библиотеки
try:
    from panorama.drivers.hackrf.hackrf_slaves.hackrf_slave_wrapper import (
        HackRFSlaveDevice, 
        HackRFSlaveError,
        HackRFSlaveDeviceError,
        HackRFSlaveConfigError,
        HackRFSlaveCaptureError,
        HackRFSlaveProcessingError,
        HackRFSlaveTimeoutError,
        list_devices
    )
    HACKRF_SLAVE_AVAILABLE = True
except Exception as e:
    HACKRF_SLAVE_AVAILABLE = False
    logging.getLogger(__name__).error(f"Failed to import hackrf_slave library: {e}")


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
        self.uri = uri  # Для HackRF это может быть серийный номер
        self.log = logger

        self.device = None
        self.is_initialized = False

        # Конфигурация устройства
        self.sample_rate = 8000000  # 8 МГц
        self.lna_gain = 16          # LNA усиление
        self.vga_gain = 20          # VGA усиление
        self.amp_enable = False     # RF усилитель
        self.bandwidth = 2500000    # 2.5 МГц
        self.frequency = 2.4e9      # Текущая частота
        self.k_cal_db = 0.0         # Калибровочная поправка

        self.is_measuring = False
        self.current_window: Optional[MeasurementWindow] = None

        # RMS calculator for trilateration
        self.rms_calculator = RMSCalculator()

        # Per-slave baseline estimation (noise tracking around measurement band)
        self.noise_baseline_dbm: float = -90.0
        self._noise_alpha: float = 0.1  # EMA for noise baseline

        # Отслеживание ошибок и автоматическое отключение
        self.error_count = 0
        self.max_errors = 20  # Увеличили порог до 20 ошибок
        self.error_timeout = 180  # Уменьшили время отключения до 3 минут
        self.disabled_until = 0  # Время до которого устройство отключено
        self.last_error_time = 0
        self.success_count = 0  # Счетчик успехов для восстановления

        self._mutex = QMutex()
        
        self._init_device()

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
            'last_error_time': self.last_error_time,
            'noise_baseline_dbm': self.noise_baseline_dbm
        }

    # --- Device init/close ---
    def _init_device(self):
        if not HACKRF_SLAVE_AVAILABLE:
            self.log.error("HackRF slave library not available")
            return
        
        try:
            # Парсинг URI для извлечения серийного номера
            serial = self._parse_serial_from_uri(self.uri)
            
            # Создаем и открываем устройство
            self.device = HackRFSlaveDevice(serial=serial, logger=self.log)
            
            if not self.device.open():
                raise HackRFSlaveDeviceError(f"Failed to open device with serial: {serial}")
            
            # Импортируем SlaveConfig из нового wrapper
            from panorama.drivers.hackrf.hackrf_slaves.hackrf_slave_wrapper import SlaveConfig
            
            # Создаем конфигурацию
            config = SlaveConfig(
                center_freq_hz=int(self.frequency),
                sample_rate_hz=self.sample_rate,
                lna_gain=self.lna_gain,
                vga_gain=self.vga_gain,
                amp_enable=self.amp_enable,
                window_type=1,  # HAMMING
                dc_offset_correction=True,
                iq_balance_correction=True,
                freq_offset_hz=0.0
            )
            
            # Конфигурируем устройство
            if not self.device.configure(config):
                raise HackRFSlaveConfigError(f"Failed to configure device")
            
            self.is_initialized = True
            self.status_changed.emit("READY")
            self.log.info(f"Slave {self.slave_id} initialized: {self.uri} (serial: {serial})")
            
        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower() or "unavailable" in error_msg.lower():
                self.log.warning(f"Slave {self.slave_id} device not available (may be in use as master): {e}")
            else:
                self.log.error(f"Failed to initialize slave {self.slave_id}: {e}")
            self.status_changed.emit("ERROR")
            if hasattr(self, 'device') and self.device:
                try:
                    self.device.close()
                except:
                    pass
                self.device = None
    
    def _parse_serial_from_uri(self, uri: str) -> str:
        """Извлекает серийный номер из URI."""
        if not uri:
            return ""
            
        # Парсинг различных форматов URI
        if 'serial=' in uri:
            # driver=hackrf,serial=XXXX или просто serial=XXXX
            for part in uri.split(','):
                if part.startswith('serial='):
                    return part[7:]  # Убираем 'serial='
        
        # Если это просто серийный номер (32 символа hex)
        if len(uri) == 32 and all(c in '0123456789abcdefABCDEF' for c in uri):
            return uri
            
        return ""

    def close(self):
        try:
            if self.device:
                self.device.close()
                self.device = None
            self.is_initialized = False
            self.log.info(f"Slave {self.slave_id} closed")
        except Exception as e:
            self.log.error(f"Error closing slave {self.slave_id}: {e}")

    def _reconfigure_device(self, new_bandwidth: Optional[int] = None) -> bool:
        """Переконфигурация устройства при необходимости."""
        try:
            if not self.device or not self.is_initialized:
                return False
            
            # При необходимости обновляем конфигурацию
            if new_bandwidth and new_bandwidth != self.bandwidth:
                self.bandwidth = new_bandwidth
                from panorama.drivers.hackrf.hackrf_slaves.hackrf_slave_wrapper import SlaveConfig
                config = SlaveConfig(
                    center_freq_hz=int(self.frequency),  # Используем текущую частоту
                    sample_rate_hz=self.sample_rate,
                    lna_gain=self.lna_gain,
                    vga_gain=self.vga_gain,
                    amp_enable=self.amp_enable,
                    window_type=1,  # HAMMING
                    dc_offset_correction=True,
                    iq_balance_correction=True,
                    freq_offset_hz=0.0
                )
                self.device.configure(config)
                self.log.debug(f"{self.slave_id}: reconfigured with bandwidth {new_bandwidth}")
            
            return True
        except Exception as e:
            self.log.error(f"{self.slave_id}: reconfiguration failed: {e}")
            return False

    # --- band measurement ---
    def measure_band_rssi(self, center_hz: float, span_hz: float, dwell_ms: int,
                          k_cal_db: Optional[float] = None) -> RSSIMeasurement:
        # Проверяем, не отключено ли устройство
        if self.is_disabled():
            current_time = time.time()
            remaining = self.disabled_until - current_time
            raise RuntimeError(f"Slave {self.slave_id} temporarily disabled. Retry in {remaining:.0f} seconds")
            
        if not self.is_initialized or not self.device:
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

            # Подготовка к измерению
            # Обновляем калибровку если передана
            calibration_db = k_cal_db if k_cal_db is not None else self.k_cal_db
            if calibration_db != self.k_cal_db:
                self.k_cal_db = calibration_db
                # Переконфигурируем устройство с новой калибровкой
                from panorama.drivers.hackrf.hackrf_slaves.hackrf_slave_wrapper import SlaveConfig
                config = SlaveConfig(
                    center_freq_hz=int(center_hz),
                    sample_rate_hz=self.sample_rate,
                    lna_gain=self.lna_gain,
                    vga_gain=self.vga_gain,
                    amp_enable=self.amp_enable,
                    window_type=1,  # HAMMING
                    dc_offset_correction=True,
                    iq_balance_correction=True,
                    freq_offset_hz=0.0
                )
                self.device.configure(config)
            
            # Автоматическая настройка полосы пропускания
            bw = max(200000, min(int(span_hz), int(self.sample_rate * 0.8)))
            self._reconfigure_device(bw)

            # Прямой доступ к устройству (только реальный SDR)
            duration_sec = dwell_ms / 1000.0
            result = self.device.measure_rssi(
                center_hz=center_hz,
                span_hz=span_hz,
                duration_sec=duration_sec
            )
            if result is None:
                raise RuntimeError("RSSI measurement returned None")
            measurement = RSSIMeasurement(
                slave_id=self.slave_id,
                center_hz=float(center_hz),
                span_hz=float(span_hz),
                band_rssi_dbm=result.rssi_dbm,
                band_noise_dbm=result.noise_floor_dbm,
                snr_db=result.snr_db,
                n_samples=result.sample_count,
                ts=result.timestamp,
                flags={"direct_device": True},
            )

            # Update per-slave noise baseline using EMA of measured noise floor
            try:
                nf = float(result.noise_floor_dbm)
                self.noise_baseline_dbm = (
                    (1.0 - self._noise_alpha) * self.noise_baseline_dbm + self._noise_alpha * nf
                )
            except Exception:
                pass
            
            # Записываем успех
            self._record_success()
            return measurement
            
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "busy" in error_msg.lower():
                # Timeout - временная проблема, не увеличиваем счетчик критических ошибок
                self.log.debug(f"Slave {self.slave_id}: temporary timeout (device busy with master)")
            else:
                # Записываем ошибку только для серьезных проблем
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
        if not self.is_initialized or not self.device:
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
            target_id = str(target.get('id', 'unknown'))
            
            # Используем новый CFFI wrapper для RSSI измерения
            span_hz = halfspan_hz * 2.0  # Полный span = 2 * halfspan
            duration_sec = dwell_ms / 1000.0  # Конвертируем в секунды
            
            result = self.device.measure_rssi(
                center_hz=center_hz,
                span_hz=span_hz,
                duration_sec=duration_sec
            )
            
            if result is None:
                raise RuntimeError("RSSI measurement returned None")
            
            # Преобразуем в наш формат RMSMeasurement
            from panorama.shared.rms_utils import RMSMeasurement
            measurement = RMSMeasurement(
                slave_id=self.slave_id,
                target_id=target_id,
                center_hz=center_hz,
                halfspan_hz=halfspan_hz,
                guard_hz=guard_hz,
                rssi_rms_dbm=result.rssi_dbm,
                noise_floor_dbm=result.noise_floor_dbm,
                snr_db=result.snr_db,
                n_samples=result.sample_count,
                timestamp=result.timestamp
            )
            
            if measurement:
                self.log.debug(f"Slave {self.slave_id}: RMS measurement for target {target_id} = {measurement.rssi_rms_dbm:.1f} dBm")
            
            return measurement
            
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "busy" in error_msg.lower():
                self.log.debug(f"Slave {self.slave_id}: RMS measurement timeout (device busy with master)")
            else:
                self.log.error(f"Slave {self.slave_id}: target RMS measurement failed: {e}")
            return None
        finally:
            self.is_measuring = False

    # --- manager glue helpers ---
    def set_k_cal(self, k_db: float):
        self.k_cal_db = float(k_db)
    
    def update_spectrum_from_master(self, freqs: np.ndarray, dbm: np.ndarray):
        # Virtual-slave режим отключён. Никаких данных от Master для слейвов.
        return
    
    def _measure_rssi_from_spectrum(self, center_hz: float, span_hz: float) -> Optional[RSSIMeasurement]:
        # Virtual-slave режим отключён
        return None


class SlaveManager(QObject):
    all_measurements_complete = pyqtSignal(list)  # List[RSSIMeasurement]
    measurement_error = pyqtSignal(str)
    # Новый сигнал для онлайновых апдейтов по мере готовности измерений
    measurement_progress = pyqtSignal(object)  # RSSIMeasurement
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
                    # Онлайн-уведомление: измерение готово — шлём в оркестратор/UI
                    try:
                        self.measurement_progress.emit(r)
                    except Exception:
                        pass
                    # Апдейтим baseline/статусы слейвов в UI
                    try:
                        self._emit_slaves_updated()
                    except Exception:
                        pass
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
                "time_until_enable": status_info['time_until_enable'],
                "noise_baseline_dbm": status_info.get('noise_baseline_dbm', -90.0)
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
