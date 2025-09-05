# -*- coding: utf-8 -*-
"""
Синхронизация и калибровка HackRF устройств
Решает проблему частотных смещений и амплитудных расхождений между устройствами
"""

from __future__ import annotations
import os
import time
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class DeviceCalibration:
    """Калибровочные параметры для одного устройства."""
    serial: str
    frequency_offset_hz: float = 0.0  # Частотное смещение в Гц
    amplitude_offset_db: float = 0.0  # Амплитудное смещение в дБ
    phase_offset_deg: float = 0.0     # Фазовое смещение в градусах
    temperature_coefficient: float = 0.0  # Температурный коэффициент ppm/°C
    last_calibration_time: float = 0.0
    reference_temperature: float = 25.0  # Опорная температура °C


@dataclass
class CalibrationTarget:
    """Цель для калибровки."""
    name: str
    frequency_hz: float
    expected_power_dbm: Optional[float] = None
    bandwidth_hz: float = 10000  # 10 кГц по умолчанию
    duration_sec: float = 1.0


class HackRFSyncCalibrator:
    """
    Калибратор для синхронизации множества HackRF устройств.
    
    Функции:
    1. Определение частотных смещений относительно мастера
    2. Амплитудная калибровка устройств
    3. Временная синхронизация измерений
    4. Компенсация температурного дрейфа
    """
    
    def __init__(self, config_path: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config_path = config_path or Path.home() / ".panorama" / "hackrf_calibration.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Калибровочные данные для каждого устройства
        self.calibrations: Dict[str, DeviceCalibration] = {}
        
        # Опорные цели для калибровки
        self.reference_targets = [
            CalibrationTarget("FM_Radio", 100e6, -30.0, 200e3),
            CalibrationTarget("GSM900", 935e6, -45.0, 200e3),
            CalibrationTarget("GSM1800", 1850e6, -50.0, 200e3),
            CalibrationTarget("WiFi_2G4", 2450e6, -40.0, 20e6),
        ]
        
        # Параметры калибровки
        self.sync_timeout_sec = 30.0
        self.frequency_search_span_hz = 50e3
        self.amplitude_tolerance_db = 3.0
        
        # Синхронизация
        self._sync_barrier: Optional[threading.Barrier] = None
        self._sync_start_time: float = 0.0
        
        self.load_calibrations()
    
    def load_calibrations(self) -> bool:
        """Загружает калибровочные данные из файла."""
        if not self.config_path.exists():
            self.logger.info("Калибровочный файл не найден, будет создан новый")
            return False
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.calibrations = {}
            for serial, cal_data in data.get('calibrations', {}).items():
                self.calibrations[serial] = DeviceCalibration(**cal_data)
            
            self.logger.info(f"Загружены калибровочные данные для {len(self.calibrations)} устройств")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки калибровочных данных: {e}")
            return False
    
    def save_calibrations(self) -> bool:
        """Сохраняет калибровочные данные в файл."""
        try:
            data = {
                'calibrations': {serial: asdict(cal) for serial, cal in self.calibrations.items()},
                'last_update': time.time()
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Сохранены калибровочные данные для {len(self.calibrations)} устройств")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения калибровочных данных: {e}")
            return False
    
    def detect_frequency_offset(self, master_device, slave_device, target: CalibrationTarget) -> float:
        """
        Определяет частотное смещение slave относительно master на заданной частоте.
        
        Алгоритм:
        1. Master измеряет спектр вокруг целевой частоты
        2. Slave ищет максимум сигнала в окрестности
        3. Вычисляется разность частот максимумов
        """
        try:
            # Master получает опорный спектр (требуется от C-слоя — без заглушек)
            master_freqs, master_powers = master_device.get_spectrum(
                center_hz=target.frequency_hz,
                span_hz=self.frequency_search_span_hz * 2,
                dwell_ms=int(target.duration_sec * 1000)
            )
            
            if not master_freqs or not master_powers:
                raise RuntimeError("Master не получил спектр")
            
            # Находим пик в master спектре
            master_freqs = np.array(master_freqs)
            master_powers = np.array(master_powers)
            master_peak_idx = np.argmax(master_powers)
            master_peak_freq = master_freqs[master_peak_idx]
            
            # Slave ищет пик в том же диапазоне
            slave_freqs, slave_powers = slave_device.get_spectrum(
                center_hz=target.frequency_hz,
                span_hz=self.frequency_search_span_hz * 2,
                dwell_ms=int(target.duration_sec * 1000)
            )
            
            if not slave_freqs or not slave_powers:
                raise RuntimeError("Slave не получил спектр")
            
            slave_freqs = np.array(slave_freqs)
            slave_powers = np.array(slave_powers)
            slave_peak_idx = np.argmax(slave_powers)
            slave_peak_freq = slave_freqs[slave_peak_idx]
            
            # Частотное смещение = разность пиков
            frequency_offset = slave_peak_freq - master_peak_freq
            
            self.logger.debug(f"Частотное смещение: Master={master_peak_freq/1e6:.3f}MHz, "
                             f"Slave={slave_peak_freq/1e6:.3f}MHz, offset={frequency_offset:.0f}Hz")
            
            return frequency_offset
            
        except Exception as e:
            self.logger.error(f"Ошибка определения частотного смещения: {e}")
            return 0.0
    
    def detect_amplitude_offset(self, master_device, slave_device, target: CalibrationTarget) -> float:
        """
        Определяет амплитудное смещение slave относительно master.
        """
        try:
            # Измерения RSSI на той же частоте с fallback
            try:
                master_measurement = master_device.measure_rssi(
                    center_hz=target.frequency_hz,
                    span_hz=target.bandwidth_hz,
                    duration_sec=target.duration_sec
                )
            except Exception:
                master_measurement = None
            try:
                slave_measurement = slave_device.measure_rssi(
                    center_hz=target.frequency_hz,
                    span_hz=target.bandwidth_hz,
                    duration_sec=target.duration_sec
                )
            except Exception:
                slave_measurement = None
            
            if not master_measurement or not slave_measurement:
                raise RuntimeError("Не удалось получить измерения RSSI")
            
            # Амплитудное смещение = разность RSSI
            amplitude_offset = slave_measurement.rssi_dbm - master_measurement.rssi_dbm
            
            self.logger.debug(f"Амплитудное смещение: Master={master_measurement.rssi_dbm:.1f}dBm, "
                             f"Slave={slave_measurement.rssi_dbm:.1f}dBm, offset={amplitude_offset:.1f}dB")
            
            return amplitude_offset
            
        except Exception as e:
            self.logger.error(f"Ошибка определения амплитудного смещения: {e}")
            return 0.0
    
    def calibrate_device_pair(self, master_device, slave_device, slave_serial: str) -> bool:
        """
        Калибрует одно slave устройство относительно master.
        """
        self.logger.info(f"Начало калибровки устройства {slave_serial}")
        
        frequency_offsets = []
        amplitude_offsets = []
        
        # Калибровка по нескольким опорным частотам
        for target in self.reference_targets:
            try:
                self.logger.info(f"Калибровка на частоте {target.frequency_hz/1e6:.1f} МГц ({target.name})")
                
                # Частотное смещение
                freq_offset = self.detect_frequency_offset(master_device, slave_device, target)
                if abs(freq_offset) < self.frequency_search_span_hz:
                    frequency_offsets.append(freq_offset)
                
                # Амплитудное смещение
                amp_offset = self.detect_amplitude_offset(master_device, slave_device, target)
                if abs(amp_offset) < 50.0:  # Санитарная проверка
                    amplitude_offsets.append(amp_offset)
                
                time.sleep(0.1)  # Пауза между измерениями
                
            except Exception as e:
                self.logger.warning(f"Не удалось откалибровать на {target.name}: {e}")
                continue
        
        if not frequency_offsets or not amplitude_offsets:
            self.logger.error(f"Недостаточно данных для калибровки {slave_serial}")
            return False
        
        # Усреднение результатов
        avg_freq_offset = float(np.median(frequency_offsets))
        avg_amp_offset = float(np.median(amplitude_offsets))
        
        # Сохранение калибровочных данных
        calibration = DeviceCalibration(
            serial=slave_serial,
            frequency_offset_hz=avg_freq_offset,
            amplitude_offset_db=avg_amp_offset,
            last_calibration_time=time.time()
        )
        
        self.calibrations[slave_serial] = calibration
        
        self.logger.info(f"Калибровка {slave_serial} завершена: "
                        f"freq_offset={avg_freq_offset:.0f}Hz, "
                        f"amp_offset={avg_amp_offset:.1f}dB")
        
        return True

    # Runtime update from dialog
    def set_targets(self, targets_mhz: List[float], dwell_ms: int, search_span_khz: int,
                    amplitude_tolerance_db: float, sync_timeout_sec: float):
        try:
            self.reference_targets = [
                CalibrationTarget(f"F{int(f)}", float(f) * 1e6, None, 200e3) for f in targets_mhz
            ]
            self.frequency_search_span_hz = float(search_span_khz) * 1e3
            # duration in sec
            for t in self.reference_targets:
                t.duration_sec = max(0.05, float(dwell_ms) / 1000.0)
            self.amplitude_tolerance_db = float(amplitude_tolerance_db)
            self.sync_timeout_sec = float(sync_timeout_sec)
            return True
        except Exception:
            return False
    
    def calibrate_all_slaves(self, master_device, slave_devices: Dict[str, object]) -> bool:
        """
        Калибрует все slave устройства относительно master.
        """
        if not slave_devices:
            self.logger.warning("Нет slave устройств для калибровки")
            return False
        
        self.logger.info(f"Начало калибровки {len(slave_devices)} slave устройств")
        
        success_count = 0
        
        # Последовательная калибровка каждого slave
        for slave_serial, slave_device in slave_devices.items():
            try:
                if self.calibrate_device_pair(master_device, slave_device, slave_serial):
                    success_count += 1
                else:
                    self.logger.error(f"Калибровка {slave_serial} не удалась")
                
            except Exception as e:
                self.logger.error(f"Ошибка калибровки {slave_serial}: {e}")
        
        # Сохранение результатов
        if success_count > 0:
            self.save_calibrations()
            self.logger.info(f"Калибровка завершена: {success_count}/{len(slave_devices)} устройств")
            return True
        else:
            self.logger.error("Калибровка не удалась ни для одного устройства")
            return False
    
    def apply_calibration(self, device, serial: str) -> bool:
        """
        Применяет сохраненную калибровку к устройству.
        """
        if serial not in self.calibrations:
            self.logger.warning(f"Нет калибровочных данных для {serial}")
            return False
        
        calibration = self.calibrations[serial]
        
        try:
            # Применение частотного смещения через конфигурацию
            if hasattr(device, 'configure'):
                # Для slave устройств через SlaveConfig
                from panorama.drivers.hackrf.hackrf_slaves.hackrf_slave_wrapper import SlaveConfig
                current_config = getattr(device, '_current_config', None)
                if current_config:
                    updated_config = SlaveConfig(
                        center_freq_hz=current_config.center_freq_hz,
                        sample_rate_hz=current_config.sample_rate_hz,
                        lna_gain=current_config.lna_gain,
                        vga_gain=current_config.vga_gain,
                        amp_enable=current_config.amp_enable,
                        window_type=current_config.window_type,
                        dc_offset_correction=current_config.dc_offset_correction,
                        iq_balance_correction=current_config.iq_balance_correction,
                        freq_offset_hz=calibration.frequency_offset_hz  # Применяем смещение
                    )
                    device.configure(updated_config)
                    device._calibration_offset_db = calibration.amplitude_offset_db
                    
            self.logger.info(f"Применена калибровка для {serial}: "
                           f"freq_offset={calibration.frequency_offset_hz:.0f}Hz, "
                           f"amp_offset={calibration.amplitude_offset_db:.1f}dB")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка применения калибровки для {serial}: {e}")
            return False
    
    def setup_synchronous_measurement(self, device_count: int) -> bool:
        """
        Настройка синхронных измерений для множества устройств.
        """
        try:
            self._sync_barrier = threading.Barrier(device_count)
            self._sync_start_time = time.time() + 2.0  # Старт через 2 секунды
            
            self.logger.info(f"Настроена синхронизация для {device_count} устройств")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка настройки синхронизации: {e}")
            return False
    
    def wait_for_sync_start(self) -> bool:
        """
        Ожидание синхронного старта измерений.
        Вызывается каждым устройством перед началом измерения.
        """
        if not self._sync_barrier:
            return True  # Без синхронизации
        
        try:
            # Ожидание барьера
            self._sync_barrier.wait(timeout=self.sync_timeout_sec)
            
            # Точное время старта
            current_time = time.time()
            if current_time < self._sync_start_time:
                time.sleep(self._sync_start_time - current_time)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка синхронизации: {e}")
            return False
    
    def get_calibration_info(self, serial: str) -> Optional[Dict]:
        """Возвращает информацию о калибровке устройства."""
        if serial not in self.calibrations:
            return None
        
        cal = self.calibrations[serial]
        return {
            'serial': cal.serial,
            'frequency_offset_hz': cal.frequency_offset_hz,
            'amplitude_offset_db': cal.amplitude_offset_db,
            'phase_offset_deg': cal.phase_offset_deg,
            'last_calibration': time.strftime('%Y-%m-%d %H:%M:%S', 
                                            time.localtime(cal.last_calibration_time)),
            'age_hours': (time.time() - cal.last_calibration_time) / 3600
        }
    
    def is_calibration_valid(self, serial: str, max_age_hours: float = 24.0) -> bool:
        """Проверяет актуальность калибровки."""
        if serial not in self.calibrations:
            return False
        
        cal = self.calibrations[serial]
        age_hours = (time.time() - cal.last_calibration_time) / 3600
        
        return age_hours < max_age_hours
    
    def get_corrected_frequency(self, serial: str, nominal_frequency: float) -> float:
        """Возвращает скорректированную частоту с учетом калибровки."""
        if serial in self.calibrations:
            return nominal_frequency + self.calibrations[serial].frequency_offset_hz
        return nominal_frequency
    
    def get_corrected_power(self, serial: str, measured_power: float) -> float:
        """Возвращает скорректированную мощность с учетом калибровки."""
        if serial in self.calibrations:
            return measured_power + self.calibrations[serial].amplitude_offset_db
        return measured_power