# panorama/features/trilateration/coordinator.py
"""
Координатор системы трилатерации - связывает Master, Slaves и движок трилатерации.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QThread

from panorama.features.detector.peak_watchlist_manager import (
    PeakWatchlistManager, VideoSignalPeak, WatchlistEntry
)
from panorama.features.trilateration.rssi_engine import (
    RSSITrilaterationEngine, TrilaterationResult
)


@dataclass
class SlaveRSSIMeasurement:
    """Измерение RSSI от одного slave."""
    slave_id: str
    peak_id: str
    center_freq_hz: float
    rssi_rms_dbm: float
    timestamp: float
    snr_db: float = 0.0
    is_valid: bool = True


class RSSICollectorThread(QThread):
    """Поток для сбора RSSI измерений от всех slaves."""
    
    measurements_ready = pyqtSignal(str, dict)  # peak_id, {slave_id: rssi}
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.measurements_queue: List[SlaveRSSIMeasurement] = []
        self.collection_interval_ms = 100  # Интервал сбора
    
    def add_measurement(self, measurement: SlaveRSSIMeasurement):
        """Добавляет измерение в очередь."""
        self.measurements_queue.append(measurement)
    
    def run(self):
        """Основной цикл сбора измерений."""
        self.running = True
        
        while self.running:
            # Группируем измерения по peak_id
            grouped: Dict[str, Dict[str, float]] = {}
            current_time = time.time()
            
            # Обрабатываем накопленные измерения
            measurements_to_process = []
            for m in self.measurements_queue:
                # Берем только свежие измерения (не старше 1 сек)
                if current_time - m.timestamp < 1.0 and m.is_valid:
                    measurements_to_process.append(m)
            
            # Очищаем очередь
            self.measurements_queue = [
                m for m in self.measurements_queue 
                if current_time - m.timestamp < 1.0
            ]
            
            # Группируем по peak_id
            for m in measurements_to_process:
                if m.peak_id not in grouped:
                    grouped[m.peak_id] = {}
                grouped[m.peak_id][m.slave_id] = m.rssi_rms_dbm
            
            # Эмитим готовые группы (где есть минимум 3 измерения)
            for peak_id, measurements in grouped.items():
                if len(measurements) >= 3:
                    self.measurements_ready.emit(peak_id, measurements)
            
            # Спим
            self.msleep(self.collection_interval_ms)
    
    def stop(self):
        """Останавливает поток."""
        self.running = False


class TrilaterationCoordinator(QObject):
    """
    Главный координатор системы трилатерации.
    Связывает:
    - Master SDR (детектор пиков)
    - Slave SDR (измерения RSSI)
    - Движок трилатерации
    - UI карты
    """
    
    # Сигналы
    target_detected = pyqtSignal(object)  # TrilaterationResult
    target_updated = pyqtSignal(object)   # TrilaterationResult
    target_lost = pyqtSignal(str)         # peak_id
    watchlist_changed = pyqtSignal(list)  # List[WatchlistEntry]
    status_message = pyqtSignal(str)
    error_message = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Компоненты системы
        self.peak_manager = PeakWatchlistManager()
        self.trilateration_engine = RSSITrilaterationEngine()
        self.rssi_collector = RSSICollectorThread()
        
        # Состояние
        self.is_running = False
        self.slave_positions: Dict[str, Tuple[float, float, float]] = {}
        self.active_targets: Dict[str, TrilaterationResult] = {}
        
        # Параметры
        self.user_span_mhz = 5.0  # Пользовательский span для watchlist
        self.path_loss_exponent = 2.5  # Коэффициент затухания
        self.target_timeout_sec = 10.0  # Таймаут потери цели
        
        # Таймер обновления
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_targets)
        self.update_timer.setInterval(500)  # Обновление 2 раза в секунду
        
        # Подключаем сигналы
        self._connect_signals()
    
    def _connect_signals(self):
        """Подключает внутренние сигналы."""
        # От менеджера пиков
        self.peak_manager.peak_detected.connect(self._on_peak_detected)
        self.peak_manager.watchlist_updated.connect(self._on_watchlist_updated)
        
        # От коллектора RSSI
        self.rssi_collector.measurements_ready.connect(self._on_rssi_measurements_ready)
        self.rssi_collector.error_occurred.connect(self.error_message.emit)
        
        # От движка трилатерации
        self.trilateration_engine.target_update.connect(self._on_target_calculated)
        self.trilateration_engine.trilateration_error.connect(self.error_message.emit)
    
    def start(self):
        """Запускает систему трилатерации."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Запускаем коллектор RSSI
        self.rssi_collector.start()
        
        # Запускаем таймер обновления
        self.update_timer.start()
        
        self.status_message.emit("Система трилатерации запущена")
        print("[Coordinator] Trilateration system started")
    
    def stop(self):
        """Останавливает систему трилатерации."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Останавливаем компоненты
        self.rssi_collector.stop()
        self.update_timer.stop()
        try:
            if self.rssi_collector.isRunning():
                self.rssi_collector.wait(500)
        except Exception:
            pass
        
        # Очищаем состояние
        self.peak_manager.clear_watchlist()
        self.active_targets.clear()
        
        self.status_message.emit("Система трилатерации остановлена")
        print("[Coordinator] Trilateration system stopped")
    
    def set_slave_positions(self, positions: Dict[str, Tuple[float, float, float]]):
        """
        Устанавливает позиции slave SDR.
        
        Args:
            positions: Словарь {slave_id: (x, y, z)} в метрах
        """
        self.slave_positions = positions.copy()
        # Гарантируем, что slave0 находится в (0,0,0)
        if 'slave0' not in self.slave_positions:
            self.slave_positions['slave0'] = (0.0, 0.0, 0.0)
        else:
            sx, sy, sz = self.slave_positions['slave0']
            self.slave_positions['slave0'] = (0.0, 0.0, float(sz))
        
        # Обновляем в движке трилатерации
        for slave_id, (x, y, z) in self.slave_positions.items():
            self.trilateration_engine.add_station(slave_id, x, y, z)
        
        print(f"[Coordinator] Updated {len(self.slave_positions)} slave positions")
        self.status_message.emit(f"Обновлено позиций slave: {len(self.slave_positions)}")
    
    def set_user_span(self, halfspan_mhz: float):
        """Устанавливает пользовательский halfspan для RMS измерений."""
        self.user_span_mhz = halfspan_mhz * 2  # Сохраняем как полную ширину для совместимости
        # Устанавливаем полную ширину для watchlist_span_hz (2 × halfspan)
        self.peak_manager.watchlist_span_hz = halfspan_mhz * 2e6
        # Устанавливаем halfspan для RMS расчетов
        if hasattr(self.peak_manager, 'rms_halfspan_hz'):
            self.peak_manager.rms_halfspan_hz = halfspan_mhz * 1e6
        print(f"[Coordinator] RMS halfspan set to {halfspan_mhz} MHz (full span: {halfspan_mhz * 2} MHz)")
    
    def set_path_loss_exponent(self, n: float):
        """Устанавливает коэффициент затухания."""
        self.path_loss_exponent = n
        self.trilateration_engine.set_path_loss_exponent(n)
        print(f"[Coordinator] Path loss exponent set to {n}")
    
    def process_master_spectrum(self, freqs_hz: np.ndarray, power_dbm: np.ndarray):
        """
        Обрабатывает спектр от Master SDR.
        Ищет пики и добавляет в watchlist.
        """
        if not self.is_running:
            return
        
        # Обрабатываем спектр через менеджер пиков
        peaks = self.peak_manager.process_spectrum(
            freqs_hz, power_dbm, 
            user_span_hz=self.user_span_mhz * 1e6
        )
        
        if peaks:
            print(f"[Coordinator] Detected {len(peaks)} peaks from Master")
    
    def add_slave_rssi_measurement(self, slave_id: str, peak_id: str, 
                                   center_freq_hz: float, rssi_rms_dbm: float):
        """
        Добавляет измерение RSSI от slave.
        
        Args:
            slave_id: ID slave устройства
            peak_id: ID пика из watchlist
            center_freq_hz: Центральная частота
            rssi_rms_dbm: Измеренный RMS RSSI в дБм
        """
        measurement = SlaveRSSIMeasurement(
            slave_id=slave_id,
            peak_id=peak_id,
            center_freq_hz=center_freq_hz,
            rssi_rms_dbm=rssi_rms_dbm,
            timestamp=time.time()
        )
        
        # Добавляем в коллектор
        self.rssi_collector.add_measurement(measurement)
        
        # Обновляем в менеджере пиков
        self.peak_manager.update_rssi_measurement(peak_id, slave_id, rssi_rms_dbm)
    
    def _on_peak_detected(self, peak: VideoSignalPeak):
        """Обработка обнаруженного пика."""
        print(f"[Coordinator] Peak detected: {peak.center_freq_hz/1e6:.1f} MHz, "
              f"BW={peak.bandwidth_hz/1e6:.1f} MHz, SNR={peak.snr_db:.1f} dB")
        self.status_message.emit(
            f"Обнаружен пик: {peak.center_freq_hz/1e6:.1f} МГц"
        )
    
    def _on_watchlist_updated(self, entries: List[WatchlistEntry]):
        """Обработка обновления watchlist."""
        self.watchlist_changed.emit(entries)
        print(f"[Coordinator] Watchlist updated: {len(entries)} entries")
    
    def _on_rssi_measurements_ready(self, peak_id: str, measurements: Dict[str, float]):
        """
        Обработка готовых измерений RSSI для трилатерации.
        
        Args:
            peak_id: ID пика
            measurements: Словарь {slave_id: rssi_dbm}
        """
        # Получаем частоту из watchlist
        freq_mhz = 0.0
        if peak_id in self.peak_manager.watchlist:
            freq_mhz = self.peak_manager.watchlist[peak_id].center_freq_hz / 1e6
        
        print(f"[Coordinator] RSSI measurements ready for peak {peak_id}: "
              f"{len(measurements)} slaves, freq={freq_mhz:.1f} MHz")
        
        # Вычисляем позицию с трекингом
        result = self.trilateration_engine.calculate_position_with_tracking(
            measurements, peak_id, freq_mhz, alpha=0.7
        )
        
        if result:
            print(f"[Coordinator] Position calculated: ({result.x:.1f}, {result.y:.1f}, {result.z:.1f}) m")
            # транслируем наружу
            self.target_detected.emit(result)
    
    def _on_target_calculated(self, result: TrilaterationResult):
        """Обработка вычисленной позиции цели."""
        peak_id = result.peak_id
        
        # Проверяем, новая ли это цель
        is_new = peak_id not in self.active_targets
        
        # Сохраняем/обновляем
        self.active_targets[peak_id] = result
        
        # Эмитим соответствующий сигнал
        if is_new:
            self.target_detected.emit(result)
            self.status_message.emit(
                f"Новая цель: {result.freq_mhz:.1f} МГц на ({result.x:.1f}, {result.y:.1f}) м"
            )
        else:
            self.target_updated.emit(result)
    
    def _update_targets(self):
        """Периодическое обновление состояния целей."""
        current_time = time.time()
        targets_to_remove = []
        
        # Проверяем таймауты
        for peak_id, target in self.active_targets.items():
            if current_time - target.timestamp > self.target_timeout_sec:
                targets_to_remove.append(peak_id)
        
        # Удаляем потерянные цели
        for peak_id in targets_to_remove:
            del self.active_targets[peak_id]
            self.target_lost.emit(peak_id)
            self.status_message.emit(f"Цель потеряна: {peak_id}")
            print(f"[Coordinator] Target lost: {peak_id}")
    
    def get_active_targets(self) -> List[TrilaterationResult]:
        """Получает список активных целей."""
        return list(self.active_targets.values())
    
    def get_target_trajectory(self, peak_id: str) -> List[Tuple[float, float, float]]:
        """Получает траекторию цели."""
        return self.trilateration_engine.get_target_trajectory(peak_id)
    
    def clear_all(self):
        """Очищает все данные."""
        self.peak_manager.clear_watchlist()
        self.trilateration_engine.clear_history()
        self.active_targets.clear()
        self.status_message.emit("Все данные очищены")