# panorama/features/trilateration/engine.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import time
import threading
from queue import Queue, Empty


@dataclass
class SignalMeasurement:
    """Измерение сигнала от одного SDR."""
    timestamp: float
    device_serial: str
    freq_mhz: float
    power_dbm: float
    bandwidth_khz: float
    noise_floor_dbm: float = -110.0
    

@dataclass
class SynchronizedMeasurement:
    """Синхронизированное измерение от всех SDR."""
    timestamp: float
    freq_mhz: float
    master_power: float
    slave1_power: float
    slave2_power: float
    confidence: float = 0.0
    

@dataclass
class TargetPosition:
    """Вычисленная позиция цели."""
    timestamp: float
    freq_mhz: float
    x: float
    y: float
    z: float = 0.0
    error_estimate: float = 0.0  # Оценка ошибки в метрах
    confidence: float = 0.0  # 0-1
    signal_type: str = "Unknown"
    measurements_count: int = 0
    

class TrilaterationEngine:
    """
    Движок трилатерации с синхронизацией измерений.
    
    Принцип работы:
    1. Каждый SDR отправляет измерения с временными метками
    2. Синхронизатор группирует измерения по частоте и времени
    3. Для синхронизированных измерений вычисляются координаты
    4. Используется фильтр Калмана для сглаживания траекторий
    """
    
    # Параметры синхронизации
    TIME_WINDOW_MS = 100  # Окно синхронизации в миллисекундах
    FREQ_TOLERANCE_MHZ = 0.5  # Допуск по частоте
    MIN_MEASUREMENTS = 3  # Минимум измерений для трилатерации
    
    # Параметры модели распространения сигнала
    PATH_LOSS_EXPONENT = 2.5  # Показатель затухания (2 для свободного пространства)
    REFERENCE_DISTANCE_M = 1.0  # Референсное расстояние
    REFERENCE_POWER_DBM = -40.0  # Мощность на референсном расстоянии
    
    def __init__(self):
        self.device_positions: Dict[str, Tuple[float, float, float]] = {}
        self.measurements_queue: Queue[SignalMeasurement] = Queue(maxsize=1000)
        self.pending_measurements: Dict[float, List[SignalMeasurement]] = {}  # freq -> measurements
        self.synchronized_measurements: Deque[SynchronizedMeasurement] = deque(maxlen=100)
        self.target_positions: Dict[float, TargetPosition] = {}  # freq -> last position
        self.kalman_filters: Dict[float, KalmanFilter] = {}  # freq -> filter
        
        self.is_running = False
        self.sync_thread: Optional[threading.Thread] = None
        self.calc_thread: Optional[threading.Thread] = None
        
    def set_device_positions(self, master_pos: Tuple[float, float, float],
                           slave1_pos: Tuple[float, float, float],
                           slave2_pos: Tuple[float, float, float],
                           master_serial: str, slave1_serial: str, slave2_serial: str):
        """Устанавливает позиции SDR устройств."""
        self.device_positions = {
            master_serial: master_pos,
            slave1_serial: slave1_pos,
            slave2_serial: slave2_pos
        }
        self.master_serial = master_serial
        self.slave1_serial = slave1_serial
        self.slave2_serial = slave2_serial
        
    def start(self):
        """Запускает движок трилатерации."""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Запускаем поток синхронизации
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()
        
        # Запускаем поток вычислений
        self.calc_thread = threading.Thread(target=self._calc_worker, daemon=True)
        self.calc_thread.start()
        
    def stop(self):
        """Останавливает движок."""
        self.is_running = False
        
        if self.sync_thread:
            self.sync_thread.join(timeout=1.0)
        if self.calc_thread:
            self.calc_thread.join(timeout=1.0)
            
    def add_measurement(self, measurement: SignalMeasurement):
        """Добавляет измерение от SDR."""
        if self.is_running:
            try:
                self.measurements_queue.put_nowait(measurement)
            except:
                pass  # Очередь полная, пропускаем
                
    def _sync_worker(self):
        """Рабочий поток синхронизации измерений."""
        while self.is_running:
            try:
                # Получаем измерение из очереди
                measurement = self.measurements_queue.get(timeout=0.1)
                
                # Округляем частоту для группировки
                freq_key = round(measurement.freq_mhz / self.FREQ_TOLERANCE_MHZ) * self.FREQ_TOLERANCE_MHZ
                
                # Добавляем в pending
                if freq_key not in self.pending_measurements:
                    self.pending_measurements[freq_key] = []
                
                self.pending_measurements[freq_key].append(measurement)
                
                # Очищаем старые измерения
                current_time = time.time()
                cutoff_time = current_time - self.TIME_WINDOW_MS / 1000.0
                
                for freq in list(self.pending_measurements.keys()):
                    # Фильтруем старые
                    self.pending_measurements[freq] = [
                        m for m in self.pending_measurements[freq]
                        if m.timestamp > cutoff_time
                    ]
                    
                    # Проверяем возможность синхронизации
                    measurements = self.pending_measurements[freq]
                    if len(measurements) >= self.MIN_MEASUREMENTS:
                        sync_result = self._try_synchronize(freq, measurements)
                        if sync_result:
                            self.synchronized_measurements.append(sync_result)
                            # Очищаем использованные измерения
                            self.pending_measurements[freq] = []
                            
            except Empty:
                continue
            except Exception as e:
                print(f"Sync error: {e}")
                
    def _try_synchronize(self, freq: float, measurements: List[SignalMeasurement]) -> Optional[SynchronizedMeasurement]:
        """Пытается синхронизировать измерения от разных SDR."""
        # Группируем по устройствам
        by_device = {}
        for m in measurements:
            if m.device_serial not in by_device:
                by_device[m.device_serial] = []
            by_device[m.device_serial].append(m)
            
        # Проверяем наличие всех устройств
        if not all(serial in by_device for serial in [self.master_serial, self.slave1_serial, self.slave2_serial]):
            return None
            
        # Берем последние измерения от каждого устройства
        master_m = by_device[self.master_serial][-1]
        slave1_m = by_device[self.slave1_serial][-1]
        slave2_m = by_device[self.slave2_serial][-1]
        
        # Проверяем временную синхронность
        timestamps = [master_m.timestamp, slave1_m.timestamp, slave2_m.timestamp]
        time_spread = max(timestamps) - min(timestamps)
        
        if time_spread > self.TIME_WINDOW_MS / 1000.0:
            return None
            
        # Создаем синхронизированное измерение
        return SynchronizedMeasurement(
            timestamp=np.mean(timestamps),
            freq_mhz=freq,
            master_power=master_m.power_dbm,
            slave1_power=slave1_m.power_dbm,
            slave2_power=slave2_m.power_dbm,
            confidence=1.0 - (time_spread * 1000.0 / self.TIME_WINDOW_MS)
        )
        
    def _calc_worker(self):
        """Рабочий поток вычисления координат."""
        while self.is_running:
            try:
                # Обрабатываем синхронизированные измерения
                if len(self.synchronized_measurements) > 0:
                    measurement = self.synchronized_measurements.popleft()
                    position = self._calculate_position(measurement)
                    
                    if position:
                        # Применяем фильтр Калмана
                        if measurement.freq_mhz not in self.kalman_filters:
                            self.kalman_filters[measurement.freq_mhz] = KalmanFilter()
                            
                        filtered_pos = self.kalman_filters[measurement.freq_mhz].update(
                            position.x, position.y, position.z
                        )
                        
                        position.x, position.y, position.z = filtered_pos
                        self.target_positions[measurement.freq_mhz] = position
                        
                time.sleep(0.01)  # Небольшая задержка
                
            except Exception as e:
                print(f"Calc error: {e}")
                
    def _calculate_position(self, measurement: SynchronizedMeasurement) -> Optional[TargetPosition]:
        """
        Вычисляет позицию цели методом трилатерации.
        
        Используется метод наименьших квадратов для решения системы уравнений:
        (x - x_i)² + (y - y_i)² + (z - z_i)² = d_i²
        """
        # Преобразуем RSSI в расстояния
        d_master = self._rssi_to_distance(measurement.master_power)
        d_slave1 = self._rssi_to_distance(measurement.slave1_power)
        d_slave2 = self._rssi_to_distance(measurement.slave2_power)
        
        # Получаем позиции SDR
        p_master = np.array(self.device_positions[self.master_serial])
        p_slave1 = np.array(self.device_positions[self.slave1_serial])
        p_slave2 = np.array(self.device_positions[self.slave2_serial])
        
        # Решаем систему методом наименьших квадратов
        # Линеаризуем систему уравнений относительно master
        A = 2 * np.array([
            p_slave1 - p_master,
            p_slave2 - p_master
        ])
        
        b = np.array([
            d_master**2 - d_slave1**2 + np.linalg.norm(p_slave1)**2 - np.linalg.norm(p_master)**2,
            d_master**2 - d_slave2**2 + np.linalg.norm(p_slave2)**2 - np.linalg.norm(p_master)**2
        ])
        
        try:
            # Решаем для 2D (x, y)
            solution_2d = np.linalg.lstsq(A[:, :2], b, rcond=None)[0]
            
            # Вычисляем z из уравнения сферы
            x, y = solution_2d
            z_squared = d_master**2 - (x - p_master[0])**2 - (y - p_master[1])**2
            z = np.sqrt(max(0, z_squared)) if z_squared > 0 else 0
            
            # Оцениваем ошибку
            distances = [
                np.linalg.norm([x, y, z] - p_master),
                np.linalg.norm([x, y, z] - p_slave1),
                np.linalg.norm([x, y, z] - p_slave2)
            ]
            
            errors = [
                abs(distances[0] - d_master),
                abs(distances[1] - d_slave1),
                abs(distances[2] - d_slave2)
            ]
            
            error_estimate = np.mean(errors)
            
            # Вычисляем уверенность на основе ошибки и уровня сигнала
            signal_strength = np.mean([measurement.master_power, measurement.slave1_power, measurement.slave2_power])
            confidence = max(0, min(1, (1.0 - error_estimate / 10.0) * (1.0 + signal_strength / 100.0)))
            
            return TargetPosition(
                timestamp=measurement.timestamp,
                freq_mhz=measurement.freq_mhz,
                x=float(x),
                y=float(y),
                z=float(z),
                error_estimate=float(error_estimate),
                confidence=float(confidence),
                measurements_count=3
            )
            
        except Exception as e:
            print(f"Trilateration failed: {e}")
            return None
            
    def _rssi_to_distance(self, rssi_dbm: float) -> float:
        """
        Преобразует RSSI в расстояние используя логарифмическую модель затухания.
        
        PL(d) = PL(d0) + 10*n*log10(d/d0)
        где:
        - PL(d) - затухание на расстоянии d
        - PL(d0) - затухание на референсном расстоянии d0
        - n - показатель затухания (2 для свободного пространства)
        """
        path_loss = self.REFERENCE_POWER_DBM - rssi_dbm
        distance = self.REFERENCE_DISTANCE_M * (10 ** (path_loss / (10 * self.PATH_LOSS_EXPONENT)))
        
        # Ограничиваем разумными пределами
        return np.clip(distance, 0.1, 1000.0)
        
    def get_current_positions(self) -> Dict[float, TargetPosition]:
        """Возвращает текущие позиции всех целей."""
        return self.target_positions.copy()
        

class KalmanFilter:
    """
    Простой фильтр Калмана для 3D позиции.
    """
    
    def __init__(self):
        # Состояние: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 100
        
        # Матрица перехода (модель постоянной скорости)
        self.F = np.array([
            [1, 0, 0, 1, 0, 0],  # x = x + vx*dt
            [0, 1, 0, 0, 1, 0],  # y = y + vy*dt
            [0, 0, 1, 0, 0, 1],  # z = z + vz*dt
            [0, 0, 0, 1, 0, 0],  # vx = vx
            [0, 0, 0, 0, 1, 0],  # vy = vy
            [0, 0, 0, 0, 0, 1],  # vz = vz
        ])
        
        # Матрица измерений (измеряем только позицию)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Шум процесса
        self.Q = np.eye(6) * 0.1
        self.Q[3:, 3:] *= 0.01  # Меньше шума для скорости
        
        # Шум измерений
        self.R = np.eye(3) * 5.0
        
        self.initialized = False
        
    def update(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Обновляет фильтр новым измерением."""
        measurement = np.array([x, y, z])
        
        if not self.initialized:
            # Первое измерение - инициализация
            self.state[:3] = measurement
            self.initialized = True
            return x, y, z
            
        # Предсказание
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        # Коррекция
        innovation = measurement - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ innovation
        self.covariance = (np.eye(6) - K @ self.H) @ self.covariance
        
        return tuple(self.state[:3])