# panorama/features/trilateration/rssi_engine.py
"""
Движок трилатерации на основе RSSI измерений от трех SDR.
Реализует алгоритм из документа с функцией ошибки и оптимизацией.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from PyQt5.QtCore import QObject, pyqtSignal
import time


@dataclass
class SDRStation:
    """Позиция SDR станции."""
    id: str
    x: float  # метры
    y: float  # метры
    z: float  # метры
    calibration_offset_db: float = 0.0  # Калибровочная поправка


@dataclass
class TrilaterationResult:
    """Результат трилатерации."""
    x: float  # метры
    y: float  # метры
    z: float  # метры
    error: float  # Ошибка оптимизации
    confidence: float  # Уверенность в результате (0-1)
    timestamp: float
    peak_id: str
    freq_mhz: float
    rssi_measurements: Dict[str, float]  # slave_id -> rssi_dbm


class RSSITrilaterationEngine(QObject):
    """
    Движок трилатерации по RSSI на основе алгоритма из документа.
    
    Формула RSSI: P_i = P_t - 10*n*log10(d_i)
    где:
    - P_i - принятая мощность на i-й станции (RSSI)
    - P_t - мощность передатчика (неизвестна, исключается)
    - n - коэффициент потерь (2-4, подбирается)
    - d_i - расстояние от передатчика до i-й станции
    """
    
    target_update = pyqtSignal(object)  # TrilaterationResult
    trilateration_error = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Позиции SDR станций
        self.stations: Dict[str, SDRStation] = {}
        
        # Параметры модели распространения
        self.path_loss_exponent = 2.5  # n - коэффициент потерь (2 для открытого пространства, 3-4 для помещений)
        self.reference_distance_m = 1.0  # Опорное расстояние
        self.reference_power_dbm = -30.0  # Опорная мощность
        
        # Параметры оптимизации
        self.max_iterations = 1000
        self.tolerance = 1e-6
        self.search_bounds = (-100, 100)  # Границы поиска в метрах
        
        # История для трекинга
        self.target_history: Dict[str, List[TrilaterationResult]] = {}
        self.max_history_length = 100
    
    def add_station(self, station_id: str, x: float, y: float, z: float = 0.0, 
                   calibration: float = 0.0):
        """
        Добавляет SDR станцию.
        
        Args:
            station_id: Идентификатор станции
            x, y, z: Координаты в метрах
            calibration: Калибровочная поправка в дБ
        """
        self.stations[station_id] = SDRStation(
            id=station_id, x=x, y=y, z=z, 
            calibration_offset_db=calibration
        )
        print(f"[Trilateration] Added station {station_id} at ({x}, {y}, {z})")
    
    def update_station_positions(self, positions: Dict[str, Tuple[float, float, float]]):
        """Обновляет позиции станций."""
        for station_id, (x, y, z) in positions.items():
            if station_id in self.stations:
                self.stations[station_id].x = x
                self.stations[station_id].y = y
                self.stations[station_id].z = z
            else:
                self.add_station(station_id, x, y, z)
    
    def calculate_position(self, rssi_measurements: Dict[str, float], 
                          peak_id: str = "", freq_mhz: float = 0.0) -> Optional[TrilaterationResult]:
        """
        Вычисляет позицию источника по измерениям RSSI.
        Реализует алгоритм из документа.
        
        Args:
            rssi_measurements: Словарь {station_id: rssi_dbm}
            peak_id: ID пика для трекинга
            freq_mhz: Частота сигнала
            
        Returns:
            Результат трилатерации или None при ошибке
        """
        # Проверяем минимум 3 измерения
        if len(rssi_measurements) < 3:
            self.trilateration_error.emit(f"Недостаточно измерений: {len(rssi_measurements)}, нужно минимум 3")
            return None
        
        # Проверяем что все станции известны
        station_ids = []
        rssi_values = []
        station_coords = []
        
        for station_id, rssi in rssi_measurements.items():
            if station_id not in self.stations:
                self.trilateration_error.emit(f"Неизвестная станция: {station_id}")
                continue
            
            station = self.stations[station_id]
            station_ids.append(station_id)
            rssi_values.append(rssi + station.calibration_offset_db)  # Применяем калибровку
            station_coords.append([station.x, station.y, station.z])
        
        if len(station_ids) < 3:
            self.trilateration_error.emit("Недостаточно известных станций")
            return None
        
        # Преобразуем в numpy массивы
        rssi = np.array(rssi_values[:3])  # Берем первые 3 для базового алгоритма
        sdr_coords = np.array(station_coords[:3])
        
        # Функция ошибки из документа
        def loss_function(pos, sdr_coords, rssi, n=None):
            """
            Функция ошибки для оптимизации.
            pos - пробуемые координаты (x, y, z)
            sdr_coords - координаты SDR [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]
            rssi - уровни сигнала [RSSI1, RSSI2, RSSI3]
            n - коэффициент затухания
            """
            if n is None:
                n = self.path_loss_exponent
            
            x, y, z = pos
            
            # Считаем расстояния от пробной точки до каждого приёмника
            d = [np.linalg.norm(np.array([x, y, z]) - np.array(sdr_coords[i])) 
                 for i in range(3)]
            
            # Проверка: если расстояние ноль (чтобы не делить на 0)
            if min(d) < 1e-6:
                return 1e6  # большая ошибка
            
            # Разности RSSI, которые реально замерены
            delta_rssi_12 = rssi[0] - rssi[1]
            delta_rssi_13 = rssi[0] - rssi[2]
            
            # Теоретические разности по модели
            model_12 = 10 * n * np.log10(d[1] / d[0])
            model_13 = 10 * n * np.log10(d[2] / d[0])
            
            # Ошибка - сумма квадратов разностей
            loss = (delta_rssi_12 - model_12)**2 + (delta_rssi_13 - model_13)**2
            return loss
        
        # Начальное приближение
        initial_guess = self._get_initial_guess(peak_id, sdr_coords)
        
        # Границы поиска
        bounds = [
            (self.search_bounds[0], self.search_bounds[1]),  # x
            (self.search_bounds[0], self.search_bounds[1]),  # y
            (0, 50)  # z (высота дрона обычно до 50м)
        ]
        
        # Оптимизация методом Nelder-Mead (как в документе)
        try:
            result = minimize(
                loss_function,
                initial_guess,
                args=(sdr_coords, rssi),
                method='Nelder-Mead',
                options={'maxiter': self.max_iterations, 'xatol': self.tolerance}
            )
            
            if not result.success:
                self.trilateration_error.emit(f"Оптимизация не сошлась: {result.message}")
                return None
            
            # Извлекаем результат
            x_opt, y_opt, z_opt = result.x
            error = result.fun
            
            # Оцениваем confidence на основе ошибки
            # Чем меньше ошибка, тем выше уверенность
            if error < 1.0:
                confidence = 0.95
            elif error < 5.0:
                confidence = 0.8
            elif error < 10.0:
                confidence = 0.6
            elif error < 20.0:
                confidence = 0.4
            else:
                confidence = 0.2
            
            # Создаем результат
            trilateration_result = TrilaterationResult(
                x=float(x_opt),
                y=float(y_opt),
                z=float(z_opt),
                error=float(error),
                confidence=float(confidence),
                timestamp=time.time(),
                peak_id=peak_id,
                freq_mhz=freq_mhz,
                rssi_measurements=rssi_measurements.copy()
            )
            
            # Сохраняем в историю для трекинга
            if peak_id:
                if peak_id not in self.target_history:
                    self.target_history[peak_id] = []
                self.target_history[peak_id].append(trilateration_result)
                
                # Ограничиваем размер истории
                if len(self.target_history[peak_id]) > self.max_history_length:
                    self.target_history[peak_id].pop(0)
            
            # Эмитим результат
            self.target_update.emit(trilateration_result)
            
            print(f"[Trilateration] Result: ({x_opt:.1f}, {y_opt:.1f}, {z_opt:.1f}) m, "
                  f"error={error:.2f}, confidence={confidence:.2f}")
            
            return trilateration_result
            
        except Exception as e:
            self.trilateration_error.emit(f"Ошибка вычисления: {e}")
            return None
    
    def _get_initial_guess(self, peak_id: str, sdr_coords: np.ndarray) -> np.ndarray:
        """
        Получает начальное приближение для оптимизации.
        Использует предыдущую позицию если есть, иначе центр между станциями.
        """
        # Если есть история - используем последнюю позицию
        if peak_id and peak_id in self.target_history and self.target_history[peak_id]:
            last = self.target_history[peak_id][-1]
            return np.array([last.x, last.y, last.z])
        
        # Иначе используем центр масс станций
        center = np.mean(sdr_coords, axis=0)
        return center
    
    def calculate_position_with_tracking(self, rssi_measurements: Dict[str, float],
                                        peak_id: str, freq_mhz: float,
                                        alpha: float = 0.7) -> Optional[TrilaterationResult]:
        """
        Вычисляет позицию с применением сглаживания для трекинга.
        
        Args:
            rssi_measurements: Измерения RSSI
            peak_id: ID цели
            freq_mhz: Частота
            alpha: Коэффициент сглаживания (0-1), больше = больше вес новых данных
        """
        # Вычисляем новую позицию
        new_result = self.calculate_position(rssi_measurements, peak_id, freq_mhz)
        
        if not new_result:
            return None
        
        # Если есть история - применяем EMA фильтр
        if peak_id in self.target_history and len(self.target_history[peak_id]) > 1:
            # Берем предпоследнюю позицию (последняя - это только что добавленная)
            prev = self.target_history[peak_id][-2]
            
            # Применяем экспоненциальное сглаживание
            new_result.x = alpha * new_result.x + (1 - alpha) * prev.x
            new_result.y = alpha * new_result.y + (1 - alpha) * prev.y
            new_result.z = alpha * new_result.z + (1 - alpha) * prev.z
            
            # Обновляем последнюю позицию в истории
            self.target_history[peak_id][-1] = new_result
        
        return new_result
    
    def get_target_trajectory(self, peak_id: str, max_points: int = 50) -> List[Tuple[float, float, float]]:
        """
        Получает траекторию цели.
        
        Returns:
            Список точек траектории [(x, y, z), ...]
        """
        if peak_id not in self.target_history:
            return []
        
        history = self.target_history[peak_id]
        points = [(r.x, r.y, r.z) for r in history[-max_points:]]
        
        return points
    
    def clear_history(self, peak_id: Optional[str] = None):
        """Очищает историю трекинга."""
        if peak_id:
            if peak_id in self.target_history:
                del self.target_history[peak_id]
        else:
            self.target_history.clear()
    
    def set_path_loss_exponent(self, n: float):
        """
        Устанавливает коэффициент потерь.
        2.0 - открытое пространство
        2.5-3.0 - пригород
        3.0-4.0 - городская застройка
        4.0-6.0 - помещения
        """
        self.path_loss_exponent = max(1.5, min(6.0, n))
        print(f"[Trilateration] Path loss exponent set to {self.path_loss_exponent}")