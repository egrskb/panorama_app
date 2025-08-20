"""
Engine for RSSI-based trilateration using multiple SDR stations.
Implements Levenberg-Marquardt algorithm for position estimation.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PyQt5.QtCore import QObject, pyqtSignal
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


@dataclass
class StationPosition:
    """Позиция SDR станции."""
    id: str
    x: float      # X координата (м)
    y: float      # Y координата (м)
    z: float      # Z координата (м, опционально)
    k_cal_db: float  # Калибровочный коэффициент (дБ)


@dataclass
class TrilaterationResult:
    """Результат трилатерации."""
    center_hz: float           # Центральная частота (Гц)
    span_hz: float            # Ширина полосы (Гц)
    x: float                  # X координата (м)
    y: float                  # Y координата (м)
    z: Optional[float]        # Z координата (м)
    cov: np.ndarray           # Ковариационная матрица
    confidence: float         # Уровень доверия (0-1)
    age_ms: float            # Возраст измерения (мс)
    n_stations: int          # Количество использованных станций
    rssi_measurements: List  # RSSI измерения


class RSSITrilaterationEngine(QObject):
    """Движок трилатерации по RSSI для множественных SDR станций."""
    
    # Сигналы
    target_update = pyqtSignal(object)  # TrilaterationResult
    trilateration_error = pyqtSignal(str)  # Ошибка трилатерации
    
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.log = logger
        
        # Параметры модели распространения
        self.path_loss_exponent = 2.7  # Показатель затухания по умолчанию
        self.reference_distance = 1.0  # Опорное расстояние (м)
        self.reference_power = -30.0   # Опорная мощность на опорном расстоянии (дБм)
        
        # Станции
        self.stations: Dict[str, StationPosition] = {}
        
        # Кэш результатов
        self.results_cache: Dict[str, TrilaterationResult] = {}
        
        # Параметры алгоритма
        self.min_stations = 3          # Минимальное количество станций
        self.max_iterations = 100      # Максимальное количество итераций
        self.tolerance = 1e-6          # Точность сходимости
        
    def add_station(self, station_id: str, x: float, y: float, 
                    z: float = 0.0, k_cal_db: float = 0.0):
        """Добавляет SDR станцию."""
        self.stations[station_id] = StationPosition(
            id=station_id,
            x=x, y=y, z=z,
            k_cal_db=k_cal_db
        )
        self.log.info(f"Added station {station_id} at ({x:.1f}, {y:.1f}, {z:.1f})")
    
    def remove_station(self, station_id: str):
        """Удаляет SDR станцию."""
        if station_id in self.stations:
            del self.stations[station_id]
            self.log.info(f"Removed station {station_id}")
    
    def set_path_loss_exponent(self, n: float):
        """Устанавливает показатель затухания."""
        self.path_loss_exponent = n
        self.log.info(f"Set path loss exponent to {n}")
    
    def set_reference_parameters(self, ref_distance: float, ref_power: float):
        """Устанавливает опорные параметры модели."""
        self.reference_distance = ref_distance
        self.reference_power = ref_power
        self.log.info(f"Set reference: distance={ref_distance}m, power={ref_power}dBm")
    
    def calculate_position(self, rssi_measurements: List) -> Optional[TrilaterationResult]:
        """Вычисляет позицию источника по RSSI измерениям."""
        if len(rssi_measurements) < self.min_stations:
            self.log.warning(f"Insufficient measurements: {len(rssi_measurements)} < {self.min_stations}")
            return None
        
        try:
            # Группируем измерения по частоте и полосе
            measurements_by_band = self._group_measurements_by_band(rssi_measurements)
            
            results = []
            for band_key, measurements in measurements_by_band.items():
                if len(measurements) >= self.min_stations:
                    result = self._trilaterate_band(measurements)
                    if result:
                        results.append(result)
            
            return results[0] if results else None
            
        except Exception as e:
            error_msg = f"Error in calculate_position: {e}"
            self.log.error(error_msg)
            self.trilateration_error.emit(error_msg)
            return None
    
    def _group_measurements_by_band(self, measurements: List) -> Dict:
        """Группирует измерения по частоте и полосе."""
        grouped = {}
        
        for measurement in measurements:
            # Создаем ключ для группировки
            band_key = f"{measurement.center_hz:.0f}_{measurement.span_hz:.0f}"
            
            if band_key not in grouped:
                grouped[band_key] = []
            
            grouped[band_key].append(measurement)
        
        return grouped
    
    def _trilaterate_band(self, measurements: List) -> Optional[TrilaterationResult]:
        """Выполняет трилатерацию для одной полосы частот."""
        try:
            # Проверяем, что все измерения от разных станций
            station_ids = [m.slave_id for m in measurements]
            if len(set(station_ids)) < self.min_stations:
                self.log.warning(f"Measurements from same stations: {station_ids}")
                return None
            
            # Получаем позиции станций
            station_positions = []
            rssi_values = []
            weights = []
            
            for measurement in measurements:
                if measurement.slave_id not in self.stations:
                    self.log.warning(f"Station {measurement.slave_id} not found")
                    continue
                
                station = self.stations[measurement.slave_id]
                station_positions.append([station.x, station.y, station.z])
                rssi_values.append(measurement.band_rssi_dbm)
                
                # Вес по SNR (чем выше SNR, тем выше вес)
                weight = max(0.1, min(10.0, measurement.snr_db / 10.0))
                weights.append(weight)
            
            if len(station_positions) < self.min_stations:
                return None
            
            # Начальная оценка позиции (центр масс станций)
            initial_guess = np.mean(station_positions, axis=0)
            
            # Выполняем оптимизацию
            result = self._optimize_position(
                station_positions, rssi_values, weights, initial_guess
            )
            
            if result is None:
                return None
            
            # Создаем результат
            center_hz = measurements[0].center_hz
            span_hz = measurements[0].span_hz
            
            trilateration_result = TrilaterationResult(
                center_hz=center_hz,
                span_hz=span_hz,
                x=result[0],
                y=result[1],
                z=result[2] if len(result) > 2 else None,
                cov=self._estimate_covariance(station_positions, rssi_values, result),
                confidence=self._calculate_confidence(measurements, result),
                age_ms=(time.time() - min(m.ts for m in measurements)) * 1000,
                n_stations=len(measurements),
                rssi_measurements=measurements
            )
            
            # Кэшируем результат
            band_key = f"{center_hz:.0f}_{span_hz:.0f}"
            self.results_cache[band_key] = trilateration_result
            
            # Эмитим сигнал
            self.target_update.emit(trilateration_result)
            
            return trilateration_result
            
        except Exception as e:
            self.log.error(f"Error in _trilaterate_band: {e}")
            return None
    
    def _optimize_position(self, station_positions: List, rssi_values: List, 
                          weights: List, initial_guess: np.ndarray) -> Optional[np.ndarray]:
        """Оптимизирует позицию источника."""
        try:
            # Функция ошибки для минимизации
            def error_function(position):
                errors = []
                for i, (station_pos, rssi, weight) in enumerate(zip(station_positions, rssi_values, weights)):
                    # Вычисляем расстояние
                    distance = np.linalg.norm(np.array(position) - np.array(station_pos))
                    
                    # Предсказанный RSSI по модели
                    predicted_rssi = self._predict_rssi(distance)
                    
                    # Ошибка
                    error = (rssi - predicted_rssi) * weight
                    errors.append(error)
                
                return np.array(errors)
            
            # Запускаем оптимизацию
            result = least_squares(
                error_function,
                initial_guess,
                method='lm',  # Levenberg-Marquardt
                max_nfev=self.max_iterations,
                ftol=self.tolerance,
                xtol=self.tolerance
            )
            
            if result.success:
                return result.x
            else:
                self.log.warning(f"Optimization failed: {result.message}")
                return None
                
        except Exception as e:
            self.log.error(f"Error in _optimize_position: {e}")
            return None
    
    def _predict_rssi(self, distance: float) -> float:
        """Предсказывает RSSI по расстоянию используя модель затухания."""
        if distance <= 0:
            return self.reference_power
        
        # Модель затухания: RSSI = P_ref - 10*n*log10(d/d_ref)
        rssi = self.reference_power - 10 * self.path_loss_exponent * np.log10(distance / self.reference_distance)
        return rssi
    
    def _estimate_covariance(self, station_positions: List, rssi_values: List, 
                           estimated_position: np.ndarray) -> np.ndarray:
        """Оценивает ковариационную матрицу позиции."""
        try:
            # Простая оценка ковариации на основе расстояний до станций
            distances = [np.linalg.norm(np.array(estimated_position) - np.array(pos)) for pos in station_positions]
            
            # Среднее расстояние
            mean_distance = np.mean(distances)
            
            # Оценка ошибки позиции (пропорциональна среднему расстоянию)
            position_error = mean_distance * 0.1  # 10% от среднего расстояния
            
            # Создаем диагональную ковариационную матрицу
            if len(estimated_position) == 2:
                cov = np.array([[position_error**2, 0], [0, position_error**2]])
            else:
                cov = np.array([[position_error**2, 0, 0], 
                              [0, position_error**2, 0], 
                              [0, 0, position_error**2]])
            
            return cov
            
        except Exception as e:
            self.log.error(f"Error estimating covariance: {e}")
            # Возвращаем единичную матрицу в случае ошибки
            if len(estimated_position) == 2:
                return np.eye(2)
            else:
                return np.eye(3)
    
    def _calculate_confidence(self, measurements: List, estimated_position: np.ndarray) -> float:
        """Вычисляет уровень доверия к результату."""
        try:
            # Факторы, влияющие на доверие:
            # 1. Количество станций
            n_stations = len(measurements)
            station_confidence = min(1.0, n_stations / 6.0)  # Максимум при 6+ станциях
            
            # 2. Качество SNR
            snr_values = [m.snr_db for m in measurements]
            avg_snr = np.mean(snr_values)
            snr_confidence = min(1.0, avg_snr / 20.0)  # Максимум при SNR > 20 дБ
            
            # 3. Согласованность измерений
            rssi_values = [m.band_rssi_dbm for m in measurements]
            rssi_std = np.std(rssi_values)
            consistency_confidence = max(0.0, 1.0 - rssi_std / 20.0)  # Максимум при малом разбросе
            
            # Взвешенная сумма
            confidence = 0.4 * station_confidence + 0.4 * snr_confidence + 0.2 * consistency_confidence
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.log.error(f"Error calculating confidence: {e}")
            return 0.5  # Среднее значение по умолчанию
    
    def get_station_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Возвращает позиции всех станций."""
        return {station_id: (station.x, station.y, station.z) 
                for station_id, station in self.stations.items()}
    
    def get_latest_results(self) -> List[TrilaterationResult]:
        """Возвращает последние результаты трилатерации."""
        return list(self.results_cache.values())
    
    def clear_cache(self):
        """Очищает кэш результатов."""
        self.results_cache.clear()
        self.log.info("Trilateration cache cleared")
    
    def get_status(self) -> Dict:
        """Возвращает статус движка трилатерации."""
        return {
            'n_stations': len(self.stations),
            'n_cached_results': len(self.results_cache),
            'path_loss_exponent': self.path_loss_exponent,
            'reference_distance': self.reference_distance,
            'reference_power': self.reference_power,
            'min_stations': self.min_stations
        }