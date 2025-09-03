"""
Менеджер карты трилатерации.
Модульная архитектура для управления картой и её компонентами.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
from collections import defaultdict, deque
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal, QTimer

from .config import MapConfig, StationConfig, VisualizationMode
from .maplibre_widget import MapLibreWidget


logger = logging.getLogger(__name__)


class MapDataManager(QObject):
    """Менеджер данных карты."""
    
    # Сигналы для обновлений
    drones_updated = pyqtSignal(list)
    stations_updated = pyqtSignal(list) 
    rssi_updated = pyqtSignal(list)
    trajectories_updated = pyqtSignal(list)
    
    def __init__(self, config: MapConfig):
        super().__init__()
        self.config = config
        
        # Данные
        self._drones: Dict[str, Dict] = {}
        self._stations: Dict[str, Dict] = {}
        self._rssi_measurements: List[Dict] = []
        self._trajectories: Dict[str, deque] = {}
        
        # Настройки обновлений
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._emit_updates)
        self._update_timer.start(int(1000 / self.config.rssi.update_rate_hz))
        
        # Флаги для оптимизации обновлений
        self._pending_updates = {
            'drones': False,
            'stations': False,
            'rssi': False,
            'trajectories': False
        }
    
    def add_or_update_drone(self, drone_id: str, x: float, y: float, 
                           freq_mhz: float = 0.0, rssi_dbm: float = -100.0,
                           confidence: float = 0.5, is_tracked: bool = False) -> None:
        """Добавляет или обновляет дрон."""
        drone_data = {
            'id': drone_id,
            'x': float(x),
            'y': float(y),
            'freq': float(freq_mhz),
            'rssi': float(rssi_dbm),
            'confidence': float(confidence),
            'is_tracked': is_tracked,
            'last_update': time.time()
        }
        
        self._drones[drone_id] = drone_data
        self._pending_updates['drones'] = True
        
        # Обновляем траекторию
        if drone_id not in self._trajectories:
            self._trajectories[drone_id] = deque(maxlen=self.config.trilateration.max_trajectory_length)
        self._trajectories[drone_id].append({'x': x, 'y': y, 'timestamp': time.time()})
        self._pending_updates['trajectories'] = True
        
        logger.debug(f"Drone {drone_id} updated: pos=({x:.1f},{y:.1f}), confidence={confidence:.2f}")
    
    def add_or_update_station(self, station_id: str, x: float, y: float, z: float = 0.0,
                             nickname: str = "", is_reference: bool = False, is_active: bool = True) -> None:
        """Добавляет или обновляет станцию."""
        station_data = {
            'id': station_id,
            'nickname': nickname or f"Станция {station_id}",
            'x': float(x),
            'y': float(y),
            'z': float(z),
            'is_reference': is_reference,
            'is_active': is_active,
            'last_update': time.time()
        }
        
        self._stations[station_id] = station_data
        self._pending_updates['stations'] = True
        
        # Обновляем конфигурацию
        self.config.add_station(station_id, x, y, z, nickname, is_reference)
        
        logger.debug(f"Station {station_id} updated: pos=({x:.1f},{y:.1f},{z:.1f})")
    
    def add_rssi_measurement(self, x: float, y: float, rssi_dbm: float, 
                            station_id: str = "", frequency: float = 0.0, radius: float = None) -> None:
        """Добавляет измерение RSSI."""
        measurement = {
            'x': float(x),
            'y': float(y),
            'rssi': float(rssi_dbm),
            'station_id': station_id,
            'frequency': float(frequency),
            'radius': radius or self.config.rssi.visualization_radius_m,
            'timestamp': time.time()
        }
        
        # Очищаем старые измерения
        current_time = time.time()
        max_age = self.config.rssi.max_signal_age_ms / 1000.0
        self._rssi_measurements = [
            m for m in self._rssi_measurements 
            if current_time - m['timestamp'] < max_age
        ]
        
        # Добавляем новое измерение если оно проходит фильтр
        if rssi_dbm >= self.config.rssi.threshold_dbm:
            self._rssi_measurements.append(measurement)
            self._pending_updates['rssi'] = True
    
    def remove_drone(self, drone_id: str) -> bool:
        """Удаляет дрон."""
        if drone_id in self._drones:
            del self._drones[drone_id]
            if drone_id in self._trajectories:
                del self._trajectories[drone_id]
            self._pending_updates['drones'] = True
            self._pending_updates['trajectories'] = True
            return True
        return False
    
    def clear_all_drones(self) -> None:
        """Очищает всех дронов."""
        self._drones.clear()
        self._trajectories.clear()
        self._pending_updates['drones'] = True
        self._pending_updates['trajectories'] = True
    
    def clear_rssi_measurements(self) -> None:
        """Очищает измерения RSSI."""
        self._rssi_measurements.clear()
        self._pending_updates['rssi'] = True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику данных."""
        active_drones = len(self._drones)
        high_confidence_drones = len([d for d in self._drones.values() if d['confidence'] > 0.7])
        active_stations = len([s for s in self._stations.values() if s['is_active']])
        rssi_measurements = len(self._rssi_measurements)
        
        return {
            'active_drones': active_drones,
            'high_confidence_drones': high_confidence_drones,
            'active_stations': active_stations,
            'total_stations': len(self._stations),
            'rssi_measurements': rssi_measurements,
            'trajectories': len(self._trajectories),
            'trilateration_ready': self.config.is_trilateration_ready() and active_stations >= 3,
            'last_update': time.time()
        }
    
    def _emit_updates(self) -> None:
        """Эмитирует сигналы обновлений при необходимости."""
        if self._pending_updates['drones']:
            self.drones_updated.emit(list(self._drones.values()))
            self._pending_updates['drones'] = False
        
        if self._pending_updates['stations']:
            self.stations_updated.emit(list(self._stations.values()))
            self._pending_updates['stations'] = False
        
        if self._pending_updates['rssi']:
            self.rssi_updated.emit(self._rssi_measurements.copy())
            self._pending_updates['rssi'] = False
        
        if self._pending_updates['trajectories']:
            traj_data = []
            for drone_id, points in self._trajectories.items():
                if len(points) > 1:
                    traj_data.append({
                        'id': f'traj_{drone_id}',
                        'points': list(points)
                    })
            self.trajectories_updated.emit(traj_data)
            self._pending_updates['trajectories'] = False


class MapManager(QObject):
    """Основной менеджер карты трилатерации."""
    
    # Сигналы состояния
    map_ready = pyqtSignal()
    drone_detected = pyqtSignal(str, float, float, float)  # id, x, y, confidence
    station_status_changed = pyqtSignal(str, bool)  # station_id, is_active
    
    def __init__(self, config: Optional[MapConfig] = None, parent=None):
        super().__init__(parent)
        
        self.config = config or MapConfig.create_default()
        
        # Компоненты
        self.data_manager = MapDataManager(self.config)
        self.widget: Optional[MapLibreWidget] = None
        
        # Состояние
        self._is_initialized = False
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Подключаем сигналы данных
        self._connect_data_signals()
        
        logger.info("MapManager initialized")
    
    def initialize_widget(self, parent=None) -> MapLibreWidget:
        """Инициализирует виджет карты."""
        if self.widget is None:
            self.widget = MapLibreWidget(parent)
            self.widget.mapReady.connect(self._on_map_ready)
            self.widget.droneSelected.connect(self._on_drone_selected)
            self.widget.stationSelected.connect(self._on_station_selected)
            
            # Настройки из конфигурации
            self._apply_config_to_widget()
            
        return self.widget
    
    def _connect_data_signals(self) -> None:
        """Подключает сигналы от менеджера данных."""
        self.data_manager.drones_updated.connect(self._update_widget_drones)
        self.data_manager.stations_updated.connect(self._update_widget_stations)
        self.data_manager.rssi_updated.connect(self._update_widget_rssi)
        self.data_manager.trajectories_updated.connect(self._update_widget_trajectories)
    
    def _apply_config_to_widget(self) -> None:
        """Применяет конфигурацию к виджету."""
        if not self.widget:
            return
        
        # Инициализируем станции из конфигурации
        for station in self.config.stations.values():
            self.data_manager.add_or_update_station(
                station.id, 
                station.position[0], 
                station.position[1], 
                station.position[2],
                station.nickname,
                station.is_reference,
                station.is_active
            )
        
        # RSSI настройки управляются детектором на master устройстве
        logger.debug("RSSI settings controlled by master device detector")
    
    def _update_widget_drones(self, drones: List[Dict]) -> None:
        """Обновляет дронов в виджете."""
        if self.widget:
            self.widget.updateDrones(drones)
    
    def _update_widget_stations(self, stations: List[Dict]) -> None:
        """Обновляет станции в виджете."""
        if self.widget:
            self.widget.updateStations(stations)
    
    def _update_widget_rssi(self, rssi_data: List[Dict]) -> None:
        """Обновляет RSSI в виджете."""
        if self.widget:
            self.widget.update_rssi_levels(rssi_data)
    
    def _update_widget_trajectories(self, trajectories: List[Dict]) -> None:
        """Обновляет траектории в виджете."""
        if self.widget:
            self.widget.updateTrajectories(trajectories)
    
    def _on_map_ready(self) -> None:
        """Обработка готовности карты."""
        self._is_initialized = True
        self.map_ready.emit()
        logger.info("Map is ready")
    
    def _on_drone_selected(self, drone_id: str) -> None:
        """Обработка выбора дрона."""
        self._trigger_callbacks('drone_selected', drone_id)
    
    def _on_station_selected(self, station_id: str) -> None:
        """Обработка выбора станции."""
        self._trigger_callbacks('station_selected', station_id)
    
    # Публичные методы для работы с данными
    
    def add_drone(self, drone_id: str, x: float, y: float, 
                  freq_mhz: float = 0.0, rssi_dbm: float = -100.0,
                  confidence: float = 0.5, is_tracked: bool = False) -> None:
        """Добавляет дрон на карту."""
        self.data_manager.add_or_update_drone(
            drone_id, x, y, freq_mhz, rssi_dbm, confidence, is_tracked
        )
        self.drone_detected.emit(drone_id, x, y, confidence)
    
    def add_station(self, station_id: str, x: float, y: float, z: float = 0.0,
                   nickname: str = "", is_reference: bool = False) -> None:
        """Добавляет станцию на карту."""
        self.data_manager.add_or_update_station(
            station_id, x, y, z, nickname, is_reference
        )
    
    def add_rssi_measurement(self, x: float, y: float, rssi_dbm: float,
                           station_id: str = "", frequency: float = 0.0) -> None:
        """Добавляет измерение RSSI."""
        self.data_manager.add_rssi_measurement(
            x, y, rssi_dbm, station_id, frequency
        )
    
    def update_config(self, new_config: MapConfig) -> None:
        """Обновляет конфигурацию."""
        self.config = new_config
        self.data_manager.config = new_config
        self._apply_config_to_widget()
    
    # Методы управления RSSI убраны - настройки контролируются детектором
    
    def clear_all_drones(self) -> None:
        """Очищает всех дронов."""
        self.data_manager.clear_all_drones()
    
    def center_on_origin(self) -> None:
        """Центрирует карту на начале координат."""
        if self.widget and hasattr(self.widget, 'center_on_origin'):
            self.widget.center_on_origin()
    
    def center_on_station(self, station_id: str) -> None:
        """Центрирует карту на станции."""
        if self.widget and hasattr(self.widget, 'center_on_station'):
            self.widget.center_on_station(station_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику карты."""
        return self.data_manager.get_statistics()
    
    # Система callbacks
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Регистрирует callback для события."""
        self._callbacks[event].append(callback)
    
    def unregister_callback(self, event: str, callback: Callable) -> None:
        """Удаляет callback для события."""
        if callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
    
    def _trigger_callbacks(self, event: str, *args, **kwargs) -> None:
        """Вызывает callbacks для события."""
        for callback in self._callbacks[event]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback for event '{event}': {e}")
    
    # Синхронизация с внешними данными
    
    def sync_with_slaves_data(self, slaves_data: Dict[str, Any]) -> None:
        """Синхронизирует карту с данными слейвов."""
        try:
            if 'stations' in slaves_data:
                for station_data in slaves_data['stations']:
                    self.data_manager.add_or_update_station(
                        station_data.get('id', 'unknown'),
                        station_data.get('x', 0.0),
                        station_data.get('y', 0.0),
                        station_data.get('z', 0.0),
                        station_data.get('nickname', ''),
                        station_data.get('is_reference', False),
                        station_data.get('is_active', True)
                    )
            
            if 'rssi_measurements' in slaves_data:
                for rssi_data in slaves_data['rssi_measurements']:
                    self.data_manager.add_rssi_measurement(
                        rssi_data.get('x', 0.0),
                        rssi_data.get('y', 0.0),
                        rssi_data.get('rssi', -100.0),
                        rssi_data.get('station_id', ''),
                        rssi_data.get('frequency', 0.0)
                    )
            
            logger.debug("Synchronized map with slaves data")
            
        except Exception as e:
            logger.error(f"Error syncing with slaves data: {e}")
    
    @property
    def is_ready(self) -> bool:
        """Проверяет готовность карты."""
        return self._is_initialized and self.widget is not None