"""
Конфигурация для модуля карт трилатерации.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MapTheme(Enum):
    """Темы для карты."""
    DARK = "dark"
    LIGHT = "light"
    HIGH_CONTRAST = "high_contrast"


class VisualizationMode(Enum):
    """Режимы визуализации."""
    RSSI_TRILATERATION = "rssi-trilateration"
    RSSI_ZONES = "rssi-zones"
    TRACKING_ONLY = "tracking-only"


@dataclass
class RSSIConfig:
    """Конфигурация для RSSI визуализации."""
    threshold_dbm: float = -70.0
    visualization_radius_m: int = 50
    update_rate_hz: int = 20
    color_scheme: str = "traffic_light"  # "traffic_light", "heatmap", "blue_gradient"
    show_weak_signals: bool = False
    min_signal_age_ms: int = 500
    max_signal_age_ms: int = 5000


@dataclass
class TrilatationConfig:
    """Конфигурация для трилатерации."""
    min_stations: int = 3
    max_uncertainty_m: float = 100.0
    confidence_threshold: float = 0.5
    track_drones: bool = True
    show_trajectories: bool = True
    max_trajectory_length: int = 100
    trajectory_retention_sec: int = 300


@dataclass
class UIConfig:
    """Конфигурация пользовательского интерфейса."""
    theme: MapTheme = MapTheme.DARK
    show_controls: bool = True
    show_legend: bool = True
    show_status: bool = True
    show_coordinates: bool = True
    animation_enabled: bool = True
    compact_mode: bool = False


@dataclass
class StationConfig:
    """Конфигурация SDR станции."""
    id: str
    nickname: str = ""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_reference: bool = False
    is_active: bool = True
    color: Optional[str] = None


@dataclass
class MapConfig:
    """Основная конфигурация карты."""
    # Базовые настройки
    projection_code: str = "local-meters"
    extent: Tuple[float, float, float, float] = (-2000.0, -2000.0, 2000.0, 2000.0)
    initial_center: Tuple[float, float] = (0.0, 0.0)
    initial_zoom: int = 15
    min_zoom: int = 12
    max_zoom: int = 20
    
    # Конфигурации модулей
    rssi: RSSIConfig = field(default_factory=RSSIConfig)
    trilateration: TrilatationConfig = field(default_factory=TrilatationConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Станции
    stations: Dict[str, StationConfig] = field(default_factory=dict)
    
    # Режим отображения
    visualization_mode: VisualizationMode = VisualizationMode.RSSI_TRILATERATION
    
    @classmethod
    def create_default(cls) -> MapConfig:
        """Создает конфигурацию по умолчанию."""
        config = cls()
        
        # Добавляем опорную станцию по умолчанию
        config.stations["slave0"] = StationConfig(
            id="slave0",
            nickname="Опорная станция",
            position=(0.0, 0.0, 0.0),
            is_reference=True,
            color="#f59e0b"
        )
        
        return config
    
    def add_station(self, station_id: str, x: float, y: float, z: float = 0.0, 
                   nickname: str = "", is_reference: bool = False) -> None:
        """Добавляет станцию в конфигурацию."""
        self.stations[station_id] = StationConfig(
            id=station_id,
            nickname=nickname or f"Станция {station_id}",
            position=(x, y, z),
            is_reference=is_reference,
            color="#22c55e" if not is_reference else "#f59e0b"
        )
    
    def update_station_position(self, station_id: str, x: float, y: float, z: float = 0.0) -> bool:
        """Обновляет позицию станции."""
        if station_id in self.stations:
            self.stations[station_id].position = (x, y, z)
            return True
        return False
    
    def get_active_stations(self) -> List[StationConfig]:
        """Возвращает список активных станций."""
        return [station for station in self.stations.values() if station.is_active]
    
    def is_trilateration_ready(self) -> bool:
        """Проверяет готовность системы к трилатерации."""
        active_stations = len(self.get_active_stations())
        return active_stations >= self.trilateration.min_stations
    
    def to_dict(self) -> Dict:
        """Конвертирует конфигурацию в словарь."""
        return {
            'projection_code': self.projection_code,
            'extent': self.extent,
            'initial_center': self.initial_center,
            'initial_zoom': self.initial_zoom,
            'rssi': {
                'threshold_dbm': self.rssi.threshold_dbm,
                'visualization_radius_m': self.rssi.visualization_radius_m,
                'update_rate_hz': self.rssi.update_rate_hz,
                'color_scheme': self.rssi.color_scheme
            },
            'trilateration': {
                'min_stations': self.trilateration.min_stations,
                'confidence_threshold': self.trilateration.confidence_threshold,
                'track_drones': self.trilateration.track_drones
            },
            'stations': [
                {
                    'id': station.id,
                    'nickname': station.nickname,
                    'x': station.position[0],
                    'y': station.position[1],
                    'z': station.position[2],
                    'is_reference': station.is_reference,
                    'is_active': station.is_active,
                    'color': station.color
                }
                for station in self.stations.values()
            ],
            'ui': {
                'theme': self.ui.theme.value,
                'show_controls': self.ui.show_controls,
                'show_legend': self.ui.show_legend
            },
            'visualization_mode': self.visualization_mode.value
        }


def load_config_from_dict(data: Dict) -> MapConfig:
    """Загружает конфигурацию из словаря."""
    config = MapConfig()
    
    # Базовые параметры
    config.projection_code = data.get('projection_code', config.projection_code)
    config.extent = tuple(data.get('extent', config.extent))
    config.initial_center = tuple(data.get('initial_center', config.initial_center))
    config.initial_zoom = data.get('initial_zoom', config.initial_zoom)
    
    # RSSI настройки
    if 'rssi' in data:
        rssi_data = data['rssi']
        config.rssi.threshold_dbm = rssi_data.get('threshold_dbm', config.rssi.threshold_dbm)
        config.rssi.visualization_radius_m = rssi_data.get('visualization_radius_m', config.rssi.visualization_radius_m)
        config.rssi.update_rate_hz = rssi_data.get('update_rate_hz', config.rssi.update_rate_hz)
        config.rssi.color_scheme = rssi_data.get('color_scheme', config.rssi.color_scheme)
    
    # Станции
    if 'stations' in data:
        config.stations.clear()
        for station_data in data['stations']:
            station = StationConfig(
                id=station_data['id'],
                nickname=station_data.get('nickname', ''),
                position=(
                    station_data.get('x', 0.0),
                    station_data.get('y', 0.0),
                    station_data.get('z', 0.0)
                ),
                is_reference=station_data.get('is_reference', False),
                is_active=station_data.get('is_active', True),
                color=station_data.get('color')
            )
            config.stations[station.id] = station
    
    # UI настройки
    if 'ui' in data:
        ui_data = data['ui']
        config.ui.theme = MapTheme(ui_data.get('theme', config.ui.theme.value))
        config.ui.show_controls = ui_data.get('show_controls', config.ui.show_controls)
        config.ui.show_legend = ui_data.get('show_legend', config.ui.show_legend)
    
    # Режим визуализации
    config.visualization_mode = VisualizationMode(
        data.get('visualization_mode', config.visualization_mode.value)
    )
    
    return config