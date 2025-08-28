# panorama/features/map/openlayers_widget_v10.py
"""
Виджет карты для детекции дронов (2D only).
Поддержка различных режимов отображения и координатной системы с центром в Slave0.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import json
import time
import numpy as np
from pathlib import Path
from collections import deque
import logging

from PyQt5.QtCore import (
    QObject, QTimer, QUrl, pyqtSignal, pyqtSlot, 
    QThread, QMutex, QMutexLocker
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel

logger = logging.getLogger(__name__)


@dataclass
class DroneUpdate:
    """Обновление информации о дроне."""
    id: str
    x: float
    y: float
    freq: float = 0.0
    rssi: float = -100.0
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)


class MapBridge(QObject):
    """Мост между Python и JavaScript."""
    
    dataUpdated = pyqtSignal(str)
    featureClicked = pyqtSignal(str)
    mapReady = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._update_buffer = deque(maxlen=100)
        self._mutex = QMutex()
        self._batch_timer = QTimer()
        self._batch_timer.timeout.connect(self._send_batched_updates)
        self._batch_timer.start(50)  # 20Hz
    
    def send_update(self, update_data: dict):
        """Добавляет обновление в буфер."""
        with QMutexLocker(self._mutex):
            self._update_buffer.append(update_data)
    
    def _send_batched_updates(self):
        """Отправляет накопленные обновления."""
        with QMutexLocker(self._mutex):
            if not self._update_buffer:
                return
            
            # Объединяем все обновления
            combined = {
                'drones': [],
                'stations': [],
                'trajectories': [],
                'rssiLevels': []
            }
            
            while self._update_buffer:
                update = self._update_buffer.popleft()
                for key in combined:
                    if key in update and update[key]:
                        combined[key].extend(update[key])
            
            # Дедупликация дронов по ID
            if combined['drones']:
                seen = set()
                unique_drones = []
                for drone in combined['drones']:
                    if drone['id'] not in seen:
                        seen.add(drone['id'])
                        unique_drones.append(drone)
                combined['drones'] = unique_drones
            
            # Отправляем
            self.dataUpdated.emit(json.dumps(combined))
    
    @pyqtSlot(str)
    def _on_feature_clicked(self, properties_json: str):
        """Обработка клика по объекту."""
        logger.debug(f"Feature clicked: {properties_json}")
        self.featureClicked.emit(properties_json)
    
    @pyqtSlot()
    def _on_map_ready(self):
        """Карта готова."""
        logger.info("Map is ready")
        self.mapReady.emit()


class OpenLayersMapWidget(QWidget):
    """
    Виджет 2D карты для детекции дронов.
    Slave0 находится в центре координат (0, 0).
    """
    
    droneSelected = pyqtSignal(str)
    stationSelected = pyqtSignal(str)
    mapReady = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._web_view = None
        self._web_channel = None
        self._bridge = MapBridge()
        self._page_loaded = False
        
        # Позиции станций (Slave0 всегда в центре)
        self._stations = {
            'slave0': (0.0, 0.0, 0.0),     # Центр координат
            'slave1': (200.0, 200.0, 0.0),  # Пример позиции
            'slave2': (-200.0, 200.0, 0.0)  # Пример позиции
        }
        
        # Активные дроны
        self._active_drones: Dict[str, DroneUpdate] = {}
        
        # Траектории дронов
        self._trajectories: Dict[str, deque] = {}
        self._max_trajectory_length = 100
        
        # Режим карты
        self._current_mode = 'detection'
        
        # Создаем UI
        self._setup_ui()
        
        # Подключаем сигналы
        self._bridge.featureClicked.connect(self._on_feature_clicked)
        self._bridge.mapReady.connect(lambda: self.mapReady.emit())
    
    def _setup_ui(self):
        """Создание интерфейса."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Панель управления
        control_panel = QHBoxLayout()
        
        # Выбор режима карты
        self.mode_label = QLabel("Режим:")
        control_panel.addWidget(self.mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Детекция",
            "Трекинг",
            "Тепловая карта",
            "Покрытие",
            "RSSI уровни"
        ])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        control_panel.addWidget(self.mode_combo)
        
        control_panel.addStretch()
        
        # Кнопки управления
        self.btn_center = QPushButton("Центрировать")
        self.btn_center.clicked.connect(self.center_on_slave0)
        control_panel.addWidget(self.btn_center)
        
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self.clear_all_drones)
        control_panel.addWidget(self.btn_clear)
        
        layout.addLayout(control_panel)
        
        # Web view с картой
        self._web_view = QWebEngineView()
        self._web_channel = QWebChannel()
        self._web_channel.registerObject('bridge', self._bridge)
        
        page = self._web_view.page()
        page.setWebChannel(self._web_channel)
        
        # Загружаем HTML
        html_path = Path(__file__).parent / 'openlayers_map_v10.html'
        if html_path.exists():
            page.load(QUrl.fromLocalFile(str(html_path)))
        else:
            logger.warning(f"HTML file not found at {html_path}")
        
        page.loadFinished.connect(self._on_page_loaded)
        
        layout.addWidget(self._web_view)
    
    def _on_page_loaded(self, success: bool):
        """Обработчик загрузки страницы."""
        if success:
            self._page_loaded = True
            logger.info("Drone detection map loaded successfully")
            self._initialize_stations()
        else:
            logger.error("Failed to load map")
    
    def _initialize_stations(self):
        """Инициализирует позиции станций на карте."""
        stations_data = []
        for station_id, (x, y, z) in self._stations.items():
            stations_data.append({
                'id': station_id,
                'x': float(x),
                'y': float(y),
                'z': float(z)
            })
        
        update = {'stations': stations_data}
        self._bridge.send_update(update)
    
    def _on_mode_changed(self, mode_text: str):
        """Изменение режима карты."""
        mode_map = {
            "Детекция": "detection",
            "Трекинг": "tracking",
            "Тепловая карта": "heatmap",
            "Покрытие": "coverage",
            "RSSI уровни": "rssi"
        }
        
        mode = mode_map.get(mode_text, "detection")
        self._current_mode = mode
        
        if self._page_loaded:
            js_code = f"if (window.mapAPI) {{ window.mapAPI.setMapMode('{mode}'); }}"
            self._web_view.page().runJavaScript(js_code)
    
    def _on_feature_clicked(self, properties_json: str):
        """Обработка клика по объекту."""
        try:
            props = json.loads(properties_json)
            
            if 'type' in props:
                if props['type'] == 'drone':
                    self.droneSelected.emit(props.get('id', ''))
                elif props['type'] == 'sdr_station':
                    self.stationSelected.emit(props.get('id', ''))
                    
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing feature properties: {e}")
    
    def update_stations_from_config(self, config: Dict):
        """Обновляет позиции станций из конфигурации."""
        try:
            # Slave0 всегда в центре
            self._stations['slave0'] = (0.0, 0.0, 0.0)
            
            # Обновляем позиции других slaves
            if 'slaves' in config:
                for idx, slave in enumerate(config['slaves']):
                    slave_id = slave.get('nickname', f'slave{idx+1}')
                    pos = slave.get('pos', [0.0, 0.0, 0.0])
                    if len(pos) >= 2:
                        # Относительно Slave0
                        self._stations[slave_id] = (
                            float(pos[0]),
                            float(pos[1]),
                            float(pos[2]) if len(pos) > 2 else 0.0
                        )
            
            self._initialize_stations()
            logger.info(f"Updated {len(self._stations)} station positions")
            
        except Exception as e:
            logger.error(f"Error updating stations: {e}")
    
    def add_drone(self, drone_id: str, x: float, y: float, 
                  freq_mhz: float = 0.0, rssi_dbm: float = -100.0,
                  confidence: float = 0.5):
        """Добавляет или обновляет дрон на карте."""
        drone = DroneUpdate(
            id=drone_id,
            x=float(x),
            y=float(y),
            freq=float(freq_mhz),
            rssi=float(rssi_dbm),
            confidence=float(confidence)
        )
        
        self._active_drones[drone_id] = drone
        
        # Добавляем в траекторию
        if drone_id not in self._trajectories:
            self._trajectories[drone_id] = deque(maxlen=self._max_trajectory_length)
        self._trajectories[drone_id].append((x, y))
        
        # Отправляем обновление
        update = {
            'drones': [{
                'id': drone.id,
                'x': drone.x,
                'y': drone.y,
                'freq': drone.freq,
                'rssi': drone.rssi,
                'confidence': drone.confidence
            }]
        }
        
        # Добавляем траекторию если есть история
        if len(self._trajectories[drone_id]) > 1:
            points = [{'x': p[0], 'y': p[1]} for p in self._trajectories[drone_id]]
            update['trajectories'] = [{
                'id': f'traj_{drone_id}',
                'points': points
            }]
        
        self._bridge.send_update(update)
    
    def add_target(self, target_id: str, x: float, y: float, 
                   freq_mhz: float = 0.0, rssi_dbm: float = -100.0,
                   confidence: float = 0.5, metadata: Optional[Dict] = None):
        """Псевдоним для add_drone для совместимости."""
        self.add_drone(target_id, x, y, freq_mhz, rssi_dbm, confidence)
    
    def add_target_from_detector(self, target_data: Dict):
        """Добавляет цель из детектора."""
        try:
            target_id = str(target_data.get('id', 'unknown'))
            
            # Извлекаем позицию
            if 'position' in target_data:
                x = float(target_data['position'].get('x', 0))
                y = float(target_data['position'].get('y', 0))
            else:
                x = float(target_data.get('x', 0))
                y = float(target_data.get('y', 0))
            
            freq_mhz = float(target_data.get('freq', 0))
            rssi_dbm = float(target_data.get('rssi', -100))
            confidence = float(target_data.get('confidence', 0.5))
            
            self.add_drone(target_id, x, y, freq_mhz, rssi_dbm, confidence)
            
        except Exception as e:
            logger.error(f"Error adding target from detector: {e}")
    
    def update_rssi_levels(self, measurements: List[Dict]):
        """Обновляет отображение уровней RSSI."""
        if self._current_mode != 'rssi':
            return
        
        rssi_data = []
        for m in measurements:
            rssi_data.append({
                'x': float(m.get('x', 0)),
                'y': float(m.get('y', 0)),
                'radius': float(m.get('radius', 100)),
                'rssi': float(m.get('rssi', -100))
            })
        
        update = {'rssiLevels': rssi_data}
        self._bridge.send_update(update)
    
    def remove_drone(self, drone_id: str):
        """Удаляет дрон с карты."""
        if drone_id in self._active_drones:
            del self._active_drones[drone_id]
        
        if drone_id in self._trajectories:
            del self._trajectories[drone_id]
        
        if self._page_loaded:
            js_code = f"if (window.mapAPI) {{ window.mapAPI.removeDrone('{drone_id}'); }}"
            self._web_view.page().runJavaScript(js_code)
    
    def clear_all_drones(self):
        """Очищает все дроны с карты."""
        self._active_drones.clear()
        self._trajectories.clear()
        
        if self._page_loaded:
            js_code = "if (window.mapAPI) { window.mapAPI.clearAll(); }"
            self._web_view.page().runJavaScript(js_code)
    
    def center_on_slave0(self):
        """Центрирует карту на Slave0."""
        if self._page_loaded:
            js_code = "if (window.mapAPI) { window.mapAPI.centerOnStation('slave0'); }"
            self._web_view.page().runJavaScript(js_code)
    
    def center_on_station(self, station_id: str):
        """Центрирует карту на указанной станции."""
        if self._page_loaded:
            js_code = f"if (window.mapAPI) {{ window.mapAPI.centerOnStation('{station_id}'); }}"
            self._web_view.page().runJavaScript(js_code)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику."""
        return {
            'active_drones': len(self._active_drones),
            'active_trajectories': len(self._trajectories),
            'stations': len(self._stations),
            'current_mode': self._current_mode
        }