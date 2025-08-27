"""
Виджет карты на OpenLayers 9.x с QWebEngineView.
Современная визуализация с WebGL, тепловыми картами и трилатерацией.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import time
import numpy as np
from pathlib import Path

from PyQt5.QtCore import (
    QObject, QTimer, QUrl, pyqtSignal, pyqtSlot, 
    QThread, QMutex, QMutexLocker
)
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import PyQt5.QtCore as QtCore


@dataclass
class MapUpdate:
    """Структура обновления для карты."""
    slaves: List[Dict] = None
    targets: List[Dict] = None
    trajectories: List[Dict] = None
    heatmap_points: List[Dict] = None
    coverage: List[Dict] = Non          
    remove: List[str] = None
    
    def to_dict(self) -> Dict:
        """Конвертация в словарь с фильтрацией None."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class MapBridge(QObject):
    """Мост между Python и JavaScript через QWebChannel."""
    
    # Сигнал для отправки данных в JS
    dataUpdated = pyqtSignal(str)
    
    # Сигнал от JS при клике на объект
    featureClicked = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self._update_buffer = []
        self._mutex = QMutex()
        self._last_update = time.time()
        self._update_interval = 0.05  # 20Hz max
        
        # Таймер батчинга
        self._batch_timer = QTimer()
        self._batch_timer.timeout.connect(self._send_batched_updates)
        self._batch_timer.start(50)  # 50ms = 20Hz
    
    def send_update(self, update: MapUpdate):
        """Добавляет обновление в буфер для батчинга."""
        with QMutexLocker(self._mutex):
            self._update_buffer.append(update)
    
    def _send_batched_updates(self):
        """Отправляет накопленные обновления."""
        with QMutexLocker(self._mutex):
            if not self._update_buffer:
                return
            
            # Объединяем все обновления
            combined = MapUpdate()
            for update in self._update_buffer:
                if update.slaves:
                    if combined.slaves is None:
                        combined.slaves = []
                    combined.slaves.extend(update.slaves)
                
                if update.targets:
                    if combined.targets is None:
                        combined.targets = []
                    combined.targets.extend(update.targets)
                
                if update.trajectories:
                    if combined.trajectories is None:
                        combined.trajectories = []
                    combined.trajectories.extend(update.trajectories)
                
                if update.heatmap_points:
                    if combined.heatmap_points is None:
                        combined.heatmap_points = []
                    combined.heatmap_points.extend(update.heatmap_points)
                
                if update.coverage:
                    if combined.coverage is None:
                        combined.coverage = []
                    combined.coverage.extend(update.coverage)
                
                if update.remove:
                    if combined.remove is None:
                        combined.remove = []
                    combined.remove.extend(update.remove)
            
            # Очищаем буфер
            self._update_buffer.clear()
            
            # Отправляем объединенное обновление
            data_json = json.dumps(combined.to_dict())
            self.dataUpdated.emit(data_json)
    
    @pyqtSlot(str)
    def _on_feature_clicked(self, properties_json: str):
        """Обработка клика по объекту на карте."""
        print(f"[MapBridge] Feature clicked: {properties_json}")


class OpenLayersMapWidget(QWidget):
    """
    Виджет карты на OpenLayers с поддержкой WebGL и тепловых карт.
    """
    
    # Сигналы
    targetSelected = pyqtSignal(str)  # ID цели
    slaveSelected = pyqtSignal(str)   # ID slave
    mapReady = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Инициализация
        self._web_view = None
        self._web_channel = None
        self._bridge = MapBridge()
        self._page_loaded = False
        
        # Данные
        self._slaves_positions: Dict[str, Tuple[float, float, float]] = {}
        self._active_targets: Dict[str, Dict] = {}
        self._trajectories: Dict[str, List[Tuple[float, float]]] = {}
        
        # Параметры визуализации
        self._show_heatmap = False
        self._show_coverage = True
        self._show_trajectories = True
        self._use_webgl = True  # Использовать WebGL для больших объемов
        self._webgl_threshold = 1000  # Порог переключения на WebGL
        
        # Создаем UI
        self._setup_ui()
        
        # Подключаем сигналы
        self._bridge.featureClicked.connect(self._on_feature_clicked)
    
    def _setup_ui(self):
        """Создание интерфейса с QWebEngineView."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Создаем web view
        self._web_view = QWebEngineView()
        
        # Настройка для слабого железа/WSL
        import os
        if os.environ.get('QT_WEBENGINE_DISABLE_GPU') == '1':
            # Отключаем GPU ускорение
            from PyQt5.QtWebEngineWidgets import QWebEngineSettings
            settings = self._web_view.settings()
            settings.setAttribute(QWebEngineSettings.WebGLEnabled, False)
            settings.setAttribute(QWebEngineSettings.Accelerated2dCanvasEnabled, False)
            print("[Map] GPU acceleration disabled")
        
        # Создаем канал связи
        self._web_channel = QWebChannel()
        self._web_channel.registerObject('bridge', self._bridge)
        
        # Настраиваем страницу
        page = self._web_view.page()
        page.setWebChannel(self._web_channel)
        
        # Загружаем HTML
        html_path = Path(__file__).parent / 'openlayers_map.html'
        if html_path.exists():
            page.load(QUrl.fromLocalFile(str(html_path)))
        else:
            print(f"[Map] Warning: HTML file not found at {html_path}")
        
        # Обработчик загрузки
        page.loadFinished.connect(self._on_page_loaded)
        
        layout.addWidget(self._web_view)
    
    def _on_page_loaded(self, success: bool):
        """Обработчик завершения загрузки страницы."""
        if success:
            self._page_loaded = True
            print("[Map] OpenLayers map loaded successfully")
            self.mapReady.emit()
        else:
            print("[Map] Failed to load OpenLayers map")
    
    def _on_feature_clicked(self, properties_json: str):
        """Обработка клика по объекту на карте."""
        try:
            props = json.loads(properties_json)
            
            if 'targetId' in props:
                self.targetSelected.emit(props['targetId'])
            elif 'slaveId' in props:
                self.slaveSelected.emit(props['slaveId'])
                
        except json.JSONDecodeError:
            print(f"[Map] Error parsing feature properties: {properties_json}")
    
    # ==================== Публичные методы ====================
    
    def update_slave_positions(self, positions: Dict[str, Tuple[float, float, float]]):
        """
        Обновляет позиции Slave SDR на карте.
        
        Args:
            positions: {slave_id: (x, y, z)} в метрах
        """
        self._slaves_positions = positions.copy()
        
        slaves_data = []
        for slave_id, (x, y, z) in positions.items():
            slaves_data.append({
                'id': slave_id,
                'name': f'Slave {slave_id[-4:]}',
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'status': 'online',
                'rssi': -50.0,  # Placeholder
                'range': 50.0    # Радиус покрытия в метрах
            })
        
        update = MapUpdate(slaves=slaves_data)
        self._bridge.send_update(update)
    
    def update_stations_from_config(self, config: Dict):
        """
        Обновляет позиции станций из конфигурации (совместимость с существующим кодом).
        
        Args:
            config: Словарь конфигурации с позициями станций
        """
        try:
            if 'slaves' in config:
                positions = {}
                for slave in config['slaves']:
                    if 'nickname' in slave and 'pos' in slave:
                        slave_id = slave['nickname']
                        pos = slave['pos']
                        if len(pos) >= 2:
                            positions[slave_id] = (float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0)
                
                if positions:
                    self.update_slave_positions(positions)
                    
        except Exception as e:
            print(f"[Map] Error updating stations from config: {e}")
            print(f"[Map] Config: {config}")
    
    def add_target(self, target_id: str, x: float, y: float, 
                   freq_mhz: float = 0.0, rssi_dbm: float = -100.0,
                   confidence: float = 0.5):
        """
        Добавляет или обновляет цель на карте.
        
        Args:
            target_id: Идентификатор цели
            x, y: Координаты в метрах
            freq_mhz: Частота в МГц
            rssi_dbm: Уровень сигнала
            confidence: Уверенность в позиции (0-1)
        """
        target_data = {
            'id': target_id,
            'x': float(x),
            'y': float(y),
            'freq': float(freq_mhz),
            'rssi': float(rssi_dbm),
            'confidence': float(confidence)
        }
        
        self._active_targets[target_id] = target_data
        
        update = MapUpdate(targets=[target_data])
        self._bridge.send_update(update)
        
        # Добавляем в траекторию
        if target_id not in self._trajectories:
            self._trajectories[target_id] = []
        self._trajectories[target_id].append((x, y))
        
        # Ограничиваем длину траектории
        if len(self._trajectories[target_id]) > 100:
            self._trajectories[target_id].pop(0)
        
        # Обновляем траекторию на карте
        if self._show_trajectories:
            self._update_trajectory(target_id)
    
    def add_target_from_detector(self, target_data):
        """
        Добавляет цель из детектора (совместимость с существующим кодом).
        
        Args:
            target_data: Словарь с данными цели от детектора
        """
        try:
            # Извлекаем данные из структуры детектора
            target_id = str(target_data.get('id', target_data.get('target_id', 'unknown')))
            x = float(target_data.get('x', target_data.get('pos_x', 0.0)))
            y = float(target_data.get('y', target_data.get('pos_y', 0.0)))
            freq_mhz = float(target_data.get('freq_mhz', target_data.get('frequency', 0.0)))
            rssi_dbm = float(target_data.get('rssi_dbm', target_data.get('power_dbm', -100.0)))
            confidence = float(target_data.get('confidence', 0.5))
            
            # Используем существующий метод
            self.add_target(target_id, x, y, freq_mhz, rssi_dbm, confidence)
            
        except Exception as e:
            print(f"[Map] Error adding target from detector: {e}")
            print(f"[Map] Target data: {target_data}")
    
    def _update_trajectory(self, target_id: str):
        """Обновляет траекторию цели на карте."""
        if target_id not in self._trajectories:
            return
        
        points = self._trajectories[target_id]
        if len(points) < 2:
            return
        
        trajectory_data = {
            'id': f'traj_{target_id}',
            'targetId': target_id,
            'points': [{'x': p[0], 'y': p[1]} for p in points]
        }
        
        update = MapUpdate(trajectories=[trajectory_data])
        self._bridge.send_update(update)
    
    def update_heatmap(self, points: List[Tuple[float, float, float]]):
        """
        Обновляет тепловую карту вероятности.
        
        Args:
            points: Список (x, y, confidence) точек
        """
        if not self._show_heatmap:
            return
        
        heatmap_data = [
            {'x': float(x), 'y': float(y), 'confidence': float(conf)}
            for x, y, conf in points
        ]
        
        # Используем WebGL для больших объемов
        if len(heatmap_data) > self._webgl_threshold and self._use_webgl:
            print(f"[Map] Using WebGL for {len(heatmap_data)} points")
        
        update = MapUpdate(heatmap_points=heatmap_data)
        self._bridge.send_update(update)
    
    def update_coverage_zones(self, zones: List[Dict]):
        """
        Обновляет зоны покрытия SDR.
        
        Args:
            zones: Список зон с типом и параметрами
        """
        if not self._show_coverage:
            return
        
        update = MapUpdate(coverage=zones)
        self._bridge.send_update(update)
    
    def remove_target(self, target_id: str):
        """Удаляет цель с карты."""
        if target_id in self._active_targets:
            del self._active_targets[target_id]
        
        if target_id in self._trajectories:
            del self._trajectories[target_id]
        
        # Удаляем цель и её траекторию
        update = MapUpdate(remove=[target_id, f'traj_{target_id}'])
        self._bridge.send_update(update)
    
    def clear_all_targets(self):
        """Очищает все цели с карты."""
        target_ids = list(self._active_targets.keys())
        trajectory_ids = [f'traj_{tid}' for tid in target_ids]
        
        self._active_targets.clear()
        self._trajectories.clear()
        
        update = MapUpdate(remove=target_ids + trajectory_ids)
        self._bridge.send_update(update)
    
    def set_heatmap_visible(self, visible: bool):
        """Включает/выключает тепловую карту."""
        self._show_heatmap = visible
        if self._page_loaded:
            self._web_view.page().runJavaScript(
                f"toggleLayer('heatmap');"
            )
    
    def set_coverage_visible(self, visible: bool):
        """Включает/выключает отображение зон покрытия."""
        self._show_coverage = visible
        if self._page_loaded:
            self._web_view.page().runJavaScript(
                f"toggleLayer('coverage');"
            )
    
    def set_trajectories_visible(self, visible: bool):
        """Включает/выключает отображение траекторий."""
        self._show_trajectories = visible
        if self._page_loaded:
            self._web_view.page().runJavaScript(
                f"toggleLayer('trajectories');"
            )
    
    def center_map(self, x: float = 0.0, y: float = 0.0):
        """Центрирует карту на указанных координатах."""
        if self._page_loaded:
            self._web_view.page().runJavaScript(
                f"map.getView().animate({{center: [{x}, {y}], duration: 500}});"
            )
    
    def set_zoom(self, zoom: int):
        """Устанавливает уровень масштабирования."""
        if self._page_loaded:
            self._web_view.page().runJavaScript(
                f"map.getView().setZoom({zoom});"
            )