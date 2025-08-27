"""
Виджет карты на OpenLayers v10.6.0 с QWebEngineView.
Современная визуализация с WebGL, тепловыми картами и трилатерацией.
Оптимизирован для новой версии OpenLayers.
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
    QThread, QMutex, QMutexLocker, QDateTime
)
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox
import PyQt5.QtCore as QtCore

# Настройка логирования
logger = logging.getLogger(__name__)


@dataclass
class MapUpdate:
    """Структура обновления для карты."""
    slaves: Optional[List[Dict]] = None
    targets: Optional[List[Dict]] = None
    trajectories: Optional[List[Dict]] = None
    heatmap_points: Optional[List[Dict]] = None
    coverage: Optional[List[Dict]] = None
    remove: Optional[List[str]] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Конвертация в словарь с фильтрацией None."""
        result = {}
        for k, v in asdict(self).items():
            if k != 'timestamp' and v is not None:
                result[k] = v
        return result
    
    def size(self) -> int:
        """Возвращает приблизительный размер обновления."""
        total = 0
        if self.slaves:
            total += len(self.slaves)
        if self.targets:
            total += len(self.targets)
        if self.trajectories:
            total += len(self.trajectories)
        if self.heatmap_points:
            total += len(self.heatmap_points)
        if self.coverage:
            total += len(self.coverage)
        if self.remove:
            total += len(self.remove)
        return total


class MapBridge(QObject):
    """Мост между Python и JavaScript через QWebChannel с оптимизацией для v10."""
    
    # Сигнал для отправки данных в JS
    dataUpdated = pyqtSignal(str)
    
    # Сигнал от JS при клике на объект
    featureClicked = pyqtSignal(str)
    
    # Сигнал о готовности карты
    mapReady = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._update_buffer = deque(maxlen=100)  # Ограничиваем размер буфера
        self._mutex = QMutex()
        self._last_update = time.time()
        self._update_interval = 0.05  # 20Hz max
        self._batch_size_limit = 1000  # Максимальный размер батча
        
        # Таймер батчинга с адаптивной частотой
        self._batch_timer = QTimer()
        self._batch_timer.timeout.connect(self._send_batched_updates)
        self._batch_timer.start(50)  # 50ms = 20Hz
        
        # Счетчики для статистики
        self._updates_sent = 0
        self._updates_dropped = 0
    
    def send_update(self, update: MapUpdate):
        """Добавляет обновление в буфер для батчинга с проверкой размера."""
        with QMutexLocker(self._mutex):
            # Проверяем возраст обновления
            age = time.time() - update.timestamp
            if age > 1.0:  # Отбрасываем устаревшие обновления
                self._updates_dropped += 1
                return
            
            self._update_buffer.append(update)
    
    def _send_batched_updates(self):
        """Отправляет накопленные обновления с оптимизацией."""
        with QMutexLocker(self._mutex):
            if not self._update_buffer:
                return
            
            # Объединяем все обновления
            combined = MapUpdate()
            batch_size = 0
            
            while self._update_buffer and batch_size < self._batch_size_limit:
                update = self._update_buffer.popleft()
                batch_size += update.size()
                
                # Объединяем slaves
                if update.slaves:
                    if combined.slaves is None:
                        combined.slaves = []
                    combined.slaves.extend(update.slaves)
                
                # Объединяем targets
                if update.targets:
                    if combined.targets is None:
                        combined.targets = []
                    combined.targets.extend(update.targets)
                
                # Объединяем trajectories
                if update.trajectories:
                    if combined.trajectories is None:
                        combined.trajectories = []
                    combined.trajectories.extend(update.trajectories)
                
                # Объединяем heatmap_points
                if update.heatmap_points:
                    if combined.heatmap_points is None:
                        combined.heatmap_points = []
                    combined.heatmap_points.extend(update.heatmap_points)
                
                # Объединяем coverage
                if update.coverage:
                    if combined.coverage is None:
                        combined.coverage = []
                    combined.coverage.extend(update.coverage)
                
                # Объединяем remove
                if update.remove:
                    if combined.remove is None:
                        combined.remove = []
                    combined.remove.extend(update.remove)
            
            # Дедупликация
            if combined.slaves:
                seen = set()
                unique_slaves = []
                for slave in combined.slaves:
                    if slave['id'] not in seen:
                        seen.add(slave['id'])
                        unique_slaves.append(slave)
                combined.slaves = unique_slaves
            
            if combined.targets:
                seen = set()
                unique_targets = []
                for target in combined.targets:
                    if target['id'] not in seen:
                        seen.add(target['id'])
                        unique_targets.append(target)
                combined.targets = unique_targets
            
            # Отправляем объединенное обновление
            data_json = json.dumps(combined.to_dict())
            self.dataUpdated.emit(data_json)
            self._updates_sent += 1
    
    @pyqtSlot(str)
    def _on_feature_clicked(self, properties_json: str):
        """Обработка клика по объекту на карте."""
        logger.debug(f"Feature clicked: {properties_json}")
        self.featureClicked.emit(properties_json)
    
    @pyqtSlot()
    def _on_map_ready(self):
        """Обработка сигнала готовности карты от JS."""
        logger.info("Map is ready")
        self.mapReady.emit()
    
    def get_statistics(self) -> Dict[str, int]:
        """Возвращает статистику работы моста."""
        return {
            'updates_sent': self._updates_sent,
            'updates_dropped': self._updates_dropped,
            'buffer_size': len(self._update_buffer)
        }


class OpenLayersMapWidget(QWidget):
    """
    Виджет карты на OpenLayers v10 с поддержкой WebGL и тепловых карт.
    Оптимизирован для высокой производительности.
    """
    
    # Сигналы
    targetSelected = pyqtSignal(str)  # ID цели
    slaveSelected = pyqtSignal(str)   # ID slave
    mapReady = pyqtSignal()
    
    def __init__(self, parent=None, show_controls: bool = False):
        super().__init__(parent)
        
        # Инициализация
        self._web_view = None
        self._web_channel = None
        self._bridge = MapBridge()
        self._page_loaded = False
        self._show_controls = show_controls
        
        # Данные
        self._slaves_positions: Dict[str, Tuple[float, float, float]] = {}
        self._active_targets: Dict[str, Dict] = {}
        self._trajectories: Dict[str, deque] = {}  # Используем deque для эффективности
        self._max_trajectory_length = 100
        
        # Параметры визуализации
        self._show_heatmap = False
        self._show_coverage = True
        self._show_trajectories = True
        self._use_webgl = True  # Использовать WebGL для больших объемов
        self._webgl_threshold = 1000  # Порог переключения на WebGL
        self._enable_clustering = False  # Кластеризация для большого числа объектов
        
        # Счетчики производительности
        self._frame_counter = 0
        self._last_fps_time = time.time()
        
        # Создаем UI
        self._setup_ui()
        
        # Подключаем сигналы
        self._bridge.featureClicked.connect(self._on_feature_clicked)
        self._bridge.mapReady.connect(lambda: self.mapReady.emit())
        
        # Таймер для обновления FPS
        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)
    
    def _setup_ui(self):
        """Создание интерфейса с QWebEngineView и опциональными контролами."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Опциональная панель управления
        if self._show_controls:
            controls_layout = QHBoxLayout()
            
            # Чекбоксы для слоев
            self._heatmap_checkbox = QCheckBox("Тепловая карта")
            self._heatmap_checkbox.setChecked(self._show_heatmap)
            self._heatmap_checkbox.toggled.connect(self.set_heatmap_visible)
            
            self._coverage_checkbox = QCheckBox("Зоны покрытия")
            self._coverage_checkbox.setChecked(self._show_coverage)
            self._coverage_checkbox.toggled.connect(self.set_coverage_visible)
            
            self._trajectories_checkbox = QCheckBox("Траектории")
            self._trajectories_checkbox.setChecked(self._show_trajectories)
            self._trajectories_checkbox.toggled.connect(self.set_trajectories_visible)
            
            self._clustering_checkbox = QCheckBox("Кластеризация")
            self._clustering_checkbox.setChecked(self._enable_clustering)
            self._clustering_checkbox.toggled.connect(self.set_clustering_enabled)
            
            # Кнопки управления
            self._center_button = QPushButton("Центрировать")
            self._center_button.clicked.connect(self.center_map)
            
            self._clear_button = QPushButton("Очистить")
            self._clear_button.clicked.connect(self.clear_all_targets)
            
            self._export_button = QPushButton("Экспорт")
            self._export_button.clicked.connect(self.export_map)
            
            controls_layout.addWidget(self._heatmap_checkbox)
            controls_layout.addWidget(self._coverage_checkbox)
            controls_layout.addWidget(self._trajectories_checkbox)
            controls_layout.addWidget(self._clustering_checkbox)
            controls_layout.addStretch()
            controls_layout.addWidget(self._center_button)
            controls_layout.addWidget(self._clear_button)
            controls_layout.addWidget(self._export_button)
            
            layout.addLayout(controls_layout)
        
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
            self._use_webgl = False
            logger.info("GPU acceleration disabled")
        
        # Создаем канал связи
        self._web_channel = QWebChannel()
        self._web_channel.registerObject('bridge', self._bridge)
        
        # Настраиваем страницу
        page = self._web_view.page()
        page.setWebChannel(self._web_channel)
        
        # Загружаем HTML
        html_path = Path(__file__).parent / 'openlayers_map_v10.html'
        if html_path.exists():
            page.load(QUrl.fromLocalFile(str(html_path)))
        else:
            logger.warning(f"HTML file not found at {html_path}")
            # Fallback: загружаем HTML из строки
            self._load_html_from_string(page)
        
        # Обработчик загрузки
        page.loadFinished.connect(self._on_page_loaded)
        
        layout.addWidget(self._web_view)
    
    def _load_html_from_string(self, page):
        """Загружает HTML напрямую из строки если файл не найден."""
        # Здесь можно вставить HTML напрямую
        logger.info("Loading HTML from embedded string")
        # page.setHtml(html_content, QUrl("file:///"))
    
    def _on_page_loaded(self, success: bool):
        """Обработчик завершения загрузки страницы."""
        if success:
            self._page_loaded = True
            logger.info("OpenLayers v10 map loaded successfully")
            self._inject_initial_settings()
        else:
            logger.error("Failed to load OpenLayers map")
    
    def _inject_initial_settings(self):
        """Инжектирует начальные настройки в карту."""
        if not self._page_loaded:
            return
        
        # Устанавливаем начальные видимости слоев
        js_code = f"""
            if (window.mapAPI) {{
                toggleLayer('heatmap', {str(self._show_heatmap).lower()});
                toggleLayer('coverage', {str(self._show_coverage).lower()});
                toggleLayer('trajectories', {str(self._show_trajectories).lower()});
                console.log('Initial settings applied');
            }}
        """
        self._web_view.page().runJavaScript(js_code)
    
    def _on_feature_clicked(self, properties_json: str):
        """Обработка клика по объекту на карте."""
        try:
            props = json.loads(properties_json)
            
            if 'targetId' in props:
                self.targetSelected.emit(props['targetId'])
            elif 'slaveId' in props:
                self.slaveSelected.emit(props['slaveId'])
            elif 'id' in props:
                # Определяем тип по префиксу
                id_str = str(props['id'])
                if id_str.startswith('slave'):
                    self.slaveSelected.emit(id_str)
                elif id_str.startswith('target'):
                    self.targetSelected.emit(id_str)
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing feature properties: {e}")
    
    def _update_fps(self):
        """Обновляет счетчик FPS."""
        current_time = time.time()
        elapsed = current_time - self._last_fps_time
        if elapsed > 0:
            fps = self._frame_counter / elapsed
            logger.debug(f"Map FPS: {fps:.1f}")
        self._frame_counter = 0
        self._last_fps_time = current_time
    
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
                'range': 80.0    # Радиус покрытия в метрах
            })
        
        update = MapUpdate(slaves=slaves_data)
        self._bridge.send_update(update)
        self._frame_counter += 1
    
    def update_stations_from_config(self, config: Dict):
        """
        Обновляет позиции станций из конфигурации.
        
        Args:
            config: Словарь конфигурации с позициями станций
        """
        try:
            positions = {}
            
            # Поддержка разных форматов конфигурации
            if 'slaves' in config:
                for slave in config['slaves']:
                    if 'nickname' in slave and 'pos' in slave:
                        slave_id = slave['nickname']
                        pos = slave['pos']
                        if len(pos) >= 2:
                            positions[slave_id] = (
                                float(pos[0]), 
                                float(pos[1]), 
                                float(pos[2]) if len(pos) > 2 else 0.0
                            )
            elif 'stations' in config:
                for station_id, station_data in config['stations'].items():
                    if 'position' in station_data:
                        pos = station_data['position']
                        positions[station_id] = (
                            float(pos.get('x', 0)),
                            float(pos.get('y', 0)),
                            float(pos.get('z', 0))
                        )
            
            if positions:
                self.update_slave_positions(positions)
                logger.info(f"Updated {len(positions)} station positions")
                    
        except Exception as e:
            logger.error(f"Error updating stations from config: {e}")
    
    def add_target(self, target_id: str, x: float, y: float, 
                   freq_mhz: float = 0.0, rssi_dbm: float = -100.0,
                   confidence: float = 0.5, metadata: Optional[Dict] = None):
        """
        Добавляет или обновляет цель на карте.
        
        Args:
            target_id: Идентификатор цели
            x, y: Координаты в метрах
            freq_mhz: Частота в МГц
            rssi_dbm: Уровень сигнала
            confidence: Уверенность в позиции (0-1)
            metadata: Дополнительные метаданные
        """
        target_data = {
            'id': target_id,
            'x': float(x),
            'y': float(y),
            'freq': float(freq_mhz),
            'rssi': float(rssi_dbm),
            'confidence': float(confidence)
        }
        
        if metadata:
            target_data.update(metadata)
        
        self._active_targets[target_id] = target_data
        
        update = MapUpdate(targets=[target_data])
        self._bridge.send_update(update)
        
        # Добавляем в траекторию
        if target_id not in self._trajectories:
            self._trajectories[target_id] = deque(maxlen=self._max_trajectory_length)
        self._trajectories[target_id].append((x, y))
        
        # Обновляем траекторию на карте
        if self._show_trajectories and len(self._trajectories[target_id]) > 1:
            self._update_trajectory(target_id)
        
        self._frame_counter += 1
    
    def add_target_from_detector(self, target_data: Dict):
        """
        Добавляет цель из детектора с поддержкой различных форматов.
        
        Args:
            target_data: Словарь с данными цели от детектора
        """
        try:
            # Извлекаем данные из различных структур
            target_id = str(target_data.get('id', 
                           target_data.get('target_id', 
                           target_data.get('uuid', 'unknown'))))
            
            # Позиция
            if 'position' in target_data:
                x = float(target_data['position'].get('x', 0))
                y = float(target_data['position'].get('y', 0))
            else:
                x = float(target_data.get('x', target_data.get('pos_x', 0)))
                y = float(target_data.get('y', target_data.get('pos_y', 0)))
            
            # Частота
            freq_mhz = float(target_data.get('freq_mhz', 
                            target_data.get('frequency', 
                            target_data.get('freq', 0))))
            
            # RSSI
            rssi_dbm = float(target_data.get('rssi_dbm', 
                            target_data.get('power_dbm', 
                            target_data.get('rssi', -100))))
            
            # Confidence
            confidence = float(target_data.get('confidence', 
                              target_data.get('probability', 0.5)))
            
            # Метаданные
            metadata = {
                'timestamp': target_data.get('timestamp', time.time()),
                'velocity': target_data.get('velocity'),
                'heading': target_data.get('heading'),
                'source': target_data.get('source', 'detector')
            }
            
            # Используем существующий метод
            self.add_target(target_id, x, y, freq_mhz, rssi_dbm, confidence, metadata)
            
        except Exception as e:
            logger.error(f"Error adding target from detector: {e}")
            logger.debug(f"Target data: {target_data}")
    
    def _update_trajectory(self, target_id: str):
        """Обновляет траекторию цели на карте."""
        if target_id not in self._trajectories:
            return
        
        points = list(self._trajectories[target_id])
        if len(points) < 2:
            return
        
        trajectory_data = {
            'id': f'traj_{target_id}',
            'targetId': target_id,
            'points': [{'x': p[0], 'y': p[1]} for p in points]
        }
        
        update = MapUpdate(trajectories=[trajectory_data])
        self._bridge.send_update(update)
    
    def update_heatmap(self, points: List[Tuple[float, float, float]],
                       use_webgl: Optional[bool] = None):
        """
        Обновляет тепловую карту вероятности.
        
        Args:
            points: Список (x, y, confidence) точек
            use_webgl: Принудительное использование WebGL
        """
        if not self._show_heatmap:
            return
        
        heatmap_data = [
            {'x': float(x), 'y': float(y), 'confidence': float(conf)}
            for x, y, conf in points
        ]
        
        # Определяем использование WebGL
        if use_webgl is None:
            use_webgl = len(heatmap_data) > self._webgl_threshold and self._use_webgl
        
        if use_webgl:
            logger.info(f"Using WebGL for {len(heatmap_data)} heatmap points")
        
        update = MapUpdate(heatmap_points=heatmap_data)
        self._bridge.send_update(update)
        self._frame_counter += 1
    
    def update_coverage_zones(self, zones: List[Dict]):
        """
        Обновляет зоны покрытия SDR.
        
        Args:
            zones: Список зон с типом и параметрами
        """
        if not self._show_coverage:
            return
        
        # Валидация зон
        valid_zones = []
        for zone in zones:
            if 'type' in zone:
                if zone['type'] == 'circle' and all(k in zone for k in ['x', 'y', 'radius']):
                    valid_zones.append(zone)
                elif zone['type'] == 'ellipse' and all(k in zone for k in ['x', 'y', 'radiusX', 'radiusY']):
                    zone.setdefault('rotation', 0)
                    valid_zones.append(zone)
                elif zone['type'] == 'polygon' and 'points' in zone:
                    valid_zones.append(zone)
        
        if valid_zones:
            update = MapUpdate(coverage=valid_zones)
            self._bridge.send_update(update)
            self._frame_counter += 1
    
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
        
        if target_ids or trajectory_ids:
            update = MapUpdate(remove=target_ids + trajectory_ids)
            self._bridge.send_update(update)
        
        # Также очищаем через JS для надежности
        if self._page_loaded:
            self._web_view.page().runJavaScript("if (window.mapAPI) { window.mapAPI.clearTargets(); }")
    
    def set_heatmap_visible(self, visible: bool):
        """Включает/выключает тепловую карту."""
        self._show_heatmap = visible
        if self._page_loaded:
            self._web_view.page().runJavaScript(
                f"if (window.mapAPI) {{ window.mapAPI.toggleLayer('heatmap'); }}"
            )
    
    def set_coverage_visible(self, visible: bool):
        """Включает/выключает отображение зон покрытия."""
        self._show_coverage = visible
        if self._page_loaded:
            self._web_view.page().runJavaScript(
                f"if (window.mapAPI) {{ window.mapAPI.toggleLayer('coverage'); }}"
            )
    
    def set_trajectories_visible(self, visible: bool):
        """Включает/выключает отображение траекторий."""
        self._show_trajectories = visible
        if self._page_loaded:
            self._web_view.page().runJavaScript(
                f"if (window.mapAPI) {{ window.mapAPI.toggleLayer('trajectories'); }}"
            )
    
    def set_clustering_enabled(self, enabled: bool):
        """Включает/выключает кластеризацию объектов."""
        self._enable_clustering = enabled
        # TODO: Реализовать в JS
        logger.info(f"Clustering {'enabled' if enabled else 'disabled'}")
    
    def center_map(self, x: float = 0.0, y: float = 0.0):
        """Центрирует карту на указанных координатах."""
        if self._page_loaded:
            self._web_view.page().runJavaScript(
                f"if (window.mapAPI) {{ window.mapAPI.setView([{x}, {y}], 14); }}"
            )
    
    def set_zoom(self, zoom: int):
        """Устанавливает уровень масштабирования."""
        if self._page_loaded:
            self._web_view.page().runJavaScript(
                f"if (map) {{ map.getView().setZoom({zoom}); }}"
            )
    
    def export_map(self):
        """Экспортирует карту в изображение."""
        if self._page_loaded:
            self._web_view.page().runJavaScript(
                "if (window.mapAPI) { window.mapAPI.exportMap(); }"
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику работы виджета."""
        stats = self._bridge.get_statistics()
        stats.update({
            'active_targets': len(self._active_targets),
            'active_trajectories': len(self._trajectories),
            'slave_count': len(self._slaves_positions)
        })
        return stats