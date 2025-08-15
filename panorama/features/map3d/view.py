# panorama/features/map3d/view.py
"""
Карта с трилатерацией. Принимает цели ТОЛЬКО из детектора.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import time


@dataclass 
class Target:
    """Цель на карте."""
    id: int
    x: float
    y: float
    freq_mhz: float
    power_dbm: float
    confidence: float
    timestamp: float
    source: str = "detector"  # Источник: "detector" или "manual"
    rssi_master: float = -100.0
    rssi_slave1: float = -100.0
    rssi_slave2: float = -100.0
    is_tracking: bool = False
    last_update: float = 0.0
    history_x: List[float] = field(default_factory=list)
    history_y: List[float] = field(default_factory=list)


class MapView(QtWidgets.QWidget):
    """Карта с отображением целей и трилатерацией."""
    
    targetDetected = QtCore.pyqtSignal(object)
    targetSelected = QtCore.pyqtSignal(object)
    trilaterationStarted = QtCore.pyqtSignal()
    trilaterationStopped = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Позиции SDR (обновляются из DeviceManager)
        self.sdr_positions = {
            'master': (0.0, 0.0),
            'slave1': (10.0, 0.0),
            'slave2': (0.0, 10.0)
        }
        
        # SDR устройства
        self.master_device = None
        self.slave1_device = None
        self.slave2_device = None
        
        # Цели
        self.targets: List[Target] = []
        self._target_id_seq = 0
        self._selected_target: Optional[Target] = None
        self._tracking_target: Optional[Target] = None
        
        # Настройки отображения
        self.show_grid = True
        self.show_labels = True
        self.show_circles = True
        self.show_trails = True
        
        # Флаг трилатерации
        self._trilateration_active = False
        
        # Создаем UI
        self._build_ui()
        
        # Таймер обновления
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self._update_tracking)
        self.update_timer.setInterval(100)

    def _build_ui(self):
        """Создание интерфейса."""
        main_layout = QtWidgets.QHBoxLayout(self)
        
        # === Левая панель ===
        left_panel = QtWidgets.QVBoxLayout()
        
        # Информация об устройствах
        devices_group = QtWidgets.QGroupBox("Устройства SDR")
        devices_layout = QtWidgets.QVBoxLayout(devices_group)
        
        self.device_info = QtWidgets.QTextEdit()
        self.device_info.setReadOnly(True)
        self.device_info.setMaximumHeight(120)
        self.device_info.setPlainText("Устройства не настроены\n\nИсточник → Настройка SDR")
        devices_layout.addWidget(self.device_info)
        
        left_panel.addWidget(devices_group)
        
        # Позиции SDR
        positions_group = QtWidgets.QGroupBox("Позиции SDR")
        positions_layout = QtWidgets.QVBoxLayout(positions_group)
        
        self.position_info = QtWidgets.QTextEdit()
        self.position_info.setReadOnly(True)
        self.position_info.setMaximumHeight(100)
        self.position_info.setPlainText("Позиции не заданы")
        positions_layout.addWidget(self.position_info)
        
        left_panel.addWidget(positions_group)
        
        # Настройки отображения
        display_group = QtWidgets.QGroupBox("Отображение")
        display_layout = QtWidgets.QVBoxLayout(display_group)
        
        self.chk_grid = QtWidgets.QCheckBox("Сетка")
        self.chk_grid.setChecked(True)
        self.chk_labels = QtWidgets.QCheckBox("Подписи")
        self.chk_labels.setChecked(True)
        self.chk_circles = QtWidgets.QCheckBox("Круги дальности")
        self.chk_circles.setChecked(True)
        self.chk_trails = QtWidgets.QCheckBox("Траектории")
        self.chk_trails.setChecked(True)
        
        display_layout.addWidget(self.chk_grid)
        display_layout.addWidget(self.chk_labels)
        display_layout.addWidget(self.chk_circles)
        display_layout.addWidget(self.chk_trails)
        
        left_panel.addWidget(display_group)
        
        # Управление
        control_group = QtWidgets.QGroupBox("Трилатерация")
        control_layout = QtWidgets.QVBoxLayout(control_group)
        
        self.lbl_status = QtWidgets.QLabel("Требуется 3 SDR")
        self.lbl_status.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        
        self.btn_start = QtWidgets.QPushButton("Старт")
        self.btn_stop = QtWidgets.QPushButton("Стоп")
        self.btn_stop.setEnabled(False)
        
        control_layout.addWidget(self.lbl_status)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        
        left_panel.addWidget(control_group)
        left_panel.addStretch()
        
        # === Центр - Карта ===
        map_container = QtWidgets.QVBoxLayout()
        
        self.cursor_label = QtWidgets.QLabel("X: —, Y: —")
        map_container.addWidget(self.cursor_label)
        
        self.map_plot = pg.PlotWidget()
        self.map_plot.showGrid(x=True, y=True, alpha=0.3)
        self.map_plot.setLabel('left', 'Y (м)')
        self.map_plot.setLabel('bottom', 'X (м)')
        self.map_plot.setAspectLocked(True)
        self.map_plot.setXRange(-50, 50)
        self.map_plot.setYRange(-50, 50)
        
        # Элементы карты
        self.sdr_scatter = pg.ScatterPlotItem(size=15)
        self.target_scatter = pg.ScatterPlotItem(size=12)
        self.map_plot.addItem(self.sdr_scatter)
        self.map_plot.addItem(self.target_scatter)
        
        self.sdr_labels = []
        self.range_circles = []
        self.trail_lines = []
        
        map_container.addWidget(self.map_plot)
        
        # === Правая панель ===
        right_panel = QtWidgets.QVBoxLayout()
        
        # Список целей
        targets_group = QtWidgets.QGroupBox("Цели (из детектора)")
        targets_layout = QtWidgets.QVBoxLayout(targets_group)
        
        self.targets_table = QtWidgets.QTableWidget(0, 4)
        self.targets_table.setHorizontalHeaderLabels(["ID", "Позиция", "Частота", "Источник"])
        self.targets_table.setMaximumHeight(200)
        self.targets_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        targets_layout.addWidget(self.targets_table)
        
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_track = QtWidgets.QPushButton("Отследить")
        self.btn_clear = QtWidgets.QPushButton("Очистить")
        btn_layout.addWidget(self.btn_track)
        btn_layout.addWidget(self.btn_clear)
        targets_layout.addLayout(btn_layout)
        
        right_panel.addWidget(targets_group)
        
        # Статистика
        stats_group = QtWidgets.QGroupBox("Статистика")
        stats_layout = QtWidgets.QFormLayout(stats_group)
        
        self.lbl_targets = QtWidgets.QLabel("0")
        self.lbl_tracking = QtWidgets.QLabel("НЕТ")
        
        stats_layout.addRow("Целей:", self.lbl_targets)
        stats_layout.addRow("Трекинг:", self.lbl_tracking)
        
        right_panel.addWidget(stats_group)
        right_panel.addStretch()
        
        # Сборка
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(300)
        
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setMaximumWidth(350)
        
        main_layout.addWidget(left_widget)
        main_layout.addLayout(map_container, stretch=1)
        main_layout.addWidget(right_widget)
        
        # Сигналы
        self.chk_grid.toggled.connect(self._update_display)
        self.chk_labels.toggled.connect(self._update_display)
        self.chk_circles.toggled.connect(self._update_display)
        self.chk_trails.toggled.connect(self._update_display)
        
        self.btn_track.clicked.connect(self._start_tracking)
        self.btn_clear.clicked.connect(self._clear_targets)
        self.btn_start.clicked.connect(self._start_trilateration)
        self.btn_stop.clicked.connect(self._stop_trilateration)
        
        self.map_plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.targets_table.itemSelectionChanged.connect(self._on_target_selected)

    def update_devices(self, master_device, slave1_device, slave2_device):
        """Обновляет информацию об устройствах из DeviceManager."""
        self.master_device = master_device
        self.slave1_device = slave1_device
        self.slave2_device = slave2_device
        
        if master_device and slave1_device and slave2_device:
            # Обновляем позиции
            self.sdr_positions['master'] = (master_device.position_x, master_device.position_y)
            self.sdr_positions['slave1'] = (slave1_device.position_x, slave1_device.position_y)
            self.sdr_positions['slave2'] = (slave2_device.position_x, slave2_device.position_y)
            
            # Обновляем информацию
            device_text = f"Master: {master_device.nickname}\n"
            device_text += f"Slave1: {slave1_device.nickname}\n"
            device_text += f"Slave2: {slave2_device.nickname}"
            self.device_info.setPlainText(device_text)
            
            position_text = f"Master: ({master_device.position_x:.1f}, {master_device.position_y:.1f})\n"
            position_text += f"Slave1: ({slave1_device.position_x:.1f}, {slave1_device.position_y:.1f})\n"
            position_text += f"Slave2: ({slave2_device.position_x:.1f}, {slave2_device.position_y:.1f})"
            self.position_info.setPlainText(position_text)
            
            self.lbl_status.setText("Готово")
            self.lbl_status.setStyleSheet("padding: 5px; background-color: #c8e6c9;")
            self.btn_start.setEnabled(True)
        else:
            self.lbl_status.setText("Требуется настройка")
            self.lbl_status.setStyleSheet("padding: 5px; background-color: #ffccbc;")
            self.btn_start.setEnabled(False)
        
        self._refresh_map()

    def update_target_position(self, freq_mhz: float, x: float, y: float,
                               confidence: float = 0.0):
        """Обновляет позицию цели по частоте."""
        # Находим цель с близкой частотой
        for t in self.targets:
            if abs(t.freq_mhz - freq_mhz) < 0.5:
                t.x = x
                t.y = y
                t.confidence = confidence
                t.last_update = time.time()
                t.history_x.append(x)
                t.history_y.append(y)
                if len(t.history_x) > 100:
                    t.history_x.pop(0)
                    t.history_y.pop(0)
                self._update_targets_table()
                self._refresh_map()
                break

    def add_target_from_detector(self, detection):
        """Добавляет цель из детектора (только через кнопку пользователя)."""
        self._target_id_seq += 1
        target = Target(
            id=self._target_id_seq,
            x=0.0,  # Начальная позиция
            y=0.0,
            freq_mhz=detection.freq_mhz,
            power_dbm=detection.power_dbm,
            confidence=detection.confidence,
            timestamp=time.time(),
            source="detector"
        )
        
        self.targets.append(target)
        self._update_targets_table()
        self._refresh_map()
        
        return target

    def process_trilateration(self, freq_mhz: float, rssi_master: float, 
                             rssi_slave1: float, rssi_slave2: float):
        """Обрабатывает данные трилатерации для широкополосных сигналов."""
        if not self._trilateration_active:
            return
        
        # Ищем цель в диапазоне ±5 МГц (для широкополосных)
        target = None
        for t in self.targets:
            if abs(t.freq_mhz - freq_mhz) < 5.0:
                target = t
                break
        
        if not target:
            return  # Не создаем новые цели автоматически
        
        # Обновляем RSSI
        target.rssi_master = rssi_master
        target.rssi_slave1 = rssi_slave1
        target.rssi_slave2 = rssi_slave2
        target.last_update = time.time()
        
        # Вычисляем позицию
        new_pos = self._calculate_position(rssi_master, rssi_slave1, rssi_slave2)
        if new_pos:
            # Сглаживание EMA
            alpha = 0.3
            target.x = alpha * new_pos[0] + (1 - alpha) * target.x
            target.y = alpha * new_pos[1] + (1 - alpha) * target.y
            
            # История для траектории
            target.history_x.append(target.x)
            target.history_y.append(target.y)
            if len(target.history_x) > 50:
                target.history_x.pop(0)
                target.history_y.pop(0)
        
        self._refresh_map()

    def _calculate_position(self, rssi_m: float, rssi_s1: float, rssi_s2: float):
        """Трилатерация по RSSI."""
        def rssi_to_distance(rssi: float) -> float:
            # Модель затухания: d = 10^((P0 - RSSI) / (10 * n))
            P0 = -40.0  # RSSI на 1м
            n = 2.5     # Коэффициент затухания
            return 10.0 ** ((P0 - rssi) / (10.0 * n))
        
        d0 = rssi_to_distance(rssi_m)
        d1 = rssi_to_distance(rssi_s1)
        d2 = rssi_to_distance(rssi_s2)
        
        p0 = np.array(self.sdr_positions['master'])
        p1 = np.array(self.sdr_positions['slave1'])
        p2 = np.array(self.sdr_positions['slave2'])
        
        try:
            A = 2 * np.array([
                [p1[0] - p0[0], p1[1] - p0[1]],
                [p2[0] - p0[0], p2[1] - p0[1]]
            ])
            
            b = np.array([
                d0**2 - d1**2 - np.linalg.norm(p0)**2 + np.linalg.norm(p1)**2,
                d0**2 - d2**2 - np.linalg.norm(p0)**2 + np.linalg.norm(p2)**2
            ])
            
            result = np.linalg.lstsq(A, b, rcond=None)[0]
            return (np.clip(result[0], -50, 50), np.clip(result[1], -50, 50))
        except:
            return None

    def _refresh_map(self):
        """Обновление карты."""
        # SDR
        sdr_points = []
        sdr_brushes = []
        
        sdr_points.append({'pos': self.sdr_positions['master']})
        sdr_brushes.append(pg.mkBrush(50, 100, 255, 200))
        
        sdr_points.append({'pos': self.sdr_positions['slave1']})
        sdr_brushes.append(pg.mkBrush(50, 255, 100, 200))
        
        sdr_points.append({'pos': self.sdr_positions['slave2']})
        sdr_brushes.append(pg.mkBrush(255, 50, 50, 200))
        
        self.sdr_scatter.setData(sdr_points, brush=sdr_brushes)
        
        # Подписи
        for label in self.sdr_labels:
            self.map_plot.removeItem(label)
        self.sdr_labels.clear()
        
        if self.chk_labels.isChecked():
            for name, pos in [('Master', self.sdr_positions['master']),
                              ('Slave1', self.sdr_positions['slave1']),
                              ('Slave2', self.sdr_positions['slave2'])]:
                label = pg.TextItem(name, anchor=(0.5, -0.5))
                label.setPos(pos[0], pos[1])
                self.map_plot.addItem(label)
                self.sdr_labels.append(label)
        
        # Цели
        if self.targets:
            target_points = []
            target_brushes = []
            
            for target in self.targets:
                target_points.append({'pos': (target.x, target.y)})
                
                if target.is_tracking:
                    target_brushes.append(pg.mkBrush(255, 0, 0, 200))
                elif target == self._selected_target:
                    target_brushes.append(pg.mkBrush(255, 255, 0, 200))
                else:
                    target_brushes.append(pg.mkBrush(255, 150, 0, 180))
            
            self.target_scatter.setData(target_points, brush=target_brushes)
            
            # Траектории
            for line in self.trail_lines:
                self.map_plot.removeItem(line)
            self.trail_lines.clear()
            
            if self.chk_trails.isChecked():
                for target in self.targets:
                    if len(target.history_x) > 1:
                        pen = pg.mkPen(color=(255, 200, 0, 100), width=2, style=QtCore.Qt.DashLine)
                        line = self.map_plot.plot(target.history_x, target.history_y, pen=pen)
                        self.trail_lines.append(line)
        
        self._update_display()

    def _update_display(self):
        """Обновление элементов отображения."""
        self.map_plot.showGrid(x=self.chk_grid.isChecked(), y=self.chk_grid.isChecked())
        
        # Круги дальности
        for circle in self.range_circles:
            self.map_plot.removeItem(circle)
        self.range_circles.clear()
        
        if self.chk_circles.isChecked():
            for radius in [10, 20, 30]:
                for pos in self.sdr_positions.values():
                    circle = pg.CircleROI(
                        (pos[0] - radius, pos[1] - radius),
                        size=(radius * 2, radius * 2),
                        pen=pg.mkPen((100, 100, 100, 50), width=1),
                        movable=False
                    )
                    self.map_plot.addItem(circle)
                    self.range_circles.append(circle)

    def _update_targets_table(self):
        """Обновление таблицы целей."""
        self.targets_table.setRowCount(len(self.targets))
        
        for row, target in enumerate(self.targets):
            self.targets_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(target.id)))
            self.targets_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"({target.x:.1f}, {target.y:.1f})"))
            self.targets_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{target.freq_mhz:.1f} МГц"))
            self.targets_table.setItem(row, 3, QtWidgets.QTableWidgetItem(target.source))
        
        self.lbl_targets.setText(str(len(self.targets)))

    def _on_mouse_moved(self, pos):
        """Курсор."""
        vb = self.map_plot.getViewBox()
        if vb:
            scene_pos = vb.mapSceneToView(pos)
            self.cursor_label.setText(f"X: {scene_pos.x():.1f} м, Y: {scene_pos.y():.1f} м")

    def _on_target_selected(self):
        """Выбор цели."""
        rows = self.targets_table.selectionModel().selectedRows()
        if rows:
            row = rows[0].row()
            if 0 <= row < len(self.targets):
                self._selected_target = self.targets[row]
                self._refresh_map()

    def _start_tracking(self):
        """Начать трекинг выбранной цели."""
        if self._selected_target:
            self._tracking_target = self._selected_target
            self._tracking_target.is_tracking = True
            self.update_timer.start()
            self.lbl_tracking.setText(f"ID {self._tracking_target.id}")

    def _update_tracking(self):
        """Обновление трекинга."""
        if self._tracking_target:
            # Проверка таймаута
            if time.time() - self._tracking_target.last_update > 5.0:
                self._tracking_target.is_tracking = False
                self._tracking_target = None
                self.update_timer.stop()
                self.lbl_tracking.setText("ПОТЕРЯНА")

    def _clear_targets(self):
        """Очистка целей."""
        self.targets.clear()
        self.targets_table.setRowCount(0)
        self.lbl_targets.setText("0")
        self.lbl_tracking.setText("НЕТ")
        self._selected_target = None
        self._tracking_target = None
        self.update_timer.stop()
        
        for line in self.trail_lines:
            self.map_plot.removeItem(line)
        self.trail_lines.clear()
        
        self._refresh_map()

    def _start_trilateration(self):
        """Запуск трилатерации."""
        if not (self.master_device and self.slave1_device and self.slave2_device):
            QtWidgets.QMessageBox.warning(self, "Трилатерация", "Настройте устройства!")
            return
        
        self._trilateration_active = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("АКТИВНА")
        self.lbl_status.setStyleSheet("padding: 5px; background-color: #ffcdd2;")
        
        self.trilaterationStarted.emit()

    def _stop_trilateration(self):
        """Остановка трилатерации."""
        self._trilateration_active = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Готово")
        self.lbl_status.setStyleSheet("padding: 5px; background-color: #c8e6c9;")
        
        self.trilaterationStopped.emit()