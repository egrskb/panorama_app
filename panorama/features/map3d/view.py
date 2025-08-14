from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl  # Для 3D если доступно
import time


@dataclass 
class Target:
    """Цель на карте."""
    id: int
    x: float
    y: float
    freq_mhz: float
    power_dbm: float
    confidence: float  # 0-1
    timestamp: float
    rssi_master: float = -100.0
    rssi_slave1: float = -100.0
    rssi_slave2: float = -100.0


@dataclass
class DetectedSignal:
    """Детектированный сигнал для трилатерации."""
    freq_mhz: float
    bandwidth_khz: float
    center_freq_mhz: float
    power_dbm: float
    timestamp: float
    duration_ms: float
    modulation_type: str = "unknown"  # FM, AM, Digital, etc
    confidence: float = 0.0


class MapView(QtWidgets.QWidget):
    """Интерактивная карта с трилатерацией."""
    
    targetDetected = QtCore.pyqtSignal(object)  # Target
    targetSelected = QtCore.pyqtSignal(object)  # Target - для демодуляции
    trilaterationStarted = QtCore.pyqtSignal()
    trilaterationStopped = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Позиции SDR
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
        
        # Настройки отображения
        self.grid_size = 50.0  # метров
        self.show_grid = True
        self.show_labels = True
        self.show_circles = True
        
        # Флаг трилатерации
        self._trilateration_active = False
        
        # Создаем UI
        self._build_ui()
        
        # Обновляем карту
        self._refresh_map()

    def _build_ui(self):
        """Создание интерфейса."""
        main_layout = QtWidgets.QHBoxLayout(self)
        
        # === Левая панель управления ===
        left_panel = QtWidgets.QVBoxLayout()
        
        # Выбор устройств
        devices_group = QtWidgets.QGroupBox("Устройства SDR")
        devices_layout = QtWidgets.QFormLayout(devices_group)
        
        self.master_combo = QtWidgets.QComboBox()
        self.master_combo.addItem("(не выбрано)")
        self.master_combo.setToolTip("Основное устройство для анализа спектра")
        
        self.slave1_combo = QtWidgets.QComboBox()
        self.slave1_combo.addItem("(не выбрано)")
        
        self.slave2_combo = QtWidgets.QComboBox()
        self.slave2_combo.addItem("(не выбрано)")
        
        self.btn_refresh_devices = QtWidgets.QPushButton("Обновить")
        self.btn_refresh_devices.clicked.connect(self._refresh_devices)
        
        devices_layout.addRow("Master SDR:", self.master_combo)
        devices_layout.addRow("Slave 1:", self.slave1_combo)
        devices_layout.addRow("Slave 2:", self.slave2_combo)
        devices_layout.addRow(self.btn_refresh_devices)
        
        left_panel.addWidget(devices_group)
        
        # Позиции SDR
        sdr_group = QtWidgets.QGroupBox("Позиции SDR")
        sdr_layout = QtWidgets.QFormLayout(sdr_group)
        
        # Master (фиксирован в 0,0)
        self.master_label = QtWidgets.QLabel("Master: (0.0, 0.0)")
        sdr_layout.addRow(self.master_label)
        
        # Slave 1
        s1_layout = QtWidgets.QHBoxLayout()
        self.s1_x = QtWidgets.QDoubleSpinBox()
        self.s1_x.setRange(-100, 100)
        self.s1_x.setValue(10.0)
        self.s1_x.setSuffix(" м")
        self.s1_y = QtWidgets.QDoubleSpinBox()
        self.s1_y.setRange(-100, 100)
        self.s1_y.setValue(0.0)
        self.s1_y.setSuffix(" м")
        s1_layout.addWidget(QtWidgets.QLabel("X:"))
        s1_layout.addWidget(self.s1_x)
        s1_layout.addWidget(QtWidgets.QLabel("Y:"))
        s1_layout.addWidget(self.s1_y)
        sdr_layout.addRow("Slave 1:", s1_layout)
        
        # Slave 2
        s2_layout = QtWidgets.QHBoxLayout()
        self.s2_x = QtWidgets.QDoubleSpinBox()
        self.s2_x.setRange(-100, 100)
        self.s2_x.setValue(0.0)
        self.s2_x.setSuffix(" м")
        self.s2_y = QtWidgets.QDoubleSpinBox()
        self.s2_y.setRange(-100, 100)
        self.s2_y.setValue(10.0)
        self.s2_y.setSuffix(" м")
        s2_layout.addWidget(QtWidgets.QLabel("X:"))
        s2_layout.addWidget(self.s2_x)
        s2_layout.addWidget(QtWidgets.QLabel("Y:"))
        s2_layout.addWidget(self.s2_y)
        sdr_layout.addRow("Slave 2:", s2_layout)
        
        left_panel.addWidget(sdr_group)
        
        # Настройки отображения
        display_group = QtWidgets.QGroupBox("Отображение")
        display_layout = QtWidgets.QVBoxLayout(display_group)
        
        self.chk_grid = QtWidgets.QCheckBox("Сетка")
        self.chk_grid.setChecked(True)
        self.chk_labels = QtWidgets.QCheckBox("Подписи")
        self.chk_labels.setChecked(True)
        self.chk_circles = QtWidgets.QCheckBox("Круги дальности")
        self.chk_circles.setChecked(True)
        self.chk_heatmap = QtWidgets.QCheckBox("Тепловая карта")
        self.chk_heatmap.setChecked(False)
        
        display_layout.addWidget(self.chk_grid)
        display_layout.addWidget(self.chk_labels)
        display_layout.addWidget(self.chk_circles)
        display_layout.addWidget(self.chk_heatmap)
        
        self.grid_spin = QtWidgets.QSpinBox()
        self.grid_spin.setRange(10, 200)
        self.grid_spin.setValue(50)
        self.grid_spin.setSuffix(" м")
        grid_row = QtWidgets.QHBoxLayout()
        grid_row.addWidget(QtWidgets.QLabel("Размер сетки:"))
        grid_row.addWidget(self.grid_spin)
        display_layout.addLayout(grid_row)
        
        left_panel.addWidget(display_group)
        
        # Список целей
        targets_group = QtWidgets.QGroupBox("Обнаруженные цели")
        targets_layout = QtWidgets.QVBoxLayout(targets_group)
        
        self.targets_table = QtWidgets.QTableWidget(0, 4)
        self.targets_table.setHorizontalHeaderLabels(["ID", "Позиция", "Частота", "Уровень"])
        self.targets_table.setMaximumHeight(200)
        self.targets_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.targets_table.itemSelectionChanged.connect(self._on_target_selected)
        targets_layout.addWidget(self.targets_table)
        
        self.btn_clear = QtWidgets.QPushButton("Очистить цели")
        self.btn_export = QtWidgets.QPushButton("Экспорт KML")
        targets_layout.addWidget(self.btn_clear)
        targets_layout.addWidget(self.btn_export)
        
        left_panel.addWidget(targets_group)
        
        # Управление трилатерацией
        control_group = QtWidgets.QGroupBox("Управление")
        control_layout = QtWidgets.QVBoxLayout(control_group)
        
        self.btn_start = QtWidgets.QPushButton("Начать трилатерацию")
        self.btn_stop = QtWidgets.QPushButton("Остановить")
        self.btn_stop.setEnabled(False)
        
        self.lbl_trilat_status = QtWidgets.QLabel("Требуется 3 SDR устройства")
        self.lbl_trilat_status.setStyleSheet("color: orange; padding: 5px;")
        
        control_layout.addWidget(self.lbl_trilat_status)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        
        left_panel.addWidget(control_group)
        left_panel.addStretch()
        
        # === Центральная карта ===
        map_container = QtWidgets.QVBoxLayout()
        
        # Информация о курсоре
        self.cursor_label = QtWidgets.QLabel("X: —, Y: —")
        self.cursor_label.setStyleSheet("font-weight: bold; padding: 5px;")
        map_container.addWidget(self.cursor_label)
        
        # 2D карта
        self.map_plot = pg.PlotWidget()
        self.map_plot.showGrid(x=True, y=True, alpha=0.3)
        self.map_plot.setLabel('left', 'Y (метры)')
        self.map_plot.setLabel('bottom', 'X (метры)')
        self.map_plot.setAspectLocked(True)
        
        # Настройка осей
        self.map_plot.setXRange(-50, 50)
        self.map_plot.setYRange(-50, 50)
        
        # Элементы карты
        self.sdr_scatter = pg.ScatterPlotItem(size=15, pen=pg.mkPen(None))
        self.target_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen('w', width=2))
        self.map_plot.addItem(self.sdr_scatter)
        self.map_plot.addItem(self.target_scatter)
        
        # Текстовые подписи для SDR
        self.sdr_labels = []
        
        # Круги дальности
        self.range_circles = []
        
        # Линии трилатерации
        self.trilat_lines = []
        
        # Тепловая карта (опционально)
        self.heatmap_img = pg.ImageItem()
        self.map_plot.addItem(self.heatmap_img)
        self.heatmap_img.hide()
        
        map_container.addWidget(self.map_plot)
        
        # === Правая панель (лог и демодуляция) ===
        right_panel = QtWidgets.QVBoxLayout()
        
        # Статистика
        stats_group = QtWidgets.QGroupBox("Статистика")
        stats_layout = QtWidgets.QFormLayout(stats_group)
        
        self.lbl_targets = QtWidgets.QLabel("0")
        self.lbl_accuracy = QtWidgets.QLabel("—")
        self.lbl_update_rate = QtWidgets.QLabel("—")
        
        stats_layout.addRow("Целей:", self.lbl_targets)
        stats_layout.addRow("Точность:", self.lbl_accuracy)
        stats_layout.addRow("Обновлений/с:", self.lbl_update_rate)
        
        right_panel.addWidget(stats_group)
        
        # Лог событий
        log_group = QtWidgets.QGroupBox("Лог событий")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        
        self.event_log = QtWidgets.QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumHeight(200)
        log_layout.addWidget(self.event_log)
        
        right_panel.addWidget(log_group)
        
        # Панель демодуляции видео
        demod_group = QtWidgets.QGroupBox("Демодуляция FPV")
        demod_layout = QtWidgets.QVBoxLayout(demod_group)
        
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumHeight(200)
        self.video_label.setStyleSheet("background-color: black; color: white; border: 1px solid #444;")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setText("Выберите цель для демодуляции")
        
        self.demod_info = QtWidgets.QLabel("Статус: Ожидание")
        self.demod_info.setStyleSheet("padding: 5px;")
        
        demod_layout.addWidget(self.video_label)
        demod_layout.addWidget(self.demod_info)
        
        right_panel.addWidget(demod_group)
        right_panel.addStretch()
        
        # === Сборка layout ===
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(300)
        
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setMaximumWidth(350)
        
        main_layout.addWidget(left_widget)
        main_layout.addLayout(map_container, stretch=1)
        main_layout.addWidget(right_widget)
        
        # Подключение сигналов
        self.s1_x.valueChanged.connect(self._on_sdr_moved)
        self.s1_y.valueChanged.connect(self._on_sdr_moved)
        self.s2_x.valueChanged.connect(self._on_sdr_moved)
        self.s2_y.valueChanged.connect(self._on_sdr_moved)
        
        self.chk_grid.toggled.connect(self._update_display)
        self.chk_labels.toggled.connect(self._update_display)
        self.chk_circles.toggled.connect(self._update_display)
        self.chk_heatmap.toggled.connect(self._update_display)
        self.grid_spin.valueChanged.connect(self._update_display)
        
        self.btn_clear.clicked.connect(self._clear_targets)
        self.btn_export.clicked.connect(self._export_kml)
        self.btn_start.clicked.connect(self._start_trilateration)
        self.btn_stop.clicked.connect(self._stop_trilateration)
        
        self.master_combo.currentTextChanged.connect(self._check_devices_ready)
        self.slave1_combo.currentTextChanged.connect(self._check_devices_ready)
        self.slave2_combo.currentTextChanged.connect(self._check_devices_ready)
        
        self.map_plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        # Убираем добавление целей по клику
        # self.map_plot.scene().sigMouseClicked.connect(self._on_map_clicked)
        
        # Обработчик клика на цель
        self.target_scatter.sigClicked.connect(self._on_target_clicked)

    def _refresh_devices(self):
        """Обновляет список доступных устройств."""
        # Здесь будет логика получения списка устройств
        # Пока заглушка
        devices = ["HackRF_001", "HackRF_002", "HackRF_003", "HackRF_004"]
        
        current_master = self.master_combo.currentText()
        current_slave1 = self.slave1_combo.currentText()
        current_slave2 = self.slave2_combo.currentText()
        
        self.master_combo.clear()
        self.slave1_combo.clear()
        self.slave2_combo.clear()
        
        self.master_combo.addItem("(не выбрано)")
        self.slave1_combo.addItem("(не выбрано)")
        self.slave2_combo.addItem("(не выбрано)")
        
        for dev in devices:
            self.master_combo.addItem(dev)
            self.slave1_combo.addItem(dev)
            self.slave2_combo.addItem(dev)
        
        # Восстанавливаем выбор
        idx = self.master_combo.findText(current_master)
        if idx >= 0:
            self.master_combo.setCurrentIndex(idx)
        idx = self.slave1_combo.findText(current_slave1)
        if idx >= 0:
            self.slave1_combo.setCurrentIndex(idx)
        idx = self.slave2_combo.findText(current_slave2)
        if idx >= 0:
            self.slave2_combo.setCurrentIndex(idx)
        
        self._log("Список устройств обновлен")

    def _check_devices_ready(self):
        """Проверяет готовность устройств для трилатерации."""
        master = self.master_combo.currentText()
        slave1 = self.slave1_combo.currentText()
        slave2 = self.slave2_combo.currentText()
        
        # Проверяем что выбраны все 3 и они разные
        if (master != "(не выбрано)" and 
            slave1 != "(не выбрано)" and 
            slave2 != "(не выбрано)" and
            len({master, slave1, slave2}) == 3):
            
            self.btn_start.setEnabled(not self._trilateration_active)
            self.lbl_trilat_status.setText("Устройства готовы")
            self.lbl_trilat_status.setStyleSheet("color: green; padding: 5px;")
            return True
        else:
            self.btn_start.setEnabled(False)
            if master == "(не выбрано)" or slave1 == "(не выбрано)" or slave2 == "(не выбрано)":
                self.lbl_trilat_status.setText("Выберите все 3 устройства")
            else:
                self.lbl_trilat_status.setText("Устройства должны быть разными")
            self.lbl_trilat_status.setStyleSheet("color: orange; padding: 5px;")
            return False

    def _on_sdr_moved(self):
        """Обновление позиций SDR."""
        self.sdr_positions['slave1'] = (self.s1_x.value(), self.s1_y.value())
        self.sdr_positions['slave2'] = (self.s2_x.value(), self.s2_y.value())
        self._refresh_map()
        self._log(f"SDR позиции обновлены")

    def _refresh_map(self):
        """Перерисовка карты."""
        # SDR позиции
        sdr_points = []
        sdr_brushes = []
        
        # Master - синий
        sdr_points.append({'pos': self.sdr_positions['master'], 'data': 'M'})
        sdr_brushes.append(pg.mkBrush(50, 100, 255, 200))
        
        # Slave 1 - зеленый
        sdr_points.append({'pos': self.sdr_positions['slave1'], 'data': 'S1'})
        sdr_brushes.append(pg.mkBrush(50, 255, 100, 200))
        
        # Slave 2 - красный
        sdr_points.append({'pos': self.sdr_positions['slave2'], 'data': 'S2'})
        sdr_brushes.append(pg.mkBrush(255, 50, 50, 200))
        
        self.sdr_scatter.setData(sdr_points, brush=sdr_brushes)
        
        # Обновляем подписи SDR
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
                target_points.append({'pos': (target.x, target.y), 'data': target})
                # Цвет по уровню уверенности
                alpha = int(100 + 155 * target.confidence)
                if target == self._selected_target:
                    # Выделенная цель - желтая с рамкой
                    target_brushes.append(pg.mkBrush(255, 255, 0, alpha))
                else:
                    target_brushes.append(pg.mkBrush(255, 200, 0, alpha))
            
            self.target_scatter.setData(target_points, brush=target_brushes)
        
        self._update_display()

    def _update_display(self):
        """Обновление элементов отображения."""
        # Сетка
        self.map_plot.showGrid(x=self.chk_grid.isChecked(), y=self.chk_grid.isChecked())
        
        # Круги дальности
        for circle in self.range_circles:
            self.map_plot.removeItem(circle)
        self.range_circles.clear()
        
        if self.chk_circles.isChecked():
            for radius in [10, 20, 30, 40, 50]:
                for sdr_name, pos in self.sdr_positions.items():
                    circle = pg.CircleROI(
                        (pos[0] - radius, pos[1] - radius),
                        size=(radius * 2, radius * 2),
                        pen=pg.mkPen((100, 100, 100, 50), width=1),
                        movable=False
                    )
                    self.map_plot.addItem(circle)
                    self.range_circles.append(circle)
        
        # Тепловая карта
        if self.chk_heatmap.isChecked():
            self._update_heatmap()
            self.heatmap_img.show()
        else:
            self.heatmap_img.hide()

    def _update_heatmap(self):
        """Создание тепловой карты вероятности."""
        # Простая тепловая карта на основе целей
        size = 100
        heatmap = np.zeros((size, size))
        
        x_range = (-50, 50)
        y_range = (-50, 50)
        
        for target in self.targets:
            # Преобразуем координаты в индексы
            xi = int((target.x - x_range[0]) / (x_range[1] - x_range[0]) * size)
            yi = int((target.y - y_range[0]) / (y_range[1] - y_range[0]) * size)
            
            if 0 <= xi < size and 0 <= yi < size:
                # Добавляем гауссово размытие вокруг цели
                sigma = 5
                for i in range(max(0, xi - 3*sigma), min(size, xi + 3*sigma)):
                    for j in range(max(0, yi - 3*sigma), min(size, yi + 3*sigma)):
                        dist = np.sqrt((i - xi)**2 + (j - yi)**2)
                        heatmap[j, i] += target.confidence * np.exp(-dist**2 / (2*sigma**2))
        
        # Нормализация
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        
        # Применяем цветовую карту
        self.heatmap_img.setImage(heatmap.T, levels=(0, 1))
        self.heatmap_img.setRect(QtCore.QRectF(x_range[0], y_range[0], 
                                                x_range[1] - x_range[0],
                                                y_range[1] - y_range[0]))
        self.heatmap_img.setOpacity(0.5)

    def _on_mouse_moved(self, pos):
        """Отслеживание курсора."""
        vb = self.map_plot.getViewBox()
        if vb:
            scene_pos = vb.mapSceneToView(pos)
            self.cursor_label.setText(f"X: {scene_pos.x():.1f} м, Y: {scene_pos.y():.1f} м")

    def _on_target_clicked(self, plot, points):
        """Обработка клика на цель."""
        if points:
            target = points[0].data()
            if isinstance(target, Target):
                self._select_target(target)

    def _on_target_selected(self):
        """Обработка выбора цели в таблице."""
        rows = self.targets_table.selectionModel().selectedRows()
        if rows:
            row = rows[0].row()
            if 0 <= row < len(self.targets):
                self._select_target(self.targets[row])

    def _select_target(self, target: Target):
        """Выбирает цель для демодуляции."""
        self._selected_target = target
        self._refresh_map()
        
        # Обновляем панель демодуляции
        self.video_label.setText(f"Демодуляция цели #{target.id}\nЧастота: {target.freq_mhz:.1f} МГц")
        self.demod_info.setText(f"Статус: Попытка демодуляции...")
        
        # Эмитим сигнал для внешней обработки
        self.targetSelected.emit(target)
        
        # Заглушка для демодуляции
        QtCore.QTimer.singleShot(2000, self._simulate_demodulation_fail)
        
        self._log(f"Выбрана цель #{target.id} для демодуляции")

    def _simulate_demodulation_fail(self):
        """Симуляция неудачной демодуляции."""
        if self._selected_target:
            self.video_label.setText(f"Не удалось демодулировать сигнал\n"
                                    f"Цель #{self._selected_target.id}\n"
                                    f"Частота: {self._selected_target.freq_mhz:.1f} МГц\n"
                                    f"Уровень: {self._selected_target.power_dbm:.1f} дБм")
            self.demod_info.setText("Статус: Демодуляция не удалась (слабый сигнал или неподдерживаемый формат)")

    def add_target_from_detection(self, detected_signal: DetectedSignal, 
                                  rssi_master: float, rssi_slave1: float, rssi_slave2: float) -> Optional[Target]:
        """
        Добавляет цель на основе детектированного сигнала и RSSI от 3 SDR.
        Выполняет трилатерацию для определения позиции.
        """
        if not self._trilateration_active:
            return None
        
        # Выполняем трилатерацию
        position = self.trilaterate(rssi_master, rssi_slave1, rssi_slave2)
        if position is None:
            self._log(f"Трилатерация не удалась для сигнала {detected_signal.freq_mhz:.1f} МГц")
            return None
        
        x, y = position
        
        # Создаем цель
        self._target_id_seq += 1
        target = Target(
            id=self._target_id_seq,
            x=x,
            y=y,
            freq_mhz=detected_signal.freq_mhz,
            power_dbm=detected_signal.power_dbm,
            confidence=detected_signal.confidence,
            timestamp=detected_signal.timestamp,
            rssi_master=rssi_master,
            rssi_slave1=rssi_slave1,
            rssi_slave2=rssi_slave2
        )
        
        self.targets.append(target)
        
        # Добавляем в таблицу
        row = self.targets_table.rowCount()
        self.targets_table.insertRow(row)
        self.targets_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(target.id)))
        self.targets_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"({target.x:.1f}, {target.y:.1f})"))
        self.targets_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{target.freq_mhz:.1f} МГц"))
        self.targets_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{target.power_dbm:.1f} дБм"))
        
        self.lbl_targets.setText(str(len(self.targets)))
        self._log(f"Цель #{target.id} обнаружена: ({x:.1f}, {y:.1f}) @ {detected_signal.freq_mhz:.1f} МГц")
        
        self._refresh_map()
        self.targetDetected.emit(target)
        
        return target

    def _clear_targets(self):
        """Очистка всех целей."""
        self.targets.clear()
        self.targets_table.setRowCount(0)
        self.lbl_targets.setText("0")
        self._selected_target = None
        self.video_label.setText("Выберите цель для демодуляции")
        self.demod_info.setText("Статус: Ожидание")
        self._refresh_map()
        self._log("Все цели удалены")

    def _export_kml(self):
        """Экспорт целей в KML для Google Earth."""
        if not self.targets:
            QtWidgets.QMessageBox.information(self, "Экспорт", "Нет целей для экспорта")
            return
        
        from PyQt5.QtCore import QDateTime
        default_name = f"targets_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.kml"
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить KML", default_name, "KML files (*.kml)"
        )
        if not path:
            return
        
        try:
            # Простой KML генератор
            kml_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>PANORAMA Targets</name>
<Style id="targetStyle">
    <IconStyle>
        <color>ff00ffff</color>
        <scale>1.0</scale>
        <Icon>
            <href>http://maps.google.com/mapfiles/kml/shapes/target.png</href>
        </Icon>
    </IconStyle>
</Style>
"""
            # Добавляем точки
            for target in self.targets:
                # Преобразуем локальные координаты в GPS (примерный расчет)
                # Предполагаем центр в Helsinki, Uusimaa, FI
                lat_base = 60.1699  # Helsinki
                lon_base = 24.9384
                
                # Примерное преобразование метров в градусы
                lat = lat_base + (target.y / 111000.0)
                lon = lon_base + (target.x / (111000.0 * np.cos(np.radians(lat_base))))
                
                kml_content += f"""
<Placemark>
    <name>Target {target.id}</name>
    <description>
        Freq: {target.freq_mhz:.1f} MHz
        Power: {target.power_dbm:.1f} dBm
        Confidence: {target.confidence:.2f}
        RSSI Master: {target.rssi_master:.1f} dBm
        RSSI Slave1: {target.rssi_slave1:.1f} dBm
        RSSI Slave2: {target.rssi_slave2:.1f} dBm
        Timestamp: {QDateTime.fromSecsSinceEpoch(int(target.timestamp)).toString()}
    </description>
    <styleUrl>#targetStyle</styleUrl>
    <Point>
        <coordinates>{lon},{lat},0</coordinates>
    </Point>
</Placemark>
"""
            
            kml_content += """
</Document>
</kml>"""
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(kml_content)
            
            QtWidgets.QMessageBox.information(self, "Экспорт", f"Сохранено {len(self.targets)} целей в KML:\n{path}")
            self._log(f"Экспортировано {len(self.targets)} целей в {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка экспорта: {e}")
            self._log(f"Ошибка экспорта KML: {e}")

    def _start_trilateration(self):
        """Запуск трилатерации."""
        if not self._check_devices_ready():
            return
        
        self._trilateration_active = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_trilat_status.setText("Трилатерация активна")
        self.lbl_trilat_status.setStyleSheet("color: green; background-color: #e0ffe0; padding: 5px;")
        
        self.master_device = self.master_combo.currentText()
        self.slave1_device = self.slave1_combo.currentText()
        self.slave2_device = self.slave2_combo.currentText()
        
        self._log(f"Трилатерация запущена")
        self._log(f"Master: {self.master_device}")
        self._log(f"Slave1: {self.slave1_device}")
        self._log(f"Slave2: {self.slave2_device}")
        
        self.trilaterationStarted.emit()
        
        # Симуляция автоматического обнаружения целей
        self._simulate_auto_detection()

    def _stop_trilateration(self):
        """Остановка трилатерации."""
        self._trilateration_active = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_trilat_status.setText("Трилатерация остановлена")
        self.lbl_trilat_status.setStyleSheet("color: orange; padding: 5px;")
        self._log("Трилатерация остановлена")
        self.trilaterationStopped.emit()

    def _simulate_auto_detection(self):
        """Симуляция автоматического обнаружения целей (заглушка)."""
        if not self._trilateration_active:
            return
        
        import random
        
        # Симулируем обнаружение сигнала
        if random.random() < 0.3:  # 30% вероятность обнаружения
            signal = DetectedSignal(
                freq_mhz=random.choice([433.0, 868.0, 915.0, 2437.0, 2450.0, 2462.0, 5800.0]),
                bandwidth_khz=random.uniform(100, 1000),
                center_freq_mhz=0,
                power_dbm=random.uniform(-80, -40),
                timestamp=time.time(),
                duration_ms=random.uniform(100, 5000),
                modulation_type=random.choice(["FM", "AM", "Digital", "FHSS", "Unknown"]),
                confidence=random.uniform(0.5, 1.0)
            )
            signal.center_freq_mhz = signal.freq_mhz
            
            # Симулируем RSSI от каждого SDR
            base_rssi = signal.power_dbm
            rssi_master = base_rssi + random.uniform(-10, 5)
            rssi_slave1 = base_rssi + random.uniform(-15, 5)
            rssi_slave2 = base_rssi + random.uniform(-15, 5)
            
            # Добавляем цель
            self.add_target_from_detection(signal, rssi_master, rssi_slave1, rssi_slave2)
        
        # Планируем следующую проверку
        if self._trilateration_active:
            QtCore.QTimer.singleShot(3000, self._simulate_auto_detection)

    def _log(self, message: str):
        """Добавление записи в лог."""
        from PyQt5.QtCore import QDateTime
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        self.event_log.append(f"[{timestamp}] {message}")
        # Автопрокрутка вниз
        scrollbar = self.event_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def trilaterate(self, rssi_master: float, rssi_slave1: float, rssi_slave2: float) -> Optional[Tuple[float, float]]:
        """
        Простая трилатерация по RSSI.
        Возвращает (x, y) позицию или None.
        """
        # Преобразуем RSSI в расстояние (упрощенная модель)
        def rssi_to_distance(rssi: float, rssi_at_1m: float = -40.0, path_loss_exp: float = 2.0) -> float:
            """Преобразование RSSI в расстояние в метрах."""
            return 10.0 ** ((rssi_at_1m - rssi) / (10.0 * path_loss_exp))
        
        d0 = rssi_to_distance(rssi_master)
        d1 = rssi_to_distance(rssi_slave1)
        d2 = rssi_to_distance(rssi_slave2)
        
        # Позиции SDR
        p0 = self.sdr_positions['master']
        p1 = self.sdr_positions['slave1']
        p2 = self.sdr_positions['slave2']
        
        # Решение системы уравнений трилатерации
        # (x - x0)² + (y - y0)² = d0²
        # (x - x1)² + (y - y1)² = d1²
        # (x - x2)² + (y - y2)² = d2²
        
        try:
            # Упрощенное решение методом наименьших квадратов
            A = 2 * np.array([
                [p1[0] - p0[0], p1[1] - p0[1]],
                [p2[0] - p0[0], p2[1] - p0[1]]
            ])
            
            b = np.array([
                d0**2 - d1**2 - p0[0]**2 + p1[0]**2 - p0[1]**2 + p1[1]**2,
                d0**2 - d2**2 - p0[0]**2 + p2[0]**2 - p0[1]**2 + p2[1]**2
            ])
            
            # Решаем Ax = b
            result = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Добавляем случайный шум для реалистичности
            import random
            x = float(result[0]) + random.uniform(-2, 2)
            y = float(result[1]) + random.uniform(-2, 2)
            
            # Ограничиваем область
            x = np.clip(x, -50, 50)
            y = np.clip(y, -50, 50)
            
            return x, y
            
        except Exception as e:
            self._log(f"Ошибка трилатерации: {e}")
            return None

    def update_from_spectrum(self, freq_mhz: float, power_dbm: float):
        """Обновление от спектра (для интеграции)."""
        # Здесь можно добавить логику обнаружения целей по спектру
        pass

    def get_devices(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Возвращает выбранные устройства."""
        master = self.master_combo.currentText()
        slave1 = self.slave1_combo.currentText()
        slave2 = self.slave2_combo.currentText()
        
        master = None if master == "(не выбрано)" else master
        slave1 = None if slave1 == "(не выбрано)" else slave1
        slave2 = None if slave2 == "(не выбрано)" else slave2
        
        return master, slave1, slave2