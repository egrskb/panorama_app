# panorama/features/detector/widget.py
from __future__ import annotations
from typing import List, Dict, Optional, Deque
from dataclasses import dataclass, field
from collections import deque
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import time
import json


@dataclass
class Detection:
    """Единичное обнаружение сигнала."""
    timestamp: float
    freq_mhz: float
    power_dbm: float
    bandwidth_khz: float
    duration_ms: float
    roi_index: int
    confidence: float = 0.0
    signal_type: str = "Unknown"  # Классификация сигнала
    sweep_count: int = 1  # Количество свипов подтверждения
    last_seen: float = 0.0
    
    
@dataclass
class ROIRegion:
    """Регион интереса для мониторинга."""
    id: int
    name: str
    start_mhz: float
    stop_mhz: float
    threshold_mode: str = "auto"  # "auto" или "manual"
    threshold_dbm: float = -80.0
    baseline_dbm: float = -110.0  # Шумовой порог
    threshold_offset: float = 15.0  # Порог = baseline + offset
    min_width_bins: int = 3
    min_sweeps: int = 3  # Минимум свипов для подтверждения
    enabled: bool = True
    detections: List[Detection] = field(default_factory=list)
    last_activity: Optional[float] = None
    history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))
    

@dataclass
class DetectorState:
    """Состояние детектора."""
    is_active: bool = False
    start_time: Optional[float] = None
    total_detections: int = 0
    confirmed_detections: int = 0
    regions: List[ROIRegion] = field(default_factory=list)
    detection_history: List[Detection] = field(default_factory=list)
    pending_detections: Dict[str, Detection] = field(default_factory=dict)  # Ключ: f"{freq_mhz}_{roi_id}"
    

class SignalClassifier:
    """Классификатор типов сигналов по частотным диапазонам."""
    
    # Базовая классификация по диапазонам (МГц)
    FREQUENCY_BANDS = [
        # FM радио
        (87.5, 108.0, "FM Radio"),
        
        # Авиация
        (108.0, 137.0, "Aviation NAV/COM"),
        (118.0, 137.0, "Air Band AM"),
        
        # Морская связь
        (156.0, 163.0, "Marine VHF"),
        
        # Любительские диапазоны
        (144.0, 148.0, "Amateur 2m"),
        (430.0, 440.0, "Amateur 70cm"),
        
        # PMR/FRS/GMRS
        (446.0, 446.2, "PMR446"),
        (462.5, 467.7, "FRS/GMRS"),
        
        # ISM диапазоны
        (433.05, 434.79, "ISM 433MHz"),
        (863.0, 870.0, "ISM 868MHz"),
        (902.0, 928.0, "ISM 915MHz"),
        
        # Сотовая связь
        (880.0, 960.0, "GSM 900"),
        (1710.0, 1880.0, "GSM 1800"),
        (1920.0, 2170.0, "UMTS/3G"),
        (703.0, 803.0, "LTE 700"),
        (2500.0, 2690.0, "LTE 2600"),
        
        # Wi-Fi / Bluetooth
        (2400.0, 2483.5, "2.4GHz ISM (WiFi/BT/ZigBee)"),
        (5150.0, 5350.0, "WiFi 5GHz UNII-1/2"),
        (5470.0, 5895.0, "WiFi 5GHz UNII-2/3"),
        
        # FPV / Видео
        (5650.0, 5950.0, "5.8GHz FPV"),
        (1200.0, 1300.0, "1.2GHz Video"),
        (2300.0, 2450.0, "2.4GHz Video"),
        
        # Спутниковая связь
        (137.0, 138.0, "Weather Satellite"),
        (1525.0, 1559.0, "Inmarsat Down"),
        (1626.5, 1660.5, "Inmarsat Up"),
        
        # GPS/GNSS
        (1559.0, 1610.0, "GPS/GNSS L1"),
        (1215.0, 1240.0, "GPS/GNSS L2"),
        
        # Радары
        (2700.0, 2900.0, "S-Band Radar"),
        (5250.0, 5850.0, "C-Band Radar"),
        (8500.0, 10550.0, "X-Band Radar"),
    ]
    
    @classmethod
    def classify(cls, freq_mhz: float, bandwidth_khz: float = 0) -> str:
        """Классифицирует сигнал по частоте и ширине полосы."""
        
        # Сначала проверяем точные диапазоны
        for start, end, name in cls.FREQUENCY_BANDS:
            if start <= freq_mhz <= end:
                # Уточняем по ширине полосы если известна
                if bandwidth_khz > 0:
                    if "WiFi" in name and bandwidth_khz > 15000:
                        return f"{name} (Wide)"
                    elif "GSM" in name and 180 < bandwidth_khz < 220:
                        return f"{name} (Channel)"
                    elif bandwidth_khz < 25 and "FM" not in name:
                        return f"{name} (Narrow)"
                return name
        
        # Общая классификация если не попали в известные диапазоны
        if freq_mhz < 30:
            return "HF Band"
        elif freq_mhz < 300:
            return "VHF Band"
        elif freq_mhz < 3000:
            return "UHF Band"
        elif freq_mhz < 30000:
            return "SHF Band"
        else:
            return "EHF Band"


class DetectorWidget(QtWidgets.QWidget):
    """Виджет детектора активности с ROI и визуализацией."""
    
    rangeSelected = QtCore.pyqtSignal(float, float)
    detectionStarted = QtCore.pyqtSignal()
    detectionStopped = QtCore.pyqtSignal()
    signalDetected = QtCore.pyqtSignal(object)  # Detection
    sendToMap = QtCore.pyqtSignal(object)  # Detection для отправки на карту
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._state = DetectorState()
        self._roi_id_seq = 0
        self._max_history = 5000
        self._classifier = SignalClassifier()
        
        self._build_ui()

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        
        # === Левая панель (настройки и управление) ===
        left_panel = QtWidgets.QVBoxLayout()
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(400)
        
        # Ручной ввод диапазона
        grp_manual = QtWidgets.QGroupBox("Ручной ввод диапазона")
        manual_layout = QtWidgets.QHBoxLayout(grp_manual)
        
        self.manual_start = QtWidgets.QDoubleSpinBox()
        self.manual_start.setRange(0, 7000)
        self.manual_start.setDecimals(3)
        self.manual_start.setValue(433.0)
        self.manual_start.setSuffix(" МГц")
        
        self.manual_stop = QtWidgets.QDoubleSpinBox()
        self.manual_stop.setRange(0, 7000)
        self.manual_stop.setDecimals(3)
        self.manual_stop.setValue(435.0)
        self.manual_stop.setSuffix(" МГц")
        
        self.manual_name = QtWidgets.QLineEdit()
        self.manual_name.setPlaceholderText("Название диапазона")
        
        self.btn_add_manual = QtWidgets.QPushButton("+ Добавить")
        self.btn_add_manual.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 6px;
            }
        """)
        
        manual_layout.addWidget(QtWidgets.QLabel("От:"))
        manual_layout.addWidget(self.manual_start)
        manual_layout.addWidget(QtWidgets.QLabel("До:"))
        manual_layout.addWidget(self.manual_stop)
        manual_layout.addWidget(self.manual_name)
        manual_layout.addWidget(self.btn_add_manual)
        
        left_panel.addWidget(grp_manual)
        
        # Пресеты диапазонов
        grp_presets = QtWidgets.QGroupBox("Быстрые пресеты ROI")
        preset_layout = QtWidgets.QGridLayout(grp_presets)
        
        presets = [
            ("FM Radio", (87.5, 108.0), "#FF6B6B"),
            ("Air Band", (118.0, 137.0), "#4ECDC4"),
            ("2m Ham", (144.0, 148.0), "#45B7D1"),
            ("Marine", (156.0, 163.0), "#96CEB4"),
            ("70cm Ham", (430.0, 440.0), "#DDA0DD"),
            ("PMR446", (446.0, 446.2), "#F4A460"),
            ("ISM 433", (433.0, 435.0), "#87CEEB"),
            ("ISM 868", (863.0, 873.0), "#98D8C8"),
            ("GSM 900", (890.0, 960.0), "#FFB6C1"),
            ("GSM 1800", (1710.0, 1880.0), "#FFA07A"),
            ("WiFi 2.4", (2400.0, 2483.5), "#20B2AA"),
            ("WiFi 5GHz", (5170.0, 5895.0), "#9370DB"),
            ("FPV 5.8", (5725.0, 5875.0), "#FF69B4"),
            ("GPS L1", (1559.0, 1610.0), "#00CED1"),
            ("Weather Sat", (137.0, 138.0), "#FFD700"),
        ]
        
        for idx, (name, (start, stop), color) in enumerate(presets):
            btn = QtWidgets.QPushButton(name)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: {color}CC;
                }}
                QPushButton:pressed {{
                    background-color: {color}99;
                }}
            """)
            btn.clicked.connect(lambda _, s=start, e=stop, n=name: self._add_roi(s, e, n))
            preset_layout.addWidget(btn, idx // 3, idx % 3)
        
        left_panel.addWidget(grp_presets)
        
        # Параметры детектора
        grp_params = QtWidgets.QGroupBox("Параметры детектора")
        param_form = QtWidgets.QFormLayout(grp_params)
        
        # Режим порога
        self.threshold_mode = QtWidgets.QComboBox()
        self.threshold_mode.addItems(["Авто (baseline + N)", "Ручной порог"])
        self.threshold_mode.currentTextChanged.connect(self._on_threshold_mode_changed)
        
        # Offset для авто-порога (по умолчанию +20 дБ)
        self.threshold_offset = QtWidgets.QDoubleSpinBox()
        self.threshold_offset.setRange(3, 50)
        self.threshold_offset.setValue(20)  # Изменено с 15 на 20
        self.threshold_offset.setSuffix(" дБ над шумом")
        self.threshold_offset.setToolTip("Порог = baseline + это значение")
        
        # Ручной порог
        self.manual_threshold = QtWidgets.QDoubleSpinBox()
        self.manual_threshold.setRange(-160, 30)
        self.manual_threshold.setValue(-70)
        self.manual_threshold.setSuffix(" дБм")
        self.manual_threshold.setEnabled(False)
        
        # Минимальная ширина сигнала
        self.min_width = QtWidgets.QSpinBox()
        self.min_width.setRange(1, 100)
        self.min_width.setValue(3)
        self.min_width.setSuffix(" бинов")
        
        # Устойчивость сигнала
        self.min_sweeps = QtWidgets.QSpinBox()
        self.min_sweeps.setRange(1, 10)
        self.min_sweeps.setValue(3)
        self.min_sweeps.setSuffix(" свипов")
        self.min_sweeps.setToolTip("Сигнал должен присутствовать минимум N свипов подряд")
        
        # Тайм-аут пропадания
        self.signal_timeout = QtWidgets.QDoubleSpinBox()
        self.signal_timeout.setRange(0.1, 10.0)
        self.signal_timeout.setValue(2.0)
        self.signal_timeout.setSuffix(" сек")
        self.signal_timeout.setToolTip("Время после которого сигнал считается пропавшим")
        
        param_form.addRow("Режим порога:", self.threshold_mode)
        param_form.addRow("Авто-порог:", self.threshold_offset)
        param_form.addRow("Ручной порог:", self.manual_threshold)
        param_form.addRow("Мин. ширина:", self.min_width)
        param_form.addRow("Устойчивость:", self.min_sweeps)
        param_form.addRow("Тайм-аут:", self.signal_timeout)
        
        left_panel.addWidget(grp_params)
        
        # Управление
        grp_control = QtWidgets.QGroupBox("Управление")
        control_layout = QtWidgets.QVBoxLayout(grp_control)
        
        self.btn_start = QtWidgets.QPushButton("▶ Начать детекцию")
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
        """)
        
        self.btn_stop = QtWidgets.QPushButton("⬛ Остановить")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        
        left_panel.addWidget(grp_control)
        left_panel.addStretch()
        
        # === Центральная панель (ROI и детекции) ===
        center_panel = QtWidgets.QVBoxLayout()
        
        # Таблица ROI
        grp_roi = QtWidgets.QGroupBox("Диапазоны мониторинга (ROI)")
        roi_layout = QtWidgets.QVBoxLayout(grp_roi)
        
        self.tbl_roi = QtWidgets.QTableWidget(0, 6)
        self.tbl_roi.setHorizontalHeaderLabels(["✓", "Название", "Начало МГц", "Конец МГц", "Порог дБм", "Активность"])
        self.tbl_roi.horizontalHeader().setStretchLastSection(True)
        self.tbl_roi.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        roi_layout.addWidget(self.tbl_roi)
        
        roi_buttons = QtWidgets.QHBoxLayout()
        self.btn_add_current = QtWidgets.QPushButton("➕ Добавить текущий")
        self.btn_delete_roi = QtWidgets.QPushButton("➖ Удалить")
        self.btn_clear_roi = QtWidgets.QPushButton("🗑 Очистить все")
        roi_buttons.addWidget(self.btn_add_current)
        roi_buttons.addWidget(self.btn_delete_roi)
        roi_buttons.addWidget(self.btn_clear_roi)
        roi_layout.addLayout(roi_buttons)
        
        center_panel.addWidget(grp_roi, stretch=1)
        
        # Таблица обнаружений
        grp_detections = QtWidgets.QGroupBox("Обнаруженные сигналы")
        det_layout = QtWidgets.QVBoxLayout(grp_detections)
        
        self.tbl_detections = QtWidgets.QTableWidget(0, 8)
        self.tbl_detections.setHorizontalHeaderLabels([
            "Время", "ROI", "Частота", "Уровень", "Ширина", 
            "Свипов", "Тип сигнала", "Действия"
        ])
        self.tbl_detections.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        det_layout.addWidget(self.tbl_detections)
        
        det_buttons = QtWidgets.QHBoxLayout()
        self.btn_send_to_map = QtWidgets.QPushButton("📍 На карту")
        self.btn_export_log = QtWidgets.QPushButton("💾 Экспорт")
        self.btn_clear_detections = QtWidgets.QPushButton("🗑 Очистить")
        det_buttons.addWidget(self.btn_send_to_map)
        det_buttons.addWidget(self.btn_export_log)
        det_buttons.addWidget(self.btn_clear_detections)
        det_layout.addLayout(det_buttons)
        
        center_panel.addWidget(grp_detections, stretch=2)
        
        # === Правая панель (статистика и визуализация) ===
        right_panel = QtWidgets.QVBoxLayout()
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setMaximumWidth(300)
        
        # Статистика
        grp_stats = QtWidgets.QGroupBox("Статистика")
        stats_form = QtWidgets.QFormLayout(grp_stats)
        
        self.lbl_total_detections = QtWidgets.QLabel("0")
        self.lbl_confirmed = QtWidgets.QLabel("0")
        self.lbl_active_roi = QtWidgets.QLabel("0")
        self.lbl_detection_rate = QtWidgets.QLabel("0/мин")
        
        stats_form.addRow("Всего обнаружений:", self.lbl_total_detections)
        stats_form.addRow("Подтверждено:", self.lbl_confirmed)
        stats_form.addRow("Активных ROI:", self.lbl_active_roi)
        stats_form.addRow("Скорость:", self.lbl_detection_rate)
        
        right_panel.addWidget(grp_stats)
        
        # График активности (заглушка для будущего)
        grp_activity = QtWidgets.QGroupBox("Активность")
        activity_layout = QtWidgets.QVBoxLayout(grp_activity)
        
        self.activity_plot = QtWidgets.QLabel("График активности")
        self.activity_plot.setMinimumHeight(200)
        self.activity_plot.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                color: #888;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        self.activity_plot.setAlignment(QtCore.Qt.AlignCenter)
        activity_layout.addWidget(self.activity_plot)
        
        right_panel.addWidget(grp_activity)
        
        # Статус
        self.status_label = QtWidgets.QLabel("⚪ Готов к работе")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                padding: 10px;
                font-weight: bold;
                border-radius: 4px;
            }
        """)
        right_panel.addWidget(self.status_label)
        
        right_panel.addStretch()
        
        # === Сборка layout ===
        main_layout.addWidget(left_widget)
        main_layout.addLayout(center_panel, stretch=1)
        main_layout.addWidget(right_widget)
        
        # Подключение сигналов
        self.btn_start.clicked.connect(self._start_detection)
        self.btn_stop.clicked.connect(self._stop_detection)
        self.btn_add_manual.clicked.connect(self._add_manual_range)
        self.btn_add_current.clicked.connect(self._add_current_range)
        self.btn_delete_roi.clicked.connect(self._delete_selected_roi)
        self.btn_clear_roi.clicked.connect(self._clear_all_roi)
        self.btn_send_to_map.clicked.connect(self._send_selected_to_map)
        self.btn_export_log.clicked.connect(self._export_log)
        self.btn_clear_detections.clicked.connect(self._clear_detections)
        self.tbl_roi.itemSelectionChanged.connect(self._on_roi_selected)

    def _on_threshold_mode_changed(self, text):
        """Переключение режима порога."""
        is_auto = "Авто" in text
        self.threshold_offset.setEnabled(is_auto)
        self.manual_threshold.setEnabled(not is_auto)

    def _add_roi(self, start_mhz: float, stop_mhz: float, name: str = ""):
        """Добавление ROI региона."""
        # Проверяем дубликаты
        for roi in self._state.regions:
            if abs(roi.start_mhz - start_mhz) < 0.1 and abs(roi.stop_mhz - stop_mhz) < 0.1:
                return
        
        self._roi_id_seq += 1
        
        # Определяем порог
        if "Авто" in self.threshold_mode.currentText():
            threshold_mode = "auto"
            threshold_dbm = -110.0 + self.threshold_offset.value()  # Будет пересчитан
        else:
            threshold_mode = "manual"
            threshold_dbm = self.manual_threshold.value()
        
        roi = ROIRegion(
            id=self._roi_id_seq,
            name=name or f"ROI-{self._roi_id_seq}",
            start_mhz=start_mhz,
            stop_mhz=stop_mhz,
            threshold_mode=threshold_mode,
            threshold_dbm=threshold_dbm,
            threshold_offset=self.threshold_offset.value(),
            min_width_bins=self.min_width.value(),
            min_sweeps=self.min_sweeps.value()
        )
        
        self._state.regions.append(roi)
        self._update_roi_table()
        self.rangeSelected.emit(start_mhz, stop_mhz)

    def _add_manual_range(self):
        """Добавление диапазона из ручного ввода."""
        start = self.manual_start.value()
        stop = self.manual_stop.value()
        
        if start >= stop:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Начальная частота должна быть меньше конечной!")
            return
        
        name = self.manual_name.text().strip()
        if not name:
            name = f"Диапазон {start:.1f}-{stop:.1f} МГц"
        
        self._add_roi(start, stop, name)
        
        # Очищаем поле названия
        self.manual_name.clear()
        
        # Смещаем диапазон для следующего ввода
        width = stop - start
        self.manual_start.setValue(stop)
        self.manual_stop.setValue(stop + width)

    def _add_current_range(self):
        """Добавление текущего диапазона из спектра."""
        if self.parent():
            try:
                spectrum = self.parent().spectrum_tab
                start = spectrum.start_mhz.value()
                stop = spectrum.stop_mhz.value()
                self._add_roi(start, stop, f"Спектр {start:.1f}-{stop:.1f}")
            except Exception:
                self._add_roi(2400.0, 2483.5, "По умолчанию")

    def _update_roi_table(self):
        """Обновление таблицы ROI."""
        self.tbl_roi.setRowCount(len(self._state.regions))
        
        for row, roi in enumerate(self._state.regions):
            # Чекбокс включения
            chk = QtWidgets.QCheckBox()
            chk.setChecked(roi.enabled)
            chk.toggled.connect(lambda checked, r=roi: setattr(r, 'enabled', checked))
            self.tbl_roi.setCellWidget(row, 0, chk)
            
            self.tbl_roi.setItem(row, 1, QtWidgets.QTableWidgetItem(roi.name))
            self.tbl_roi.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{roi.start_mhz:.3f}"))
            self.tbl_roi.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{roi.stop_mhz:.3f}"))
            
            threshold_text = f"{roi.threshold_dbm:.1f}" if roi.threshold_mode == "manual" else "auto"
            self.tbl_roi.setItem(row, 4, QtWidgets.QTableWidgetItem(threshold_text))
            
            # Индикатор активности
            activity_item = QtWidgets.QTableWidgetItem("—")
            activity_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tbl_roi.setItem(row, 5, activity_item)
        
        # Обновляем счетчик активных ROI
        active_count = sum(1 for roi in self._state.regions if roi.enabled)
        self.lbl_active_roi.setText(str(active_count))

    def _delete_selected_roi(self):
        """Удаление выбранных ROI."""
        rows = sorted({i.row() for i in self.tbl_roi.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self._state.regions):
                del self._state.regions[r]
        self._update_roi_table()

    def _clear_all_roi(self):
        """Очистка всех ROI."""
        self._state.regions.clear()
        self._update_roi_table()
        self.rangeSelected.emit(0, 0)

    def _on_roi_selected(self):
        """При выборе ROI в таблице."""
        rows = self.tbl_roi.selectionModel().selectedRows()
        if rows and self._state.regions:
            r = rows[0].row()
            if 0 <= r < len(self._state.regions):
                roi = self._state.regions[r]
                self.rangeSelected.emit(roi.start_mhz, roi.stop_mhz)

    def _start_detection(self):
        """Запуск детектора."""
        if not self._state.regions:
            QtWidgets.QMessageBox.warning(self, "Детектор", "Добавьте диапазоны ROI для мониторинга")
            return
        
        self._state.is_active = True
        self._state.start_time = time.time()
        self._state.pending_detections.clear()
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        self.status_label.setText("🔴 Детекция активна")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #ffcccc;
                padding: 10px;
                font-weight: bold;
                border-radius: 4px;
            }
        """)
        
        self.detectionStarted.emit()

    def _stop_detection(self):
        """Остановка детектора."""
        self._state.is_active = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        self.status_label.setText("⚪ Детекция остановлена")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                padding: 10px;
                font-weight: bold;
                border-radius: 4px;
            }
        """)
        
        self.detectionStopped.emit()

    def push_data(self, freqs_hz: np.ndarray, row_dbm: np.ndarray, device_serial=None):
        """Обработка данных от спектра."""
        if not self._state.is_active or not self._state.regions:
            return
        
        freqs_mhz = freqs_hz / 1e6
        current_time = time.time()
        
        # Обрабатываем каждый активный ROI
        for roi_idx, roi in enumerate(self._state.regions):
            if not roi.enabled:
                continue
            
            # Находим индексы в диапазоне ROI
            mask = (freqs_mhz >= roi.start_mhz) & (freqs_mhz <= roi.stop_mhz)
            if not np.any(mask):
                continue
            
            roi_freqs = freqs_mhz[mask]
            roi_power = row_dbm[mask]
            
            # Обновляем историю для расчета baseline
            roi.history.append(roi_power.copy())
            
            # Вычисляем baseline (медиана по истории)
            if len(roi.history) >= 3:
                history_array = np.array(roi.history)
                roi.baseline_dbm = float(np.median(history_array))
            
            # Определяем порог
            if roi.threshold_mode == "auto":
                roi.threshold_dbm = roi.baseline_dbm + roi.threshold_offset
            
            # Детектируем сигналы выше порога
            above_threshold = roi_power > roi.threshold_dbm
            
            if np.any(above_threshold):
                # Находим связные области (сигналы)
                signals = self._find_signals(roi_freqs, roi_power, above_threshold, roi.min_width_bins)
                
                for sig_freq, sig_power, sig_width in signals:
                    # Классифицируем сигнал
                    signal_type = self._classifier.classify(sig_freq, sig_width * 1000)
                    
                    # Ключ для отслеживания
                    key = f"{sig_freq:.3f}_{roi.id}"
                    
                    if key in self._state.pending_detections:
                        # Обновляем существующее обнаружение
                        detection = self._state.pending_detections[key]
                        detection.power_dbm = max(detection.power_dbm, sig_power)
                        detection.bandwidth_khz = max(detection.bandwidth_khz, sig_width * 1000)
                        detection.last_seen = current_time
                        detection.sweep_count += 1
                        
                        # Проверяем достаточно ли свипов для подтверждения
                        if detection.sweep_count >= roi.min_sweeps:
                            detection.duration_ms = (current_time - detection.timestamp) * 1000
                            detection.confidence = min(1.0, detection.sweep_count / 10.0)
                            
                            # Добавляем в подтвержденные
                            self._confirm_detection(detection, roi)
                    else:
                        # Создаем новое обнаружение
                        detection = Detection(
                            timestamp=current_time,
                            freq_mhz=sig_freq,
                            power_dbm=sig_power,
                            bandwidth_khz=sig_width * 1000,
                            duration_ms=0,
                            roi_index=roi_idx,
                            signal_type=signal_type,
                            sweep_count=1,
                            last_seen=current_time,
                            confidence=0.1
                        )
                        self._state.pending_detections[key] = detection
            
            # Очищаем устаревшие pending detections
            self._cleanup_pending_detections(current_time)
            
            # Обновляем индикатор активности ROI
            self._update_roi_activity(roi_idx, roi_power.max() if np.any(above_threshold) else None)

    def _find_signals(self, freqs: np.ndarray, powers: np.ndarray, mask: np.ndarray, min_width: int):
        """Находит отдельные сигналы в маске."""
        signals = []
        
        # Находим начала и концы связных областей
        diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            width = end - start
            if width >= min_width:
                # Находим пик в этой области
                region_powers = powers[start:end]
                peak_idx = np.argmax(region_powers)
                peak_freq = freqs[start + peak_idx]
                peak_power = region_powers[peak_idx]
                
                # Оцениваем ширину на уровне -3dB
                threshold_3db = peak_power - 3.0
                above_3db = region_powers > threshold_3db
                width_3db = np.sum(above_3db) * (freqs[1] - freqs[0]) if len(freqs) > 1 else width
                
                signals.append((peak_freq, peak_power, width_3db))
        
        return signals

    def _cleanup_pending_detections(self, current_time: float):
        """Удаляет устаревшие неподтвержденные обнаружения."""
        timeout = self.signal_timeout.value()
        to_remove = []
        
        for key, detection in self._state.pending_detections.items():
            if current_time - detection.last_seen > timeout:
                to_remove.append(key)
        
        for key in to_remove:
            del self._state.pending_detections[key]

    def _confirm_detection(self, detection: Detection, roi: ROIRegion):
        """Подтверждает обнаружение и добавляет в историю."""
        # Добавляем в историю
        roi.detections.append(detection)
        roi.last_activity = detection.timestamp
        self._state.detection_history.append(detection)
        self._state.total_detections += 1
        self._state.confirmed_detections += 1
        
        # Ограничиваем размер истории
        if len(self._state.detection_history) > self._max_history:
            self._state.detection_history = self._state.detection_history[-self._max_history:]
        
        # Добавляем в таблицу
        self._add_detection_to_table(detection, roi)
        
        # Обновляем статистику
        self._update_statistics()
        
        # Эмитим сигнал
        self.signalDetected.emit(detection)
        
        # Удаляем из pending
        key = f"{detection.freq_mhz:.3f}_{roi.id}"
        if key in self._state.pending_detections:
            del self._state.pending_detections[key]

    def _add_detection_to_table(self, detection: Detection, roi: ROIRegion):
        """Добавление обнаружения в таблицу."""
        from PyQt5.QtCore import QDateTime
        
        # Ограничиваем количество записей
        if self.tbl_detections.rowCount() >= 100:
            self.tbl_detections.removeRow(0)
        
        row = self.tbl_detections.rowCount()
        self.tbl_detections.insertRow(row)
        
        # Время
        time_str = QDateTime.fromSecsSinceEpoch(int(detection.timestamp)).toString("HH:mm:ss")
        self.tbl_detections.setItem(row, 0, QtWidgets.QTableWidgetItem(time_str))
        
        # ROI
        self.tbl_detections.setItem(row, 1, QtWidgets.QTableWidgetItem(roi.name))
        
        # Частота
        freq_item = QtWidgets.QTableWidgetItem(f"{detection.freq_mhz:.3f} МГц")
        freq_item.setData(QtCore.Qt.UserRole, detection)  # Сохраняем объект Detection
        self.tbl_detections.setItem(row, 2, freq_item)
        
        # Уровень
        self.tbl_detections.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{detection.power_dbm:.1f} дБм"))
        
        # Ширина
        self.tbl_detections.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{detection.bandwidth_khz:.1f} кГц"))
        
        # Количество свипов
        sweeps_item = QtWidgets.QTableWidgetItem(str(detection.sweep_count))
        sweeps_item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tbl_detections.setItem(row, 5, sweeps_item)
        
        # Тип сигнала
        type_item = QtWidgets.QTableWidgetItem(detection.signal_type)
        self.tbl_detections.setItem(row, 6, type_item)
        
        # Кнопка действий
        action_btn = QtWidgets.QPushButton("→ Карта")
        action_btn.clicked.connect(lambda: self.sendToMap.emit(detection))
        self.tbl_detections.setCellWidget(row, 7, action_btn)
        
        # Цветовая индикация по уровню
        if detection.power_dbm >= -50:
            color = QtGui.QColor(255, 200, 200)  # Красный - сильный
        elif detection.power_dbm >= -70:
            color = QtGui.QColor(255, 255, 200)  # Желтый - средний
        else:
            color = QtGui.QColor(200, 255, 200)  # Зеленый - слабый
        
        for col in range(7):
            item = self.tbl_detections.item(row, col)
            if item:
                item.setBackground(QtGui.QBrush(color))
        
        # Прокручиваем вниз
        self.tbl_detections.scrollToBottom()

    def _update_roi_activity(self, roi_idx: int, power_dbm: Optional[float]):
        """Обновление индикатора активности ROI."""
        if roi_idx >= self.tbl_roi.rowCount():
            return
        
        activity_item = self.tbl_roi.item(roi_idx, 5)
        if not activity_item:
            return
        
        if power_dbm is None:
            activity_item.setText("—")
            activity_item.setBackground(QtGui.QBrush())
        else:
            activity_item.setText(f"{power_dbm:.1f} дБм")
            
            # Цвет по уровню
            if power_dbm >= -50:
                color = QtGui.QColor(255, 200, 200)
            elif power_dbm >= -70:
                color = QtGui.QColor(255, 255, 200)
            else:
                color = QtGui.QColor(200, 255, 200)
            
            activity_item.setBackground(QtGui.QBrush(color))

    def _update_statistics(self):
        """Обновление статистики."""
        self.lbl_total_detections.setText(str(self._state.total_detections))
        self.lbl_confirmed.setText(str(self._state.confirmed_detections))
        
        # Вычисляем скорость обнаружений
        if self._state.start_time:
            elapsed = time.time() - self._state.start_time
            if elapsed > 0:
                rate = (self._state.total_detections / elapsed) * 60
                self.lbl_detection_rate.setText(f"{rate:.1f}/мин")

    def _send_selected_to_map(self):
        """Отправка выбранных обнаружений на карту."""
        selected_rows = set()
        for item in self.tbl_detections.selectedItems():
            selected_rows.add(item.row())
        
        if not selected_rows:
            QtWidgets.QMessageBox.information(self, "На карту", "Выберите обнаружения для отправки")
            return
        
        sent_count = 0
        for row in selected_rows:
            freq_item = self.tbl_detections.item(row, 2)
            if freq_item:
                detection = freq_item.data(QtCore.Qt.UserRole)
                if detection:
                    self.sendToMap.emit(detection)
                    sent_count += 1
        
        if sent_count > 0:
            QtWidgets.QMessageBox.information(self, "На карту", f"Отправлено целей: {sent_count}")

    def _clear_detections(self):
        """Очистка таблицы обнаружений."""
        self.tbl_detections.setRowCount(0)
        self._state.detection_history.clear()
        self._state.pending_detections.clear()

    def _export_log(self):
        """Экспорт лога обнаружений."""
        if not self._state.detection_history:
            QtWidgets.QMessageBox.information(self, "Экспорт", "Нет обнаружений для экспорта")
            return
        
        from PyQt5.QtCore import QDateTime
        default_name = f"detections_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.csv"
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить лог", default_name, "CSV files (*.csv);;JSON files (*.json)"
        )
        if not path:
            return
        
        try:
            if path.endswith('.json'):
                # Экспорт в JSON
                data = {
                    'metadata': {
                        'export_time': time.time(),
                        'total_detections': self._state.total_detections,
                        'confirmed_detections': self._state.confirmed_detections,
                        'session_start': self._state.start_time,
                        'regions': [
                            {
                                'id': roi.id,
                                'name': roi.name,
                                'start_mhz': roi.start_mhz,
                                'stop_mhz': roi.stop_mhz,
                                'threshold_mode': roi.threshold_mode,
                                'threshold_dbm': roi.threshold_dbm,
                                'detections_count': len(roi.detections)
                            }
                            for roi in self._state.regions
                        ]
                    },
                    'detections': [
                        {
                            'timestamp': d.timestamp,
                            'freq_mhz': d.freq_mhz,
                            'power_dbm': d.power_dbm,
                            'bandwidth_khz': d.bandwidth_khz,
                            'duration_ms': d.duration_ms,
                            'roi_index': d.roi_index,
                            'signal_type': d.signal_type,
                            'sweep_count': d.sweep_count,
                            'confidence': d.confidence
                        }
                        for d in self._state.detection_history
                    ]
                }
                
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            else:
                # Экспорт в CSV
                import csv
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'datetime', 'roi_name', 'freq_mhz', 
                                   'power_dbm', 'bandwidth_khz', 'duration_ms', 
                                   'signal_type', 'sweep_count', 'confidence'])
                    
                    for d in self._state.detection_history:
                        roi_name = self._state.regions[d.roi_index].name if d.roi_index < len(self._state.regions) else "Unknown"
                        dt_str = QDateTime.fromSecsSinceEpoch(int(d.timestamp)).toString("yyyy-MM-dd HH:mm:ss")
                        writer.writerow([
                            d.timestamp,
                            dt_str,
                            roi_name,
                            f"{d.freq_mhz:.6f}",
                            f"{d.power_dbm:.2f}",
                            f"{d.bandwidth_khz:.1f}",
                            f"{d.duration_ms:.0f}",
                            d.signal_type,
                            d.sweep_count,
                            f"{d.confidence:.3f}"
                        ])
            
            QtWidgets.QMessageBox.information(self, "Экспорт", 
                f"Сохранено {len(self._state.detection_history)} обнаружений:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка экспорта: {e}")

    def get_state(self) -> DetectorState:
        """Получить текущее состояние детектора."""
        return self._state

    def get_active_rois(self) -> List[ROIRegion]:
        """Получить список активных ROI."""
        return [roi for roi in self._state.regions if roi.enabled]