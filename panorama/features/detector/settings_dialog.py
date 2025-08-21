#!/usr/bin/env python3
"""
Диалог настройки детектора для работы с Master sweep.
Позволяет настроить пороги детекции и параметры watchlist для Slave.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PyQt5 import QtWidgets, QtCore, QtGui
import json
from pathlib import Path


@dataclass
class DetectorSettings:
    """Настройки детектора."""
    # Основные параметры
    enabled: bool = True
    auto_start: bool = True  # Автоматически запускать при старте Master
    
    # Пороги детекции
    threshold_mode: str = "adaptive"  # "adaptive" или "fixed"
    baseline_offset_db: float = 20.0  # Для adaptive: baseline + offset
    fixed_threshold_dbm: float = -70.0  # Для fixed режима
    
    # Параметры детекции
    min_snr_db: float = 10.0  # Минимальный SNR для детекции
    min_peak_width_bins: int = 3  # Минимальная ширина пика
    min_peak_distance_bins: int = 5  # Минимальное расстояние между пиками
    
    # Параметры для Slave watchlist
    watchlist_span_mhz: float = 2.0  # Ширина полосы для измерения RSSI
    watchlist_dwell_ms: int = 150  # Время измерения
    max_watchlist_size: int = 20  # Максимальный размер списка
    
    # Временные параметры
    peak_timeout_sec: float = 5.0  # Таймаут для удаления неактивных пиков
    measurement_interval_sec: float = 1.0  # Интервал между измерениями
    
    # Фильтры частот
    frequency_ranges: List[Tuple[float, float]] = None  # Список (start_mhz, stop_mhz)
    exclude_ranges: List[Tuple[float, float]] = None  # Исключаемые диапазоны
    
    def __post_init__(self):
        if self.frequency_ranges is None:
            self.frequency_ranges = []
        if self.exclude_ranges is None:
            self.exclude_ranges = []
    
    def to_dict(self) -> dict:
        return {
            'enabled': self.enabled,
            'auto_start': self.auto_start,
            'threshold_mode': self.threshold_mode,
            'baseline_offset_db': self.baseline_offset_db,
            'fixed_threshold_dbm': self.fixed_threshold_dbm,
            'min_snr_db': self.min_snr_db,
            'min_peak_width_bins': self.min_peak_width_bins,
            'min_peak_distance_bins': self.min_peak_distance_bins,
            'watchlist_span_mhz': self.watchlist_span_mhz,
            'watchlist_dwell_ms': self.watchlist_dwell_ms,
            'max_watchlist_size': self.max_watchlist_size,
            'peak_timeout_sec': self.peak_timeout_sec,
            'measurement_interval_sec': self.measurement_interval_sec,
            'frequency_ranges': self.frequency_ranges,
            'exclude_ranges': self.exclude_ranges
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class DetectorSettingsDialog(QtWidgets.QDialog):
    """Диалог настройки детектора."""
    
    settingsChanged = QtCore.pyqtSignal(object)  # DetectorSettings
    
    def __init__(self, parent=None, current_settings: DetectorSettings = None):
        super().__init__(parent)
        self.setWindowTitle("Настройки детектора")
        self.resize(800, 600)
        
        self._settings_path = Path.home() / ".panorama" / "detector_settings.json"
        self.settings = current_settings or self._load_from_disk() or DetectorSettings()
        
        self._build_ui()
        self._load_settings()
        self._connect_signals()
    
    def _build_ui(self):
        """Создает интерфейс."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Создаем вкладки
        tabs = QtWidgets.QTabWidget()
        
        # Вкладка основных настроек
        basic_tab = self._create_basic_tab()
        tabs.addTab(basic_tab, "Основные")
        
        # Вкладка порогов
        threshold_tab = self._create_threshold_tab()
        tabs.addTab(threshold_tab, "Пороги")
        
        # Вкладка параметров для Slave
        slave_tab = self._create_slave_tab()
        tabs.addTab(slave_tab, "Slave Watchlist")
        
        # Вкладка фильтров частот
        filter_tab = self._create_filter_tab()
        tabs.addTab(filter_tab, "Фильтры частот")
        
        layout.addWidget(tabs)
        
        # Кнопки
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | 
            QtWidgets.QDialogButtonBox.Cancel |
            QtWidgets.QDialogButtonBox.Apply
        )
        buttons.accepted.connect(self._save_and_close)
        buttons.rejected.connect(self.reject)
        buttons.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self._apply_settings)
        
        # Кнопки пресетов
        preset_layout = QtWidgets.QHBoxLayout()
        btn_preset_sensitive = QtWidgets.QPushButton("📡 Чувствительный")
        btn_preset_sensitive.clicked.connect(lambda: self._load_preset("sensitive"))
        btn_preset_normal = QtWidgets.QPushButton("⚖️ Нормальный")
        btn_preset_normal.clicked.connect(lambda: self._load_preset("normal"))
        btn_preset_robust = QtWidgets.QPushButton("🛡️ Устойчивый")
        btn_preset_robust.clicked.connect(lambda: self._load_preset("robust"))
        
        preset_layout.addWidget(QtWidgets.QLabel("Пресеты:"))
        preset_layout.addWidget(btn_preset_sensitive)
        preset_layout.addWidget(btn_preset_normal)
        preset_layout.addWidget(btn_preset_robust)
        preset_layout.addStretch()
        
        layout.addLayout(preset_layout)
        layout.addWidget(buttons)
    
    def _create_basic_tab(self):
        """Создает вкладку основных настроек."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        
        # Включение детектора
        self.chk_enabled = QtWidgets.QCheckBox("Детектор включен")
        self.chk_enabled.setChecked(self.settings.enabled)
        layout.addRow(self.chk_enabled)
        
        # Автозапуск
        self.chk_auto_start = QtWidgets.QCheckBox("Автоматически запускать при старте Master sweep")
        self.chk_auto_start.setChecked(self.settings.auto_start)
        layout.addRow(self.chk_auto_start)
        
        # Разделитель
        layout.addRow(QtWidgets.QLabel())
        
        # Временные параметры
        layout.addRow(QtWidgets.QLabel("<b>Временные параметры:</b>"))
        
        self.spin_peak_timeout = QtWidgets.QDoubleSpinBox()
        self.spin_peak_timeout.setRange(1.0, 60.0)
        self.spin_peak_timeout.setValue(self.settings.peak_timeout_sec)
        self.spin_peak_timeout.setSuffix(" сек")
        self.spin_peak_timeout.setToolTip("Время, после которого неактивный пик удаляется")
        layout.addRow("Таймаут пика:", self.spin_peak_timeout)
        
        self.spin_measurement_interval = QtWidgets.QDoubleSpinBox()
        self.spin_measurement_interval.setRange(0.1, 10.0)
        self.spin_measurement_interval.setValue(self.settings.measurement_interval_sec)
        self.spin_measurement_interval.setSuffix(" сек")
        self.spin_measurement_interval.setToolTip("Интервал между измерениями RSSI")
        layout.addRow("Интервал измерений:", self.spin_measurement_interval)
        
        # Информация
        info_label = QtWidgets.QLabel(
            "<i>Детектор автоматически обнаруживает пики в спектре от Master sweep "
            "и передает их координаты в Slave watchlist для измерения RSSI и трилатерации.</i>"
        )
        info_label.setWordWrap(True)
        layout.addRow(info_label)
        
        return widget
    
    def _create_threshold_tab(self):
        """Создает вкладку настройки порогов."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        # Режим порога
        mode_group = QtWidgets.QGroupBox("Режим определения порога")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        
        self.radio_adaptive = QtWidgets.QRadioButton("Адаптивный (baseline + offset)")
        self.radio_fixed = QtWidgets.QRadioButton("Фиксированный порог")
        
        if self.settings.threshold_mode == "adaptive":
            self.radio_adaptive.setChecked(True)
        else:
            self.radio_fixed.setChecked(True)
        
        mode_layout.addWidget(self.radio_adaptive)
        mode_layout.addWidget(self.radio_fixed)
        layout.addWidget(mode_group)
        
        # Параметры адаптивного порога
        adaptive_group = QtWidgets.QGroupBox("Адаптивный порог")
        adaptive_layout = QtWidgets.QFormLayout(adaptive_group)
        
        self.spin_baseline_offset = QtWidgets.QDoubleSpinBox()
        self.spin_baseline_offset.setRange(5.0, 50.0)
        self.spin_baseline_offset.setValue(self.settings.baseline_offset_db)
        self.spin_baseline_offset.setSuffix(" дБ")
        self.spin_baseline_offset.setToolTip("Порог = baseline + это значение")
        adaptive_layout.addRow("Offset над baseline:", self.spin_baseline_offset)
        
        adaptive_info = QtWidgets.QLabel(
            "<i>Baseline вычисляется как медиана последних N свипов. "
            "Рекомендуемое значение: 15-25 дБ</i>"
        )
        adaptive_info.setWordWrap(True)
        adaptive_layout.addRow(adaptive_info)
        
        layout.addWidget(adaptive_group)
        
        # Параметры фиксированного порога
        fixed_group = QtWidgets.QGroupBox("Фиксированный порог")
        fixed_layout = QtWidgets.QFormLayout(fixed_group)
        
        self.spin_fixed_threshold = QtWidgets.QDoubleSpinBox()
        self.spin_fixed_threshold.setRange(-120.0, 0.0)
        self.spin_fixed_threshold.setValue(self.settings.fixed_threshold_dbm)
        self.spin_fixed_threshold.setSuffix(" дБм")
        fixed_layout.addRow("Порог:", self.spin_fixed_threshold)
        
        layout.addWidget(fixed_group)
        
        # Параметры детекции пиков
        detection_group = QtWidgets.QGroupBox("Параметры детекции пиков")
        detection_layout = QtWidgets.QFormLayout(detection_group)
        
        self.spin_min_snr = QtWidgets.QDoubleSpinBox()
        self.spin_min_snr.setRange(3.0, 50.0)
        self.spin_min_snr.setValue(self.settings.min_snr_db)
        self.spin_min_snr.setSuffix(" дБ")
        self.spin_min_snr.setToolTip("Минимальное отношение сигнал/шум для детекции")
        detection_layout.addRow("Минимальный SNR:", self.spin_min_snr)
        
        self.spin_min_width = QtWidgets.QSpinBox()
        self.spin_min_width.setRange(1, 20)
        self.spin_min_width.setValue(self.settings.min_peak_width_bins)
        self.spin_min_width.setSuffix(" бинов")
        self.spin_min_width.setToolTip("Минимальная ширина пика в бинах")
        detection_layout.addRow("Минимальная ширина:", self.spin_min_width)
        
        self.spin_min_distance = QtWidgets.QSpinBox()
        self.spin_min_distance.setRange(1, 50)
        self.spin_min_distance.setValue(self.settings.min_peak_distance_bins)
        self.spin_min_distance.setSuffix(" бинов")
        self.spin_min_distance.setToolTip("Минимальное расстояние между пиками")
        detection_layout.addRow("Минимальное расстояние:", self.spin_min_distance)
        
        layout.addWidget(detection_group)
        layout.addStretch()
        
        # Обновляем состояние виджетов
        self._update_threshold_widgets()
        
        return widget
    
    def _create_slave_tab(self):
        """Создает вкладку настроек для Slave watchlist."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        # Параметры watchlist
        watchlist_group = QtWidgets.QGroupBox("Параметры Watchlist")
        watchlist_layout = QtWidgets.QFormLayout(watchlist_group)
        
        self.spin_watchlist_span = QtWidgets.QDoubleSpinBox()
        self.spin_watchlist_span.setRange(0.1, 20.0)
        self.spin_watchlist_span.setValue(self.settings.watchlist_span_mhz)
        self.spin_watchlist_span.setSuffix(" МГц")
        self.spin_watchlist_span.setToolTip("Ширина полосы для измерения RSSI на Slave")
        watchlist_layout.addRow("Span для RSSI:", self.spin_watchlist_span)
        
        self.spin_watchlist_dwell = QtWidgets.QSpinBox()
        self.spin_watchlist_dwell.setRange(10, 1000)
        self.spin_watchlist_dwell.setValue(self.settings.watchlist_dwell_ms)
        self.spin_watchlist_dwell.setSuffix(" мс")
        self.spin_watchlist_dwell.setToolTip("Время накопления для измерения RSSI")
        watchlist_layout.addRow("Dwell time:", self.spin_watchlist_dwell)
        
        self.spin_max_watchlist = QtWidgets.QSpinBox()
        self.spin_max_watchlist.setRange(1, 100)
        self.spin_max_watchlist.setValue(self.settings.max_watchlist_size)
        self.spin_max_watchlist.setToolTip("Максимальное количество целей в watchlist")
        watchlist_layout.addRow("Макс. размер списка:", self.spin_max_watchlist)
        
        layout.addWidget(watchlist_group)
        
        # Визуализация
        viz_group = QtWidgets.QGroupBox("Визуализация")
        viz_layout = QtWidgets.QVBoxLayout(viz_group)
        
        # График примера (как подсказка)
        self.example_plot = QtWidgets.QLabel()
        self.example_plot.setMinimumHeight(200)
        self.example_plot.setStyleSheet(
            """
            QLabel {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 4px;
            }
            """
        )
        self.example_plot.setAlignment(QtCore.Qt.AlignCenter)
        self._update_example_plot()
        viz_layout.addWidget(self.example_plot)
        
        layout.addWidget(viz_group)
        
        # Информация
        info_label = QtWidgets.QLabel(
            "<i>Когда Master обнаруживает пик, его частота добавляется в watchlist. "
            "Каждый Slave измеряет RSSI в указанной полосе вокруг пика для трилатерации.</i>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        return widget
    
    def _create_filter_tab(self):
        """Создает вкладку фильтров частот."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        # Разрешенные диапазоны
        include_group = QtWidgets.QGroupBox("Разрешенные диапазоны частот")
        include_layout = QtWidgets.QVBoxLayout(include_group)
        
        self.include_table = QtWidgets.QTableWidget()
        self.include_table.setColumnCount(3)
        self.include_table.setHorizontalHeaderLabels(["Начало (МГц)", "Конец (МГц)", "Удалить"])
        include_layout.addWidget(self.include_table)
        
        include_buttons = QtWidgets.QHBoxLayout()
        self.btn_add_include = QtWidgets.QPushButton("➕ Добавить диапазон")
        self.btn_add_include.clicked.connect(lambda: self._add_range(self.include_table))
        include_buttons.addWidget(self.btn_add_include)
        include_layout.addLayout(include_buttons)
        
        layout.addWidget(include_group)
        
        # Исключаемые диапазоны
        exclude_group = QtWidgets.QGroupBox("Исключаемые диапазоны частот")
        exclude_layout = QtWidgets.QVBoxLayout(exclude_group)
        
        self.exclude_table = QtWidgets.QTableWidget()
        self.exclude_table.setColumnCount(3)
        self.exclude_table.setHorizontalHeaderLabels(["Начало (МГц)", "Конец (МГц)", "Удалить"])
        exclude_layout.addWidget(self.exclude_table)
        
        exclude_buttons = QtWidgets.QHBoxLayout()
        self.btn_add_exclude = QtWidgets.QPushButton("➕ Добавить исключение")
        self.btn_add_exclude.clicked.connect(lambda: self._add_range(self.exclude_table))
        exclude_buttons.addWidget(self.btn_add_exclude)
        exclude_layout.addLayout(exclude_buttons)
        
        layout.addWidget(exclude_group)
        
        # Пресеты диапазонов
        preset_buttons = QtWidgets.QHBoxLayout()
        btn_ism = QtWidgets.QPushButton("📻 ISM диапазоны")
        btn_ism.clicked.connect(self._preset_ism)
        btn_wifi = QtWidgets.QPushButton("📶 WiFi")
        btn_wifi.clicked.connect(self._preset_wifi)
        btn_cellular = QtWidgets.QPushButton("📱 Сотовая связь")
        btn_cellular.clicked.connect(self._preset_cellular)
        
        preset_buttons.addWidget(QtWidgets.QLabel("Пресеты:"))
        preset_buttons.addWidget(btn_ism)
        preset_buttons.addWidget(btn_wifi)
        preset_buttons.addWidget(btn_cellular)
        preset_buttons.addStretch()
        
        layout.addLayout(preset_buttons)
        
        # Загружаем текущие фильтры
        self._load_filters()
        
        return widget
    
    def _connect_signals(self):
        """Подключает сигналы."""
        # Переключение режима порога
        self.radio_adaptive.toggled.connect(self._update_threshold_widgets)
        self.radio_fixed.toggled.connect(self._update_threshold_widgets)
        
        # Обновление примера при изменении параметров
        self.spin_watchlist_span.valueChanged.connect(self._update_example_plot)
        self.spin_watchlist_dwell.valueChanged.connect(self._update_example_plot)
    
    def _update_threshold_widgets(self):
        """Обновляет доступность виджетов порогов."""
        adaptive_enabled = self.radio_adaptive.isChecked()
        self.spin_baseline_offset.setEnabled(adaptive_enabled)
        self.spin_fixed_threshold.setEnabled(not adaptive_enabled)
    
    def _update_example_plot(self):
        """Обновляет пример визуализации."""
        span = self.spin_watchlist_span.value()
        dwell = self.spin_watchlist_dwell.value()
        left = 2450 - span / 2
        right = 2450 + span / 2
        text = (
            "<div style='color: #ffffff; padding: 20px;'>"
            "<h3>Пример измерения RSSI</h3>"
            f"<p>Центральная частота: 2450.0 МГц</p>"
            f"<p>Полоса: ±{span/2:.1f} МГц</p>"
            f"<p>Диапазон: {left:.1f} - {right:.1f} МГц</p>"
            f"<p>Время накопления: {dwell} мс</p>"
            "</div>"
        )
        self.example_plot.setText(text)
    
    def _add_range(self, table: QtWidgets.QTableWidget):
        r = table.rowCount()
        table.setRowCount(r + 1)
        for c in range(2):
            it = QtWidgets.QTableWidgetItem("0.0")
            table.setItem(r, c, it)
        btn = QtWidgets.QPushButton("🗑")
        def _rem():
            row = btn.property("row_index")
            if row is None:
                # fallback: search
                for i in range(table.rowCount()):
                    if table.cellWidget(i, 2) is btn:
                        row = i
                        break
            if row is not None and 0 <= row < table.rowCount():
                table.removeRow(int(row))
        btn.clicked.connect(_rem)
        btn.setProperty("row_index", r)
        table.setCellWidget(r, 2, btn)
    
    def _get_ranges_from_table(self, table: QtWidgets.QTableWidget) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for r in range(table.rowCount()):
            try:
                a = float(table.item(r, 0).text())
                b = float(table.item(r, 1).text())
                if b < a:
                    a, b = b, a
                out.append((a, b))
            except Exception:
                continue
        return out
    
    def _load_filters(self):
        # Заполняем таблицы из текущих настроек
        self.include_table.setRowCount(0)
        for rng in (self.settings.frequency_ranges or []):
            self._add_range(self.include_table)
            r = self.include_table.rowCount() - 1
            self.include_table.item(r, 0).setText(f"{rng[0]:.3f}")
            self.include_table.item(r, 1).setText(f"{rng[1]:.3f}")
        self.exclude_table.setRowCount(0)
        for rng in (self.settings.exclude_ranges or []):
            self._add_range(self.exclude_table)
            r = self.exclude_table.rowCount() - 1
            self.exclude_table.item(r, 0).setText(f"{rng[0]:.3f}")
            self.exclude_table.item(r, 1).setText(f"{rng[1]:.3f}")
    
    def _preset_ism(self):
        # 433/868/915/2400
        self.settings.frequency_ranges = [(433.0, 434.0), (868.0, 870.0), (902.0, 928.0), (2400.0, 2483.5)]
        self._load_filters()
    
    def _preset_wifi(self):
        self.settings.frequency_ranges = [(2400.0, 2483.5), (5150.0, 5350.0), (5470.0, 5725.0), (5725.0, 5875.0)]
        self._load_filters()
    
    def _preset_cellular(self):
        # Простейшие заглушки по сотовым диапазонам
        self.settings.frequency_ranges = [(700.0, 900.0), (1700.0, 2200.0)]
        self._load_filters()
    
    def _load_preset(self, preset_name: str):
        """Загружает пресет настроек."""
        if preset_name == "sensitive":
            # Чувствительный режим - низкие пороги, быстрые измерения
            self.settings.threshold_mode = "adaptive"
            self.settings.baseline_offset_db = 10.0
            self.settings.min_snr_db = 5.0
            self.settings.min_peak_width_bins = 2
            self.settings.min_peak_distance_bins = 3
            self.settings.watchlist_span_mhz = 1.0
            self.settings.watchlist_dwell_ms = 100
            self.settings.peak_timeout_sec = 3.0
            self.settings.measurement_interval_sec = 0.5
        elif preset_name == "normal":
            # Нормальный режим - сбалансированные настройки
            self.settings.threshold_mode = "adaptive"
            self.settings.baseline_offset_db = 20.0
            self.settings.min_snr_db = 10.0
            self.settings.min_peak_width_bins = 3
            self.settings.min_peak_distance_bins = 5
            self.settings.watchlist_span_mhz = 2.0
            self.settings.watchlist_dwell_ms = 150
            self.settings.peak_timeout_sec = 5.0
            self.settings.measurement_interval_sec = 1.0
        elif preset_name == "robust":
            # Устойчивый режим - высокие пороги, медленные измерения
            self.settings.threshold_mode = "fixed"
            self.settings.fixed_threshold_dbm = -60.0
            self.settings.min_snr_db = 15.0
            self.settings.min_peak_width_bins = 5
            self.settings.min_peak_distance_bins = 8
            self.settings.watchlist_span_mhz = 3.0
            self.settings.watchlist_dwell_ms = 200
            self.settings.peak_timeout_sec = 10.0
            self.settings.measurement_interval_sec = 2.0
        
        # Применяем настройки к UI
        self._load_settings()
    
    def _load_settings(self):
        # Проставляем значения в виджеты
        self.chk_enabled.setChecked(self.settings.enabled)
        self.chk_auto_start.setChecked(self.settings.auto_start)
        if self.settings.threshold_mode == "adaptive":
            self.radio_adaptive.setChecked(True)
        else:
            self.radio_fixed.setChecked(True)
        self.spin_baseline_offset.setValue(self.settings.baseline_offset_db)
        self.spin_fixed_threshold.setValue(self.settings.fixed_threshold_dbm)
        self.spin_min_snr.setValue(self.settings.min_snr_db)
        self.spin_min_width.setValue(self.settings.min_peak_width_bins)
        self.spin_min_distance.setValue(self.settings.min_peak_distance_bins)
        self.spin_watchlist_span.setValue(self.settings.watchlist_span_mhz)
        self.spin_watchlist_dwell.setValue(self.settings.watchlist_dwell_ms)
        self.spin_max_watchlist.setValue(self.settings.max_watchlist_size)
        self.spin_peak_timeout.setValue(self.settings.peak_timeout_sec)
        self.spin_measurement_interval.setValue(self.settings.measurement_interval_sec)
        self._load_filters()
        self._update_example_plot()
    
    def _gather_settings(self) -> DetectorSettings:
        s = DetectorSettings(
            enabled=self.chk_enabled.isChecked(),
            auto_start=self.chk_auto_start.isChecked(),
            threshold_mode=("adaptive" if self.radio_adaptive.isChecked() else "fixed"),
            baseline_offset_db=self.spin_baseline_offset.value(),
            fixed_threshold_dbm=self.spin_fixed_threshold.value(),
            min_snr_db=self.spin_min_snr.value(),
            min_peak_width_bins=int(self.spin_min_width.value()),
            min_peak_distance_bins=int(self.spin_min_distance.value()),
            watchlist_span_mhz=self.spin_watchlist_span.value(),
            watchlist_dwell_ms=int(self.spin_watchlist_dwell.value()),
            max_watchlist_size=int(self.spin_max_watchlist.value()),
            peak_timeout_sec=self.spin_peak_timeout.value(),
            measurement_interval_sec=self.spin_measurement_interval.value(),
            frequency_ranges=self._get_ranges_from_table(self.include_table),
            exclude_ranges=self._get_ranges_from_table(self.exclude_table),
        )
        return s
    
    def _apply_settings(self):
        self.settings = self._gather_settings()
        self._save_to_disk(self.settings)
        self.settingsChanged.emit(self.settings)
    
    def _save_and_close(self):
        self._apply_settings()
        self.accept()
    
    def _load_from_disk(self) -> Optional[DetectorSettings]:
        try:
            if self._settings_path.exists():
                data = json.loads(self._settings_path.read_text(encoding="utf-8"))
                return DetectorSettings.from_dict(data)
        except Exception:
            pass
        return None
    
    def _save_to_disk(self, s: DetectorSettings) -> None:
        try:
            self._settings_path.parent.mkdir(exist_ok=True)
            self._settings_path.write_text(json.dumps(s.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


