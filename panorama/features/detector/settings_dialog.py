#!/usr/bin/env python3
"""
Диалог настройки детектора для работы с Master sweep.
Позволяет настроить пороги детекции и параметры watchlist для Slave.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets


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

    # Расширенные параметры детекции (гистерезис и границы)
    hysteresis_high_db: float = 10.0
    hysteresis_low_db: float = 5.0
    bridge_gap_bins: int = 4
    boundary_end_delta_db: float = 1.2
    edge_min_run_bins: int = 3
    prefilter_median_bins: int = 5
    min_region_occupancy: float = 0.40
    min_region_area: float = 3.0

    # Параметры для Slave watchlist и RMS трилатерации
    watchlist_dwell_ms: int = 150  # Время измерения
    max_watchlist_size: int = 20  # Максимальный размер списка

    # RMS параметры для трилатерации (единый параметр)
    rms_halfspan_mhz: float = 2.5  # Полуширина полосы для RMS расчета
    center_mode: str = "fmax"  # "fmax" | "centroid"

    # Временные параметры
    peak_timeout_sec: float = 5.0  # Таймаут для удаления неактивных пиков
    measurement_interval_sec: float = 1.0  # Интервал между измерениями
    min_confirmation_sweeps: int = 3  # Минимум свипов для подтверждения пика

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
            'hysteresis_high_db': self.hysteresis_high_db,
            'hysteresis_low_db': self.hysteresis_low_db,
            'bridge_gap_bins': self.bridge_gap_bins,
            'boundary_end_delta_db': self.boundary_end_delta_db,
            'edge_min_run_bins': self.edge_min_run_bins,
            'prefilter_median_bins': self.prefilter_median_bins,
            'min_region_occupancy': self.min_region_occupancy,
            'min_region_area': self.min_region_area,
            'watchlist_dwell_ms': self.watchlist_dwell_ms,
            'max_watchlist_size': self.max_watchlist_size,
            'rms_halfspan_mhz': self.rms_halfspan_mhz,
            'center_mode': self.center_mode,
            'peak_timeout_sec': self.peak_timeout_sec,
            'measurement_interval_sec': self.measurement_interval_sec,
            'min_confirmation_sweeps': self.min_confirmation_sweeps,
            'frequency_ranges': self.frequency_ranges,
            'exclude_ranges': self.exclude_ranges
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DetectorSettingsDialog(QtWidgets.QDialog):
    """Диалог настройки детектора."""

    settingsChanged = QtCore.pyqtSignal(object)  # DetectorSettings

    def __init__(self, parent=None, current_settings: DetectorSettings = None):
        super().__init__(parent)
        self.setWindowTitle("Настройки детектора")
        self.resize(800, 600)

        self._settings_path = Path.home() / ".panorama" / "signal_processing_settings.json"
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

        # Вкладка параметров (ранее: Пороги)
        threshold_tab = self._create_threshold_tab()
        tabs.addTab(threshold_tab, "Параметры")

        # Вкладка параметров для Slave
        slave_tab = self._create_slave_tab()
        tabs.addTab(slave_tab, "Slave Watchlist")

        # Удаляем вкладку фильтров частот (сканируем весь мастер-диапазон)

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

        # Информация
        info_label = QtWidgets.QLabel(
            "<i>Детектор автоматически обнаруживает пики в спектре от Master sweep "
            "и передает их координаты в Slave watchlist для измерения RSSI и трилатерации.</i>"
        )
        info_label.setWordWrap(True)
        layout.addRow(info_label)

        return widget

    def _create_threshold_tab(self):
        """Создает вкладку настройки порогов (адаптивный layout с прокруткой)."""
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

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
        # Пояснение
        mode_info = QtWidgets.QLabel(
            "<i>Адаптивный: порог вычисляется как baseline + offset. \n"
            "Фиксированный: используется заданное значение в дБм независимо от окружения.</i>"
        )
        mode_info.setWordWrap(True)
        layout.addWidget(mode_info)

        # Параметры адаптивного порога
        adaptive_group = QtWidgets.QGroupBox("Адаптивный порог")
        adaptive_layout = QtWidgets.QFormLayout(adaptive_group)
        adaptive_layout.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        adaptive_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        adaptive_layout.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        adaptive_layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.spin_baseline_offset = QtWidgets.QDoubleSpinBox()
        self.spin_baseline_offset.setRange(5.0, 50.0)
        self.spin_baseline_offset.setValue(self.settings.baseline_offset_db)
        self.spin_baseline_offset.setSuffix(" дБ")
        self.spin_baseline_offset.setMaximumWidth(180)
        self.spin_baseline_offset.setToolTip("Порог = baseline + это значение")
        adaptive_layout.addRow("Offset над baseline:", self.spin_baseline_offset)
        adaptive_expl = QtWidgets.QLabel("<i>Насколько выше шумового пола (baseline) должен быть сигнал, чтобы считаться кандидатом.</i>")
        adaptive_expl.setWordWrap(True)
        adaptive_expl.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        adaptive_layout.addRow("", adaptive_expl)

        adaptive_info = QtWidgets.QLabel(
            "<i>Baseline вычисляется как медиана последних N свипов. "
            "Рекомендуемое значение: 15-25 дБ для обычных условий, "
            "10-15 дБ для слабых сигналов.</i>"
        )
        adaptive_info.setWordWrap(True)
        adaptive_layout.addRow(adaptive_info)

        layout.addWidget(adaptive_group)

        # Параметры фиксированного порога
        fixed_group = QtWidgets.QGroupBox("Фиксированный порог")
        fixed_layout = QtWidgets.QFormLayout(fixed_group)
        fixed_layout.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        fixed_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        fixed_layout.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        fixed_layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.spin_fixed_threshold = QtWidgets.QDoubleSpinBox()
        self.spin_fixed_threshold.setRange(-120.0, 0.0)
        self.spin_fixed_threshold.setValue(self.settings.fixed_threshold_dbm)
        self.spin_fixed_threshold.setSuffix(" дБм")
        self.spin_fixed_threshold.setMaximumWidth(180)
        fixed_layout.addRow("Порог:", self.spin_fixed_threshold)
        fixed_expl = QtWidgets.QLabel("<i>Абсолютный уровень мощности, при превышении которого считается, что есть сигнал.</i>")
        fixed_expl.setWordWrap(True)
        fixed_expl.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        fixed_layout.addRow("", fixed_expl)

        layout.addWidget(fixed_group)

        # Параметры детекции пиков
        detection_group = QtWidgets.QGroupBox("Параметры детекции пиков")
        detection_layout = QtWidgets.QFormLayout(detection_group)
        detection_layout.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        detection_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        detection_layout.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        detection_layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.spin_min_snr = QtWidgets.QDoubleSpinBox()
        self.spin_min_snr.setRange(3.0, 50.0)
        self.spin_min_snr.setValue(self.settings.min_snr_db)
        self.spin_min_snr.setSuffix(" дБ")
        self.spin_min_snr.setMaximumWidth(180)
        self.spin_min_snr.setToolTip("Минимальное отношение сигнал/шум для детекции")
        detection_layout.addRow("Минимальный SNR:", self.spin_min_snr)
        lbl_snr = QtWidgets.QLabel("<i>Минимальная разница между пиковым уровнем и baseline для учета пика (в дБ).</i>")
        lbl_snr.setWordWrap(True)
        lbl_snr.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        detection_layout.addRow("", lbl_snr)

        self.spin_min_width = QtWidgets.QSpinBox()
        self.spin_min_width.setRange(1, 20)
        self.spin_min_width.setValue(self.settings.min_peak_width_bins)
        self.spin_min_width.setSuffix(" бинов")
        self.spin_min_width.setMaximumWidth(180)
        self.spin_min_width.setToolTip("Минимальная ширина пика в бинах")
        detection_layout.addRow("Минимальная ширина:", self.spin_min_width)
        lbl_width = QtWidgets.QLabel("<i>Минимальное число соседних бинов над порогом, чтобы регион считался сигналом.</i>")
        lbl_width.setWordWrap(True)
        detection_layout.addRow("", lbl_width)

        self.spin_min_distance = QtWidgets.QSpinBox()
        self.spin_min_distance.setRange(1, 50)
        self.spin_min_distance.setValue(self.settings.min_peak_distance_bins)
        self.spin_min_distance.setSuffix(" бинов")
        self.spin_min_distance.setMaximumWidth(180)
        self.spin_min_distance.setToolTip("Минимальное расстояние между пиками")
        detection_layout.addRow("Минимальное расстояние:", self.spin_min_distance)
        lbl_dist = QtWidgets.QLabel("<i>Минимальный зазор между пиками (в бинах), чтобы они не сливались в один.</i>")
        lbl_dist.setWordWrap(True)
        detection_layout.addRow("", lbl_dist)

        # Гистерезис (упрощённые настройки)
        self.spin_hyst_high = QtWidgets.QDoubleSpinBox()
        self.spin_hyst_high.setRange(5.0, 30.0)
        self.spin_hyst_high.setValue(self.settings.hysteresis_high_db)
        self.spin_hyst_high.setSuffix(" дБ")
        self.spin_hyst_high.setMaximumWidth(180)
        detection_layout.addRow("Гистерезис — высокий порог:", self.spin_hyst_high)

        self.spin_hyst_low = QtWidgets.QDoubleSpinBox()
        self.spin_hyst_low.setRange(1.0, 25.0)
        self.spin_hyst_low.setValue(self.settings.hysteresis_low_db)
        self.spin_hyst_low.setSuffix(" дБ")
        self.spin_hyst_low.setMaximumWidth(180)
        detection_layout.addRow("Гистерезис — низкий порог:", self.spin_hyst_low)

        self.spin_bridge = QtWidgets.QSpinBox()
        self.spin_bridge.setRange(0, 20)
        self.spin_bridge.setValue(self.settings.bridge_gap_bins)
        self.spin_bridge.setSuffix(" бинов")
        self.spin_bridge.setMaximumWidth(180)
        detection_layout.addRow("Допустимый разрыв внутри полосы:", self.spin_bridge)

        self.spin_boundary_delta = QtWidgets.QDoubleSpinBox()
        self.spin_boundary_delta.setRange(0.1, 5.0)
        self.spin_boundary_delta.setSingleStep(0.1)
        self.spin_boundary_delta.setValue(self.settings.boundary_end_delta_db)
        self.spin_boundary_delta.setSuffix(" дБ")
        self.spin_boundary_delta.setMaximumWidth(180)
        detection_layout.addRow("Завершение при возврате к baseline ±:", self.spin_boundary_delta)

        self.spin_edge_run = QtWidgets.QSpinBox()
        self.spin_edge_run.setRange(1, 20)
        self.spin_edge_run.setValue(self.settings.edge_min_run_bins)
        self.spin_edge_run.setSuffix(" бинов")
        self.spin_edge_run.setMaximumWidth(180)
        detection_layout.addRow("Подряд бинов для фиксации края:", self.spin_edge_run)

        # Анти‑выбросы
        self.spin_pref_median = QtWidgets.QSpinBox()
        self.spin_pref_median.setRange(0, 99)
        self.spin_pref_median.setValue(self.settings.prefilter_median_bins)
        self.spin_pref_median.setSuffix(" бинов")
        self.spin_pref_median.setMaximumWidth(180)
        detection_layout.addRow("Скользящая медиана (предфильтр):", self.spin_pref_median)

        self.spin_occupancy = QtWidgets.QDoubleSpinBox()
        self.spin_occupancy.setRange(0.0, 1.0)
        self.spin_occupancy.setSingleStep(0.05)
        self.spin_occupancy.setValue(self.settings.min_region_occupancy)
        self.spin_occupancy.setSuffix(" доля")
        self.spin_occupancy.setMaximumWidth(180)
        detection_layout.addRow("Мин. занятость региона (0..1):", self.spin_occupancy)

        self.spin_area = QtWidgets.QDoubleSpinBox()
        self.spin_area.setRange(0.0, 1000.0)
        self.spin_area.setSingleStep(0.5)
        self.spin_area.setValue(self.settings.min_region_area)
        self.spin_area.setSuffix(" дБ·бин")
        self.spin_area.setMaximumWidth(180)
        detection_layout.addRow("Мин. площадь SNR:", self.spin_area)

        # Подтверждение детекции (перенесено сюда из Основных)
        confirm_group = QtWidgets.QGroupBox("Подтверждение")
        confirm_layout = QtWidgets.QFormLayout(confirm_group)
        confirm_layout.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        confirm_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        confirm_layout.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        confirm_layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.spin_confirmation_sweeps = QtWidgets.QSpinBox()
        self.spin_confirmation_sweeps.setRange(1, 10)
        self.spin_confirmation_sweeps.setValue(self.settings.min_confirmation_sweeps)
        self.spin_confirmation_sweeps.setSuffix(" свипов")
        self.spin_confirmation_sweeps.setMaximumWidth(180)
        self.spin_confirmation_sweeps.setToolTip("Сколько свипов подряд сигнал должен наблюдаться для подтверждения")
        confirm_layout.addRow("Свипов для подтверждения:", self.spin_confirmation_sweeps)
        lbl_conf = QtWidgets.QLabel("<i>Значение повышает надежность детекции, уменьшая ложные пуски.</i>")
        lbl_conf.setWordWrap(True)
        confirm_layout.addRow("", lbl_conf)
        layout.addWidget(confirm_group)

        layout.addWidget(detection_group)
        layout.addStretch()

        # Обновляем состояние виджетов
        self._update_threshold_widgets()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _create_slave_tab(self):
        """Создает вкладку настроек для Slave watchlist (с прокруткой и адаптивными формами)."""
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Параметры watchlist
        watchlist_group = QtWidgets.QGroupBox("Параметры Watchlist")
        watchlist_layout = QtWidgets.QFormLayout(watchlist_group)
        watchlist_layout.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        watchlist_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        watchlist_layout.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        watchlist_layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # RMS Halfspan parameter (единый параметр для всех измерений)
        self.spin_rms_halfspan = QtWidgets.QDoubleSpinBox()
        self.spin_rms_halfspan.setRange(1.0, 10.0)
        self.spin_rms_halfspan.setSingleStep(0.1)
        self.spin_rms_halfspan.setValue(getattr(self.settings, 'rms_halfspan_mhz', 2.5))
        self.spin_rms_halfspan.setSuffix(" МГц")
        self.spin_rms_halfspan.setToolTip("Полуширина полосы для расчета RMS в трилатерации (от F_max ± halfspan)")
        self.spin_rms_halfspan.setMaximumWidth(180)
        watchlist_layout.addRow("RMS полуширина:", self.spin_rms_halfspan)
        lbl_rms = QtWidgets.QLabel("<i>Полуширина полосы для RMS измерений Slaves. Полная ширина = 2×halfspan. Slaves измеряют среднеквадратичный RSSI в диапазоне F_max ± halfspan для трилатерации.</i>")
        lbl_rms.setWordWrap(True)
        lbl_rms.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        watchlist_layout.addRow("", lbl_rms)

        # Режим центра: F_max или центроид
        self.combo_center_mode = QtWidgets.QComboBox()
        self.combo_center_mode.addItems(["F_max", "Centroid"])
        try:
            current_mode = getattr(self.settings, 'center_mode', 'fmax')
            self.combo_center_mode.setCurrentIndex(0 if current_mode == 'fmax' else 1)
        except Exception:
            self.combo_center_mode.setCurrentIndex(0)
        self.combo_center_mode.setToolTip("Чем центрироваться при измерении: F_max или центроид кластера")
        watchlist_layout.addRow("Центр полосы:", self.combo_center_mode)

        self.spin_watchlist_dwell = QtWidgets.QSpinBox()
        self.spin_watchlist_dwell.setRange(10, 1000)
        self.spin_watchlist_dwell.setValue(self.settings.watchlist_dwell_ms)
        self.spin_watchlist_dwell.setSuffix(" мс")
        self.spin_watchlist_dwell.setToolTip("Время накопления для измерения RSSI")
        self.spin_watchlist_dwell.setMaximumWidth(180)
        watchlist_layout.addRow("Dwell time:", self.spin_watchlist_dwell)
        lbl_dwell = QtWidgets.QLabel("<i>Время накопления в каждой частотной полосе для усреднения RSSI.</i>")
        lbl_dwell.setWordWrap(True)
        lbl_dwell.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        watchlist_layout.addRow("", lbl_dwell)

        self.spin_max_watchlist = QtWidgets.QSpinBox()
        self.spin_max_watchlist.setRange(1, 100)
        self.spin_max_watchlist.setValue(self.settings.max_watchlist_size)
        self.spin_max_watchlist.setToolTip("Максимальное количество целей в watchlist")
        self.spin_max_watchlist.setMaximumWidth(180)
        watchlist_layout.addRow("Макс. размер списка:", self.spin_max_watchlist)
        lbl_max = QtWidgets.QLabel("<i>Максимальное число одновременно отслеживаемых целей (последние имеют приоритет).</i>")
        lbl_max.setWordWrap(True)
        lbl_max.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        watchlist_layout.addRow("", lbl_max)

        # Интервал обновления RMS (секунды)
        self.spin_rms_interval_sec = QtWidgets.QDoubleSpinBox()
        self.spin_rms_interval_sec.setRange(0.1, 10.0)
        self.spin_rms_interval_sec.setSingleStep(0.1)
        self.spin_rms_interval_sec.setValue(float(getattr(self.settings, 'measurement_interval_sec', 1.0)))
        self.spin_rms_interval_sec.setSuffix(" c")
        self.spin_rms_interval_sec.setToolTip("Как часто слейвы переизмеряют RMS для каждой цели (сек)")
        self.spin_rms_interval_sec.setMaximumWidth(180)
        watchlist_layout.addRow("Интервал обновления RMS:", self.spin_rms_interval_sec)
        lbl_int = QtWidgets.QLabel("<i>Период постановки задач измерения RMS. Меньше — чаще обновления и выше нагрузка.</i>")
        lbl_int.setWordWrap(True)
        lbl_int.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        watchlist_layout.addRow("", lbl_int)

        layout.addWidget(watchlist_group)

        # Визуализация
        viz_group = QtWidgets.QGroupBox("Визуализация")
        viz_layout = QtWidgets.QVBoxLayout(viz_group)

        # График примера
        self.example_plot = QtWidgets.QLabel()
        self.example_plot.setMinimumHeight(200)
        self.example_plot.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 4px;
            }
        """)
        self.example_plot.setAlignment(QtCore.Qt.AlignCenter)
        self._update_example_plot()
        viz_layout.addWidget(self.example_plot)

        layout.addWidget(viz_group)

        # Информация
        info_label = QtWidgets.QLabel(
            "<i>Когда Master обнаруживает пик, его частота добавляется в watchlist. "
            "Каждый Slave измеряет RSSI в указанной полосе вокруг пика для трилатерации. "
            "Ширина окна определяет диапазон (пик ± span/2) для измерения среднеквадратичного RSSI.</i>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addStretch()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(content)
        return scroll

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
        self.spin_watchlist_dwell.valueChanged.connect(self._update_example_plot)
        self.spin_rms_halfspan.valueChanged.connect(self._update_example_plot)
        self.combo_center_mode.currentIndexChanged.connect(self._update_example_plot)

    def _update_threshold_widgets(self):
        """Обновляет доступность виджетов порогов."""
        adaptive_enabled = self.radio_adaptive.isChecked()
        self.spin_baseline_offset.setEnabled(adaptive_enabled)
        self.spin_fixed_threshold.setEnabled(not adaptive_enabled)

    def _update_example_plot(self):
        """Обновляет пример визуализации."""
        dwell = self.spin_watchlist_dwell.value()
        rms_halfspan = self.spin_rms_halfspan.value() if hasattr(self, 'spin_rms_halfspan') else 2.5

        # Полная ширина = 2 × halfspan
        full_span = 2 * rms_halfspan
        left = 2450 - rms_halfspan
        right = 2450 + rms_halfspan

        center_label = "F_max" if self.combo_center_mode.currentIndex() == 0 else "Centroid"
        text = (
            "<div style='color: #ffffff; padding: 20px;'>"
            "<h3>Пример RMS измерения</h3>"
            f"<p>Центральная частота ({center_label}): 2450.0 МГц</p>"
            f"<p><b>RMS полоса: ±{rms_halfspan:.1f} МГц ({left:.1f} - {right:.1f} МГц)</b></p>"
            f"<p>Полная ширина измерения: {full_span:.1f} МГц</p>"
            f"<p>Время накопления: {dwell} мс</p>"
            "<p><b>Все Slaves измеряют RMS RSSI в едином диапазоне ±halfspan вокруг F_max</b></p>"
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
            for i in range(table.rowCount()):
                if table.cellWidget(i, 2) is btn:
                    table.removeRow(i)
                    break
        btn.clicked.connect(_rem)
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
        # ISM диапазоны
        self.settings.frequency_ranges = [
            (433.0, 434.8),     # 433 МГц ISM
            (868.0, 870.0),     # 868 МГц EU
            (902.0, 928.0),     # 915 МГц US
            (2400.0, 2483.5),   # 2.4 ГГц
            (5725.0, 5875.0)    # 5.8 ГГц
        ]
        self._load_filters()

    def _preset_wifi(self):
        self.settings.frequency_ranges = [
            (2400.0, 2483.5),   # 2.4 ГГц
            (5150.0, 5350.0),   # 5 ГГц UNII-1/2
            (5470.0, 5725.0),   # 5 ГГц UNII-2e/3
            (5725.0, 5875.0)    # 5 ГГц UNII-4
        ]
        self._load_filters()

    def _preset_cellular(self):
        # Основные сотовые диапазоны
        self.settings.frequency_ranges = [
            (791.0, 821.0),     # LTE Band 20
            (832.0, 862.0),     # LTE Band 20
            (880.0, 915.0),     # GSM 900
            (925.0, 960.0),     # GSM 900
            (1710.0, 1785.0),   # GSM 1800
            (1805.0, 1880.0),   # GSM 1800
            (1920.0, 1980.0),   # UMTS
            (2110.0, 2170.0),   # UMTS
            (2500.0, 2570.0),   # LTE Band 7
            (2620.0, 2690.0)    # LTE Band 7
        ]
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
            self.settings.rms_halfspan_mhz = 1.0  # Более чувствительный режим
            self.settings.watchlist_dwell_ms = 100
            self.settings.peak_timeout_sec = 3.0
            self.settings.measurement_interval_sec = 0.5
            self.settings.min_confirmation_sweeps = 2
        elif preset_name == "normal":
            # Нормальный режим - сбалансированные настройки
            self.settings.threshold_mode = "adaptive"
            self.settings.baseline_offset_db = 20.0
            self.settings.min_snr_db = 10.0
            self.settings.min_peak_width_bins = 3
            self.settings.min_peak_distance_bins = 5
            self.settings.rms_halfspan_mhz = 2.5  # Стандартный режим
            self.settings.watchlist_dwell_ms = 150
            self.settings.peak_timeout_sec = 5.0
            self.settings.measurement_interval_sec = 1.0
            self.settings.min_confirmation_sweeps = 3
        elif preset_name == "robust":
            # Устойчивый режим - высокие пороги, медленные измерения
            self.settings.threshold_mode = "fixed"
            self.settings.fixed_threshold_dbm = -60.0
            self.settings.min_snr_db = 15.0
            self.settings.min_peak_width_bins = 5
            self.settings.min_peak_distance_bins = 8
            self.settings.rms_halfspan_mhz = 5.0  # Широкая полоса для надежности
            self.settings.watchlist_dwell_ms = 200
            self.settings.peak_timeout_sec = 10.0
            self.settings.measurement_interval_sec = 2.0
            self.settings.min_confirmation_sweeps = 3

        # Применяем настройки к UI
        self._load_settings()

    def _load_settings(self):
        """Загружает настройки в UI виджеты."""
        # Основные
        self.chk_enabled.setChecked(self.settings.enabled)
        self.chk_auto_start.setChecked(self.settings.auto_start)

        # Пороги
        if self.settings.threshold_mode == "adaptive":
            self.radio_adaptive.setChecked(True)
        else:
            self.radio_fixed.setChecked(True)
        self.spin_baseline_offset.setValue(self.settings.baseline_offset_db)
        self.spin_fixed_threshold.setValue(self.settings.fixed_threshold_dbm)
        self.spin_min_snr.setValue(self.settings.min_snr_db)
        self.spin_min_width.setValue(self.settings.min_peak_width_bins)
        self.spin_min_distance.setValue(self.settings.min_peak_distance_bins)
        self.spin_hyst_high.setValue(self.settings.hysteresis_high_db)
        self.spin_hyst_low.setValue(self.settings.hysteresis_low_db)
        self.spin_bridge.setValue(self.settings.bridge_gap_bins)
        self.spin_boundary_delta.setValue(self.settings.boundary_end_delta_db)
        self.spin_edge_run.setValue(self.settings.edge_min_run_bins)
        self.spin_pref_median.setValue(self.settings.prefilter_median_bins)
        self.spin_occupancy.setValue(self.settings.min_region_occupancy)
        self.spin_area.setValue(self.settings.min_region_area)

        # Watchlist
        self.spin_watchlist_dwell.setValue(self.settings.watchlist_dwell_ms)
        self.spin_max_watchlist.setValue(self.settings.max_watchlist_size)
        self.spin_rms_halfspan.setValue(self.settings.rms_halfspan_mhz)
        self.spin_rms_interval_sec.setValue(self.settings.measurement_interval_sec)

        # Временные параметры (таймаут/интервал убраны из UI — сохраняем только подтверждение)
        self.spin_confirmation_sweeps.setValue(self.settings.min_confirmation_sweeps)

        # Фильтры отключены – сканируем весь мастер-диапазон

        # Обновляем визуализацию
        self._update_example_plot()

    def _gather_settings(self) -> DetectorSettings:
        """Собирает настройки из UI."""
        freq_ranges = []
        excl_ranges = []
        try:
            if hasattr(self, 'include_table') and self.include_table is not None:
                freq_ranges = self._get_ranges_from_table(self.include_table)
            if hasattr(self, 'exclude_table') and self.exclude_table is not None:
                excl_ranges = self._get_ranges_from_table(self.exclude_table)
        except Exception:
            freq_ranges = []
            excl_ranges = []

        s = DetectorSettings(
            enabled=self.chk_enabled.isChecked(),
            auto_start=self.chk_auto_start.isChecked(),
            threshold_mode=("adaptive" if self.radio_adaptive.isChecked() else "fixed"),
            baseline_offset_db=self.spin_baseline_offset.value(),
            fixed_threshold_dbm=self.spin_fixed_threshold.value(),
            min_snr_db=self.spin_min_snr.value(),
            min_peak_width_bins=int(self.spin_min_width.value()),
            min_peak_distance_bins=int(self.spin_min_distance.value()),
            hysteresis_high_db=self.spin_hyst_high.value(),
            hysteresis_low_db=self.spin_hyst_low.value(),
            bridge_gap_bins=int(self.spin_bridge.value()),
            boundary_end_delta_db=self.spin_boundary_delta.value(),
            edge_min_run_bins=int(self.spin_edge_run.value()),
            prefilter_median_bins=int(self.spin_pref_median.value()),
            min_region_occupancy=self.spin_occupancy.value(),
            min_region_area=self.spin_area.value(),
            watchlist_dwell_ms=int(self.spin_watchlist_dwell.value()),
            max_watchlist_size=int(self.spin_max_watchlist.value()),
            rms_halfspan_mhz=self.spin_rms_halfspan.value(),
            center_mode=('fmax' if self.combo_center_mode.currentIndex() == 0 else 'centroid'),
            # Таймаут остаётся прежним, интервал RMS берём из UI
            peak_timeout_sec=self.settings.peak_timeout_sec,
            measurement_interval_sec=float(self.spin_rms_interval_sec.value()),
            min_confirmation_sweeps=int(self.spin_confirmation_sweeps.value()),
            frequency_ranges=freq_ranges,
            exclude_ranges=excl_ranges,
        )
        return s

    def _apply_settings(self):
        """Применяет настройки."""
        self.settings = self._gather_settings()
        self._save_to_disk(self.settings)
        self.settingsChanged.emit(self.settings)

    def _save_and_close(self):
        """Сохраняет и закрывает диалог."""
        self._apply_settings()
        self.accept()

    def _load_from_disk(self) -> Optional[DetectorSettings]:
        """Загружает настройки с диска."""
        try:
            if self._settings_path.exists():
                data = json.loads(self._settings_path.read_text(encoding="utf-8"))
                return DetectorSettings.from_dict(data)
        except Exception:
            pass
        return None

    def _save_to_disk(self, s: DetectorSettings) -> None:
        """Сохраняет настройки на диск."""
        try:
            self._settings_path.parent.mkdir(exist_ok=True)
            self._settings_path.write_text(
                json.dumps(s.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception:
            pass


# Вспомогательная функция для интеграции с PeakWatchlistManager
def load_detector_settings() -> Optional[DetectorSettings]:
    """Загружает настройки детектора из файла."""
    settings_path = Path.home() / ".panorama" / "signal_processing_settings.json"
    try:
        if settings_path.exists():
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            return DetectorSettings.from_dict(data)
    except Exception:
        pass
    return None


def apply_settings_to_watchlist_manager(settings: DetectorSettings, manager):
    """Применяет настройки к PeakWatchlistManager."""
    if not settings or not manager:
        return

    # Основные параметры
    manager.threshold_mode = settings.threshold_mode
    manager.baseline_offset_db = settings.baseline_offset_db
    manager.threshold_dbm = settings.fixed_threshold_dbm
    manager.min_snr_db = settings.min_snr_db
    manager.min_peak_width_bins = settings.min_peak_width_bins
    manager.min_peak_distance_bins = settings.min_peak_distance_bins
    # Расширенные параметры детекции (синхронизация с менеджером)
    if hasattr(manager, 'hysteresis_high_db'):
        manager.hysteresis_high_db = settings.hysteresis_high_db
    if hasattr(manager, 'hysteresis_low_db'):
        manager.hysteresis_low_db = settings.hysteresis_low_db
    if hasattr(manager, 'bridge_gap_bins_default'):
        manager.bridge_gap_bins_default = int(settings.bridge_gap_bins)
    if hasattr(manager, 'boundary_end_delta_db'):
        manager.boundary_end_delta_db = settings.boundary_end_delta_db
    if hasattr(manager, 'edge_min_run_bins'):
        manager.edge_min_run_bins = int(settings.edge_min_run_bins)
    if hasattr(manager, 'prefilter_median_bins'):
        manager.prefilter_median_bins = int(settings.prefilter_median_bins)
    if hasattr(manager, 'min_region_occupancy'):
        manager.min_region_occupancy = float(settings.min_region_occupancy)
    if hasattr(manager, 'min_region_area'):
        manager.min_region_area = float(settings.min_region_area)

    # Параметры watchlist (используем RMS halfspan для определения полосы)
    manager.rms_halfspan_hz = settings.rms_halfspan_mhz * 1e6
    # Для совместимости со старым кодом - устанавливаем watchlist_span как полную ширину
    if hasattr(manager, 'watchlist_span_hz'):
        manager.watchlist_span_hz = settings.rms_halfspan_mhz * 2e6  # Полная ширина = 2 × halfspan

    manager.max_watchlist_size = settings.max_watchlist_size
    if hasattr(manager, 'center_mode'):
        manager.center_mode = settings.center_mode
    manager.peak_timeout_sec = settings.peak_timeout_sec
    manager.min_confirmation_sweeps = settings.min_confirmation_sweeps

    print(f"[DetectorSettings] Applied settings to watchlist manager: "
          f"mode={manager.threshold_mode}, "
          f"offset={manager.baseline_offset_db} dB, "
          f"rms_halfspan={manager.rms_halfspan_hz/1e6} MHz, "
          f"confirmations={manager.min_confirmation_sweeps}")
