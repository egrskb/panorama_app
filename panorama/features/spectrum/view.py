"""
panorama/features/spectrum/view.py
Полный исправленный виджет спектра с водопадом
Оптимизирован для 50-6000 МГц с конфигурацией через JSON
"""

from __future__ import annotations
from typing import Optional, Deque, Dict, Tuple, Any, List
from collections import deque
import time
import json
import numpy as np
import pyqtgraph as pg
from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer

from panorama.drivers.base import SweepConfig, SourceBackend
from panorama.features.spectrum.model import SpectrumModel
from panorama.shared.palettes import get_colormap

# Константы оптимизации для плавного отображения 50-6000 МГц
# Адаптивное разрешение с авто-детализацией при зуме
MIN_DISPLAY_COLS = 1024   # Минимум колонок для отображения (панорама)
MAX_DISPLAY_COLS = 8192   # Максимум колонок для отображения (детализация)
WATER_ROWS_DEFAULT = 100  # Строк водопада по умолчанию
MAX_SPECTRUM_POINTS = 8192  # Максимум точек для линий спектра (увеличено для широких диапазонов)


class SDRConfig:
    """Конфигурация SDR параметров для стабильной работы в диапазоне 50-6000 МГц."""
    def __init__(self):
        self.freq_start_mhz = 50.0
        self.freq_stop_mhz = 6000.0
        self.bin_khz = 800.0  # Минимальный bin 800 кГц для производительности
        self.lna_db = 24
        self.vga_db = 20
        self.amp_on = False
        self.segments = 4
        self.fft_size = 32
        self.waterfall_rows = WATER_ROWS_DEFAULT
        self.max_display_points = MAX_DISPLAY_COLS
        self.smoothing_enabled = True
        self.smoothing_window = 7
        self.ema_enabled = True
        self.ema_alpha = 0.3
        self.interpolation_enabled = True
        
    def load_from_file(self, path: Path):
        """Загружает конфигурацию из JSON файла."""
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for key, value in data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                return True
        except Exception as e:
            print(f"[SDRConfig] Ошибка загрузки: {e}")
        return False
    
    def save_to_file(self, path: Path):
        """Сохраняет конфигурацию в JSON файл."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[SDRConfig] Ошибка сохранения: {e}")
        return False


class SpectrumView(QtWidgets.QWidget):
    """Главный виджет спектра с водопадом, оптимизированный для высокого разрешения широких диапазонов 50-6000 МГц."""

    newRowReady = QtCore.pyqtSignal(object, object)  # (freqs_hz, row_dbm)
    rangeSelected = QtCore.pyqtSignal(float, float)
    configChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None, orchestrator=None):
        super().__init__(parent)
        
        # Конфигурация
        self.config = SDRConfig()
        self.config_path = Path.home() / ".panorama" / "sdr_config.json"
        self.config.load_from_file(self.config_path)
        
        # Модель данных
        self._model = SpectrumModel(rows=self.config.waterfall_rows)
        self._source: Optional[SourceBackend] = None
        self._current_cfg: Optional[SweepConfig] = None
        self._orchestrator = orchestrator
        
        # Буферы водопада
        self._water_view: Optional[np.ndarray] = None
        self._water_initialized = False
        self._wf_ds_factor = 1
        self._wf_cols_ds = 0
        self._wf_x_ds: Optional[np.ndarray] = None
        
        # Статистика
        self._sweep_count = 0
        self._last_full_ts: Optional[float] = None
        self._dt_ema: Optional[float] = None
        self._running = False
        
        # EMA состояние
        self._ema_last: Optional[np.ndarray] = None
        
        # Min/Max hold
        self._minhold: Optional[np.ndarray] = None
        self._maxhold: Optional[np.ndarray] = None
        
        # Среднее окно
        self._avg_queue = deque(maxlen=10)
        
        # Строим интерфейс
        self._build_ui()
        self._apply_config_to_ui()
        self._connect_signals()
        
        # Инициализация
        self._on_reset_view()
        self._apply_visibility()

    def _build_ui(self):
        """Создает интерфейс."""
        # === Верхняя панель параметров ===
        self._create_top_panel()
        
        # === Левая часть: графики ===
        self._create_plots()
        
        # === Правая панель настроек ===
        self.panel = self._build_right_panel()
        
        # === Общая компоновка ===
        graphs_layout = QtWidgets.QVBoxLayout()
        graphs_layout.addLayout(self.top_layout)
        graphs_layout.addWidget(self.plot, stretch=2)
        
        glw = pg.GraphicsLayoutWidget()
        glw.addItem(self.water_plot)
        graphs_layout.addWidget(glw, stretch=3)
        
        graphs_widget = QtWidgets.QWidget()
        graphs_widget.setLayout(graphs_layout)
        
        # Сплиттер
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        splitter.addWidget(graphs_widget)
        splitter.addWidget(self.panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        
        # Главная раскладка
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.addWidget(splitter)
        
        # Таймер обновления водопада
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setInterval(100)
        self._update_timer.timeout.connect(self._refresh_water)
        self._pending_water_update = False
        
        # Накопители для линий
        self._avg_queue: Deque[np.ndarray] = deque(maxlen=8)
        self._minhold: Optional[np.ndarray] = None
        self._maxhold: Optional[np.ndarray] = None
        self._ema_last: Optional[np.ndarray] = None
        
        # Адаптивное разрешение для отрисовки
        self._display_cols = MIN_DISPLAY_COLS  # динамическое число колонок для отрисовки
        self._zoom_debounce = QTimer(self)
        self._zoom_debounce.setInterval(120)   # мс
        self._zoom_debounce.setSingleShot(True)
        self._zoom_debounce.timeout.connect(self._apply_auto_resolution)
        
        # Кеш для перерисовки при смене масштаба
        self._last_freqs_hz: Optional[np.ndarray] = None
        self._last_power_dbm: Optional[np.ndarray] = None
        
        # Маркеры и ROI
        self._marker_seq = 0
        self._markers: Dict[int, Dict[str, Any]] = {}
        self._marker_colors = ["#FF5252", "#40C4FF", "#FFD740", "#69F0AE", "#B388FF",
                              "#FFAB40", "#18FFFF", "#FF6E40", "#64FFDA", "#EEFF41"]
        self._roi_regions = []

    def _create_top_panel(self):
        """Создает верхнюю панель с параметрами."""
        self.top_layout = QtWidgets.QHBoxLayout()
        
        # Параметры частоты
        self.start_mhz = QtWidgets.QDoubleSpinBox()
        self.start_mhz.setRange(50, 6000)  # Ограничиваем диапазон 50-6000 МГц
        self.start_mhz.setDecimals(1)
        self.start_mhz.setValue(50.0)
        self.start_mhz.setSuffix(" МГц")
        
        self.stop_mhz = QtWidgets.QDoubleSpinBox()
        self.stop_mhz.setRange(50, 6000)  # Ограничиваем диапазон 50-6000 МГц
        self.stop_mhz.setDecimals(1)
        self.stop_mhz.setValue(6000.0)
        self.stop_mhz.setSuffix(" МГц")
        
        self.bin_khz = QtWidgets.QDoubleSpinBox()
        self.bin_khz.setRange(500, 5000)  # Минимальный bin ограничен 500 кГц для производительности
        self.bin_khz.setDecimals(0)
        self.bin_khz.setValue(800)
        self.bin_khz.setSuffix(" кГц")
        
        # Параметры усиления
        self.lna_db = QtWidgets.QSpinBox()
        self.lna_db.setRange(0, 40)
        self.lna_db.setSingleStep(8)
        self.lna_db.setValue(24)
        
        self.vga_db = QtWidgets.QSpinBox()
        self.vga_db.setRange(0, 62)
        self.vga_db.setSingleStep(2)
        self.vga_db.setValue(20)
        
        self.amp_on = QtWidgets.QCheckBox("AMP")
        
        # Параметры для оркестратора (если используется)
        self.spanSpin = QtWidgets.QDoubleSpinBox()
        self.spanSpin.setRange(1, 1000)
        self.spanSpin.setValue(5)
        self.spanSpin.setSuffix(" МГц")
        
        self.dwellSpin = QtWidgets.QSpinBox()
        self.dwellSpin.setRange(100, 10000)
        self.dwellSpin.setValue(1000)
        self.dwellSpin.setSuffix(" мс")
        
        # Кнопки управления
        self.btn_start = QtWidgets.QPushButton("▶ Старт")
        self.btn_start.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        
        self.btn_stop = QtWidgets.QPushButton("⬛ Стоп")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("QPushButton:disabled { background-color: #cccccc; }")
        
        self.btn_reset = QtWidgets.QPushButton("↻ Сброс вида")
        
        # Компоновка
        def add_param(label, widget):
            vbox = QtWidgets.QVBoxLayout()
            vbox.addWidget(QtWidgets.QLabel(label))
            vbox.addWidget(widget)
            return vbox
        
        self.top_layout.addLayout(add_param("F нач", self.start_mhz))
        self.top_layout.addLayout(add_param("F кон", self.stop_mhz))
        self.top_layout.addLayout(add_param("Bin", self.bin_khz))
        self.top_layout.addLayout(add_param("LNA", self.lna_db))
        self.top_layout.addLayout(add_param("VGA", self.vga_db))
        self.top_layout.addWidget(self.amp_on)
        self.top_layout.addLayout(add_param("Span", self.spanSpin))
        self.top_layout.addLayout(add_param("Dwell", self.dwellSpin))
        self.top_layout.addStretch(1)
        self.top_layout.addWidget(self.btn_start)
        self.top_layout.addWidget(self.btn_stop)
        self.top_layout.addWidget(self.btn_reset)

    def _create_plots(self):
        """Создает графики спектра и водопада."""
        # График спектра
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Частота (МГц)")
        self.plot.setLabel("left", "Мощность (дБм)")
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        
        vb = self.plot.getViewBox()
        vb.enableAutoRange(x=False, y=False)
        vb.setMouseEnabled(x=True, y=False)
        
        # Линии спектра
        self.curve_now = pg.PlotCurveItem([], [], pen=pg.mkPen('#FFFFFF', width=1))
        self.curve_avg = pg.PlotCurveItem([], [], pen=pg.mkPen('#00FF00', width=1))
        self.curve_min = pg.PlotCurveItem([], [], pen=pg.mkPen((120, 120, 255), width=1))
        self.curve_max = pg.PlotCurveItem([], [], pen=pg.mkPen('#FFC800', width=1))
        
        self.plot.addItem(self.curve_now)
        self.plot.addItem(self.curve_avg)
        self.plot.addItem(self.curve_min)
        self.plot.addItem(self.curve_max)
        
        # Курсор
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((80, 80, 80, 120)))
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((80, 80, 80, 120)))
        self.plot.addItem(self._vline, ignoreBounds=True)
        self.plot.addItem(self._hline, ignoreBounds=True)
        
        self._cursor_text = pg.TextItem(color=pg.mkColor(255, 255, 255), anchor=(0, 1))
        self.plot.addItem(self._cursor_text)
        
        # График водопада
        self.water_plot = pg.PlotItem()
        self.water_plot.setLabel("bottom", "Частота (МГц)")
        self.water_plot.setLabel("left", "Время →")
        self.water_plot.invertY(True)  # Новые данные внизу
        self.water_plot.setMouseEnabled(x=True, y=False)
        
        self.water_img = pg.ImageItem(axisOrder="row-major")
        self.water_img.setAutoDownsample(False)
        self.water_plot.addItem(self.water_img)
        self.water_plot.setXLink(self.plot)
        
        # Палитра и уровни
        self._lut_name = "turbo"
        self._lut = get_colormap(self._lut_name, 256)
        self._wf_levels = (-110.0, -20.0)
        
        # Подписываемся на изменение диапазона графика для авто-разрешения
        self.plot.getViewBox().sigRangeChanged.connect(lambda *_: self._zoom_debounce.start())

    def _build_right_panel(self) -> QtWidgets.QWidget:
        """Создает правую панель настроек."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        
        # Группа линий
        grp_lines = QtWidgets.QGroupBox("Линии")
        gl = QtWidgets.QFormLayout(grp_lines)
        
        self.chk_now = QtWidgets.QCheckBox("Текущая")
        self.chk_now.setChecked(True)
        self.chk_avg = QtWidgets.QCheckBox("Средняя")
        self.chk_avg.setChecked(True)
        self.chk_min = QtWidgets.QCheckBox("Мин")
        self.chk_min.setChecked(False)
        self.chk_max = QtWidgets.QCheckBox("Макс")
        self.chk_max.setChecked(True)
        
        self.avg_win = QtWidgets.QSpinBox()
        self.avg_win.setRange(1, 200)
        self.avg_win.setValue(8)
        self.avg_win.setSuffix(" свипов")
        
        gl.addRow(self.chk_now)
        gl.addRow(self.chk_avg)
        gl.addRow("Окно усредн.:", self.avg_win)
        gl.addRow(self.chk_min)
        gl.addRow(self.chk_max)
        
        # Группа сглаживания
        grp_smooth = QtWidgets.QGroupBox("Сглаживание")
        gs = QtWidgets.QFormLayout(grp_smooth)
        
        self.chk_smooth = QtWidgets.QCheckBox("По частоте")
        self.chk_smooth.setChecked(True)
        
        self.smooth_win = QtWidgets.QSpinBox()
        self.smooth_win.setRange(3, 301)
        self.smooth_win.setSingleStep(2)
        self.smooth_win.setValue(7)
        
        self.chk_ema = QtWidgets.QCheckBox("EMA по времени")
        self.chk_ema.setChecked(True)
        
        self.alpha = QtWidgets.QDoubleSpinBox()
        self.alpha.setRange(0.01, 1.00)
        self.alpha.setSingleStep(0.05)
        self.alpha.setValue(0.30)
        
        self.chk_interpolation = QtWidgets.QCheckBox("Интерполяция")
        self.chk_interpolation.setChecked(True)
        
        gs.addRow(self.chk_smooth)
        gs.addRow("Окно:", self.smooth_win)
        gs.addRow(self.chk_ema)
        gs.addRow("α:", self.alpha)
        gs.addRow(self.chk_interpolation)
        
        # Группа водопада
        grp_wf = QtWidgets.QGroupBox("Водопад")
        gw = QtWidgets.QFormLayout(grp_wf)
        
        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(["turbo", "viridis", "inferno", "plasma", "magma", "gray"])
        self.cmb_cmap.setCurrentText(self._lut_name)
        
        self.sp_wf_min = QtWidgets.QDoubleSpinBox()
        self.sp_wf_min.setRange(-200, 50)
        self.sp_wf_min.setValue(-110)
        self.sp_wf_min.setSuffix(" дБм")
        
        self.sp_wf_max = QtWidgets.QDoubleSpinBox()
        self.sp_wf_max.setRange(-200, 50)
        self.sp_wf_max.setValue(-20)
        self.sp_wf_max.setSuffix(" дБм")
        
        self.btn_auto_levels = QtWidgets.QPushButton("Авто уровни")
        
        self.chk_wf_invert = QtWidgets.QCheckBox("Инверт. (новые снизу)")
        self.chk_wf_invert.setChecked(True)
        
        gw.addRow("Палитра:", self.cmb_cmap)
        gw.addRow("Мин:", self.sp_wf_min)
        gw.addRow("Макс:", self.sp_wf_max)
        gw.addRow(self.chk_wf_invert)
        gw.addRow(self.btn_auto_levels)
        
        # Группа маркеров
        grp_mrk = QtWidgets.QGroupBox("Маркеры")
        gm = QtWidgets.QVBoxLayout(grp_mrk)
        
        self.list_markers = QtWidgets.QListWidget()
        self.list_markers.setMaximumHeight(150)
        gm.addWidget(self.list_markers)
        
        btn_clear = QtWidgets.QPushButton("Очистить все")
        gm.addWidget(btn_clear)
        btn_clear.clicked.connect(self._clear_markers)
        
        # Статус
        self.lbl_sweep = QtWidgets.QLabel("Свипов: 0")
        
        # Сборка панели
        layout.addWidget(grp_lines)
        layout.addWidget(grp_smooth)
        layout.addWidget(grp_wf)
        layout.addWidget(grp_mrk)
        layout.addStretch(1)
        layout.addWidget(self.lbl_sweep)
        
        return panel

    def _connect_signals(self):
        """Подключает сигналы."""
        # Кнопки управления
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_reset.clicked.connect(self._on_reset_view)
        
        # Параметры - сохраняем при изменении
        for widget in [self.start_mhz, self.stop_mhz, self.bin_khz, 
                      self.lna_db, self.vga_db]:
            widget.valueChanged.connect(self._save_config)
        self.amp_on.toggled.connect(self._save_config)
        
        # Настройки отображения
        for widget in [self.chk_now, self.chk_avg, self.chk_min, self.chk_max]:
            widget.toggled.connect(self._apply_visibility)
        
        self.chk_smooth.toggled.connect(self._save_config)
        self.chk_ema.toggled.connect(self._save_config)
        self.chk_interpolation.toggled.connect(self._save_config)
        self.smooth_win.valueChanged.connect(self._ensure_odd_window)
        self.smooth_win.valueChanged.connect(self._save_config)
        self.alpha.valueChanged.connect(self._save_config)
        
        # Водопад
        self.cmb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        self.sp_wf_min.valueChanged.connect(self._on_wf_levels)
        self.sp_wf_max.valueChanged.connect(self._on_wf_levels)
        self.btn_auto_levels.clicked.connect(self._auto_levels)
        self.chk_wf_invert.toggled.connect(self._on_wf_invert)
        
        # Мышь
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(lambda ev: self._on_add_marker(ev, False))
        self.water_plot.scene().sigMouseClicked.connect(lambda ev: self._on_add_marker(ev, True))
        self.list_markers.itemDoubleClicked.connect(self._jump_to_marker)

    def _apply_config_to_ui(self):
        """Применяет конфигурацию к UI элементам."""
        self.start_mhz.setValue(self.config.freq_start_mhz)
        self.stop_mhz.setValue(self.config.freq_stop_mhz)
        self.bin_khz.setValue(self.config.bin_khz)
        self.lna_db.setValue(self.config.lna_db)
        self.vga_db.setValue(self.config.vga_db)
        self.amp_on.setChecked(self.config.amp_on)
        
        self.chk_smooth.setChecked(self.config.smoothing_enabled)
        self.smooth_win.setValue(self.config.smoothing_window)
        self.chk_ema.setChecked(self.config.ema_enabled)
        self.alpha.setValue(self.config.ema_alpha)
        self.chk_interpolation.setChecked(self.config.interpolation_enabled)

    def _save_config(self):
        """Сохраняет текущую конфигурацию."""
        self.config.freq_start_mhz = self.start_mhz.value()
        self.config.freq_stop_mhz = self.stop_mhz.value()
        self.config.bin_khz = self.bin_khz.value()
        self.config.lna_db = self.lna_db.value()
        self.config.vga_db = self.vga_db.value()
        self.config.amp_on = self.amp_on.isChecked()
        
        self.config.smoothing_enabled = self.chk_smooth.isChecked()
        self.config.smoothing_window = self.smooth_win.value()
        self.config.ema_enabled = self.chk_ema.isChecked()
        self.config.ema_alpha = self.alpha.value()
        self.config.interpolation_enabled = self.chk_interpolation.isChecked()
        
        self.config.save_to_file(self.config_path)

    def _target_display_cols(self, span_hz: float) -> int:
        """
        Подбираем число точек ТОЛЬКО для отрисовки в зависимости от текущего видимого span.
        Логика:
          - обзорная панорама: мало точек (1024–2048)
          - при зуме автоматом повышаем детализацию (до 8192), чтобы ~<=100 кГц/бин
        """
        if span_hz <= 0:
            return MIN_DISPLAY_COLS

        # хотим <=100 кГц на точку в «детализированном» режиме
        target_bin_hz = 100_000.0
        cols_by_resolution = int(max(1, span_hz / target_bin_hz))

        # ступенчатые пороги, чтобы картинка не «дёргалась» при лёгком скролле
        # > 1 ГГц — 1024; 200М–1Г — 2048; 50–200М — 4096; <50М — 8192
        span_mhz = span_hz * 1e-6
        if span_mhz > 1000:
            tier_cols = 1024
        elif span_mhz > 200:
            tier_cols = 2048
        elif span_mhz > 50:
            tier_cols = 4096
        else:
            tier_cols = 8192

        target = max(MIN_DISPLAY_COLS, min(MAX_DISPLAY_COLS, max(tier_cols, cols_by_resolution)))
        return target

    def _on_zoom_changed(self):
        """Обработчик изменения зума для адаптивного разрешения."""
        if self._last_freqs_hz is None or self._last_freqs_hz.size == 0:
            return
            
        x0, x1 = self.plot.getViewBox().viewRange()[0]
        span_mhz = max(0.001, x1 - x0)
        full_span_mhz = (self._last_freqs_hz[-1] - self._last_freqs_hz[0]) * 1e-6
        zoom_ratio = full_span_mhz / span_mhz

        # Простая шкала: чем глубже зум, тем больше точек
        new_cols = MIN_DISPLAY_COLS  # базовое значение
        if zoom_ratio > 4:
            new_cols = 4096
        if zoom_ratio > 8:
            new_cols = 8192
        if zoom_ratio > 16:
            new_cols = 16384
            
        # Ограничиваем максимальным значением
        new_cols = min(new_cols, MAX_DISPLAY_COLS)
        
        if new_cols != self._display_cols:
            self._display_cols = new_cols
            # перерисовать с новым даунсемплом
            self._refresh_spectrum()

    def _apply_auto_resolution(self):
        """Вызывается с дебаунсом при каждом зуме/скролле."""
        self._on_zoom_changed()

    def set_source(self, src: SourceBackend):
        """Устанавливает источник данных."""
        if self._source is not None:
            # Отключаем старый источник
            for sig, slot in [
                (self._source.fullSweepReady, self._on_full_sweep),
                (self._source.status, self._on_status),
                (self._source.error, self._on_error),
                (self._source.started, self._on_started),
                (self._source.finished, self._on_finished),
            ]:
                try:
                    sig.disconnect(slot)
                except:
                    pass
        
        self._source = src
        self._source.fullSweepReady.connect(self._on_full_sweep)
        self._source.status.connect(self._on_status)
        self._source.error.connect(self._on_error)
        self._source.started.connect(self._on_started)
        self._source.finished.connect(self._on_finished)

    def _on_full_sweep(self, freqs_hz: np.ndarray, power_dbm: np.ndarray):
        """Обработка полного свипа данных."""
        # Проверка входных данных
        if freqs_hz is None or power_dbm is None:
            return
        
        # Конвертируем в numpy если нужно
        if not isinstance(freqs_hz, np.ndarray):
            freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        if not isinstance(power_dbm, np.ndarray):
            power_dbm = np.asarray(power_dbm, dtype=np.float32)
        
        if freqs_hz.size == 0 or power_dbm.size == 0 or freqs_hz.size != power_dbm.size:
            return
        
        # Фильтруем невалидные значения и ограничиваем диапазон пользовательскими границами
        valid_mask = np.isfinite(power_dbm) & np.isfinite(freqs_hz)
        
        # Используем текущую конфигурацию для фильтрации
        if self._current_cfg:
            freq_mask = (freqs_hz >= self._current_cfg.freq_start_hz) & (freqs_hz <= self._current_cfg.freq_end_hz)
        else:
            # Fallback на разумные границы если конфигурация не задана
            freq_mask = (freqs_hz >= 50e6) & (freqs_hz <= 6000e6)
        
        valid_mask = valid_mask & freq_mask
        
        if not np.all(valid_mask):
            freqs_hz = freqs_hz[valid_mask]
            power_dbm = power_dbm[valid_mask]
            if freqs_hz.size == 0:
                return
        
        # Инициализация при первом запуске или изменении размера
        if (self._model.freqs_hz is None or 
            self._model.freqs_hz.size != freqs_hz.size or 
            self._water_view is None or 
            not self._water_initialized):
            self._initialize_display(freqs_hz, power_dbm)
            self._water_initialized = True
        
        # Обновляем модель
        self._model.freqs_hz = freqs_hz.astype(np.float64, copy=True)
        self._model.power_dbm = power_dbm.astype(np.float32, copy=True)
        self._model._last_row = power_dbm.astype(np.float32, copy=True)
        
        if hasattr(self._model, 'waterfall'):
            self._model.waterfall.append(power_dbm.astype(np.float32, copy=True))
        
        # Обновляем счетчики
        self._update_statistics()
        
        # Обновляем отображение
        self._refresh_spectrum()
        self._update_waterfall_display()
        
        # Эмитим для внешних обработчиков
        self.newRowReady.emit(freqs_hz, power_dbm)

    def _initialize_display(self, freqs_hz: np.ndarray, power_dbm: np.ndarray):
        """Инициализация отображения."""
        n = int(freqs_hz.size)
        
        # Сохраняем полную сетку
        self._model.freqs_hz = freqs_hz.astype(np.float64, copy=True)
        self._model.power_dbm = power_dbm.astype(np.float32, copy=True)
        self._model.waterfall.clear()
        
        # Настройка осей
        x_mhz = self._model.freqs_hz / 1e6
        x0, x1 = float(x_mhz[0]), float(x_mhz[-1])
        
        # Используем пользовательские границы для стабильности
        if self._current_cfg:
            user_x0 = self._current_cfg.freq_start_hz / 1e6
            user_x1 = self._current_cfg.freq_end_hz / 1e6
            x0 = max(user_x0, x0)
            x1 = min(user_x1, x1)
        else:
            # Fallback на разумные границы если конфигурация не задана
            x0 = max(50.0, x0)
            x1 = min(6000.0, x1)
        
        # Дополнительная проверка размера данных
        if x_mhz.size > 10000:  # Если слишком много точек, принудительно даунсемплируем
            self._wf_max_cols = 512
            waterfall_rows = 50
        
        self.plot.setLimits(xMin=x0, xMax=x1)
        self.water_plot.setLimits(xMin=x0, xMax=x1)
        
        self.plot.setXRange(x0, x1, padding=0.0)
        self.plot.setYRange(-110.0, -20.0, padding=0)
        self.water_plot.setXRange(x0, x1, padding=0.0)
        
        # Адаптивный даунсемплинг для больших диапазонов
        # Используем динамическое разрешение для waterfall
        self._wf_max_cols = self._display_cols
        waterfall_rows = 100  # Фиксированное количество строк
        
        self._wf_ds_factor = max(1, int(np.ceil(n / self._wf_max_cols)))
        self._wf_cols_ds = int(np.ceil(n / self._wf_ds_factor))
        
        # X-ось для водопада
        x_mhz_full = self._model.freqs_hz.astype(np.float64) / 1e6
        self._wf_x_ds = self._downsample_array(x_mhz_full, self._wf_ds_factor, method='mean')
        
        # Инициализируем буфер водопада
        self._water_view = np.full((waterfall_rows, self._wf_cols_ds), -120.0, dtype=np.float32)
        
        # Настройка отображения водопада
        self.water_img.setLookupTable(self._lut)
        self.water_img.setLevels(self._wf_levels)
        self.water_img.setImage(self._water_view, autoLevels=False)
        
        # Устанавливаем координаты
        rect = QtCore.QRectF(
            float(self._wf_x_ds[0]), 
            0.0,
            float(self._wf_x_ds[-1] - self._wf_x_ds[0]),
            float(self._water_view.shape[0])
        )
        self.water_img.setRect(rect)
        
        # Сброс накопителей
        self._avg_queue.clear()
        self._minhold = None
        self._maxhold = None
        self._ema_last = None

    def _update_waterfall_display(self):
        """Обновление водопада из накопленного массива модели."""
        if self._model.waterfall is None or len(self._model.waterfall) == 0:
            return
        
        # Получаем накопленные строки водопада
        waterfall_data = list(self._model.waterfall)
        if not waterfall_data:
            return
            
        # Даунсемплируем каждую строку
        waterfall_ds = []
        for row in waterfall_data:
            row_ds = self._downsample_array(row, self._wf_ds_factor, method='max')
            if row_ds.size != self._wf_cols_ds:
                if row_ds.size > self._wf_cols_ds:
                    row_ds = row_ds[:self._wf_cols_ds]
                else:
                    row_ds = np.pad(row_ds, (0, self._wf_cols_ds - row_ds.size), mode='edge')
            waterfall_ds.append(row_ds)
        
        # Создаем матрицу водопада
        if waterfall_ds:
            waterfall_matrix = np.array(waterfall_ds, dtype=np.float32)
            # Обновляем изображение из накопленного массива
            self.water_img.setImage(waterfall_matrix, autoLevels=False)
            
            # Устанавливаем правильный прямоугольник водопада
            if self._last_freqs_hz is not None and self._last_freqs_hz.size > 0:
                self.water_img.setRect(QtCore.QRectF(
                    self._last_freqs_hz[0]*1e-6, 0.0,
                    (self._last_freqs_hz[-1]-self._last_freqs_hz[0])*1e-6,
                    float(waterfall_matrix.shape[0])
                ))
        
        # Автонастройка уровней после первых свипов
        if self._sweep_count == 10:
            self._auto_levels()

    def _render_spectrum(self, freqs_hz, power_dbm):
        """Отрисовка спектра с адаптивным разрешением."""
        # кешируем «сырые» массивы, чтобы можно было перерисовать при смене масштаба
        self._last_freqs_hz = freqs_hz
        self._last_power_dbm = power_dbm
        
        if freqs_hz is None or power_dbm is None or freqs_hz.size == 0:
            return
            
        # первый валидный свип — фиксируем обзорный диапазон
        if not hasattr(self, '_last_freqs_hz') or self._last_freqs_hz is None:
            vb = self.plot.getViewBox()
            vb.setLimits(xMin=freqs_hz[0]*1e-6, xMax=freqs_hz[-1]*1e-6)  # запретить выход за края
            vb.setXRange(freqs_hz[0]*1e-6, freqs_hz[-1]*1e-6, padding=0)
        
        x_mhz = freqs_hz.astype(np.float64) / 1e6
        y = power_dbm.astype(np.float32, copy=False)
        
        # Применяем сглаживание
        if self.chk_smooth.isChecked():
            y = self._smooth_freq(y)
        
        if self.chk_ema.isChecked():
            y = self._smooth_time_ema(y)
        
        # Просто обновляем данные без полного reset
        self.curve_now.setData(freqs_hz * 1e-6, power_dbm)

        # EMA, min/max считаем на полных данных:
        if getattr(self, "_ema_last", None) is None:
            self._ema_last = power_dbm.astype("float32", copy=True)
        else:
            alpha = float(self.alpha.value()) if self.chk_ema.isChecked() else 1.0
            self._ema_last = (1.0 - alpha) * self._ema_last + alpha * power_dbm

        if self.chk_ema.isChecked():
            self.curve_avg.setData(freqs_hz * 1e-6, self._ema_last)
        else:
            self.curve_avg.setData([], [])

        if self.chk_min.isChecked():
            if getattr(self, "_minhold", None) is None:
                self._minhold = power_dbm.copy()
            else:
                self._minhold = np.minimum(self._minhold, power_dbm)
            self.curve_min.setData(freqs_hz * 1e-6, self._minhold)
        else:
            self.curve_min.setData([], [])

        if self.chk_max.isChecked():
            if getattr(self, "_maxhold", None) is None:
                self._maxhold = power_dbm.copy()
            else:
                self._maxhold = np.maximum(self._maxhold, power_dbm)
            self.curve_max.setData(freqs_hz * 1e-6, self._maxhold)
        else:
            self.curve_max.setData([], [])
        
        self._update_cursor_label()
        
        # Обновляем бейдж с разрешением
        self._update_resolution_badge()

    def _refresh_spectrum(self):
        """Обновление линий спектра (совместимость)."""
        if self._model.freqs_hz is not None and self._model.power_dbm is not None:
            self._render_spectrum(self._model.freqs_hz, self._model.power_dbm)

    def _downsample_array(self, arr: np.ndarray, factor: int, method: str = 'max') -> np.ndarray:
        """Универсальный даунсемплинг массива."""
        if factor <= 1 or arr.size == 0:
            return arr.astype(arr.dtype, copy=False)
        
        # Дополняем до кратного размера
        pad_size = (-arr.size) % factor
        if pad_size > 0:
            arr_padded = np.pad(arr, (0, pad_size), mode='edge')
        else:
            arr_padded = arr
        
        # Решейпим и агрегируем
        blocks = arr_padded.reshape(-1, factor)
        
        if method == 'max':
            result = np.max(blocks, axis=1)
        elif method == 'mean':
            result = np.mean(blocks, axis=1)
        elif method == 'min':
            result = np.min(blocks, axis=1)
        else:
            result = np.max(blocks, axis=1)
        
        return result.astype(arr.dtype, copy=False)

    def _downsample_line(self, x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Равномерно прореживает x/y до max_points для быстрой отрисовки."""
        if x is None or y is None:
            return x, y
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        n = x.size
        if n <= max_points or max_points <= 0:
            return x, y
        # Индексы равномерной выборки (включая последний)
        idx = np.linspace(0, n - 1, num=max_points).astype(np.int64)
        return x[idx], y[idx]

    def _smooth_freq(self, y: np.ndarray) -> np.ndarray:
        """Сглаживание по частоте."""
        if not self.chk_smooth.isChecked() or y.size < 3:
            return y
        
        w = int(self.smooth_win.value())
        if w < 3:
            w = 3
        if w % 2 == 0:
            w += 1
        
        kernel = np.ones(w, dtype=np.float32) / float(w)
        smoothed = np.convolve(y, kernel, mode='same')
        
        # Корректировка краев
        half_w = w // 2
        for i in range(half_w):
            smoothed[i] = np.mean(y[0:i+half_w+1])
            smoothed[-(i+1)] = np.mean(y[-(i+half_w+1):])
        
        return smoothed.astype(np.float32)

    def _smooth_time_ema(self, y: np.ndarray) -> np.ndarray:
        """EMA сглаживание по времени."""
        if not self.chk_ema.isChecked():
            return y
        
        a = float(self.alpha.value())
        if self._ema_last is None or self._ema_last.shape != y.shape:
            self._ema_last = y.copy()
            return y
        
        self._ema_last = (a * y) + ((1.0 - a) * self._ema_last)
        return self._ema_last.copy()

    def _update_resolution_badge(self):
        """Обновляет бейдж с текущим разрешением."""
        if not hasattr(self, '_resolution_badge'):
            # Создаем бейдж при первом вызове
            self._resolution_badge = pg.TextItem(
                text="", 
                color=pg.mkColor(255, 255, 255), 
                anchor=(1, 0),  # правый верхний угол
                border=pg.mkPen(color=(100, 100, 100), width=1),
                fill=pg.mkBrush(color=(50, 50, 50, 180))
            )
            self.plot.addItem(self._resolution_badge)
        
        # Вычисляем текущее разрешение
        if self._last_freqs_hz is not None and self._last_freqs_hz.size > 1:
            span_hz = self._last_freqs_hz[-1] - self._last_freqs_hz[0]
            bin_width_khz = span_hz / self._display_cols / 1e3
            resolution_text = f"Разрешение: {bin_width_khz:.1f} кГц/бин"
        else:
            resolution_text = f"Точек: {self._display_cols}"
        
        self._resolution_badge.setText(resolution_text)

    def _update_statistics(self):
        """Обновление статистики."""
        t = time.time()
        if self._last_full_ts is not None:
            dt = t - self._last_full_ts
            self._dt_ema = dt if self._dt_ema is None else (0.3 * dt + 0.7 * self._dt_ema)
        self._last_full_ts = t
        self._sweep_count += 1
        
        fps_text = f" • {1.0/self._dt_ema:.1f} св/с" if self._dt_ema and self._dt_ema > 0 else ""
        self.lbl_sweep.setText(f"Свипов: {self._sweep_count}{fps_text}")

    def _on_start_clicked(self):
        """Запуск сканирования."""
        if not self._source:
            QtWidgets.QMessageBox.information(self, "Источник", "Источник данных не подключён")
            return
        
        # Проверяем настройку SDR Master
        if not self._check_sdr_master_configured():
            QtWidgets.QMessageBox.critical(self, "SDR Master не настроен", 
                "Для запуска спектра необходимо настроить SDR Master устройство.\n\n"
                "Перейдите в Настройки → Диспетчер устройств и выберите HackRF устройство как Master.")
            return
        
        # Сохраняем конфигурацию
        self._save_config()
        
        # Корректируем границы
        f0, f1, bw = self._snap_bounds(
            self.start_mhz.value(),
            self.stop_mhz.value(),
            self.bin_khz.value() * 1e3
        )
        self.start_mhz.setValue(f0 / 1e6)
        self.stop_mhz.setValue(f1 / 1e6)
        
        # Получаем серийный номер master
        master_serial = None
        try:
            from panorama.features.settings.storage import load_sdr_settings
            sdr_settings = load_sdr_settings()
            if sdr_settings and 'master' in sdr_settings:
                master_config = sdr_settings['master']
                if master_config and 'serial' in master_config:
                    master_serial = master_config['serial']
        except:
            pass
        
        # Создаем конфигурацию
        cfg = SweepConfig(
            # Используем частоты из UI вместо захардкоженных
            freq_start_hz=int(self.start_mhz.value() * 1e6),
            freq_end_hz=int(self.stop_mhz.value() * 1e6),
            bin_hz=int(bw),
            lna_db=int(self.lna_db.value()),
            vga_db=int(self.vga_db.value()),
            amp_on=bool(self.amp_on.isChecked()),
            # локальное окно только для детектора/слейвов, мастер его игнорирует
            detector_span_hz=int(self.spanSpin.value() * 1e6),
            serial=master_serial
        )
        self._current_cfg = cfg
        
        # Инициализация модели с пользовательскими границами
        self._model.set_grid(cfg.freq_start_hz, cfg.freq_end_hz, cfg.bin_hz)
        
        # Сброс счетчиков
        self._sweep_count = 0
        self._last_full_ts = None
        self._dt_ema = None
        self._minhold = None
        self._maxhold = None
        self._avg_queue.clear()
        self._ema_last = None
        self._water_initialized = False
        
        # Параметры для оркестратора
        if self._orchestrator:
            span_mhz = float(self.spanSpin.value())
            dwell_ms = int(self.dwellSpin.value())
            self._orchestrator.set_global_parameters(span_hz=span_mhz * 1e6, dwell_ms=dwell_ms)
        
        # Запускаем источник
        self._source.start(cfg)
        self.configChanged.emit()
        self._set_controls_enabled(False)
        self._running = True

    def _on_stop_clicked(self):
        """Остановка сканирования."""
        if self._source and self._source.is_running():
            self._source.stop()
        self._set_controls_enabled(True)
        self._running = False

    def _check_sdr_master_configured(self) -> bool:
        """Проверка настройки SDR Master."""
        try:
            from panorama.features.settings.storage import load_sdr_settings
            sdr_settings = load_sdr_settings()
            
            if not sdr_settings or 'master' not in sdr_settings:
                return False
            
            master_config = sdr_settings['master']
            if not master_config or 'serial' not in master_config:
                return False
            
            master_serial = master_config['serial']
            if not master_serial or len(master_serial) < 16:
                return False
            
            return True
            
        except Exception as e:
            print(f"[SpectrumView] Ошибка проверки SDR master: {e}")
            return False

    def _snap_bounds(self, f0_mhz: float, f1_mhz: float, bin_hz: float) -> Tuple[float, float, float]:
        """Корректировка границ диапазона."""
        bin_mhz = bin_hz / 1e6
        seg_mhz = 5.0
        
        f0 = np.floor(f0_mhz + 1e-9)
        f1 = np.floor(f1_mhz + 1e-9)
        
        width = max(seg_mhz, f1 - f0)
        width = np.floor(width / seg_mhz) * seg_mhz
        if width < seg_mhz:
            width = seg_mhz
        width = np.round(width / bin_mhz) * bin_mhz
        
        return float(f0) * 1e6, float(f0 + width) * 1e6, bin_hz

    def _set_controls_enabled(self, enabled: bool):
        """Управление доступностью контролов."""
        self.start_mhz.setEnabled(enabled)
        self.stop_mhz.setEnabled(enabled)
        self.bin_khz.setEnabled(enabled)
        self.lna_db.setEnabled(enabled)
        self.vga_db.setEnabled(enabled)
        self.amp_on.setEnabled(enabled)
        self.btn_start.setEnabled(enabled)
        self.btn_stop.setEnabled(not enabled)

    def _on_status(self, msg: object):
        """Обработка статусного сообщения."""
        try:
            text = str(msg)
        except:
            text = repr(msg)
        
        if hasattr(self, "lbl_sweep") and self.lbl_sweep is not None:
            base = self.lbl_sweep.text().split(" • ")[0]
            self.lbl_sweep.setText(f"{base} • {text}")
        print(f"[Spectrum] Статус: {text}")

    def _on_error(self, err: object):
        """Обработка ошибки."""
        try:
            text = str(err)
        except:
            text = repr(err)
        
        print(f"[Spectrum] ОШИБКА: {text}")
        QtWidgets.QMessageBox.critical(self, "Ошибка источника", text)
        self._set_controls_enabled(True)
        self._running = False

    def _on_started(self):
        """Источник запущен."""
        self._set_controls_enabled(False)
        self._running = True
        print("[Spectrum] Запущен")

    def _on_finished(self, code: int):
        """Источник остановлен."""
        self._set_controls_enabled(True)
        self._running = False
        print(f"[Spectrum] Завершен, код={code}")

    def _on_reset_view(self):
        """Сброс вида графиков."""
        self.plot.setYRange(-110.0, -20.0, padding=0)
        self._water_view = None
        self._water_initialized = False

    def _apply_visibility(self):
        """Управление видимостью линий."""
        self.curve_now.setVisible(self.chk_now.isChecked())
        self.curve_avg.setVisible(self.chk_avg.isChecked())
        self.curve_min.setVisible(self.chk_min.isChecked())
        self.curve_max.setVisible(self.chk_max.isChecked())

    def _ensure_odd_window(self, value):
        """Гарантирует нечетное окно."""
        if value % 2 == 0:
            self.smooth_win.setValue(value + 1)

    def _on_wf_invert(self, checked):
        """Инверсия водопада."""
        self.water_plot.invertY(checked)

    def _on_cmap_changed(self, name: str):
        """Изменение палитры."""
        self._lut_name = name
        self._lut = get_colormap(name, 256)
        self.water_img.setLookupTable(self._lut)

    def _on_wf_levels(self):
        """Изменение уровней водопада."""
        self._wf_levels = (float(self.sp_wf_min.value()), float(self.sp_wf_max.value()))
        self.water_img.setLevels(self._wf_levels)

    def _auto_levels(self):
        """Автонастройка уровней."""
        if self._water_view is not None:
            Z = self._water_view
        else:
            Z = self._model.waterfall
        
        if Z is None or (isinstance(Z, np.ndarray) and Z.size == 0):
            return
        
        if isinstance(Z, deque):
            if len(Z) == 0:
                return
            Z = np.array(Z)
        
        data = Z[np.isfinite(Z) & (Z > -200.0)]
        if data.size == 0:
            return
        
        vmin_new = float(np.percentile(data, 5))
        vmax_new = float(np.percentile(data, 99))
        
        if vmax_new - vmin_new < 5.0:
            mid = 0.5 * (vmin_new + vmax_new)
            vmin_new = mid - 2.5
            vmax_new = mid + 2.5
        
        # Сглаживание
        vmin_old, vmax_old = self._wf_levels
        alpha = 0.3
        vmin_s = alpha * vmin_new + (1.0 - alpha) * vmin_old
        vmax_s = alpha * vmax_new + (1.0 - alpha) * vmax_old
        
        if abs(vmin_s - vmin_old) > 0.2:
            self.sp_wf_min.setValue(vmin_s)
        if abs(vmax_s - vmax_old) > 0.2:
            self.sp_wf_max.setValue(vmax_s)

    def _refresh_water(self):
        """Обновление водопада по таймеру."""
        if not self._pending_water_update:
            self._update_timer.stop()
            return
        self._pending_water_update = False
        
        if self._water_view is not None:
            self.water_img.setImage(self._water_view, autoLevels=False)

    def _on_mouse_moved(self, pos):
        """Движение мыши."""
        vb = self.plot.getViewBox()
        if vb is None:
            return
        
        p = vb.mapSceneToView(pos)
        fx = float(p.x())
        fy = float(p.y())
        
        self._vline.setPos(fx)
        self._hline.setPos(fy)
        self._update_cursor_label()

    def _update_cursor_label(self):
        """Обновление подписи курсора."""
        freqs = self._model.freqs_hz
        row = self._model._last_row if hasattr(self._model, '_last_row') else None
        
        fx = float(self._vline.value())
        fy = float(self._hline.value())
        
        if freqs is None or row is None or row.size == 0:
            self._cursor_text.setText(f"{fx:.3f} МГц, {fy:.1f} дБм")
            self._cursor_text.setPos(fx, self.plot.getViewBox().viewRange()[1][1] - 5)
            return
        
        x_mhz = freqs.astype(np.float64) / 1e6
        i = int(np.clip(np.searchsorted(x_mhz, fx), 1, len(x_mhz)-1))
        x0, x1 = x_mhz[i-1], x_mhz[i]
        y0, y1 = row[i-1], row[i]
        
        y_at = float(y0) if x1 == x0 else float((fx - x0) / (x1 - x0) * (y1 - y0) + y0)
        
        self._cursor_text.setText(f"{fx:.3f} МГц, {y_at:.1f} дБм")
        self._cursor_text.setPos(fx, self.plot.getViewBox().viewRange()[1][1] - 5)

    def _on_add_marker(self, ev, from_water: bool):
        """Добавление маркера по двойному клику."""
        if not ev.double() or ev.button() != QtCore.Qt.LeftButton:
            return
        
        vb = (self.water_plot if from_water else self.plot).getViewBox()
        p = vb.mapSceneToView(ev.scenePos())
        fx = float(p.x())
        
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Новый маркер",
            f"Частота: {fx:.6f} МГц\nНазвание:",
            QtWidgets.QLineEdit.Normal,
            f"M{self._marker_seq + 1}"
        )
        
        if ok and name:
            self._add_marker(fx, name)

    def _add_marker(self, f_mhz: float, name: str, color: Optional[str] = None):
        """Добавление маркера."""
        self._marker_seq += 1
        mid = self._marker_seq
        
        if color is None:
            color = self._marker_colors[mid % len(self._marker_colors)]
        
        line_spec = pg.InfiniteLine(
            pos=f_mhz, angle=90, movable=True,
            pen=pg.mkPen(color, width=2, style=QtCore.Qt.DashLine)
        )
        line_water = pg.InfiniteLine(
            pos=f_mhz, angle=90, movable=True,
            pen=pg.mkPen(color, width=1.5, style=QtCore.Qt.DashLine)
        )
        label = pg.TextItem(name, anchor=(0.5, 1), color=color)
        label.setPos(f_mhz, self.plot.getViewBox().viewRange()[1][1])
        
        self.plot.addItem(line_spec)
        self.plot.addItem(label)
        self.water_plot.addItem(line_water)
        
        line_spec.sigPositionChanged.connect(lambda: self._sync_marker(mid, line_spec.value(), "spec"))
        line_water.sigPositionChanged.connect(lambda: self._sync_marker(mid, line_water.value(), "water"))
        
        self._markers[mid] = {
            "freq": f_mhz,
            "name": name,
            "color": color,
            "line_spec": line_spec,
            "line_water": line_water,
            "label": label
        }
        
        item = QtWidgets.QListWidgetItem(f"{name}: {f_mhz:.6f} МГц")
        item.setData(QtCore.Qt.UserRole, mid)
        item.setForeground(QtGui.QBrush(QtGui.QColor(color)))
        self.list_markers.addItem(item)

    def _sync_marker(self, mid: int, new_freq: float, source: str):
        """Синхронизация маркера между графиками."""
        if mid not in self._markers:
            return
        
        m = self._markers[mid]
        m["freq"] = new_freq
        
        if source == "spec":
            m["line_water"].setValue(new_freq)
        else:
            m["line_spec"].setValue(new_freq)
        
        m["label"].setPos(new_freq, self.plot.getViewBox().viewRange()[1][1])
        
        for i in range(self.list_markers.count()):
            item = self.list_markers.item(i)
            if item.data(QtCore.Qt.UserRole) == mid:
                item.setText(f"{m['name']}: {new_freq:.6f} МГц")
                break

    def _jump_to_marker(self, item: QtWidgets.QListWidgetItem):
        """Переход к маркеру."""
        mid = item.data(QtCore.Qt.UserRole)
        if mid not in self._markers:
            return
        
        freq = self._markers[mid]["freq"]
        vb = self.plot.getViewBox()
        x0, x1 = vb.viewRange()[0]
        center = (x0 + x1) / 2
        offset = freq - center
        self.plot.setXRange(x0 + offset, x1 + offset, padding=0)

    def _clear_markers(self):
        """Очистка всех маркеров."""
        for m in self._markers.values():
            try:
                self.plot.removeItem(m["line_spec"])
                self.plot.removeItem(m["label"])
                self.water_plot.removeItem(m["line_water"])
            except:
                pass
        
        self._markers.clear()
        self.list_markers.clear()

    def closeEvent(self, event):
        """При закрытии виджета."""
        # Сохраняем конфигурацию
        self._save_config()
        
        # Останавливаем если запущен
        if self._running and self._source:
            self._source.stop()
        
        event.accept()

    # === Публичные методы для внешнего управления ===
    
    def set_cursor_freq(self, f: float, center: bool = True):
        """Установить курсор на частоту."""
        try:
            f = float(f)
        except:
            return
        
        f_hz = f * 1e6 if f < 1e7 else f
        
        if self._model.freqs_hz is None or self._model.freqs_hz.size == 0:
            return
        
        x_mhz = self._model.freqs_hz.astype(np.float64) / 1e6
        x_val = f_hz / 1e6
        
        self._vline.setPos(x_val)
        
        if center:
            vb = self.plot.getViewBox()
            xr = vb.viewRange()[0] if vb is not None else (x_mhz[0], x_mhz[-1])
            width = xr[1] - xr[0]
            if width <= 0:
                width = max((x_mhz[-1] - x_mhz[0]) * 0.25, 1.0)
            x0 = max(x_val - width / 2.0, float(x_mhz[0]))
            x1 = min(x_val + width / 2.0, float(x_mhz[-1]))
            self.plot.setXRange(x0, x1, padding=0.0)

    def reload_detector_settings(self):
        """Перезагрузка настроек детектора."""
        if hasattr(self._model, 'reload_detector_settings'):
            self._model.reload_detector_settings()
            print("[SpectrumView] Настройки детектора перезагружены")