"""
panorama/features/spectrum/view.py
Полный исправленный виджет спектра с водопадом
Оптимизирован для 50-6000 МГц с конфигурацией через JSON
"""

from __future__ import annotations
from typing import Optional, Deque, Dict, Tuple, Any, List
from collections import deque
import time

import numpy as np
import pyqtgraph as pg

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer

from panorama.drivers.base import SweepConfig, SourceBackend
from panorama.features.spectrum.model import SpectrumModel
from panorama.shared.palettes import get_colormap
from panorama.features.spectrum.master_adapter import MasterSourceAdapter

# Константы оптимизации для плавного отображения 50-6000 МГц
# Адаптивное разрешение с авто-детализацией при зуме
MIN_DISPLAY_COLS = 1024   # Минимум колонок для отображения (панорама)
MAX_DISPLAY_COLS = 8192   # Максимум колонок для отображения (детализация)
WATER_ROWS_DEFAULT = 400  # Строк водопада по умолчанию (как в прототипе)
MAX_SPECTRUM_POINTS = 8192  # Максимум точек для линий спектра (увеличено для широких диапазонов)


# Убираем дублирующий SDRConfig - используем основной из sdr_config.py


class SpectrumView(QtWidgets.QWidget):
    """Главный виджет спектра с водопадом, оптимизированный для высокого разрешения широких диапазонов 50-6000 МГц."""

    newRowReady = QtCore.pyqtSignal(object, object)  # (freqs_hz, row_dbm)
    rangeSelected = QtCore.pyqtSignal(float, float)
    configChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None, orchestrator=None):
        super().__init__(parent)
        
        # Конфигурация - используем основной SDRConfigManager
        try:
            from panorama.features.settings.sdr_config import SDRConfigManager
            self.config_manager = SDRConfigManager()
            self.config = self.config_manager.config
        except Exception as e:
            print(f"[SpectrumView] Ошибка загрузки SDR конфигурации: {e}")
            # Fallback к дефолтам
            self.config = type('Config', (), {
                'freq_start_mhz': 50.0,
                'freq_stop_mhz': 6000.0,
                'bin_khz': 5.0,
                'lna_db': 24,
                'vga_db': 20,
                'amp_on': False,
                'smoothing_enabled': True,
                'smoothing_window': 7,
                'ema_enabled': True,
                'ema_alpha': 0.3
            })()
        
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
        self._water_idx_cache: Optional[np.ndarray] = None
        
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

        # Полные/отображаемые данные спектра
        self.full_spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.display_spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None
        
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
        self._update_timer.setInterval(80)
        self._update_timer.timeout.connect(self._refresh_water)
        self._pending_water_update = False
        
        # Таймер короткой анимации «плавного опускания» водопада
        self._wf_anim_timer = QtCore.QTimer(self)
        self._wf_anim_timer.setInterval(16)  # ~60 FPS
        self._wf_anim_timer.timeout.connect(self._wf_anim_step)
        self._wf_anim_active = False
        self._wf_anim_steps = 6
        self._wf_anim_i = 0
        
        # Накопители для линий
        self._avg_queue: Deque[np.ndarray] = deque(maxlen=8)
        self._minhold: Optional[np.ndarray] = None
        self._maxhold: Optional[np.ndarray] = None
        self._ema_last: Optional[np.ndarray] = None
        
        # Адаптивное разрешение для отрисовки
        self._display_cols = MIN_DISPLAY_COLS  # динамическое число колонок для отрисовки
        # Авто-разрешение отключено: используем фиксированный даунсэмпл под ширину виджета
        self._zoom_debounce = None
        
        # Кеш для перерисовки при смене масштаба
        self._last_freqs_hz: Optional[np.ndarray] = None
        self._last_power_dbm: Optional[np.ndarray] = None
        # Счётчик текущих точек на экране (для плашки)
        self._last_plotted_points: int = 0
        # Троттлинг перерисовки UI
        self._ui_min_interval_s: float = 0.15
        self._last_plot_ts: float = 0.0

        # Целевая ширина водопада (точек по X) для ресемплинга
        self._wf_target_cols: int = 2048
        
        # Маркеры и ROI
        self._marker_seq = 0
        self._markers: Dict[int, Dict[str, Any]] = {}
        self._marker_colors = ["#FF5252", "#40C4FF", "#FFD740", "#69F0AE", "#B388FF",
                              "#FFAB40", "#18FFFF", "#FF6E40", "#64FFDA", "#EEFF41"]
        self._roi_regions = []
        # Подсветка диапазонов мастера
        self._highlight_regions_spec: List[pg.LinearRegionItem] = []
        self._highlight_regions_water: List[pg.GraphicsObject] = []
        self._highlight_ranges_mhz: List[Tuple[float, float]] = []

    def _create_top_panel(self):
        """Создает верхнюю панель с параметрами."""
        self.top_layout = QtWidgets.QHBoxLayout()
        
        # Параметры частоты
        self.start_mhz = QtWidgets.QDoubleSpinBox()
        self.start_mhz.setRange(50, 6000)
        self.start_mhz.setDecimals(3)
        self.start_mhz.setValue(2400.000)
        self.start_mhz.setSuffix(" МГц")
        
        self.stop_mhz = QtWidgets.QDoubleSpinBox()
        self.stop_mhz.setRange(50, 6000)
        self.stop_mhz.setDecimals(3)
        self.stop_mhz.setValue(2483.500)
        self.stop_mhz.setSuffix(" МГц")
        
        self.bin_khz = QtWidgets.QDoubleSpinBox()
        self.bin_khz.setRange(1, 5000)  # Разрешаем от 1 кГц для высокого разрешения (⚠️ мелкий bin может замедлить GUI)
        self.bin_khz.setDecimals(0)
        self.bin_khz.setValue(5)  # 5 кГц по умолчанию для высокого разрешения
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
        
        # Параметры span/dwell перенесены в раздел слейвов/детектора
        
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
        # Добавляем подсказку для bin
        bin_hint = QtWidgets.QLabel("⚠️ <5 кГц может замедлить GUI")
        bin_hint.setStyleSheet("color: #FFA500; font-size: 10px;")
        bin_layout = add_param("Bin", self.bin_khz)
        bin_layout.addWidget(bin_hint)
        self.top_layout.addLayout(bin_layout)
        self.top_layout.addLayout(add_param("LNA", self.lna_db))
        self.top_layout.addLayout(add_param("VGA", self.vga_db))
        self.top_layout.addWidget(self.amp_on)
        # Span/Dwell удалены из спектра (управляются из раздела слейвов)
        self.top_layout.addStretch(1)
        self.top_layout.addWidget(self.btn_start)
        self.top_layout.addWidget(self.btn_stop)
        self.top_layout.addWidget(self.btn_reset)

    def _create_plots(self):
        """Создает графики спектра и водопада."""
        # График спектра
        self.plot = pg.PlotWidget()
        try:
            self.plot.setAntialiasing(True)
        except Exception:
            pass
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Частота (МГц)")
        self.plot.setLabel("left", "Мощность (дБм)")
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        
        vb = self.plot.getViewBox()
        vb.enableAutoRange(x=False, y=False)
        vb.setMouseEnabled(x=True, y=False)
        # Жёсткие границы частот 50–6000 МГц и запрет выхода за них
        vb.setLimits(xMin=50.0, xMax=6000.0, maxXRange=6000.0-50.0)
        
        # Линии спектра
        # Текущая линия как PlotDataItem, чтобы включить заливку под кривой
        self.curve_now = pg.PlotDataItem([], [], pen=pg.mkPen('#FFFFFF', width=1))
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
        self.water_plot.invertY(False)  # Свежие сверху, как в прототипе
        self.water_plot.setMouseEnabled(x=True, y=False)
        self.water_plot.setLimits(xMin=50.0, xMax=6000.0, maxXRange=6000.0-50.0)
        
        self.water_img = pg.ImageItem(axisOrder="row-major")
        self.water_img.setAutoDownsample(False)
        self.water_plot.addItem(self.water_img)
        self.water_plot.setXLink(self.plot)
        
        # Палитра и уровни
        self._lut_name = "turbo"
        self._lut = get_colormap(self._lut_name, 256)
        self._wf_levels = (-110.0, -20.0)
        
        # Подписываемся только для обновления заливки
        self.plot.getViewBox().sigRangeChanged.connect(lambda *_: self._update_fill_under_curve())

        # Бейдж со span и разрешением
        self._span_badge = pg.TextItem(text="", color=pg.mkColor(180, 220, 255), anchor=(1, 1),
                                       border=pg.mkPen(color=(60, 90, 140), width=1),
                                       fill=pg.mkBrush(color=(20, 30, 50, 160)))
        self.plot.addItem(self._span_badge)

        # Значки/бейджи
        if not hasattr(self, '_resolution_badge'):
            self._resolution_badge = pg.TextItem(
                text="",
                color=pg.mkColor(255, 255, 255),
                anchor=(1, 0),
                border=pg.mkPen(color=(100, 100, 100), width=1),
                fill=pg.mkBrush(color=(50, 50, 50, 180))
            )
            self.plot.addItem(self._resolution_badge)
            # размещаем бейдж в правом верхнем углу текущего видимого диапазона
            vr = self.plot.getViewBox().viewRange()
            if vr:
                self._resolution_badge.setPos(vr[0][1], vr[1][0])
        if not hasattr(self, '_points_badge'):
            self._points_badge = pg.TextItem(
                text="",
                color=pg.mkColor(200, 200, 255),
                anchor=(1, 0),
                border=pg.mkPen(color=(80, 80, 120), width=1),
                fill=pg.mkBrush(color=(40, 40, 80, 160))
            )
            self.plot.addItem(self._points_badge)

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
        # По умолчанию сглаживание пользователь включает сам (как в прототипе)
        self.chk_smooth.setChecked(False)
        
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
        
        gs.addRow(self.chk_smooth)
        gs.addRow("Окно:", self.smooth_win)
        gs.addRow(self.chk_ema)
        gs.addRow("α:", self.alpha)

        # (убрано) отдельная анимация линий
        
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
        
        self.chk_wf_invert = QtWidgets.QCheckBox("Инвертировать (свежие снизу)")
        self.chk_wf_invert.setChecked(False)
        
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

    # === Подсветка диапазонов мастера ===
    def set_highlight_ranges(self, ranges_mhz: List[Tuple[float, float]]):
        """Устанавливает подсветку диапазонов на графике спектра и водопаде.
        ranges_mhz: список (start_mhz, stop_mhz)
        """
        try:
            # Сохраним и приведём к валидному формату
            cleaned: List[Tuple[float, float]] = []
            for a, b in ranges_mhz or []:
                a = float(a); b = float(b)
                if b < a:
                    a, b = b, a
                if b <= a:
                    continue
                # Клип в глобальные границы
                a = max(50.0, min(6000.0, a))
                b = max(50.0, min(6000.0, b))
                if b > a:
                    cleaned.append((a, b))
            self._highlight_ranges_mhz = cleaned
            self._refresh_highlight_regions()
        except Exception:
            pass

    def _clear_highlight_regions(self):
        """Удаляет старые объекты подсветки."""
        try:
            for item in self._highlight_regions_spec:
                try:
                    self.plot.removeItem(item)
                except Exception:
                    pass
            self._highlight_regions_spec.clear()
        except Exception:
            pass
        try:
            for item in self._highlight_regions_water:
                try:
                    self.water_plot.removeItem(item)
                except Exception:
                    pass
            self._highlight_regions_water.clear()
        except Exception:
            pass

    def _refresh_highlight_regions(self):
        """Перерисовывает подсветку диапазонов на обоих графиках."""
        self._clear_highlight_regions()
        if not self._highlight_ranges_mhz:
            return
        # Спектр: LinearRegionItem полупрозрачным красным
        for (f0, f1) in self._highlight_ranges_mhz:
            try:
                region = pg.LinearRegionItem(values=(f0, f1), orientation=pg.LinearRegionItem.Vertical,
                                             brush=pg.mkBrush(255, 0, 0, 60), pen=pg.mkPen(255, 60, 60, 150))
                region.setMovable(False)
                self.plot.addItem(region)
                self._highlight_regions_spec.append(region)
            except Exception:
                pass
        # Водопад: такие же вертикальные полосы через LinearRegionItem поверх PlotItem
        for (f0, f1) in self._highlight_ranges_mhz:
            try:
                region_w = pg.LinearRegionItem(values=(f0, f1), orientation=pg.LinearRegionItem.Vertical,
                                               brush=pg.mkBrush(255, 0, 0, 50), pen=pg.mkPen(255, 60, 60, 160))
                region_w.setMovable(False)
                region_w.setZValue(10)
                self.water_plot.addItem(region_w)
                self._highlight_regions_water.append(region_w)
            except Exception:
                pass

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
        self.chk_smooth.toggled.connect(self._on_smoothing_toggled)
        self.chk_ema.toggled.connect(self._save_config)
        self.chk_ema.toggled.connect(self._on_ema_toggled)
        self.smooth_win.valueChanged.connect(self._ensure_odd_window)
        self.smooth_win.valueChanged.connect(self._save_config)
        self.alpha.valueChanged.connect(self._save_config)
        
        # Подключения для проброса настроек в бэкенд
        self.chk_smooth.toggled.connect(lambda _: self._apply_backend_processing())
        self.smooth_win.valueChanged.connect(lambda _: self._apply_backend_processing())
        self.chk_ema.toggled.connect(lambda _: self._apply_backend_processing())
        self.alpha.valueChanged.connect(lambda _: self._apply_backend_processing())
        
        # (убрано) переключатель анимации линий
        
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

    def _on_smoothing_toggled(self):
        """Обработчик переключения сглаживания по частоте."""
        self.smooth_win.setEnabled(self.chk_smooth.isChecked() and not self._running)

    def _on_ema_toggled(self):
        """Обработчик переключения EMA по времени."""
        self.alpha.setEnabled(self.chk_ema.isChecked() and not self._running)

    def _apply_backend_processing(self):
        """Проброс настроек сглаживания и EMA в бэкенд HackRF."""
        try:
            if isinstance(self._source, MasterSourceAdapter):
                self._source.set_freq_smoothing(self.chk_smooth.isChecked(), int(self.smooth_win.value()))
                self._source.set_ema_alpha(float(self.alpha.value()) if self.chk_ema.isChecked() else 1.0)
        except Exception:
            pass

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

    def _save_config(self):
        """Сохраняет текущую конфигурацию."""
        # Обновляем конфигурацию
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
        
        # Сохраняем через SDRConfigManager
        if hasattr(self, 'config_manager'):
            self.config_manager.save()
        else:
            print("[SpectrumView] SDRConfigManager недоступен")

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
        """(Отключено) Раньше меняло разрешение при зуме."""
        return

    def _apply_auto_resolution(self):
        """(Отключено) Раньше вызывалось с дебаунсом."""
        return

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
        """Обработка полного свипа данных (логика как в прототипе для водопада/спектра)."""
        if freqs_hz is None or power_dbm is None:
            return
        if not isinstance(freqs_hz, np.ndarray):
            freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        if not isinstance(power_dbm, np.ndarray):
            power_dbm = np.asarray(power_dbm, dtype=np.float32)
        if freqs_hz.size == 0 or power_dbm.size == 0 or freqs_hz.size != power_dbm.size:
            return

        # Очистка NaN/inf и калибровка в допустимые пределы
        valid_mask = np.isfinite(power_dbm) & np.isfinite(freqs_hz)
        if not np.all(valid_mask):
            power_dbm = power_dbm.copy()
            power_dbm[~valid_mask] = -120.0

        # Приведение к пользовательскому диапазону (если задан)
        if self._current_cfg:
            freq_mask = (freqs_hz >= self._current_cfg.freq_start_hz) & (freqs_hz <= self._current_cfg.freq_end_hz)
            freqs_hz = freqs_hz[freq_mask]
            power_dbm = power_dbm[freq_mask]
            if freqs_hz.size == 0:
                return

        # Инициализируем сетку/водопад при первом приходе данных (с пониженным разрешением для воды)
        if self._model.freqs_hz.size == 0 or self._model.water is None:
            if self._current_cfg is not None:
                f0 = float(self._current_cfg.freq_start_hz)
                f1 = float(self._current_cfg.freq_end_hz)
            else:
                f0 = float(freqs_hz[0])
                f1 = float(freqs_hz[-1])
            # Для водопада используем ограниченное число колонок, независимое от bin входных данных
            target_cols = int(max(512, min(4096, self._wf_target_cols)))
            water_bin_hz = max(1.0, (f1 - f0) / max(1, target_cols - 1))
            self._model.set_grid(f0, f1, water_bin_hz)

            # Предрасчёт индексов для даунсэмпла строки воды
            n_src = int(freqs_hz.size)
            if n_src > target_cols:
                self._water_idx_cache = (np.linspace(0, n_src - 1, num=target_cols).astype(np.int64))
            else:
                self._water_idx_cache = None  # прямое копирование

            # Выставляем оси строго по пользовательским границам
            cfg0 = float(self._current_cfg.freq_start_hz) * 1e-6 if self._current_cfg else float(f0) * 1e-6
            cfg1 = float(self._current_cfg.freq_end_hz) * 1e-6 if self._current_cfg else float(f1) * 1e-6
            if cfg1 <= cfg0:
                cfg0, cfg1 = float(f0) * 1e-6, float(f1) * 1e-6
            vb = self.plot.getViewBox()
            vb.enableAutoRange(x=False, y=False)
            vb.setLimits(xMin=50.0, xMax=6000.0, minXRange=1e-3, maxXRange=6000.0-50.0)
            self.water_plot.setLimits(xMin=50.0, xMax=6000.0, minXRange=1e-3, maxXRange=6000.0-50.0)
            self.plot.setXRange(cfg0, cfg1, padding=0)
            self.plot.setYRange(-110.0, -20.0, padding=0)
            self.water_plot.setXRange(cfg0, cfg1, padding=0)
            self.water_plot.setYRange(0, self._model.rows, padding=0)
            if self._model.water is not None:
                self.water_img.setImage(self._model.water, autoLevels=False, levels=self._wf_levels, lut=self._lut)
                x_mhz = self._model.freqs_hz.astype(np.float64) / 1e6
                self.water_img.setRect(QtCore.QRectF(float(x_mhz[0]), 0.0, float(x_mhz[-1] - x_mhz[0]), float(self._model.rows)))

        # Сохраняем ПОЛНЫЙ спектр для анализа/детектора и обновляем модель/водопад
        try:
            self.full_spectrum = (freqs_hz.astype(np.float64) * 1e-6, power_dbm.astype(np.float32, copy=False))
        except Exception:
            self.full_spectrum = (freqs_hz * 1e-6, power_dbm)

        # Обновление модели и водопада: кладём даунсэмпленную строку для воды
        try:
            if self._water_idx_cache is not None and self._model.freqs_hz.size > 0:
                row_ds = power_dbm[self._water_idx_cache]
            else:
                row_ds = power_dbm
            self._model.append_row(row_ds)
        except Exception:
            self._model.append_row(power_dbm)

        # Троттлинг UI: ограничиваем частоту тяжёлой перерисовки
        now_ts = time.time()
        do_plot = (self._last_plot_ts == 0.0) or ((now_ts - self._last_plot_ts) >= self._ui_min_interval_s)
        if do_plot:
            self._update_statistics()
            self._refresh_spectrum()
            self._last_plot_ts = now_ts
        # Передаём ПОЛНЫЙ спектр в координатор детектора, если подключен
        try:
            if self._orchestrator and self.full_spectrum is not None:
                fx, fy = self.full_spectrum
                # fx в МГц → требуются Гц для детектора
                freqs_hz_full = (fx * 1e6).astype(np.float64, copy=False)
                self._orchestrator.process_master_spectrum(freqs_hz_full, fy.astype(np.float32, copy=False))
        except Exception:
            pass
        self._pending_water_update = True
        if not self._update_timer.isActive():
            self._update_timer.start()

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
        
        # Ограничения и видимый диапазон: глобальные пределы остаются 50–6000,
        # но видим сразу пересечение данных с пользовательским диапазоном, если задан.
        global_min, global_max = 50.0, 6000.0
        vis0, vis1 = x0, x1
        if self._current_cfg is not None:
            cfg0 = float(self._current_cfg.freq_start_hz) * 1e-6
            cfg1 = float(self._current_cfg.freq_end_hz) * 1e-6
            vis0 = max(vis0, cfg0)
            vis1 = min(vis1, cfg1)
            if vis1 <= vis0:
                vis0, vis1 = x0, x1
        
        # Дополнительная проверка размера данных
        if x_mhz.size > 10000:  # Если слишком много точек, принудительно даунсемплируем
            self._wf_max_cols = 512
            waterfall_rows = 50
        
        self.plot.setLimits(xMin=global_min, xMax=global_max)
        self.water_plot.setLimits(xMin=global_min, xMax=global_max)
        
        self.plot.setXRange(vis0, vis1, padding=0.0)
        self.plot.setYRange(-110.0, -20.0, padding=0)
        self.water_plot.setXRange(vis0, vis1, padding=0.0)
        
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
            
        # Даунсемплируем каждую строку до фиксированной ширины водопада
        waterfall_ds = []
        for row in waterfall_data:
            # Ресемплинг индексацией до _wf_target_cols
            target = int(self._wf_target_cols)
            n = row.size
            if n <= target:
                # паддинг до целевой ширины для согласованной матрицы
                row_ds = np.pad(row, (0, target - n), mode='edge') if n < target else row
            else:
                step = float(n) / float(target)
                idx = (np.arange(target) * step).astype(np.int64)
                row_ds = row[idx]
            waterfall_ds.append(row_ds.astype(np.float32, copy=False))
        
        # Создаем матрицу водопада
        if waterfall_ds:
            waterfall_matrix = np.array(waterfall_ds, dtype=np.float32)
            
            # Клипируем значения для лучшего отображения
            waterfall_matrix = np.clip(waterfall_matrix, -120, -20)
            
            # Обновляем изображение
            self.water_img.setImage(waterfall_matrix, autoLevels=False)
            
            # Устанавливаем правильный прямоугольник водопада
            if self._model.freqs_hz is not None and self._model.freqs_hz.size > 0:
                x_mhz = self._model.freqs_hz / 1e6
                self.water_img.setRect(QtCore.QRectF(
                    float(x_mhz[0]), 0.0,
                    float(x_mhz[-1] - x_mhz[0]),
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
            
        # первый валидный свип — устанавливаем видимый диапазон, не меняя глобальные границы 50–6000 МГц
        if not hasattr(self, '_last_freqs_hz') or self._last_freqs_hz is None:
            vb = self.plot.getViewBox()
            try:
                vb.setXRange(freqs_hz[0]*1e-6, freqs_hz[-1]*1e-6, padding=0)
            except Exception:
                pass
        
        x_mhz = freqs_hz.astype(np.float64) / 1e6
        y = power_dbm.astype(np.float32, copy=False)
        
        # Применяем сглаживание
        if self.chk_smooth.isChecked():
            y = self._smooth_freq(y)
        if self.chk_ema.isChecked():
            y = self._smooth_time_ema(y)
        
        # Подавление зубчатого шума: вычитаем baseline, если выбран режим SNR
        try:
            if getattr(self, 'chk_snr_mode', None) and self.chk_snr_mode.isChecked():
                base = self._rolling_median(y, int(self.spin_base_win.value())) if getattr(self, 'spin_base_win', None) else self._rolling_median(y, 31)
                y = (y - base).astype(np.float32)
        except Exception:
            pass
        
        # Сохраняем полные данные и формируем даунсэмпл под ширину виджета
        x_mhz = freqs_hz.astype(np.float64) * 1e-6
        self.full_spectrum = (x_mhz, y.copy())
        max_pts = max(256, int(self.plot.width()))
        xd, yd = self._downsample_for_display(x_mhz, y, max_points=max_pts)
        # Доп. защита от случайной подачи Гц в x: если значения слишком большие, нормализуем в МГц
        try:
            if xd.size > 0 and float(np.max(xd)) > 10000.0:
                xd = (xd * 1e-6).astype(np.float64, copy=False)
        except Exception:
            pass

        # Плавная визуализация без таймера — экспоненциальное визуальное сглаживание
        try:
            self._last_plotted_points = int(yd.size)
            if not hasattr(self, '_smooth_display_y') or self._smooth_display_y is None or self._smooth_display_y.shape != yd.shape:
                self._smooth_display_y = yd.copy()
            alpha_viz = 0.28  # чуть плавнее визуально
            self._smooth_display_y = (1.0 - alpha_viz) * self._smooth_display_y + alpha_viz * yd
            self.curve_now.setData(xd, self._smooth_display_y.astype(np.float32, copy=False))
            # Включаем красивую заливку под текущей кривой относительно нижней границы графика
            self._apply_fill_under_curve()
        except Exception:
            self.curve_now.setData(xd, yd)
            self._apply_fill_under_curve()

        # EMA, min/max считаем на полных данных:
        if getattr(self, "_ema_last", None) is None:
            self._ema_last = power_dbm.astype("float32", copy=True)
        else:
            alpha = float(self.alpha.value()) if self.chk_ema.isChecked() else 1.0
            self._ema_last = (1.0 - alpha) * self._ema_last + alpha * power_dbm

        if self.chk_ema.isChecked():
            # Для средней линии применим такой же даунсэмпл под ширину экрана
            x_avg = freqs_hz.astype(np.float64) * 1e-6
            xa, ya = self._downsample_for_display(x_avg, self._ema_last, max_points=max_pts)
            self.curve_avg.setData(xa, ya)
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

    def _apply_fill_under_curve(self):
        """Настраивает заливку под текущей кривой с учётом текущего диапазона Y."""
        try:
            vb = self.plot.getViewBox()
            if vb is None:
                return
            y0, y1 = vb.viewRange()[1]
            # Базовая линия для заливки — нижняя граница видимого диапазона
            base = float(y0)
            brush = pg.mkBrush(255, 255, 255, 45)  # мягкая белая заливка
            self.curve_now.setFillLevel(base)
            self.curve_now.setBrush(brush)
        except Exception:
            pass

    def _update_fill_under_curve(self):
        """Обновляет базовый уровень заливки при изменении масштаба/диапазона."""
        self._apply_fill_under_curve()

    def _refresh_spectrum(self):
        """Обновление линий спектра."""
        if self._model.freqs_hz is None or self._model.power_dbm is None:
            return
        
        x_mhz = self._model.freqs_hz / 1e6
        
        # Адаптивное разрешение и даунсемплинг для линий
        span_hz = float((self._model.freqs_hz[-1] - self._model.freqs_hz[0])) if self._model.freqs_hz.size > 1 else 0.0
        target_cols = self._target_display_cols(span_hz)

        # Текущая линия
        if self.chk_now.isChecked():
            y = self._model.power_dbm
            # Применяем сглаживание если включено
            if self.chk_smooth.isChecked():
                y = self._smooth_freq(y)
            # Даунсемплинг линии
            xd, yd = self._downsample_line(x_mhz, y, target_cols)
            # Плавная визуализация без таймера — экспоненциальное сглаживание
            try:
                self._last_plotted_points = int(yd.size)
                if not hasattr(self, '_smooth_display_y') or self._smooth_display_y is None or self._smooth_display_y.shape != yd.shape:
                    self._smooth_display_y = yd.copy()
                alpha_viz = 0.35
                self._smooth_display_y = (1.0 - alpha_viz) * self._smooth_display_y + alpha_viz * yd
                self.curve_now.setData(xd, self._smooth_display_y.astype(np.float32, copy=False))
            except Exception:
                self.curve_now.setData(xd, yd)
        else:
            self.curve_now.setData([], [])
        
        # Средняя линия (исправлено: используем накопитель _avg_queue, актуализируем на каждом свипе)
        try:
            # Обновляем очередь усреднения сырыми (несглаженными) данными
            self._avg_queue.append(self._model.power_dbm.copy())
        except Exception:
            pass

        if self.chk_avg.isChecked() and len(self._avg_queue) > 0:
            avg_window = int(max(1, min(self.avg_win.value(), len(self._avg_queue))))
            recent_sweeps = list(self._avg_queue)[-avg_window:]
            avg_power = np.mean(recent_sweeps, axis=0).astype(np.float32, copy=False)
            if self.chk_smooth.isChecked():
                avg_power = self._smooth_freq(avg_power)
            xd, yd = self._downsample_line(x_mhz, avg_power, target_cols)
            self.curve_avg.setData(xd, yd)
        else:
            self.curve_avg.setData([], [])
        
        # Min hold
        if self.chk_min.isChecked():
            if self._minhold is None:
                self._minhold = self._model.power_dbm.copy()
            else:
                self._minhold = np.minimum(self._minhold, self._model.power_dbm)
            xd, yd = self._downsample_line(x_mhz, self._minhold, target_cols)
            self.curve_min.setData(xd, yd)
        else:
            self.curve_min.setData([], [])
        
        # Max hold
        if self.chk_max.isChecked():
            if self._maxhold is None:
                self._maxhold = self._model.power_dbm.copy()
            else:
                self._maxhold = np.maximum(self._maxhold, self._model.power_dbm)
            xd, yd = self._downsample_line(x_mhz, self._maxhold, target_cols)
            self.curve_max.setData(xd, yd)
        else:
            self.curve_max.setData([], [])
        
        self._update_cursor_label()

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

    def _downsample_for_display(self, x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Простой быстрый даунсэмплинг под ширину экрана (шаговая выборка)."""
        try:
            if x is None or y is None:
                return x, y
            n = int(min(len(x), len(y)))
            if n <= 0 or max_points <= 0 or n <= max_points:
                return x, y
            step = max(1, n // max_points)
            return x[::step], y[::step]
        except Exception:
            return x, y

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

    def _rolling_median(self, y: np.ndarray, window: int) -> np.ndarray:
        """Быстрая скользящая медиана (окно нечетное)."""
        try:
            import numpy as _np
            if window < 3:
                return y
            if (window % 2) == 0:
                window += 1
            pad = window // 2
            yp = np.pad(y, (pad, pad), mode='edge')
            # построим матрицу сдвигов и возьмем медиану по оси
            strides = (yp.strides[0], yp.strides[0])
            shape = (y.size, window)
            as_strided = np.lib.stride_tricks.as_strided
            M = as_strided(yp, shape=shape, strides=strides)
            return np.median(M, axis=1).astype(np.float32)
        except Exception:
            # запасной вариант: свертка по частям
            w = max(3, window | 1)
            out = y.copy()
            half = w // 2
            for i in range(y.size):
                l = max(0, i - half)
                r = min(y.size, i + half + 1)
                out[i] = float(np.median(y[l:r]))
            return out.astype(np.float32)

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
            span_hz = (self.plot.getViewBox().viewRange()[0][1] - self.plot.getViewBox().viewRange()[0][0]) * 1e6
            bin_width_khz = span_hz / max(1, self._last_plotted_points) / 1e3
            resolution_text = f"Разрешение: {bin_width_khz:.2f} кГц/точка"
        else:
            resolution_text = f"Разрешение: —"
        
        self._resolution_badge.setText(resolution_text)
        # Позиционируем бейдж в правом верхнем углу текущего видимого окна
        try:
            vb = self.plot.getViewBox()
            if vb is not None:
                (x0, x1), (y0, y1) = vb.viewRange()
                self._resolution_badge.setPos(x1, y1)
                # Плашка с числом точек
                self._points_badge.setText(f"Точек: {self._last_plotted_points}")
                self._points_badge.setPos(x1, y1 - 3)
                # Обновляем бейдж span
                span_mhz = max(0.0, x1 - x0)
                self._span_badge.setText(f"Span: {span_mhz:.1f} МГц")
                self._span_badge.setPos(x1, y0)
        except Exception:
            pass

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
        
        # Берем ровно то, что ввёл пользователь, только мягко клипим в рабочий диапазон 50–6000 МГц
        f0 = max(50.0, min(6000.0, float(self.start_mhz.value()))) * 1e6
        f1 = max(50.0, min(6000.0, float(self.stop_mhz.value()))) * 1e6
        bw = float(self.bin_khz.value() * 1e3)
        self.start_mhz.setValue(f0 * 1e-6)
        self.stop_mhz.setValue(f1 * 1e-6)

        # Отладка: текущий сканируемый диапазон
        try:
            print(f"[DEBUG] Scanning range: {self.start_mhz.value():.3f}-{self.stop_mhz.value():.3f} MHz")
        except Exception:
            pass
        
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
            # Используем ровно пользовательские значения без округления
            freq_start_hz=int(f0),
            freq_end_hz=int(f1),
            bin_hz=int(bw),
            lna_db=int(self.lna_db.value()),
            vga_db=int(self.vga_db.value()),
            amp_on=bool(self.amp_on.isChecked()),
            serial=master_serial
        )
        self._current_cfg = cfg
        
        # Инициализация модели с пользовательскими границами
        self._model.set_grid(cfg.freq_start_hz, cfg.freq_end_hz, cfg.bin_hz)

        # Немедленно выставляем видимый диапазон в соответствии с выбором пользователя
        try:
            x0 = float(cfg.freq_start_hz) * 1e-6
            x1 = float(cfg.freq_end_hz) * 1e-6
            if x1 > x0:
                vb = self.plot.getViewBox()
                vb.enableAutoRange(x=False, y=False)
                vb.setLimits(xMin=50.0, xMax=6000.0, minXRange=1e-3, maxXRange=6000.0-50.0)
                self.water_plot.setLimits(xMin=50.0, xMax=6000.0, minXRange=1e-3, maxXRange=6000.0-50.0)
                self.plot.setXRange(x0, x1, padding=0.0)
                self.water_plot.setXRange(x0, x1, padding=0.0)
        except Exception:
            pass
        
        # Сброс счетчиков
        self._sweep_count = 0
        self._last_full_ts = None
        self._dt_ema = None
        self._minhold = None
        self._maxhold = None
        self._avg_queue.clear()
        self._ema_last = None
        self._water_initialized = False
        
        # Параметры span/dwell задаются из диалога детектора/слейвов
        
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
        
        # Блокируем/разблокируем настройки сглаживания
        self._set_smoothing_controls_enabled(enabled)

    def _set_smoothing_controls_enabled(self, enabled: bool):
        """Блокирует/разблокирует настройки сглаживания."""
        # Настройки сглаживания по частоте
        self.chk_smooth.setEnabled(enabled)
        self.smooth_win.setEnabled(enabled and self.chk_smooth.isChecked())
        
        # Настройки EMA по времени
        self.chk_ema.setEnabled(enabled)
        self.alpha.setEnabled(enabled and self.chk_ema.isChecked())

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
        # Зафиксируем X в пределах 50–6000 МГц и не позволим выходить
        vb = self.plot.getViewBox()
        vb.enableAutoRange(x=False, y=False)
        vb.setLimits(xMin=50.0, xMax=6000.0, minXRange=1e-3, maxXRange=6000.0-50.0)
        self.plot.setXRange(50.0, 6000.0, padding=0)
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
        
        # Отрисовываем из основной матрицы модели, как в прототипе
        z = None
        if hasattr(self._model, 'water'):
            z = self._model.water
        if z is None:
            return
        
        # Включаем короткую анимацию «плавного опускания» — 5-6 кадров
        self.water_img.setImage(z, autoLevels=False, levels=self._wf_levels, lut=self._lut)
        self._wf_anim_i = 0
        if not self._wf_anim_active:
            self._wf_anim_active = True
            self._wf_anim_timer.start()
        # Обновляем прямоугольник по X-координате частот
        freqs = self._model.freqs_hz
        if freqs is not None and freqs.size:
            x_mhz = freqs.astype(np.float64) / 1e6
            self.water_img.setRect(QtCore.QRectF(float(x_mhz[0]), 0.0, float(x_mhz[-1] - x_mhz[0]), float(z.shape[0])))

    def _wf_anim_step(self):
        """Небольшая анимация уровня водопада для визуальной плавности."""
        try:
            self._wf_anim_i += 1
            if self._wf_anim_i >= self._wf_anim_steps:
                self._wf_anim_timer.stop()
                self._wf_anim_active = False
                return
            # Лёгкая интерполяция уровней
            vmin, vmax = self._wf_levels
            k = 0.12  # шаг сглаживания
            self.water_img.setLevels((vmin + k, vmax + k))
            self.water_img.setLevels((vmin, vmax))
        except Exception:
            self._wf_anim_timer.stop()
            self._wf_anim_active = False

    def _on_mouse_moved(self, pos):
        """Движение мыши."""
        vb = self.plot.getViewBox()
        if vb is None:
            return
        
        p = vb.mapSceneToView(pos)
        fx = float(p.x())
        fy = float(p.y())
        
        # Защита от ошибочного масштаба: принудительно работаем в МГц
        if fx > 60000.0:  # если приходят Гц (случайная ошибка), переводим в МГц
            fx = fx * 1e-6
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

    def _line_anim_step(self):
        """Кадр анимации плавного перехода основной линии спектра."""
        try:
            if self._anim_x is None or self._anim_prev_y is None or self._anim_target_y is None:
                self._line_anim_timer.stop()
                self._line_anim_active = False
                return
            self._line_anim_i += 1
            t = self._line_anim_i / float(max(1, self._line_anim_steps))
            # easeOutCubic
            ease = 1.0 - pow(1.0 - t, 3.0)
            y_now = (1.0 - ease) * self._anim_prev_y + ease * self._anim_target_y
            self.curve_now.setData(self._anim_x, y_now)
            if self._line_anim_i >= self._line_anim_steps:
                self._line_anim_timer.stop()
                self._line_anim_active = False
                self._anim_prev_y = self._anim_target_y.copy()
        except Exception:
            self._line_anim_timer.stop()
            self._line_anim_active = False

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
        
        # Клэмп частоты в глобальные пределы 50–6000 МГц
        try:
            new_freq = float(new_freq)
        except Exception:
            new_freq = 50.0
        if new_freq < 50.0:
            new_freq = 50.0
        elif new_freq > 6000.0:
            new_freq = 6000.0

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