from __future__ import annotations
from typing import Optional, Deque, Dict, Tuple, Any
from collections import deque
import time, json, os

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui

from panorama.drivers.base import SweepConfig, SourceBackend
from panorama.features.spectrum.model import SpectrumModel
from panorama.shared.palettes import get_colormap


class SpectrumView(QtWidgets.QWidget):
    """Главный виджет спектра с водопадом, маркерами и полной функциональностью."""
    
    newRowReady = QtCore.pyqtSignal(object, object)  # (freqs_hz, row_dbm)
    rangeSelected = QtCore.pyqtSignal(float, float)  # (start_mhz, stop_mhz) для детектора
    
    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- данные ----
        self._model = SpectrumModel(rows=400)
        self._source: Optional[SourceBackend] = None
        self._current_cfg: Optional[SweepConfig] = None

        # статистика свипов
        self._sweep_count = 0
        self._last_full_ts: Optional[float] = None
        self._dt_ema: Optional[float] = None

        # --- верхняя панель параметров ---
        self.start_mhz = QtWidgets.QDoubleSpinBox()
        self._cfg_dsb(self.start_mhz, 50, 6000, 3, 50.000, " МГц")
        self.stop_mhz = QtWidgets.QDoubleSpinBox()
        self._cfg_dsb(self.stop_mhz, 50, 6000, 3, 6000.000, " МГц")
        self.bin_khz = QtWidgets.QDoubleSpinBox()
        self._cfg_dsb(self.bin_khz, 1, 5000, 0, 200, " кГц")
        self.lna_db = QtWidgets.QSpinBox()
        self.lna_db.setRange(0, 40)
        self.lna_db.setSingleStep(8)
        self.lna_db.setValue(24)
        self.vga_db = QtWidgets.QSpinBox()
        self.vga_db.setRange(0, 62)
        self.vga_db.setSingleStep(2)
        self.vga_db.setValue(20)
        self.amp_on = QtWidgets.QCheckBox("AMP")
        self.btn_start = QtWidgets.QPushButton("Старт")
        self.btn_stop = QtWidgets.QPushButton("Стоп")
        self.btn_stop.setEnabled(False)
        self.btn_reset = QtWidgets.QPushButton("Сброс вида")

        # раскладка верхней панели
        top = QtWidgets.QHBoxLayout()
        def col(lbl, w):
            v = QtWidgets.QVBoxLayout()
            v.addWidget(QtWidgets.QLabel(lbl))
            v.addWidget(w)
            return v
        for w, lab in [
            (self.start_mhz, "F нач (МГц)"),
            (self.stop_mhz, "F конец (МГц)"),
            (self.bin_khz, "Bin (кГц)"),
            (self.lna_db, "LNA"),
            (self.vga_db, "VGA"),
        ]:
            top.addLayout(col(lab, w))
        top.addWidget(self.amp_on)
        top.addStretch(1)
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_stop)
        top.addWidget(self.btn_reset)

        # --------- левый блок: графики ----------
        # основной спектр
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Частота (МГц)")
        self.plot.setLabel("left", "Мощность (дБм)")
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        vb = self.plot.getViewBox()
        vb.enableAutoRange(x=False, y=False)
        vb.setMouseEnabled(x=True, y=False)

        # линии спектра
        self.curve_now = self.plot.plot([], [], pen=pg.mkPen('#FFFFFF', width=1))
        self.curve_avg = self.plot.plot([], [], pen=pg.mkPen('#00FF00', width=1))
        self.curve_min = self.plot.plot([], [], pen=pg.mkPen((120, 120, 255), width=1))
        self.curve_max = self.plot.plot([], [], pen=pg.mkPen('#FFC800', width=1))

        # курсор
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((80, 80, 80, 120)))
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((80, 80, 80, 120)))
        self.plot.addItem(self._vline, ignoreBounds=True)
        self.plot.addItem(self._hline, ignoreBounds=True)
        
        # подсказка курсора
        self._cursor_text = pg.TextItem(color=pg.mkColor(255, 255, 255), anchor=(0, 1))
        self.plot.addItem(self._cursor_text)

        # водопад
        self.water_plot = pg.PlotItem()
        self.water_plot.setLabel("bottom", "Частота (МГц)")
        self.water_plot.setLabel("left", "Время →")
        # ВАЖНО: НЕ инвертируем Y - свежие строки будут СВЕРХУ
        self.water_plot.invertY(False)  # <-- Исправление: False вместо True
        self.water_plot.setMouseEnabled(x=True, y=False)
        self.water_img = pg.ImageItem(axisOrder="row-major")
        self.water_plot.addItem(self.water_img)
        self.water_plot.setXLink(self.plot)

        # палитра и уровни
        self._lut_name = "turbo"
        self._lut = get_colormap(self._lut_name, 256)
        self._wf_levels = (-110.0, -20.0)

        # контейнер для графиков
        graphs = QtWidgets.QVBoxLayout()
        graphs.addLayout(top)
        graphs.addWidget(self.plot, stretch=2)
        glw = pg.GraphicsLayoutWidget()
        glw.addItem(self.water_plot)
        graphs.addWidget(glw, stretch=3)
        graphs_w = QtWidgets.QWidget()
        graphs_w.setLayout(graphs)

        # --------- правый блок: панель настроек ----------
        self.panel = self._build_right_panel()

        # общий сплиттер
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        split.addWidget(graphs_w)
        split.addWidget(self.panel)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)

        # главная раскладка
        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(split)

        # обработчики
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_reset.clicked.connect(self._on_reset_view)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(lambda ev: self._on_add_marker(ev, from_water=False))
        self.water_plot.scene().sigMouseClicked.connect(lambda ev: self._on_add_marker(ev, from_water=True))

        # таймер обновления водопада
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setInterval(80)
        self._update_timer.timeout.connect(self._refresh_water)
        self._pending_water_update = False

        # накопители
        self._avg_queue: Deque[np.ndarray] = deque(maxlen=8)
        self._minhold: Optional[np.ndarray] = None
        self._maxhold: Optional[np.ndarray] = None
        self._ema_last: Optional[np.ndarray] = None

        # маркеры
        self._marker_seq = 0
        self._markers: Dict[int, Dict[str, Any]] = {}
        self._marker_colors = ["#FF5252", "#40C4FF", "#FFD740", "#69F0AE", "#B388FF",
                               "#FFAB40", "#18FFFF", "#FF6E40", "#64FFDA", "#EEFF41"]

        # ROI регионы для детектора
        self._roi_regions = []

        # стартовый вид
        self._on_reset_view()
        self._apply_visibility()

    # ----- правая панель -----
    def _build_right_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)

        # Линии
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
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Окно:"))
        h.addWidget(self.avg_win)
        gl.addRow(h)
        gl.addRow(self.chk_min)
        gl.addRow(self.chk_max)
        for w in (self.chk_now, self.chk_avg, self.chk_min, self.chk_max):
            w.toggled.connect(self._apply_visibility)

        # Сглаживание - ИСПРАВЛЕННОЕ
        grp_smooth = QtWidgets.QGroupBox("Сглаживание")
        gs = QtWidgets.QFormLayout(grp_smooth)
        
        self.chk_smooth = QtWidgets.QCheckBox("По частоте")
        self.chk_smooth.setChecked(False)
        self.smooth_win = QtWidgets.QSpinBox()
        self.smooth_win.setRange(3, 301)  # Минимум 3 для корректного сглаживания
        self.smooth_win.setSingleStep(2)
        self.smooth_win.setValue(5)
        
        # Убеждаемся что окно всегда нечетное
        self.smooth_win.valueChanged.connect(self._ensure_odd_window)
        
        self.chk_ema = QtWidgets.QCheckBox("EMA по времени")
        self.chk_ema.setChecked(False)
        self.alpha = QtWidgets.QDoubleSpinBox()
        self.alpha.setRange(0.01, 1.00)
        self.alpha.setSingleStep(0.05)
        self.alpha.setValue(0.30)
        
        gs.addRow(self.chk_smooth)
        gs.addRow("Окно:", self.smooth_win)
        gs.addRow(self.chk_ema)
        gs.addRow("α:", self.alpha)

        # Водопад
        grp_wf = QtWidgets.QGroupBox("Водопад")
        gw = QtWidgets.QFormLayout(grp_wf)
        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(["qsa", "turbo", "viridis", "inferno", "plasma", "magma", "gray"])
        self.cmb_cmap.setCurrentText(self._lut_name)
        self.sp_wf_min = QtWidgets.QDoubleSpinBox()
        self.sp_wf_min.setRange(-200, 50)
        self.sp_wf_min.setValue(self._wf_levels[0])
        self.sp_wf_min.setSuffix(" дБм")
        self.sp_wf_max = QtWidgets.QDoubleSpinBox()
        self.sp_wf_max.setRange(-200, 50)
        self.sp_wf_max.setValue(self._wf_levels[1])
        self.sp_wf_max.setSuffix(" дБм")
        self.btn_auto_levels = QtWidgets.QPushButton("Авто уровни")
        
        # Чекбокс для инверсии водопада
        self.chk_wf_invert = QtWidgets.QCheckBox("Инвертировать (свежие снизу)")
        self.chk_wf_invert.setChecked(False)
        
        gw.addRow("Палитра:", self.cmb_cmap)
        gw.addRow("Мин:", self.sp_wf_min)
        gw.addRow("Макс:", self.sp_wf_max)
        gw.addRow(self.chk_wf_invert)
        gw.addRow(self.btn_auto_levels)

        self.cmb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        self.sp_wf_min.valueChanged.connect(self._on_wf_levels)
        self.sp_wf_max.valueChanged.connect(self._on_wf_levels)
        self.btn_auto_levels.clicked.connect(self._auto_levels)
        self.chk_wf_invert.toggled.connect(self._on_wf_invert)

        # Маркеры
        grp_mrk = QtWidgets.QGroupBox("Маркеры")
        gm = QtWidgets.QVBoxLayout(grp_mrk)
        self.list_markers = QtWidgets.QListWidget()
        self.list_markers.setMaximumHeight(150)
        gm.addWidget(self.list_markers)
        btn_clear = QtWidgets.QPushButton("Очистить все")
        gm.addWidget(btn_clear)
        btn_clear.clicked.connect(self._clear_markers)
        self.list_markers.itemDoubleClicked.connect(self._jump_to_marker)

        # Статус
        self.lbl_sweep = QtWidgets.QLabel("Свипов: 0")

        # собрать
        lay.addWidget(grp_lines)
        lay.addWidget(grp_smooth)
        lay.addWidget(grp_wf)
        lay.addWidget(grp_mrk)
        lay.addStretch(1)
        lay.addWidget(self.lbl_sweep)
        return panel

    # ---------- сервис ----------
    def _ensure_odd_window(self, value):
        """Убеждаемся что окно сглаживания всегда нечетное."""
        if value % 2 == 0:
            self.smooth_win.setValue(value + 1)

    def _on_wf_invert(self, checked):
        """Инвертирует направление водопада."""
        self.water_plot.invertY(checked)

    @staticmethod
    def _cfg_dsb(sp: QtWidgets.QDoubleSpinBox, a, b, dec, val, suffix=""):
        sp.setRange(a, b)
        sp.setDecimals(dec)
        sp.setValue(val)
        if suffix:
            sp.setSuffix(suffix)

    def _apply_visibility(self):
        self.curve_now.setVisible(self.chk_now.isChecked())
        self.curve_avg.setVisible(self.chk_avg.isChecked())
        self.curve_min.setVisible(self.chk_min.isChecked())
        self.curve_max.setVisible(self.chk_max.isChecked())

    def _auto_levels(self):
        """Автоматическая настройка уровней водопада."""
        if self._model.water is None:
            return
        data = self._model.water[self._model.water > -200]
        if data.size == 0:
            return
        vmin = float(np.percentile(data, 5))
        vmax = float(np.percentile(data, 99))
        self.sp_wf_min.setValue(vmin)
        self.sp_wf_max.setValue(vmax)

    # ---------- ROI для детектора ----------
    def add_roi_region(self, start_mhz: float, stop_mhz: float):
        """Добавляет визуальный регион ROI на графики."""
        # Удаляем старые регионы
        self.clear_roi_regions()
        
        # Создаем новые регионы
        brush = pg.mkBrush(255, 255, 0, 40)  # Желтый с прозрачностью
        pen = pg.mkPen(255, 220, 0, 150, width=2)
        
        # Регион на спектре
        reg_spec = pg.LinearRegionItem(
            values=(start_mhz, stop_mhz),
            orientation=pg.LinearRegionItem.Vertical,
            brush=brush,
            pen=pen,
            movable=False
        )
        reg_spec.setZValue(-10)
        
        # Регион на водопаде
        reg_water = pg.LinearRegionItem(
            values=(start_mhz, stop_mhz),
            orientation=pg.LinearRegionItem.Vertical,
            brush=brush,
            pen=pen,
            movable=False
        )
        reg_water.setZValue(-10)
        
        self.plot.addItem(reg_spec)
        self.water_plot.addItem(reg_water)
        
        self._roi_regions.append((reg_spec, reg_water))

    def clear_roi_regions(self):
        """Удаляет все ROI регионы."""
        for reg_spec, reg_water in self._roi_regions:
            try:
                self.plot.removeItem(reg_spec)
                self.water_plot.removeItem(reg_water)
            except Exception:
                pass
        self._roi_regions.clear()

    # ---------- Сглаживание - ИСПРАВЛЕННОЕ ----------
    def _smooth_freq(self, y: np.ndarray) -> np.ndarray:
        """Сглаживание по частоте с корректной обработкой краев."""
        if not self.chk_smooth.isChecked() or y.size < 3:
            return y
        
        w = int(self.smooth_win.value())
        if w < 3:
            w = 3
        if w % 2 == 0:
            w += 1
        
        # Используем mode='edge' для корректной обработки краев
        kernel = np.ones(w, dtype=np.float32) / float(w)
        smoothed = np.convolve(y, kernel, mode='same')
        
        # Корректируем края вручную для избежания артефактов
        half_w = w // 2
        for i in range(half_w):
            # Левый край
            smoothed[i] = np.mean(y[0:i+half_w+1])
            # Правый край
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

    # ---------- остальной код без изменений ----------
    # [Здесь остается весь остальной код из SpectrumView без изменений]
    # Я не копирую его чтобы сэкономить место, но он должен быть там же
    
    def _refresh_spectrum(self):
        """Обновляет линии спектра с исправленным сглаживанием."""
        freqs = self._model.freqs_hz
        row = self._model.last_row
        if freqs is None or row is None or not freqs.size:
            return
        
        x_mhz = freqs.astype(np.float64) / 1e6
        y = row.astype(np.float32)

        # Применяем сглаживание
        y_smoothed = self._smooth_freq(y)
        y_now = self._smooth_time_ema(y_smoothed)
        self.curve_now.setData(x_mhz, y_now)
        
        # Среднее
        self._avg_queue.append(y.copy())
        if len(self._avg_queue) > self.avg_win.value():
            while len(self._avg_queue) > self.avg_win.value():
                self._avg_queue.popleft()
        
        if self._avg_queue:
            y_avg = np.mean(self._avg_queue, axis=0).astype(np.float32)
            y_avg_smooth = self._smooth_freq(y_avg)
            self.curve_avg.setData(x_mhz, y_avg_smooth)
        
        # Min/Max hold (без сглаживания для точности)
        if self._maxhold is None:
            self._maxhold = y.copy()
        else:
            self._maxhold = np.maximum(self._maxhold, y)
        
        if self._minhold is None:
            self._minhold = y.copy()
        else:
            self._minhold = np.minimum(self._minhold, y)
        
        self.curve_min.setData(x_mhz, self._minhold)
        self.curve_max.setData(x_mhz, self._maxhold)
        
        self._update_cursor_label()