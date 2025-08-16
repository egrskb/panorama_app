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
    configChanged = QtCore.pyqtSignal()  # Сигнал об изменении конфигурации
    
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
        self._running = False

        # --- верхняя панель параметров ---
        self.start_mhz = QtWidgets.QDoubleSpinBox()
        self._cfg_dsb(self.start_mhz, 50, 6000, 3, 2400.000, " МГц")
        self.stop_mhz = QtWidgets.QDoubleSpinBox()
        self._cfg_dsb(self.stop_mhz, 50, 6000, 3, 2483.500, " МГц")
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
        self.water_plot.invertY(False)  # Свежие сверху
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

        # Сглаживание
        grp_smooth = QtWidgets.QGroupBox("Сглаживание")
        gs = QtWidgets.QFormLayout(grp_smooth)
        
        self.chk_smooth = QtWidgets.QCheckBox("По частоте")
        self.chk_smooth.setChecked(False)
        self.smooth_win = QtWidgets.QSpinBox()
        self.smooth_win.setRange(3, 301)
        self.smooth_win.setSingleStep(2)
        self.smooth_win.setValue(5)
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

    def _on_cmap_changed(self, name: str):
        """Изменение цветовой палитры водопада."""
        self._lut_name = name
        self._lut = get_colormap(name, 256)
        # Применяем новую палитру к водопаду
        if self._model.water is not None:
            self.water_img.setLookupTable(self._lut)

    def _on_wf_levels(self):
        """Изменение уровней водопада."""
        self._wf_levels = (float(self.sp_wf_min.value()), float(self.sp_wf_max.value()))
        if self._model.water is not None:
            self.water_img.setLevels(self._wf_levels)

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


    def _on_stop_clicked(self):
        """Полная остановка с гарантией освобождения ресурсов."""
        if self._source and self._source.is_running():
            self._source.stop()
            # Ждем завершения
            timeout = time.time() + 2.0
            while self._source.is_running() and time.time() < timeout:
                QtCore.QCoreApplication.processEvents()
                time.sleep(0.01)
            
            # Разблокируем UI
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            
            # Обновляем статус
            self.status.emit("Остановлено")

    # ---------- ROI для детектора ----------
    def add_roi_region(self, start_mhz: float, stop_mhz: float):
        """Добавляет визуальный регион ROI на графики."""
        self.clear_roi_regions()
        
        brush = pg.mkBrush(255, 255, 0, 40)
        pen = pg.mkPen(255, 220, 0, 150, width=2)
        
        reg_spec = pg.LinearRegionItem(
            values=(start_mhz, stop_mhz),
            orientation=pg.LinearRegionItem.Vertical,
            brush=brush,
            pen=pen,
            movable=False
        )
        reg_spec.setZValue(-10)
        
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

    # ---------- Сглаживание ----------
    def _smooth_freq(self, y: np.ndarray) -> np.ndarray:
        """Сглаживание по частоте с корректной обработкой краев."""
        if not self.chk_smooth.isChecked() or y.size < 3:
            return y
        
        w = int(self.smooth_win.value())
        if w < 3:
            w = 3
        if w % 2 == 0:
            w += 1
        
        kernel = np.ones(w, dtype=np.float32) / float(w)
        smoothed = np.convolve(y, kernel, mode='same')
        
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

    # ---------- внешнее API ----------
    def set_source(self, src: SourceBackend):
        """Подключает источник данных."""
        if self._source is not None:
            for sig, slot in [
                (self._source.fullSweepReady, self._on_full_sweep),
                (self._source.status, self._on_status),
                (self._source.error, self._on_error),
                (self._source.started, self._on_started),
                (self._source.finished, self._on_finished),
            ]:
                try: 
                    sig.disconnect(slot)
                except Exception: 
                    pass
        
        self._source = src
        
        self._source.fullSweepReady.connect(self._on_full_sweep)
        self._source.status.connect(self._on_status)
        self._source.error.connect(self._on_error)
        self._source.started.connect(self._on_started)
        self._source.finished.connect(self._on_finished)

    def set_cursor_freq(self, f_hz: float):
        """Устанавливает позицию курсора."""
        self._vline.setPos(float(f_hz)/1e6)
        self._update_cursor_label()

    def get_current_row(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Возвращает текущую строку спектра."""
        return self._model.freqs_hz, self._model.last_row

    # ---------- события источника ----------
    def _on_started(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._avg_queue.clear()
        self._minhold = None
        self._maxhold = None
        self._ema_last = None
        self._sweep_count = 0
        self._last_full_ts = None
        self._dt_ema = None
        self.lbl_sweep.setText("Свипов: 0")

    def _on_finished(self, code: int):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_status(self, text: str): 
        pass

    def _on_error(self, text: str):
        QtWidgets.QMessageBox.warning(self, "Источник", text)

    def _on_full_sweep(self, freqs_hz: np.ndarray, power_dbm: np.ndarray):
        """Обработка полного прохода от источника."""
        # Проверяем размеры
        if freqs_hz.size != power_dbm.size:
            print(f"Warning: Size mismatch in full sweep - freqs: {freqs_hz.size}, power: {power_dbm.size}")
            return
        
        # Проверяем стабильность размеров
        if freqs_hz.size > 50000:
            print(f"Warning: Very large frequency array ({freqs_hz.size} points), this may cause instability")
        
        # Проверяем, изменились ли размеры данных
        if self._model.freqs_hz is None:
            # Первый запуск - инициализируем модель
            self._model.freqs_hz = freqs_hz
            self._model.last_row = power_dbm
            n = len(freqs_hz)
            self._model.water = np.full((self._model.rows, n), -120.0, dtype=np.float32)
            
            # Сбрасываем все при первом запуске
            self._minhold = None
            self._maxhold = None
            self._avg_queue.clear()
            self._ema_last = None
            
            print(f"Initialized spectrum model with {n} frequency points")
            print(f"Frequency range: {freqs_hz[0]/1e6:.1f} - {freqs_hz[-1]/1e6:.1f} MHz")
        elif len(self._model.freqs_hz) != len(freqs_hz):
            # Размеры изменились - нужно переинициализировать
            print(f"Frequency array size changed from {len(self._model.freqs_hz)} to {len(freqs_hz)}")
            
            # ВСЕГДА быстро сбрасываем водопад при изменении размеров
            # Перенос данных слишком медленный для больших массивов
            print(f"Resetting waterfall for size change: {len(self._model.freqs_hz)} → {len(freqs_hz)}")
            
            # Переинициализируем модель
            self._model.freqs_hz = freqs_hz
            self._model.last_row = power_dbm
            n = len(freqs_hz)
            self._model.water = np.full((self._model.rows, n), -120.0, dtype=np.float32)
            
            # Сбрасываем min/max hold при изменении размера данных
            self._minhold = None
            self._maxhold = None
            self._avg_queue.clear()
            self._ema_last = None
        else:
            # Размеры не изменились - добавляем строку в водопад
            self._model.append_row(power_dbm)
            
            # Логируем прогресс для отладки
            if self._sweep_count % 10 == 0:  # Каждые 10 свипов
                active_points = np.sum(power_dbm > -120)
                print(f"Sweep {self._sweep_count}: {active_points}/{len(power_dbm)} active points")
        
        t = time.time()
        if self._last_full_ts is not None:
            dt = t - self._last_full_ts
            self._dt_ema = dt if self._dt_ema is None else (0.3*dt + 0.7*self._dt_ema)
        self._last_full_ts = t
        self._sweep_count += 1
        
        fps_text = f" • {1.0/self._dt_ema:.1f} св/с" if self._dt_ema else ""
        self.lbl_sweep.setText(f"Свипов: {self._sweep_count}{fps_text}")
        
        self._refresh_spectrum()
        self._pending_water_update = True
        if not self._update_timer.isActive():
            self._update_timer.start()
        
        self.newRowReady.emit(freqs_hz, power_dbm)

    # ---------- запуск/останов ----------
    def _on_start_clicked(self):
        if not self._source:
            QtWidgets.QMessageBox.information(self, "Источник", "Источник данных не подключён")
            return
        
        f0, f1, bw = self._snap_bounds(
            self.start_mhz.value(), 
            self.stop_mhz.value(), 
            self.bin_khz.value() * 1e3
        )
        
        self.start_mhz.setValue(f0 / 1e6)
        self.stop_mhz.setValue(f1 / 1e6)
        
        cfg = SweepConfig(
            freq_start_hz=int(f0),
            freq_end_hz=int(f1),
            bin_hz=int(bw),
            lna_db=int(self.lna_db.value()),
            vga_db=int(self.vga_db.value()),
            amp_on=bool(self.amp_on.isChecked()),
        )
        
        self._current_cfg = cfg
        self._model.set_grid(cfg.freq_start_hz, cfg.freq_end_hz, cfg.bin_hz)
        
        # Сбрасываем min/max hold при изменении конфигурации
        self._minhold = None
        self._maxhold = None
        self._avg_queue.clear()
        self._ema_last = None
        
        freqs = self._model.freqs_hz
        if freqs is not None and freqs.size:
            x_mhz = freqs.astype(np.float64) / 1e6
            self.plot.setXRange(float(x_mhz[0]), float(x_mhz[-1]), padding=0)
            self.plot.setYRange(-110.0, -20.0, padding=0)
            self.water_plot.setXRange(float(x_mhz[0]), float(x_mhz[-1]), padding=0)
            self.water_plot.setYRange(0, self._model.rows, padding=0)
            
            z = self._model.water
            if z is not None:
                self.water_img.setImage(z, autoLevels=False, levels=self._wf_levels, lut=self._lut)
                self._update_water_rect(x_mhz, z)
        
        self._source.start(cfg)
        
        # Уведомляем об изменении конфигурации
        self.configChanged.emit()
        
        # Блокируем настройки во время работы
        self._set_controls_enabled(False)
        
        # Устанавливаем флаг запуска
        self._running = True

    def _on_stop_clicked(self):
        if self._source and self._source.is_running():
            self._source.stop()
            
        # Разблокируем настройки
        self._set_controls_enabled(True)
        
        # Сбрасываем флаг запуска
        self._running = False

    def _on_reset_view(self):
        self.plot.setXRange(self.start_mhz.value(), self.stop_mhz.value(), padding=0)
        self.plot.setYRange(-110.0, -20.0, padding=0)
        self.water_plot.setXRange(self.start_mhz.value(), self.stop_mhz.value(), padding=0)
        self.water_plot.setYRange(0, self._model.rows, padding=0)

    def _snap_bounds(self, f0_mhz: float, f1_mhz: float, bin_hz: float) -> Tuple[float, float, float]:
        """Привязывает границы к сетке для ровного покрытия."""
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

    # ---------- рисование ----------
    def _refresh_spectrum(self):
        """Обновляет линии спектра с исправленным сглаживанием."""
        freqs = self._model.freqs_hz
        row = self._model.last_row
        if freqs is None or row is None or not freqs.size:
            return
        
        x_mhz = freqs.astype(np.float64) / 1e6
        y = row.astype(np.float32)
        
        # Дополнительная проверка размеров
        if y.size != x_mhz.size:
            print(f"Warning: Size mismatch - x_mhz: {x_mhz.size}, y: {y.size}")
            return

        y_smoothed = self._smooth_freq(y)
        y_now = self._smooth_time_ema(y_smoothed)
        self.curve_now.setData(x_mhz, y_now)
        
        # Проверяем, что длина массива соответствует ожидаемой
        if y.size == x_mhz.size:
            self._avg_queue.append(y.copy())
            if len(self._avg_queue) > self.avg_win.value():
                while len(self._avg_queue) > self.avg_win.value():
                    self._avg_queue.popleft()
            
            if self._avg_queue:
                # Проверяем, что все массивы в очереди имеют одинаковую длину
                if all(arr.size == x_mhz.size for arr in self._avg_queue):
                    y_avg = np.mean(self._avg_queue, axis=0).astype(np.float32)
                    y_avg_smooth = self._smooth_freq(y_avg)
                    self.curve_avg.setData(x_mhz, y_avg_smooth)
        
        # Min/Max hold (без сглаживания для точности)
        if self._maxhold is None:
            self._maxhold = y.copy()
        elif self._maxhold.size == y.size:
            self._maxhold = np.maximum(self._maxhold, y)
        
        if self._minhold is None:
            self._minhold = y.copy()
        elif self._minhold.size == y.size:
            self._minhold = np.minimum(self._minhold, y)
        
        # Проверяем размеры перед отображением min/max hold
        if self._minhold is not None and self._minhold.size == x_mhz.size:
            self.curve_min.setData(x_mhz, self._minhold)
        if self._maxhold is not None and self._maxhold.size == x_mhz.size:
            self.curve_max.setData(x_mhz, self._maxhold)
        
        self._update_cursor_label()

    def _transfer_waterfall_data(self, old_freqs: np.ndarray, old_water: np.ndarray, new_freqs: np.ndarray):
        """Переносит данные водопада со старой сетки на новую."""
        try:
            if old_water is None or old_freqs is None:
                return
            
            old_rows, old_cols = old_water.shape
            new_cols = len(new_freqs)
            
            print(f"Transferring waterfall: {old_cols} → {new_cols} points, {old_rows} rows")
            
            # Создаем новую матрицу водопада
            new_water = np.full((old_rows, new_cols), -120.0, dtype=np.float32)
            
            # Для каждой строки переносим данные
            for row in range(min(old_rows, self._model.rows)):
                # Интерполируем данные со старой сетки на новую
                for new_idx, new_freq in enumerate(new_freqs):
                    # Ищем ближайшие точки на старой сетке
                    old_indices = np.searchsorted(old_freqs, new_freq)
                    
                    # Проверяем границы массива
                    if old_indices == 0:
                        # Меньше минимальной частоты
                        if old_cols > 0:
                            new_water[row, new_idx] = old_water[row, 0]
                    elif old_indices >= old_cols:
                        # Больше максимальной частоты
                        if old_cols > 0:
                            new_water[row, new_idx] = old_water[row, old_cols - 1]
                    else:
                        # Интерполируем между двумя точками
                        idx_low = max(0, old_indices - 1)
                        idx_high = min(old_cols - 1, old_indices)
                        
                        # Проверяем, что индексы в пределах массива
                        if 0 <= idx_low < old_cols and 0 <= idx_high < old_cols:
                            freq_low = old_freqs[idx_low]
                            freq_high = old_freqs[idx_high]
                            
                            if freq_high > freq_low:
                                # Линейная интерполяция
                                weight = (new_freq - freq_low) / (freq_high - freq_low)
                                new_water[row, new_idx] = (
                                    old_water[row, idx_low] * (1 - weight) + 
                                    old_water[row, idx_high] * weight
                                )
                            else:
                                new_water[row, new_idx] = old_water[row, idx_low]
                        else:
                            # Fallback - берем ближайшую точку
                            if 0 <= old_indices < old_cols:
                                new_water[row, new_idx] = old_water[row, old_indices]
            
            # Обновляем модель
            self._model.water = new_water
            
            print(f"✓ Waterfall data transferred successfully")
            
        except Exception as e:
            print(f"Error transferring waterfall data: {e}")
            # В случае ошибки просто создаем пустой водопад
            self._model.water = np.full((self._model.rows, len(new_freqs)), -120.0, dtype=np.float32)

    def _set_controls_enabled(self, enabled: bool):
        """Блокирует/разблокирует настройки во время работы."""
        self.start_mhz.setEnabled(enabled)
        self.stop_mhz.setEnabled(enabled)
        self.bin_khz.setEnabled(enabled)
        self.lna_db.setEnabled(enabled)
        self.vga_db.setEnabled(enabled)
        self.amp_on.setEnabled(enabled)
        
        # Кнопки
        self.btn_start.setEnabled(enabled)
        self.btn_stop.setEnabled(not enabled)

    def _update_water_rect(self, x_mhz: np.ndarray, z: np.ndarray):
        """Устанавливает правильные координаты для изображения водопада."""
        self.water_img.setRect(QtCore.QRectF(
            float(x_mhz[0]), 0.0,
            float(x_mhz[-1] - x_mhz[0]),
            float(z.shape[0])
        ))

    def _refresh_water(self):
        """Обновляет водопад."""
        if not self._pending_water_update:
            self._update_timer.stop()
            return
        
        self._pending_water_update = False
        z = self._model.water
        if z is None:
            return
        
        self.water_img.setImage(z, autoLevels=False, levels=self._wf_levels, lut=self._lut)
        
        freqs = self._model.freqs_hz
        if freqs is not None and freqs.size:
            self._update_water_rect(freqs.astype(np.float64)/1e6, z)

    # ---------- курсор ----------
    def _on_mouse_moved(self, pos):
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
        """Обновляет подпись курсора с частотой и уровнем."""
        freqs = self._model.freqs_hz
        row = self._model.last_row
        
        fx = float(self._vline.value())
        fy = float(self._hline.value())
        
        if freqs is None or row is None or row.size == 0:
            self._cursor_text.setText(f"{fx:.3f} МГц, {fy:.1f} дБм")
            self._cursor_text.setPos(fx, self.plot.getViewBox().viewRange()[1][1] - 5)
            return
        
        x_mhz = freqs.astype(np.float64) / 1e6
        
        # Интерполируем значение на позиции курсора
        i = int(np.clip(np.searchsorted(x_mhz, fx), 1, len(x_mhz)-1))
        x0, x1 = x_mhz[i-1], x_mhz[i]
        y0, y1 = row[i-1], row[i]
        
        if x1 == x0:
            y_at = float(y0)
        else:
            t = (fx - x0) / (x1 - x0)
            y_at = float((1.0 - t) * y0 + t * y1)
        
        self._cursor_text.setText(f"{fx:.3f} МГц, {y_at:.1f} дБм")
        self._cursor_text.setPos(fx, self.plot.getViewBox().viewRange()[1][1] - 5)

    # ---------- маркеры ----------
    def _on_add_marker(self, ev, from_water: bool):
        """Добавляет маркер по двойному клику."""
        if not ev.double() or ev.button() != QtCore.Qt.LeftButton:
            return
        
        vb = (self.water_plot if from_water else self.plot).getViewBox()
        p = vb.mapSceneToView(ev.scenePos())
        fx = float(p.x())
        
        # Диалог для имени
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Новый маркер",
            f"Частота: {fx:.6f} МГц\nНазвание:",
            QtWidgets.QLineEdit.Normal,
            f"M{self._marker_seq + 1}"
        )
        if not ok or not name:
            return
        
        self._add_marker(fx, name)

    def _add_marker(self, f_mhz: float, name: str, color: Optional[str] = None):
        """Добавляет именованный маркер."""
        self._marker_seq += 1
        mid = self._marker_seq
        
        if color is None:
            color = self._marker_colors[mid % len(self._marker_colors)]
        
        # Линии на обоих графиках
        line_spec = pg.InfiniteLine(
            pos=f_mhz, angle=90, movable=True,
            pen=pg.mkPen(color, width=2, style=QtCore.Qt.DashLine)
        )
        line_water = pg.InfiniteLine(
            pos=f_mhz, angle=90, movable=True,
            pen=pg.mkPen(color, width=1.5, style=QtCore.Qt.DashLine)
        )
        
        # Подпись
        label = pg.TextItem(name, anchor=(0.5, 1), color=color)
        label.setPos(f_mhz, self.plot.getViewBox().viewRange()[1][1])
        
        self.plot.addItem(line_spec)
        self.plot.addItem(label)
        self.water_plot.addItem(line_water)
        
        # Синхронизация движения
        line_spec.sigPositionChanged.connect(lambda: self._sync_marker(mid, line_spec.value(), "spec"))
        line_water.sigPositionChanged.connect(lambda: self._sync_marker(mid, line_water.value(), "water"))
        
        # Сохраняем
        self._markers[mid] = {
            "freq": f_mhz,
            "name": name,
            "color": color,
            "line_spec": line_spec,
            "line_water": line_water,
            "label": label
        }
        
        # Добавляем в список
        item = QtWidgets.QListWidgetItem(f"{name}: {f_mhz:.6f} МГц")
        item.setData(QtCore.Qt.UserRole, mid)
        item.setForeground(QtGui.QBrush(QtGui.QColor(color)))
        self.list_markers.addItem(item)

    def _sync_marker(self, mid: int, new_freq: float, source: str):
        """Синхронизирует положение маркера между графиками."""
        if mid not in self._markers:
            return
        
        m = self._markers[mid]
        m["freq"] = new_freq
        
        if source == "spec":
            m["line_water"].setValue(new_freq)
        else:
            m["line_spec"].setValue(new_freq)
        
        m["label"].setPos(new_freq, self.plot.getViewBox().viewRange()[1][1])
        
        # Обновляем текст в списке
        for i in range(self.list_markers.count()):
            item = self.list_markers.item(i)
            if item.data(QtCore.Qt.UserRole) == mid:
                item.setText(f"{m['name']}: {new_freq:.6f} МГц")
                break

    def _jump_to_marker(self, item: QtWidgets.QListWidgetItem):
        """Центрирует вид на маркере."""
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
        """Удаляет все маркеры."""
        for m in self._markers.values():
            try:
                self.plot.removeItem(m["line_spec"])
                self.plot.removeItem(m["label"])
                self.water_plot.removeItem(m["line_water"])
            except Exception:
                pass
        
        self._markers.clear()
        self.list_markers.clear()

    # ---------- настройки ----------
    def restore_settings(self, settings, defaults: dict):
        """Восстанавливает настройки из QSettings."""
        d = (defaults or {}).get("spectrum", {})
        settings.beginGroup("spectrum")
        try:
            self.start_mhz.setValue(float(settings.value("start_mhz", d.get("start_mhz", 2400.0))))
            self.stop_mhz.setValue(float(settings.value("stop_mhz", d.get("stop_mhz", 2483.5))))
            self.bin_khz.setValue(float(settings.value("bin_khz", d.get("bin_khz", 200.0))))
            self.lna_db.setValue(int(settings.value("lna_db", d.get("lna_db", 24))))
            self.vga_db.setValue(int(settings.value("vga_db", d.get("vga_db", 20))))
            self.amp_on.setChecked(str(settings.value("amp_on", d.get("amp_on", False))).lower() in ("1","true","yes"))
            
            # Настройки отображения
            self.chk_now.setChecked(settings.value("chk_now", True, type=bool))
            self.chk_avg.setChecked(settings.value("chk_avg", True, type=bool))
            self.chk_min.setChecked(settings.value("chk_min", False, type=bool))
            self.chk_max.setChecked(settings.value("chk_max", True, type=bool))
            self.avg_win.setValue(int(settings.value("avg_win", 8)))
            
            self.chk_smooth.setChecked(settings.value("chk_smooth", False, type=bool))
            self.smooth_win.setValue(int(settings.value("smooth_win", 5)))
            self.chk_ema.setChecked(settings.value("chk_ema", False, type=bool))
            self.alpha.setValue(float(settings.value("alpha", 0.3)))
            
            self._lut_name = settings.value("cmap", "turbo")
            self.cmb_cmap.setCurrentText(self._lut_name)
            self._lut = get_colormap(self._lut_name, 256)
            
            self.sp_wf_min.setValue(float(settings.value("wf_min", -110.0)))
            self.sp_wf_max.setValue(float(settings.value("wf_max", -20.0)))
            self._wf_levels = (self.sp_wf_min.value(), self.sp_wf_max.value())
            
            # Загружаем маркеры
            markers_json = settings.value("markers", "[]")
            try:
                markers = json.loads(markers_json)
                for m in markers:
                    self._add_marker(
                        float(m.get("freq", 0)),
                        str(m.get("name", "M")),
                        str(m.get("color", "#FFFFFF"))
                    )
            except Exception:
                pass
                
        finally:
            settings.endGroup()
        
        self._on_reset_view()
        self._apply_visibility()

    def save_settings(self, settings):
        """Сохраняет настройки в QSettings."""
        settings.beginGroup("spectrum")
        try:
            settings.setValue("start_mhz", float(self.start_mhz.value()))
            settings.setValue("stop_mhz", float(self.stop_mhz.value()))
            settings.setValue("bin_khz", float(self.bin_khz.value()))
            settings.setValue("lna_db", int(self.lna_db.value()))
            settings.setValue("vga_db", int(self.vga_db.value()))
            settings.setValue("amp_on", bool(self.amp_on.isChecked()))
            
            settings.setValue("chk_now", self.chk_now.isChecked())
            settings.setValue("chk_avg", self.chk_avg.isChecked())
            settings.setValue("chk_min", self.chk_min.isChecked())
            settings.setValue("chk_max", self.chk_max.isChecked())
            settings.setValue("avg_win", self.avg_win.value())
            
            settings.setValue("chk_smooth", self.chk_smooth.isChecked())
            settings.setValue("smooth_win", self.smooth_win.value())
            settings.setValue("chk_ema", self.chk_ema.isChecked())
            settings.setValue("alpha", self.alpha.value())
            
            settings.setValue("cmap", self._lut_name)
            settings.setValue("wf_min", self.sp_wf_min.value())
            settings.setValue("wf_max", self.sp_wf_max.value())
            
            # Сохраняем маркеры
            markers = []
            for m in self._markers.values():
                markers.append({
                    "freq": m["freq"],
                    "name": m["name"],
                    "color": m["color"]
                })
            settings.setValue("markers", json.dumps(markers))
            
        finally:
            settings.endGroup()

    # ---------- экспорт ----------
    def export_waterfall_png(self, path: str):
        """Экспортирует водопад в PNG."""
        from PyQt5.QtWidgets import QGraphicsScene
        from PyQt5.QtGui import QPixmap, QPainter
        from PyQt5.QtCore import QRectF
        
        # Создаем сцену с водопадом
        scene = QGraphicsScene()
        scene.addItem(self.water_plot)
        
        # Рендерим в pixmap
        rect = self.water_plot.boundingRect()
        pixmap = QPixmap(int(rect.width()), int(rect.height()))
        pixmap.fill(QtCore.Qt.white)
        
        painter = QPainter(pixmap)
        scene.render(painter, QRectF(pixmap.rect()), rect)
        painter.end()
        
        pixmap.save(path)

    def export_current_csv(self, path: str):
        """Экспортирует текущий свип в CSV."""
        freqs = self._model.freqs_hz
        row = self._model.last_row
        
        if freqs is None or row is None:
            raise ValueError("Нет данных для экспорта")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("freq_hz,freq_mhz,dbm\n")
            for fz, y in zip(freqs, row):
                f.write(f"{float(fz):.3f},{float(fz)/1e6:.6f},{float(y):.2f}\n")