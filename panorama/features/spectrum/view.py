from __future__ import annotations
from typing import Optional, Deque, Dict, Tuple
from collections import deque
import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui

from panorama.drivers.base import SweepConfig, SourceBackend
from panorama.features.spectrum.model import SpectrumModel
from panorama.features.spectrum.service import SweepAssembler
from panorama.shared.palettes import get_colormap


class SpectrumView(QtWidgets.QWidget):
    newRowReady = QtCore.pyqtSignal(object, object)  # (freqs_hz, row_dbm)

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- данные / сборка полного свипа ----
        self._model = SpectrumModel(rows=400)
        self._asm = SweepAssembler(coverage_threshold=0.995)  # ждём почти 100%
        self._source: Optional[SourceBackend] = None
        self._current_cfg: Optional[SweepConfig] = None

        # статистика свипов
        self._sweep_count = 0
        self._last_full_ts: Optional[float] = None
        self._dt_ema: Optional[float] = None

        # --- верхняя панель параметров (МГц/кГц) ---
        self.start_mhz = QtWidgets.QDoubleSpinBox(); self._cfg_dsb(self.start_mhz, 50, 6000, 3, 50.000, " МГц")
        self.stop_mhz  = QtWidgets.QDoubleSpinBox(); self._cfg_dsb(self.stop_mhz,  50, 6000, 3, 6000.000, " МГц")
        self.bin_khz   = QtWidgets.QDoubleSpinBox(); self._cfg_dsb(self.bin_khz,   1, 5000,  0, 200,      " кГц")
        self.lna_db = QtWidgets.QSpinBox(); self.lna_db.setRange(0,40); self.lna_db.setSingleStep(8); self.lna_db.setValue(24)
        self.vga_db = QtWidgets.QSpinBox(); self.vga_db.setRange(0,62); self.vga_db.setSingleStep(2); self.vga_db.setValue(20)
        self.amp_on = QtWidgets.QCheckBox("AMP")
        self.btn_start = QtWidgets.QPushButton("Старт")
        self.btn_stop  = QtWidgets.QPushButton("Стоп"); self.btn_stop.setEnabled(False)
        self.btn_reset = QtWidgets.QPushButton("Сброс вида")

        # раскладка шапки
        top = QtWidgets.QHBoxLayout()
        def col(lbl, w):
            v = QtWidgets.QVBoxLayout(); v.addWidget(QtWidgets.QLabel(lbl)); v.addWidget(w); return v
        for w, lab in [
            (self.start_mhz, "F нач (МГц)"),
            (self.stop_mhz,  "F конец (МГц)"),
            (self.bin_khz,   "Bin (кГц)"),
            (self.lna_db, "LNA"), (self.vga_db, "VGA"),
        ]:
            top.addLayout(col(lab, w))
        top.addWidget(self.amp_on); top.addStretch(1)
        top.addWidget(self.btn_start); top.addWidget(self.btn_stop); top.addWidget(self.btn_reset)

        # --------- левый блок: графики ----------
        # основной спектр
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Частота (МГц)")
        self.plot.setLabel("left", "Мощность (дБм)")
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        vb = self.plot.getViewBox()
        vb.enableAutoRange(x=False, y=False)
        vb.setMouseEnabled(x=True, y=False)  # фиксируем Y

        # линии
        self.curve_now = self.plot.plot([], [], pen=pg.mkPen('#FFFFFF', width=1))
        self.curve_avg = self.plot.plot([], [], pen=pg.mkPen('#00AAFF', width=1))
        self.curve_min = self.plot.plot([], [], pen=pg.mkPen((120,120,120), width=1))
        self.curve_max = self.plot.plot([], [], pen=pg.mkPen('#FF8C00', width=1))

        # направляющие
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((60,60,60,160)))
        self._hline = pg.InfiniteLine(angle=0,  movable=False, pen=pg.mkPen((60,60,60,120)))
        self.plot.addItem(self._vline, ignoreBounds=True)
        self.plot.addItem(self._hline, ignoreBounds=True)

        self._cursor_lbl = QtWidgets.QLabel("— МГц, — дБм"); self._cursor_lbl.setStyleSheet("color:#333;")

        # водопад
        self.water_plot = pg.PlotItem()
        self.water_plot.setLabel("bottom", "Частота (МГц)")
        self.water_plot.setLabel("left", "Время →")
        # свежие строки сверху (как просил) — значит инвертируем Y
        self.water_plot.invertY(True)
        self.water_plot.setMouseEnabled(x=True, y=False)
        self.water_img = pg.ImageItem(axisOrder="row-major")
        self.water_plot.addItem(self.water_img)
        # синхронизация X
        self.water_plot.setXLink(self.plot)

        # палитра и уровни
        self._lut_name = "qsa"
        self._lut = get_colormap(self._lut_name, 256)
        self._wf_levels = (-110.0, -20.0)

        # контейнер для графиков
        graphs = QtWidgets.QVBoxLayout()
        graphs.addLayout(top)
        graphs.addWidget(self._cursor_lbl)
        graphs.addWidget(self.plot, stretch=2)
        glw = pg.GraphicsLayoutWidget()
        glw.addItem(self.water_plot)
        graphs.addWidget(glw, stretch=3)
        graphs_w = QtWidgets.QWidget(); graphs_w.setLayout(graphs)

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

        # таймер «редких» апдейтов водопада
        self._update_timer = QtCore.QTimer(self); self._update_timer.setInterval(80)
        self._update_timer.timeout.connect(self._refresh_water)
        self._pending_water_update = False

        # накопители
        self._avg_queue: Deque[np.ndarray] = deque(maxlen=8)
        self._minhold: Optional[np.ndarray] = None
        self._maxhold: Optional[np.ndarray] = None
        self._ema_last: Optional[np.ndarray] = None

        # маркеры id -> (line_spec, line_water, label_spec)
        self._marker_seq = 0
        self._markers: Dict[int, Tuple[pg.InfiniteLine, pg.InfiniteLine, pg.TextItem]] = {}

        # стартовый вид
        self._on_reset_view()
        self._apply_visibility()  # выставит видимость линий

    # ----- правая панель -----
    def _build_right_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)

        # Линии
        grp_lines = QtWidgets.QGroupBox("Линии")
        gl = QtWidgets.QFormLayout(grp_lines)
        self.chk_now = QtWidgets.QCheckBox("Текущая"); self.chk_now.setChecked(True)
        self.chk_avg = QtWidgets.QCheckBox("Средняя"); self.chk_avg.setChecked(True)
        self.chk_min = QtWidgets.QCheckBox("Мин");     self.chk_min.setChecked(False)
        self.chk_max = QtWidgets.QCheckBox("Макс");    self.chk_max.setChecked(True)
        gl.addRow(self.chk_now); gl.addRow(self.chk_avg); gl.addRow(self.chk_min); gl.addRow(self.chk_max)
        for w in (self.chk_now, self.chk_avg, self.chk_min, self.chk_max):
            w.toggled.connect(self._apply_visibility)

        # Сглаживание
        grp_smooth = QtWidgets.QGroupBox("Сглаживание")
        gs = QtWidgets.QFormLayout(grp_smooth)
        self.chk_smooth = QtWidgets.QCheckBox("По частоте (окно)")
        self.smooth_win = QtWidgets.QSpinBox(); self.smooth_win.setRange(1, 301); self.smooth_win.setValue(5); self.smooth_win.setSingleStep(2)
        self.chk_ema = QtWidgets.QCheckBox("EMA по времени")
        self.alpha   = QtWidgets.QDoubleSpinBox(); self.alpha.setRange(0.00, 1.00); self.alpha.setSingleStep(0.05); self.alpha.setValue(0.50)
        gs.addRow(self.chk_smooth)
        gs.addRow("Окно:", self.smooth_win)
        gs.addRow(self.chk_ema)
        gs.addRow("α:", self.alpha)

        # Водопад
        grp_wf = QtWidgets.QGroupBox("Водопад")
        gw = QtWidgets.QFormLayout(grp_wf)
        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(["qsa", "turbo", "viridis", "inferno", "plasma"])
        self.cmb_cmap.setCurrentText(self._lut_name)
        self.sp_wf_min = QtWidgets.QDoubleSpinBox(); self.sp_wf_min.setRange(-200, 50); self.sp_wf_min.setValue(self._wf_levels[0]); self.sp_wf_min.setSuffix(" дБм")
        self.sp_wf_max = QtWidgets.QDoubleSpinBox(); self.sp_wf_max.setRange(-200, 50); self.sp_wf_max.setValue(self._wf_levels[1]); self.sp_wf_max.setSuffix(" дБм")
        self.chk_wf_topfresh = QtWidgets.QCheckBox("Свежие сверху"); self.chk_wf_topfresh.setChecked(True)
        self.chk_wf_auto = QtWidgets.QCheckBox("Авто уровни"); self.chk_wf_auto.setChecked(False)
        gw.addRow("Палитра:", self.cmb_cmap)
        gw.addRow("Мин:", self.sp_wf_min)
        gw.addRow("Макс:", self.sp_wf_max)
        gw.addRow(self.chk_wf_topfresh)
        gw.addRow(self.chk_wf_auto)

        self.cmb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        self.sp_wf_min.valueChanged.connect(self._on_wf_levels)
        self.sp_wf_max.valueChanged.connect(self._on_wf_levels)
        self.chk_wf_topfresh.toggled.connect(self._on_wf_invert)
        # auto-levels — просто включаем/выключаем autoLevels при обновлении кадра

        # Маркеры
        grp_mrk = QtWidgets.QGroupBox("Маркеры")
        gm = QtWidgets.QVBoxLayout(grp_mrk)
        btn_add = QtWidgets.QPushButton("Добавить по курсору")
        btn_clear = QtWidgets.QPushButton("Очистить все")
        gm.addWidget(btn_add); gm.addWidget(btn_clear)
        btn_add.clicked.connect(lambda: self._add_named_marker(float(self._vline.value())))
        btn_clear.clicked.connect(self._clear_markers)

        # Статус свипа
        self.lbl_sweep = QtWidgets.QLabel("Δt: — • Свипов: 0")
        self.lbl_sweep.setStyleSheet("color:#888;")

        # собрать
        lay.addWidget(grp_lines)
        lay.addWidget(grp_smooth)
        lay.addWidget(grp_wf)
        lay.addWidget(grp_mrk)
        lay.addStretch(1)
        lay.addWidget(self.lbl_sweep)
        return panel

    # ---------- сервис ----------
    @staticmethod
    def _cfg_dsb(sp: QtWidgets.QDoubleSpinBox, a, b, dec, val, suffix=""):
        sp.setRange(a, b); sp.setDecimals(dec); sp.setValue(val)
        if suffix: sp.setSuffix(suffix)

    def _apply_visibility(self):
        self.curve_now.setVisible(self.chk_now.isChecked())
        self.curve_avg.setVisible(self.chk_avg.isChecked())
        self.curve_min.setVisible(self.chk_min.isChecked())
        self.curve_max.setVisible(self.chk_max.isChecked())

    # ---------- внешнее API ----------
    def set_source(self, src: SourceBackend):
        if self._source is not None:
            for sig, slot in [
                (self._source.sweepLine, self._on_sweep_line),
                (self._source.status,    self._on_status),
                (self._source.error,     self._on_error),
                (self._source.started,   self._on_started),
                (self._source.finished,  self._on_finished),
            ]:
                try: sig.disconnect(slot)
                except Exception: pass
        self._source = src
        self._source.sweepLine.connect(self._on_sweep_line)
        self._source.status.connect(self._on_status)
        self._source.error.connect(self._on_error)
        self._source.started.connect(self._on_started)
        self._source.finished.connect(self._on_finished)

    def set_cursor_freq(self, f_hz: float):
        self._vline.setPos(float(f_hz)/1e6)
        self._update_cursor_label()

    # ---------- события источника ----------
    def _on_started(self):
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self._avg_queue.clear(); self._minhold=None; self._maxhold=None; self._ema_last=None
        self._sweep_count = 0; self._last_full_ts = None; self._dt_ema = None
        self.lbl_sweep.setText("Δt: — • Свипов: 0")

    def _on_finished(self, code: int):
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)

    def _on_status(self, text: str): pass
    def _on_error(self, text: str):
        QtWidgets.QMessageBox.warning(self, "Источник", text)

    def _on_sweep_line(self, sw):
        full, _ = self._asm.feed(sw)
        if full is None:
            return
        # статистика
        t = time.time()
        if self._last_full_ts is not None:
            dt = t - self._last_full_ts
            self._dt_ema = dt if self._dt_ema is None else (0.3*dt + 0.7*self._dt_ema)
        self._last_full_ts = t
        self._sweep_count += 1
        self.lbl_sweep.setText(f"Δt≈{self._dt_ema:.2f} c • Свипов: {self._sweep_count}" if self._dt_ema else f"Δt: — • Свипов: {self._sweep_count}")

        # модель и перерисовка
        self._model.append_row(full)
        self._refresh_spectrum()
        self._pending_water_update = True
        if not self._update_timer.isActive():
            self._update_timer.start()
        self.newRowReady.emit(self._model.freqs_hz, self._model.last_row)

    # ---------- запуск/останов ----------
    def _on_start_clicked(self):
        if not self._source:
            QtWidgets.QMessageBox.information(self, "Источник", "Источник данных не подключён"); return
        cfg = SweepConfig(
            freq_start_hz=int(round(self.start_mhz.value()*1e6)),
            freq_end_hz=int(round(self.stop_mhz.value()*1e6)),
            bin_hz=int(round(self.bin_khz.value()*1e3)),
            lna_db=int(self.lna_db.value()),
            vga_db=int(self.vga_db.value()),
            amp_on=bool(self.amp_on.isChecked()),
        )
        self._current_cfg = cfg
        self._model.set_grid(cfg.freq_start_hz, cfg.freq_end_hz, cfg.bin_hz)
        self._asm.configure(cfg.freq_start_hz, cfg.freq_end_hz, cfg.bin_hz, lut=None)
        self._asm.reset_pass()

        # оси и пустой водопад
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

    def _on_stop_clicked(self):
        if self._source and self._source.is_running():
            self._source.stop()

    def _on_reset_view(self):
        self.plot.setXRange(self.start_mhz.value(), self.stop_mhz.value(), padding=0)
        self.plot.setYRange(-110.0, -20.0, padding=0)
        self.water_plot.setXRange(self.start_mhz.value(), self.stop_mhz.value(), padding=0)
        self.water_plot.setYRange(0, self._model.rows, padding=0)

    # ---------- рисование ----------
    def _smooth_freq(self, y: np.ndarray) -> np.ndarray:
        if not self.chk_smooth.isChecked(): return y
        w = int(self.smooth_win.value())
        if w <= 1: return y
        k = np.ones(w, np.float32) / float(w)
        return np.convolve(y, k, mode="same").astype(np.float32)

    def _smooth_time_ema(self, y: np.ndarray) -> np.ndarray:
        if not self.chk_ema.isChecked(): return y
        a = float(self.alpha.value())
        if self._ema_last is None or self._ema_last.shape != y.shape:
            self._ema_last = y.copy(); return y
        self._ema_last = (a * y) + ((1.0 - a) * self._ema_last)
        return self._ema_last

    def _refresh_spectrum(self):
        freqs = self._model.freqs_hz
        row = self._model.last_row
        if freqs is None or row is None or not freqs.size:
            return
        x_mhz = freqs.astype(np.float64) / 1e6
        y = row.astype(np.float32)

        y_now = self._smooth_time_ema(self._smooth_freq(y))
        self._avg_queue.append(y.copy())
        y_avg = np.mean(self._avg_queue, axis=0).astype(np.float32) if self._avg_queue else None
        if y_avg is not None: y_avg = self._smooth_freq(y_avg)

        self._maxhold = y if self._maxhold is None else np.maximum(self._maxhold, y)
        self._minhold = y if self._minhold is None else np.minimum(self._minhold, y)

        self.curve_now.setData(x_mhz, y_now)
        if y_avg is not None: self.curve_avg.setData(x_mhz, y_avg)
        self.curve_min.setData(x_mhz, self._minhold)
        self.curve_max.setData(x_mhz, self._maxhold)
        self._update_cursor_label()

    def _update_water_rect(self, x_mhz: np.ndarray, z: np.ndarray):
        self.water_img.setRect(QtCore.QRectF(float(x_mhz[0]), 0.0,
                                             float(x_mhz[-1]-x_mhz[0]),
                                             float(z.shape[0])))

    def _refresh_water(self):
        if not self._pending_water_update:
            self._update_timer.stop(); return
        self._pending_water_update = False
        z = self._model.water
        if z is None: return
        levels = None if self.chk_wf_auto.isChecked() else (float(self.sp_wf_min.value()), float(self.sp_wf_max.value()))
        self.water_img.setImage(z, autoLevels=self.chk_wf_auto.isChecked(), levels=levels, lut=self._lut)
        freqs = self._model.freqs_hz
        if freqs is not None and freqs.size:
            self._update_water_rect(freqs.astype(np.float64)/1e6, z)

    # ---------- курсор ----------
    def _on_mouse_moved(self, pos):
        vb = self.plot.getViewBox()
        if vb is None: return
        p = vb.mapSceneToView(pos)
        fx = float(p.x()); fy = float(p.y())
        self._vline.setPos(fx); self._hline.setPos(fy)
        self._cursor_lbl.setText(f"{fx:.3f} МГц, {fy:.1f} дБм")

    def _update_cursor_label(self):
        freqs = self._model.freqs_hz; row = self._model.last_row
        if freqs is None or row is None or row.size == 0:
            self._cursor_lbl.setText("— МГц, — дБм"); return
        x_mhz = freqs.astype(np.float64) / 1e6
        fx = float(self._vline.value())
        i = int(np.clip(np.searchsorted(x_mhz, fx), 1, len(x_mhz)-1))
        x0, x1 = x_mhz[i-1], x_mhz[i]; y0, y1 = row[i-1], row[i]
        t = 0.0 if x1 == x0 else (fx - x0) / (x1 - x0)
        y_at = float((1.0 - t) * y0 + t * y1)
        self._cursor_lbl.setText(f"{fx:.3f} МГц, {y_at:.1f} дБм")

    # ---------- маркеры (связанные на 2 графиках) ----------
    def _on_add_marker(self, ev, from_water: bool):
        if not ev.double() or ev.button() != QtCore.Qt.LeftButton:
            return
        vb = (self.water_plot if from_water else self.plot).getViewBox()
        p = vb.mapSceneToView(ev.scenePos())
        fx = float(p.x())
        self._add_named_marker(fx)

    def _add_named_marker(self, f_mhz: float, name: str | None = None):
        self._marker_seq += 1
        name = name or f"M{self._marker_seq}"
        line_spec  = pg.InfiniteLine(pos=f_mhz, angle=90, movable=True, pen=pg.mkPen((255, 255, 0, 180), width=1.2))
        line_water = pg.InfiniteLine(pos=f_mhz, angle=90, movable=True, pen=pg.mkPen((255, 255, 0, 120), width=1.0))
        label = pg.TextItem(name, anchor=(0,1), color=(255,255,0))
        self.plot.addItem(line_spec); self.plot.addItem(label)
        self.water_plot.addItem(line_water)
        label.setPos(f_mhz, self.plot.getViewBox().viewRange()[1][1])
        line_spec.sigPositionChanged.connect(lambda: line_water.setPos(line_spec.value()))
        line_water.sigPositionChanged.connect(lambda: line_spec.setPos(line_water.value()))
        for ln in (line_spec, line_water):
            ln.setZValue(100)
        line_spec.mouseClickEvent = lambda ev: self._marker_menu(ev, line_spec, line_water, label)
        self._markers[self._marker_seq] = (line_spec, line_water, label)

    def _marker_menu(self, ev, line_spec: pg.InfiniteLine, line_water: pg.InfiniteLine, label: pg.TextItem):
        if ev.button() != QtCore.Qt.RightButton: return
        ev.accept()
        menu = QtWidgets.QMenu()
        act_rename = menu.addAction("Переименовать…")
        act_delete = menu.addAction("Удалить")
        chosen = menu.exec_(QtGui.QCursor.pos())
        if chosen == act_rename:
            text, ok = QtWidgets.QInputDialog.getText(self, "Маркер", "Название:")
            if ok and text: label.setText(str(text))
        elif chosen == act_delete:
            keys = [k for k, v in self._markers.items() if v[0] is line_spec]
            for k in keys:
                ls, lw, lb = self._markers.pop(k)
                for item, host in [(ls, self.plot), (lw, self.water_plot), (lb, self.plot)]:
                    try: host.removeItem(item)
                    except Exception: pass

    def _clear_markers(self):
        for k in list(self._markers.keys()):
            ls, lw, lb = self._markers.pop(k)
            for item, host in [(ls, self.plot), (lw, self.water_plot), (lb, self.plot)]:
                try: host.removeItem(item)
                except Exception: pass

    # ----- панель событий -----
    def _on_cmap_changed(self, name: str):
        self._lut_name = name
        self._lut = get_colormap(name, 256)

    def _on_wf_levels(self):
        self._wf_levels = (float(self.sp_wf_min.value()), float(self.sp_wf_max.value()))

    def _on_wf_invert(self, on: bool):
        self.water_plot.invertY(bool(on))

    # ---------- настройки persist ----------
    def restore_settings(self, settings, defaults: dict):
        d = (defaults or {}).get("spectrum", {})
        settings.beginGroup("spectrum")
        try:
            self.start_mhz.setValue(float(settings.value("start_mhz", d.get("start_mhz", 50.000))))
            self.stop_mhz.setValue(float(settings.value("stop_mhz",   d.get("stop_mhz",   6000.000))))
            self.bin_khz.setValue(float(settings.value("bin_khz",     d.get("bin_khz",   200.0))))
            self.lna_db.setValue(int(settings.value("lna_db", d.get("lna_db", 24))))
            self.vga_db.setValue(int(settings.value("vga_db", d.get("vga_db", 20))))
            self.amp_on.setChecked(str(settings.value("amp_on", d.get("amp_on", False))).lower() in ("1","true","yes"))
        finally:
            settings.endGroup()
        self._on_reset_view()

    def save_settings(self, settings):
        settings.beginGroup("spectrum")
        try:
            settings.setValue("start_mhz", float(self.start_mhz.value()))
            settings.setValue("stop_mhz",  float(self.stop_mhz.value()))
            settings.setValue("bin_khz",   float(self.bin_khz.value()))
            settings.setValue("lna_db", int(self.lna_db.value()))
            settings.setValue("vga_db", int(self.vga_db.value()))
            settings.setValue("amp_on", bool(self.amp_on.isChecked()))
        finally:
            settings.endGroup()

    # служебка для экспорта
    def get_current_row(self):
        return self._model.freqs_hz, self._model.last_row
