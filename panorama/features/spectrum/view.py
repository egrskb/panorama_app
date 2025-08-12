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

        # Данные
        self._model = SpectrumModel(rows=400)
        self._asm = SweepAssembler(coverage_threshold=0.98)
        self._source: Optional[SourceBackend] = None
        self._current_cfg: Optional[SweepConfig] = None

        # Служебное для статистики свипов
        self._sweep_count = 0
        self._last_full_ts = None     # type: Optional[float]
        self._dt_ema = None           # type: Optional[float]

        # ---------- верхняя панель ----------
        top = QtWidgets.QHBoxLayout()
        self.start_mhz = QtWidgets.QDoubleSpinBox(); self._cfg_dsb(self.start_mhz, 50, 6000, 3, 50.000, " МГц")
        self.stop_mhz  = QtWidgets.QDoubleSpinBox(); self._cfg_dsb(self.stop_mhz,  50, 6000, 3, 6000.000, " МГц")
        self.bin_khz   = QtWidgets.QDoubleSpinBox(); self._cfg_dsb(self.bin_khz,   1, 5000,  0, 200,      " кГц")
        self.lna_db = QtWidgets.QSpinBox(); self.lna_db.setRange(0,40); self.lna_db.setSingleStep(8); self.lna_db.setValue(24)
        self.vga_db = QtWidgets.QSpinBox(); self.vga_db.setRange(0,62); self.vga_db.setSingleStep(2); self.vga_db.setValue(20)
        self.amp_on = QtWidgets.QCheckBox("AMP")
        self.btn_start = QtWidgets.QPushButton("Старт")
        self.btn_stop  = QtWidgets.QPushButton("Стоп"); self.btn_stop.setEnabled(False)
        self.btn_reset = QtWidgets.QPushButton("Сброс вида")

        def col(lbl, w):
            box = QtWidgets.QVBoxLayout(); box.addWidget(QtWidgets.QLabel(lbl)); box.addWidget(w); return box
        for w, lab in [
            (self.start_mhz, "F нач (МГц)"),
            (self.stop_mhz,  "F конец (МГц)"),
            (self.bin_khz,   "Bin (кГц)"),
            (self.lna_db, "LNA"), (self.vga_db, "VGA"),
        ]:
            top.addLayout(col(lab, w))

        top.addStretch(1)
        top.addWidget(self.btn_start); top.addWidget(self.btn_stop); top.addWidget(self.btn_reset)

        # ---------- спектр ----------
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Частота (МГц)")
        self.plot.setLabel("left", "Мощность (дБм)")
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        vb = self.plot.getViewBox()
        vb.enableAutoRange(x=False, y=False)
        vb.setMouseEnabled(x=True, y=False)  # фиксируем Y

        self.curve_now = self.plot.plot([], [], pen=pg.mkPen('#FFFFFF', width=1))
        self.curve_avg = self.plot.plot([], [], pen=pg.mkPen('#00AAFF', width=1))
        self.curve_min = self.plot.plot([], [], pen=pg.mkPen((120,120,120), width=1))
        self.curve_max = self.plot.plot([], [], pen=pg.mkPen('#FF8C00', width=1))

        # направляющие
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((60,60,60,160)))
        self._hline = pg.InfiniteLine(angle=0,  movable=False, pen=pg.mkPen((60,60,60,120)))
        self.plot.addItem(self._vline, ignoreBounds=True)
        self.plot.addItem(self._hline, ignoreBounds=True)

        self._cursor_lbl = QtWidgets.QLabel("— МГц, — дБм")
        self._cursor_lbl.setStyleSheet("color:#333;")

        # ---------- водопад ----------
        self.water_plot = pg.PlotItem()
        self.water_plot.setLabel("bottom", "Частота (МГц)")
        self.water_plot.setLabel("left", "Время →")
        self.water_plot.invertY(True)  # 0 сверху (последние строки вверху)
        self.water_plot.setMouseEnabled(x=True, y=False)
        self.water_img = pg.ImageItem(axisOrder="row-major")  # arr[row, col] = Y,X
        self.water_plot.addItem(self.water_img)
        self.water_plot.setXLink(self.plot)

        self._lut = get_colormap("qsa", 256)
        self._wf_levels = (-110.0, -20.0)

        # ---------- низ: статус свипов ----------
        bottom = QtWidgets.QHBoxLayout()
        self.lbl_sweep = QtWidgets.QLabel("Δt: — • Свипов: 0")
        self.lbl_sweep.setStyleSheet("color:#888;")
        bottom.addWidget(self.lbl_sweep); bottom.addStretch(1)

        # ---------- компоновка ----------
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self._cursor_lbl)
        lay.addWidget(self.plot, stretch=2)
        glw = pg.GraphicsLayoutWidget()
        glw.addItem(self.water_plot)
        lay.addWidget(glw, stretch=3)
        lay.addLayout(bottom)

        # обработчики
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_reset.clicked.connect(self._on_reset_view)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(lambda ev: self._on_add_marker(ev, self.plot))
        self.water_plot.scene().sigMouseClicked.connect(lambda ev: self._on_add_marker(ev, self.water_plot))

        # таймер «редких» апдейтов водопада
        self._update_timer = QtCore.QTimer(self); self._update_timer.setInterval(80)
        self._update_timer.timeout.connect(self._refresh_water)
        self._pending_water_update = False

        # накопители
        self._avg_queue: Deque[np.ndarray] = deque(maxlen=8)
        self._minhold: Optional[np.ndarray] = None
        self._maxhold: Optional[np.ndarray] = None
        self._ema_last: Optional[np.ndarray] = None

        # маркеры: id → (line, label)
        self._marker_seq = 0
        self._markers: Dict[int, Tuple[pg.InfiniteLine, pg.TextItem]] = {}

        # стартовый вид
        self._on_reset_view()
        self._apply_visibility(True, True, False, True)

    # ---------- вспом. ----------
    @staticmethod
    def _cfg_dsb(sp: QtWidgets.QDoubleSpinBox, a, b, dec, val, suffix=""):
        sp.setRange(a, b); sp.setDecimals(dec); sp.setValue(val)
        if suffix: sp.setSuffix(suffix)

    def _apply_visibility(self, now: bool, avg: bool, minv: bool, maxv: bool):
        self.curve_now.setVisible(now)
        self.curve_avg.setVisible(avg)
        self.curve_min.setVisible(minv)
        self.curve_max.setVisible(maxv)

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
        self._vline.setPos(float(f_hz) / 1e6)
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
        # отдаём в ассемблер; ждём только «полный» свип
        full, _ = self._asm.feed(sw)
        if full is None:
            return

        # статистика свипов
        t = time.time()
        if self._last_full_ts is not None:
            dt = t - self._last_full_ts
            self._dt_ema = dt if self._dt_ema is None else (0.3*dt + 0.7*self._dt_ema)
        self._last_full_ts = t
        self._sweep_count += 1
        if self._dt_ema is None:
            self.lbl_sweep.setText(f"Δt: — • Свипов: {self._sweep_count}")
        else:
            self.lbl_sweep.setText(f"Δt≈{self._dt_ema:.2f} c • Свипов: {self._sweep_count}")

        # обновляем модель и графики
        self._model.append_row(full)
        self._refresh_spectrum()
        self._pending_water_update = True
        if not self._update_timer.isActive():
            self._update_timer.start()
        self.newRowReady.emit(self._model.freqs_hz, self._model.last_row)

    # ---------- запуск/останов ----------
    def _on_start_clicked(self):
        if not self._source:
            QtWidgets.QMessageBox.information(self, "Источник", "Источник данных не подключён")
            return
        try:
            cfg = SweepConfig(
                freq_start_hz=int(round(self.start_mhz.value()*1e6)),
                freq_end_hz=int(round(self.stop_mhz.value()*1e6)),
                bin_hz=int(round(self.bin_khz.value()*1e3)),
                lna_db=int(self.lna_db.value()),
                vga_db=int(self.vga_db.value()),
                amp_on=bool(self.amp_on.isChecked()),
            )
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Параметры", "Неверные значения")
            return

        self._current_cfg = cfg
        self._model.set_grid(cfg.freq_start_hz, cfg.freq_end_hz, cfg.bin_hz)
        self._asm.configure(cfg.freq_start_hz, cfg.freq_end_hz, cfg.bin_hz, lut=None)
        self._asm.reset_pass()

        freqs = self._model.freqs_hz
        if freqs is not None and freqs.size:
            x_mhz = freqs.astype(np.float64) / 1e6
            self.plot.setXRange(float(x_mhz[0]), float(x_mhz[-1]), padding=0)
            self.plot.setYRange(-110.0, -20.0, padding=0)
            self.water_plot.setXRange(float(x_mhz[0]), float(x_mhz[-1]), padding=0)
            self.water_plot.setYRange(0, self._model.rows, padding=0)

            z = self._model.water
            if z is not None:
                # arr[row, col] — без transpose; rect: ширина по X, высота = rows
                self.water_img.setImage(z, autoLevels=False, levels=self._wf_levels, lut=self._lut)
                self.water_img.setRect(QtCore.QRectF(float(x_mhz[0]), 0.0,
                                                     float(x_mhz[-1] - x_mhz[0]),
                                                     float(z.shape[0])))

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
        # простое скользящее окно (по частоте)
        w = 5
        k = np.ones(w, np.float32) / float(w)
        return np.convolve(y, k, mode="same").astype(np.float32)

    def _refresh_spectrum(self):
        freqs = self._model.freqs_hz
        row = self._model.last_row
        if freqs is None or row is None or not freqs.size:
            return
        x_mhz = freqs.astype(np.float64) / 1e6
        y = row.astype(np.float32)

        y_now = self._smooth_freq(y)
        self._avg_queue.append(y.copy())
        y_avg = np.mean(self._avg_queue, axis=0).astype(np.float32) if self._avg_queue else None
        self._maxhold = y if self._maxhold is None else np.maximum(self._maxhold, y)
        self._minhold = y if self._minhold is None else np.minimum(self._minhold, y)

        self.curve_now.setData(x_mhz, y_now)
        if y_avg is not None: self.curve_avg.setData(x_mhz, y_avg)
        self.curve_min.setData(x_mhz, self._minhold)
        self.curve_max.setData(x_mhz, self._maxhold)
        self._update_cursor_label()

    def _refresh_water(self):
        if not self._pending_water_update:
            self._update_timer.stop(); return
        self._pending_water_update = False
        z = self._model.water
        if z is None: return
        self.water_img.setImage(z, autoLevels=False, levels=self._wf_levels, lut=self._lut)
        freqs = self._model.freqs_hz
        if freqs is not None and freqs.size:
            x_mhz = freqs.astype(np.float64) / 1e6
            self.water_img.setRect(QtCore.QRectF(float(x_mhz[0]), 0.0,
                                                 float(x_mhz[-1] - x_mhz[0]),
                                                 float(z.shape[0])))

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

    # ---------- маркеры ----------
    def _on_add_marker(self, ev, which_plot):
        if not ev.double() or ev.button() != QtCore.Qt.LeftButton:
            return
        vb = which_plot.getViewBox()
        p = vb.mapSceneToView(ev.scenePos())
        fx = float(p.x())
        self._add_named_marker(fx)

    def _add_named_marker(self, f_mhz: float, name: str | None = None):
        self._marker_seq += 1
        name = name or f"M{self._marker_seq}"
        line = pg.InfiniteLine(pos=f_mhz, angle=90, movable=True,
                               pen=pg.mkPen((255, 255, 0, 180), width=1.2))
        label = pg.TextItem(name, anchor=(0,1), color=(255,255,0))
        self.plot.addItem(line); self.plot.addItem(label)
        label.setPos(f_mhz, self.plot.getViewBox().viewRange()[1][1])  # привязка сверху

        # контекст-меню
        line.sigPositionChangeFinished.connect(lambda: label.setPos(line.value(), label.pos().y()))
        line.setZValue(100)
        line.mouseClickEvent = lambda ev: self._marker_menu(ev, line)

        self._markers[self._marker_seq] = (line, label)

    def _marker_menu(self, ev, line: pg.InfiniteLine):
        if ev.button() != QtCore.Qt.RightButton:
            return
        ev.accept()
        menu = QtWidgets.QMenu()
        act_rename = menu.addAction("Переименовать…")
        act_delete = menu.addAction("Удалить")
        chosen = menu.exec_(QtGui.QCursor.pos())
        if chosen == act_rename:
            text, ok = QtWidgets.QInputDialog.getText(self, "Маркер", "Название:")
            if ok and text:
                # найти его label и обновить
                for (l, lab) in self._markers.values():
                    if l is line:
                        lab.setText(str(text))
                        break
        elif chosen == act_delete:
            keys = [k for k, (l, _) in self._markers.items() if l is line]
            for k in keys:
                l, lab = self._markers.pop(k)
                try: self.plot.removeItem(l)
                except Exception: pass
                try: self.plot.removeItem(lab)
                except Exception: pass

    # ---------- настройки ----------
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
