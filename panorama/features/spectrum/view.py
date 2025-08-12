from __future__ import annotations
from typing import Optional, Deque, Tuple
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

from panorama.drivers.base import SweepConfig, SourceBackend
from panorama.features.spectrum.model import SpectrumModel
from panorama.features.spectrum.service import SweepAssembler
from panorama.shared.palettes import get_colormap


class SpectrumView(QtWidgets.QWidget):
    """
    Классический вид: поля в МГц/кГц, ось X в МГц, фиксированная вертикальная метка,
    «Сброс вида», линии текущая/средняя/мин/макс, водопад. Склейка через SweepAssembler.
    """
    newRowReady = QtCore.pyqtSignal(object, object)  # (freqs_hz, row_dbm)

    def __init__(self, parent=None):
        super().__init__(parent)

        # модель + ассемблер
        self._model = SpectrumModel(rows=300)
        self._asm = SweepAssembler(coverage_threshold=0.95)
        self._source: Optional[SourceBackend] = None
        self._current_cfg: Optional[SweepConfig] = None

        # ---- верхняя панель (МГц/кГц) ----
        top = QtWidgets.QHBoxLayout()
        self.start_mhz = QtWidgets.QDoubleSpinBox(); self.start_mhz.setRange(0, 10_000); self.start_mhz.setDecimals(3); self.start_mhz.setValue(50.000);  self.start_mhz.setSuffix(" МГц")
        self.stop_mhz  = QtWidgets.QDoubleSpinBox(); self.stop_mhz.setRange(0, 10_000); self.stop_mhz.setDecimals(3);  self.stop_mhz.setValue(6000.000); self.stop_mhz.setSuffix(" МГц")
        self.bin_khz   = QtWidgets.QDoubleSpinBox(); self.bin_khz.setRange(1, 5_000);   self.bin_khz.setDecimals(0);   self.bin_khz.setValue(200);       self.bin_khz.setSuffix(" кГц")

        self.lna_db = QtWidgets.QSpinBox(); self.lna_db.setRange(0,40); self.lna_db.setSingleStep(8); self.lna_db.setValue(24)
        self.vga_db = QtWidgets.QSpinBox(); self.vga_db.setRange(0,62); self.vga_db.setSingleStep(2); self.vga_db.setValue(20)
        self.amp_on = QtWidgets.QCheckBox("AMP")

        self.btn_start = QtWidgets.QPushButton("Старт")
        self.btn_stop  = QtWidgets.QPushButton("Стоп"); self.btn_stop.setEnabled(False)
        self.btn_reset = QtWidgets.QPushButton("Сброс вида")

        self.lbl_cov = QtWidgets.QLabel("Покрытие: —")
        self.lbl_src = QtWidgets.QLabel(""); self.lbl_src.setStyleSheet("color:#666;")

        # сглаживание
        self.chk_smooth = QtWidgets.QCheckBox("Smooth")
        self.smooth_win = QtWidgets.QSpinBox(); self.smooth_win.setRange(1, 999); self.smooth_win.setValue(5)
        self.chk_ema = QtWidgets.QCheckBox("EMA α")
        self.alpha   = QtWidgets.QDoubleSpinBox(); self.alpha.setRange(0.0, 1.0); self.alpha.setSingleStep(0.05); self.alpha.setValue(0.5)

        # фиксированная вертикальная метка (МГц)
        self.mark_mhz = QtWidgets.QDoubleSpinBox(); self.mark_mhz.setRange(0.0, 10_000.0); self.mark_mhz.setDecimals(3); self.mark_mhz.setValue(2400.000); self.mark_mhz.setSuffix(" МГц")

        # палитра
        self.cmap = QtWidgets.QComboBox(); self.cmap.addItems(["turbo","viridis","magma","gray","qsa"]); self.cmap.setCurrentText("turbo")

        def col(lbl, w):
            box = QtWidgets.QVBoxLayout(); box.addWidget(QtWidgets.QLabel(lbl)); box.addWidget(w); return box
        for w, lab in [
            (self.start_mhz, "F нач (МГц)"), (self.stop_mhz, "F конец (МГц)"), (self.bin_khz, "Bin (кГц)"),
            (self.lna_db, "LNA"), (self.vga_db, "VGA")
        ]:
            top.addLayout(col(lab, w))
        top.addLayout(col("Палитра", self.cmap))
        w1 = QtWidgets.QWidget(); r1 = QtWidgets.QHBoxLayout(w1); r1.setContentsMargins(0,0,0,0); r1.addWidget(self.chk_smooth); r1.addWidget(self.smooth_win)
        w2 = QtWidgets.QWidget(); r2 = QtWidgets.QHBoxLayout(w2); r2.setContentsMargins(0,0,0,0); r2.addWidget(self.chk_ema); r2.addWidget(self.alpha)
        top.addLayout(col("Сглаживание", w1)); top.addLayout(col("EMA", w2))
        top.addLayout(col("Метка (МГц)", self.mark_mhz))
        top.addStretch(1)
        top.addWidget(self.lbl_cov); top.addSpacing(8); top.addWidget(self.lbl_src); top.addSpacing(8)
        top.addWidget(self.btn_start); top.addWidget(self.btn_stop); top.addWidget(self.btn_reset)

        # ---- верхний график: ось X в МГц ----
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Частота (МГц)")
        self.plot.setLabel("left", "Уровень (дБм)")
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)

        self.curve_now = self.plot.plot([], [], pen=pg.mkPen('w', width=1))
        self.curve_avg = self.plot.plot([], [], pen=pg.mkPen((0,170,255), width=1))
        self.curve_min = self.plot.plot([], [], pen=pg.mkPen((120,120,120), width=1))
        self.curve_max = self.plot.plot([], [], pen=pg.mkPen((255,140,0), width=1))

        # фиксированные курсорные линии
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((60,60,60,160)))
        self._hline = pg.InfiniteLine(angle=0,  movable=False, pen=pg.mkPen((60,60,60,120)))
        self.plot.addItem(self._vline, ignoreBounds=True)
        self.plot.addItem(self._hline, ignoreBounds=True)

        self._cursor_lbl = QtWidgets.QLabel("Метка: — МГц, — дБм"); self._cursor_lbl.setStyleSheet("color:#333;")

        # ---- водопад (ось X тоже МГц) ----
        self.water_plot = pg.PlotItem()
        self.water_plot.setLabel("bottom", "Частота (МГц)")
        self.water_plot.setLabel("left", "Время →")
        self.water_plot.invertY(True); self.water_plot.setMouseEnabled(x=True, y=False)

        self.water_img = pg.ImageItem(axisOrder="row-major")
        self.water_plot.addItem(self.water_img)

        # раскладка
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self._cursor_lbl)
        lay.addWidget(self.plot, stretch=2)
        glw = pg.GraphicsLayoutWidget(); glw.addItem(self.water_plot)
        lay.addWidget(glw, stretch=3)

        # управление
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_reset.clicked.connect(self._on_reset_view)
        self.cmap.currentTextChanged.connect(self._on_cmap_changed)
        self.mark_mhz.valueChanged.connect(self._on_mark_changed)

        # LUT
        self._colormap_name = "turbo"
        self._lut = get_colormap(self._colormap_name, 256)

        # накопители
        self._avg_queue: Deque[np.ndarray] = deque(maxlen=8)
        self._minhold: Optional[np.ndarray] = None
        self._maxhold: Optional[np.ndarray] = None
        self._ema_last: Optional[np.ndarray] = None

        self._update_timer = QtCore.QTimer(self); self._update_timer.setInterval(80)
        self._update_timer.timeout.connect(self._refresh_water)
        self._pending_water_update = False

        # стартовый вид
        self._on_reset_view()

    # ---------- API ----------
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

    # совместимость со «Списком пиков»
    def set_cursor_freq(self, f_hz: float):
        try: f_mhz = float(f_hz) / 1e6
        except Exception: return
        self.mark_mhz.setValue(f_mhz)
        self._place_mark_line(); self._update_cursor_label()

    # ---------- события источника ----------
    def _on_started(self):
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.lbl_cov.setText("Покрытие: 0%"); self.lbl_src.setText("Источник: запущен")
        self._avg_queue.clear(); self._minhold = None; self._maxhold = None; self._ema_last = None

    def _on_finished(self, code: int):
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.lbl_src.setText(f"Источник: завершён (код {code})")

    def _on_status(self, text: str): self.lbl_src.setText(text)
    def _on_error(self, text: str):
        QtWidgets.QMessageBox.warning(self, "Источник", text)
        self.lbl_src.setText("Ошибка: " + text)

    def _on_sweep_line(self, sw):
        full, cov = self._asm.feed(sw)
        self.lbl_cov.setText(f"Покрытие: {int(cov*100)}%")
        if full is None:
            return
        self._model.append_row(full)
        self._refresh_spectrum()
        self._pending_water_update = True
        if not self._update_timer.isActive():
            self._update_timer.start()
        self.newRowReady.emit(self._model.freqs_hz, self._model.last_row)

    # ---------- UI handlers ----------
    def _on_start_clicked(self):
        if not self._source:
            QtWidgets.QMessageBox.information(self, "Источник", "Источник данных не подключён")
            return
        try:
            f0_hz  = int(round(self.start_mhz.value() * 1e6))
            f1_hz  = int(round(self.stop_mhz.value()  * 1e6))
            bin_hz = int(round(self.bin_khz.value()   * 1e3))
            cfg = SweepConfig(
                freq_start_hz=f0_hz, freq_end_hz=f1_hz, bin_hz=bin_hz,
                lna_db=int(self.lna_db.value()), vga_db=int(self.vga_db.value()),
                amp_on=bool(self.amp_on.isChecked()),
            )
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Параметры", "Неверные значения")
            return

        # сетка модели и ассемблер
        self._current_cfg = cfg
        self._model.set_grid(cfg.freq_start_hz, cfg.freq_end_hz, cfg.bin_hz)
        self._asm.configure(cfg.freq_start_hz, cfg.freq_end_hz, cfg.bin_hz, lut=None)
        self._asm.reset_pass()

        # оси/водопад (X в МГц)
        freqs = self._model.freqs_hz
        if freqs is not None and freqs.size:
            x_mhz = freqs.astype(np.float64) / 1e6
            self.plot.setXRange(float(x_mhz[0]), float(x_mhz[-1]), padding=0)
            z = self._model.water
            if z is not None:
                self.water_img.setImage(z, autoLevels=True, lut=self._lut)
                self.water_img.setRect(QtCore.QRectF(float(x_mhz[0]), 0.0, float(x_mhz[-1]-x_mhz[0]), float(z.shape[0])))

        self._place_mark_line()
        self._source.start(cfg)

    def _on_stop_clicked(self):
        if self._source and self._source.is_running():
            self._source.stop()

    def _on_reset_view(self):
        # X — из полей; Y — аккуратный дефолт
        self.plot.setXRange(self.start_mhz.value(), self.stop_mhz.value(), padding=0)
        self.plot.setYRange(-110.0, -20.0, padding=0)
        self.water_plot.setXRange(self.start_mhz.value(), self.stop_mhz.value(), padding=0)
        self.water_plot.setYRange(0, self._model.rows, padding=0)
        self._place_mark_line()

    def _on_cmap_changed(self, name: str):
        self._colormap_name = name
        self._lut = get_colormap(self._colormap_name, 256)
        self._pending_water_update = True
        if not self._update_timer.isActive():
            self._update_timer.start()

    def _on_mark_changed(self, _):
        self._place_mark_line(); self._update_cursor_label()

    # ---------- рисование ----------
    def _smooth_freq(self, y: np.ndarray) -> np.ndarray:
        if not self.chk_smooth.isChecked():
            return y
        w = int(self.smooth_win.value())
        if w <= 1:
            return y
        k = np.ones(w, np.float32) / float(w)
        return np.convolve(y, k, mode="same").astype(np.float32)

    def _smooth_time_ema(self, y: np.ndarray) -> np.ndarray:
        if not self.chk_ema.isChecked():
            return y
        a = float(self.alpha.value())
        if self._ema_last is None or self._ema_last.shape != y.shape:
            self._ema_last = y.copy(); return y
        self._ema_last = (a * y) + ((1.0 - a) * self._ema_last)
        return self._ema_last

    def _refresh_spectrum(self):
        freqs = self._model.freqs_hz
        row = self._model.last_row
        if freqs is None or row is None:
            return

        x_mhz = freqs.astype(np.float64) / 1e6
        y = row.astype(np.float32)

        y_now = self._smooth_freq(y)
        y_now = self._smooth_time_ema(y_now)

        # среднее по окну последних N свипов
        self._avg_queue.append(y.copy())
        while len(self._avg_queue) > self._avg_queue.maxlen:
            self._avg_queue.popleft()
        y_avg = np.mean(self._avg_queue, axis=0).astype(np.float32) if self._avg_queue else None
        if y_avg is not None:
            y_avg = self._smooth_freq(y_avg)

        # мин/макс удержание
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
        self.water_img.setImage(z, autoLevels=False, lut=self._lut)
        freqs = self._model.freqs_hz
        if freqs is not None and freqs.size:
            x_mhz = freqs.astype(np.float64) / 1e6
            self.water_img.setRect(QtCore.QRectF(float(x_mhz[0]), 0.0, float(x_mhz[-1]-x_mhz[0]), float(z.shape[0])))

    # ---------- курсор ----------
    def _place_mark_line(self):
        self._vline.setPos(float(self.mark_mhz.value()))

    def _update_cursor_label(self):
        freqs = self._model.freqs_hz; row = self._model.last_row
        if freqs is None or row is None or row.size == 0:
            self._cursor_lbl.setText("Метка: — МГц, — дБм"); return
        x_mhz = freqs.astype(np.float64) / 1e6
        f_mhz = float(self.mark_mhz.value())
        idx = int(np.clip(np.searchsorted(x_mhz, f_mhz), 1, len(x_mhz)-1))
        x0, x1 = x_mhz[idx-1], x_mhz[idx]; y0, y1 = row[idx-1], row[idx]
        t = 0.0 if x1 == x0 else (f_mhz - x0) / (x1 - x0)
        y_at = float((1.0 - t) * y0 + t * y1)
        self._cursor_lbl.setText(f"Метка: {f_mhz:.3f} МГц, {y_at:.1f} дБм")

    # ---------- вспомогательное ----------
    def get_current_row(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._model.freqs_hz, self._model.last_row

    def save_settings(self, settings):  # опционально
        settings.beginGroup("spectrum")
        settings.setValue("start_mhz", float(self.start_mhz.value()))
        settings.setValue("stop_mhz",  float(self.stop_mhz.value()))
        settings.setValue("bin_khz",   float(self.bin_khz.value()))
        settings.setValue("lna_db", int(self.lna_db.value())); settings.setValue("vga_db", int(self.vga_db.value()))
        settings.setValue("amp_on", bool(self.amp_on.isChecked()))
        settings.setValue("cmap", self._colormap_name)
        settings.setValue("smooth", self.chk_smooth.isChecked()); settings.setValue("smooth_win", int(self.smooth_win.value()))
        settings.setValue("ema", self.chk_ema.isChecked()); settings.setValue("alpha", float(self.alpha.value()))
        settings.setValue("mark_mhz", float(self.mark_mhz.value()))
        settings.endGroup()

    def restore_settings(self, settings, defaults: dict):
        d = defaults.get("spectrum", {})
        settings.beginGroup("spectrum")
        self.start_mhz.setValue(float(settings.value("start_mhz", d.get("start_mhz", 50.000))))
        self.stop_mhz.setValue(float(settings.value("stop_mhz",   d.get("stop_mhz",   6000.000))))
        self.bin_khz.setValue(float(settings.value("bin_khz",     d.get("bin_khz",   200.0))))
        self.lna_db.setValue(int(settings.value("lna_db", d.get("lna_db", 24))))
        self.vga_db.setValue(int(settings.value("vga_db", d.get("vga_db", 20))))
        self.amp_on.setChecked(bool(str(settings.value("amp_on", d.get("amp_on", False))).lower() in ("1","true","yes")))
        self._colormap_name = str(settings.value("cmap", d.get("cmap", "turbo"))); self._lut = get_colormap(self._colormap_name, 256)
        ci = self.cmap.findText(self._colormap_name)
        if ci >= 0: self.cmap.setCurrentIndex(ci)
        self.chk_smooth.setChecked(bool(str(settings.value("smooth", d.get("smooth", False))).lower() in ("1","true","yes")))
        self.smooth_win.setValue(int(settings.value("smooth_win", d.get("smooth_win", 5))))
        self.chk_ema.setChecked(bool(str(settings.value("ema", d.get("ema", False))).lower() in ("1","true","yes")))
        self.alpha.setValue(float(settings.value("alpha", d.get("alpha", 0.5))))
        self.mark_mhz.setValue(float(settings.value("mark_mhz", d.get("mark_mhz", 2400.000))))
        settings.endGroup()
