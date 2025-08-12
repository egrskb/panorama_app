from __future__ import annotations
from typing import Optional
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np

from panorama.drivers.base import SweepConfig, SourceBackend
from panorama.shared.palettes import get_colormap
from panorama.features.spectrum.model import SpectrumModel
from panorama.features.spectrum.service import SweepAssembler


class SpectrumView(QtWidgets.QWidget):
    """Вкладка «Спектр»: верх — линейный спектр, низ — водопад.
    Сборка строки — через SweepAssembler.
    """

    # наружу отдаём полную строку (для вкладки «Пики» и т.п.)
    newRowReady = QtCore.pyqtSignal(object, object)  # (freqs_hz, row_dbm)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source: Optional[SourceBackend] = None
        self._model = SpectrumModel(rows=300)
        self._asm = SweepAssembler(coverage_threshold=0.95)
        self._current_cfg: Optional[SweepConfig] = None

        # -------- верхняя панель (параметры) --------
        top = QtWidgets.QHBoxLayout()
        self.freq_start = QtWidgets.QLineEdit("2400000000")  # 2.4 ГГц
        self.freq_end   = QtWidgets.QLineEdit("2480000000")  # 2.48 ГГц
        self.bin_hz     = QtWidgets.QLineEdit("200000")      # 200 кГц
        self.lna_db     = QtWidgets.QSpinBox(); self.lna_db.setRange(0, 40); self.lna_db.setSingleStep(8); self.lna_db.setValue(24)
        self.vga_db     = QtWidgets.QSpinBox(); self.vga_db.setRange(0, 62); self.vga_db.setSingleStep(2); self.vga_db.setValue(20)
        self.amp_on     = QtWidgets.QCheckBox("AMP")
        self.btn_start  = QtWidgets.QPushButton("Старт")
        self.btn_stop   = QtWidgets.QPushButton("Стоп"); self.btn_stop.setEnabled(False)
        self.lbl_cov    = QtWidgets.QLabel("Покрытие: —")
        self.lbl_src    = QtWidgets.QLabel("")  # статус источника
        self.lbl_src.setStyleSheet("color:#666;")

        # палитра водопада
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(["turbo", "viridis", "magma", "gray"])
        self.cmap.setCurrentText("turbo")
        self.cmap.currentTextChanged.connect(self._on_cmap_changed)

        for w, lab in [
            (self.freq_start, "F нач (Гц)"),
            (self.freq_end,   "F конец (Гц)"),
            (self.bin_hz,     "Bin (Гц)"),
            (self.lna_db,     "LNA"),
            (self.vga_db,     "VGA"),
        ]:
            box = QtWidgets.QVBoxLayout()
            box.addWidget(QtWidgets.QLabel(lab))
            box.addWidget(w)
            top.addLayout(box)

        box_cmap = QtWidgets.QVBoxLayout()
        box_cmap.addWidget(QtWidgets.QLabel("Палитра"))
        box_cmap.addWidget(self.cmap)
        top.addLayout(box_cmap)

        top.addStretch(1)
        top.addWidget(self.lbl_cov)
        top.addSpacing(12)
        top.addWidget(self.lbl_src)
        top.addSpacing(12)
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_stop)

        # -------- графики --------
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Частота", units="Гц")
        self.plot.setLabel("left", "Уровень", units="дБм")
        self._curve = self.plot.plot([], [], pen=pg.mkPen(width=1))

        self._cursor_lbl = QtWidgets.QLabel("— МГц, — дБм")
        self._cursor_lbl.setStyleSheet("color:#333;")

        # водопад
        self.water_plot = pg.PlotItem()
        self.water_plot.setLabel("bottom", "Частота", units="Гц")
        self.water_plot.setLabel("left", "Время →", units="")
        self.water_img = pg.ImageItem()
        self.water_plot.addItem(self.water_img)
        self.water_plot.invertY(True)
        self.water_plot.setMouseEnabled(x=True, y=False)

        # курсор по спектру
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((50, 50, 50, 120)))
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((50, 50, 50, 120)))
        self.plot.addItem(self._vline, ignoreBounds=True)
        self.plot.addItem(self._hline, ignoreBounds=True)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # раскладка
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self._cursor_lbl)
        lay.addWidget(self.plot, stretch=2)
        self._water_view = pg.GraphicsLayoutWidget()
        self._water_view.addItem(self.water_plot)
        lay.addWidget(self._water_view, stretch=3)

        # управление
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

        # LUT для водопада
        self._colormap_name = "turbo"
        self._lut = get_colormap(self._colormap_name, 256)

        # «редкие» перерисовки водопада
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setInterval(80)
        self._update_timer.timeout.connect(self._refresh_water)
        self._pending_water_update = False

    # ---------- публичное API ----------
    def set_source(self, src: SourceBackend):
        if self._source is not None:
            try:
                self._source.sweepLine.disconnect(self._on_sweep_line)
                self._source.status.disconnect(self._on_status)
                self._source.error.disconnect(self._on_error)
                self._source.started.disconnect(self._on_started)
                self._source.finished.disconnect(self._on_finished)
            except Exception:
                pass

        self._source = src
        self._source.sweepLine.connect(self._on_sweep_line)
        self._source.status.connect(self._on_status)
        self._source.error.connect(self._on_error)
        self._source.started.connect(self._on_started)
        self._source.finished.connect(self._on_finished)

    # ---------- события источника ----------
    def _on_started(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_cov.setText("Покрытие: 0%")
        self.lbl_src.setText("Источник: запущен")

    def _on_finished(self, code: int):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_src.setText(f"Источник: завершён (код {code})")

    def _on_status(self, text: str):
        self.lbl_src.setText(text)

    def _on_error(self, text: str):
        QtWidgets.QMessageBox.warning(self, "Источник", text)
        self.lbl_src.setText("Ошибка: " + text)

    def _on_sweep_line(self, sw):
        # каждый сегмент кладём в ассемблер
        full, cov = self._asm.feed(sw)
        self.lbl_cov.setText(f"Покрытие: {int(cov*100)}%")
        if full is not None:
            # готова полная строка — обновляем модель и графики
            self._model.append_row(full)
            self._refresh_spectrum()
            self._pending_water_update = True
            if not self._update_timer.isActive():
                self._update_timer.start()
            # сообщаем наружу (для "Пиков")
            self.newRowReady.emit(self._model.freqs_hz, self._model.last_row)

    # ---------- UI handlers ----------
    def _on_start_clicked(self):
        if not self._source:
            QtWidgets.QMessageBox.information(self, "Источник", "Источник данных не подключён")
            return
        try:
            cfg = SweepConfig(
                freq_start_hz=int(self.freq_start.text()),
                freq_end_hz=int(self.freq_end.text()),
                bin_hz=int(self.bin_hz.text()),
                lna_db=int(self.lna_db.value()),
                vga_db=int(self.vga_db.value()),
                amp_on=bool(self.amp_on.isChecked()),
            )
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Параметры", "Неверные числовые значения")
            return

        # Настраиваем сетку модели и ассемблер
        self._current_cfg = cfg
        self._model.set_grid(cfg.freq_start_hz, cfg.freq_end_hz, cfg.bin_hz)
        self._asm.configure(cfg.freq_start_hz, cfg.freq_end_hz, cfg.bin_hz, lut=None)
        self._asm.reset_pass()

        # Обновим оси сразу и подготовим водопад
        freqs = self._model.freqs_hz
        if freqs is not None and freqs.size:
            self.plot.setXRange(float(freqs[0]), float(freqs[-1]), padding=0.01)
            z = self._model.water
            if z is not None:
                # Сначала ставим изображение, затем rect (иначе нет width/height)
                self.water_img.setImage(z.T, autoLevels=True, lut=self._lut)
                self.water_img.setRect(QtCore.QRectF(
                    float(freqs[0]),
                    0.0,
                    float(freqs[-1] - freqs[0]),
                    float(z.shape[0])
                ))

        self.lbl_src.setText("Запуск источника…")
        self._source.start(cfg)

    def _on_stop_clicked(self):
        if self._source and self._source.is_running():
            self._source.stop()

    # ---------- отрисовка ----------
    def _refresh_spectrum(self):
        freqs = self._model.freqs_hz
        y = self._model.last_row
        if freqs is None or y is None:
            return
        self._curve.setData(freqs, y)

    def _refresh_water(self):
        if not self._pending_water_update:
            self._update_timer.stop()
            return
        self._pending_water_update = False
        z = self._model.water
        if z is None:
            return
        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z))
        self.water_img.setImage(z.T, autoLevels=False, levels=(zmin, zmax), lut=self._lut)

        # держим rect в актуальном состоянии
        freqs = self._model.freqs_hz
        if freqs is not None and freqs.size:
            self.water_img.setRect(QtCore.QRectF(
                float(freqs[0]),
                0.0,
                float(freqs[-1] - freqs[0]),
                float(z.shape[0])
            ))

    # ---------- курсор ----------
    def _on_mouse_moved(self, pos):
        vb = self.plot.getViewBox()
        if vb is None:
            return
        mouse_point = vb.mapSceneToView(pos)
        fx = mouse_point.x()
        fy = mouse_point.y()
        self._vline.setPos(fx)
        self._hline.setPos(fy)
        self._cursor_lbl.setText(f"{fx/1e6:.3f} МГц, {fy:.1f} дБм")

    def set_cursor_freq(self, f_hz: float):
        self._vline.setPos(float(f_hz))
        vb = self.plot.getViewBox()
        if vb:
            x0, x1 = vb.viewRange()[0]
            if not (x0 <= f_hz <= x1):
                width = x1 - x0
                vb.setXRange(f_hz - width * 0.5, f_hz + width * 0.5, padding=0)

    # ---------- экспорт изображений / текущей строки ----------
    def get_current_row(self):
        """Возвращает (freqs_hz, row_dbm) или (None, None), если ещё нет данных."""
        return self._model.freqs_hz, self._model.last_row

    def save_plots(self, base_path: str):
        """
        Сохраняет PNG: <base>_spectrum.png и <base>_waterfall.png.
        """
        spec_path = f"{base_path}_spectrum.png"
        water_path = f"{base_path}_waterfall.png"
        self.plot.grab().save(spec_path)
        self._water_view.grab().save(water_path)
        return spec_path, water_path

    # ---------- настройки ----------
    def restore_settings(self, settings, defaults: dict):
        d = defaults.get("spectrum", {})
        settings.beginGroup("spectrum")

        def _v(key, fallback):
            return settings.value(key, fallback)

        # поля UI
        self.freq_start.setText(str(int(_v("freq_start_hz", d.get("freq_start_hz", 2_400_000_000)))))
        self.freq_end.setText(str(int(_v("freq_end_hz",   d.get("freq_end_hz",   2_480_000_000)))))
        self.bin_hz.setText(str(int(_v("bin_hz",         d.get("bin_hz",        200_000)))))
        self.lna_db.setValue(int(_v("lna_db", d.get("lna_db", 24))))
        self.vga_db.setValue(int(_v("vga_db", d.get("vga_db", 20))))
        amp = _v("amp_on", d.get("amp_on", False))
        self.amp_on.setChecked(str(amp).lower() in ("1", "true", "yes"))

        # покрытие и палитра
        cov = float(_v("coverage_threshold", d.get("coverage_threshold", 0.95)))
        self._asm.coverage_threshold = cov
        cmap = str(_v("colormap", d.get("colormap", "turbo")))
        self._colormap_name = cmap
        self._lut = get_colormap(self._colormap_name, 256)

        # синхронизируем combobox
        idx = self.cmap.findText(self._colormap_name)
        if idx >= 0:
            self.cmap.setCurrentIndex(idx)

        settings.endGroup()

    def save_settings(self, settings):
        settings.beginGroup("spectrum")
        settings.setValue("freq_start_hz", int(self.freq_start.text() or 0))
        settings.setValue("freq_end_hz",   int(self.freq_end.text() or 0))
        settings.setValue("bin_hz",        int(self.bin_hz.text() or 0))
        settings.setValue("lna_db",        int(self.lna_db.value()))
        settings.setValue("vga_db",        int(self.vga_db.value()))
        settings.setValue("amp_on",        bool(self.amp_on.isChecked()))
        settings.setValue("coverage_threshold", float(self._asm.coverage_threshold))
        settings.setValue("colormap", self._colormap_name)
        settings.endGroup()

    # ---------- палитра ----------
    def _on_cmap_changed(self, name: str):
        self._colormap_name = name
        self._lut = get_colormap(self._colormap_name, 256)
        self._pending_water_update = True
        if not self._update_timer.isActive():
            self._update_timer.start()
