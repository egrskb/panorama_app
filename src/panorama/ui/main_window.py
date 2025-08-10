import csv
import json
import random
import time
import collections

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QColorDialog
import numpy as np
import pyqtgraph as pg

from ..core.colormaps import get_colormap
from ..core.sdr import SweepWorker, LibWorker
from ..core import settings_store
from ..detection.peaks import find_peaks
from .peaks_dialog import PeaksDialog
from .trilateration_window import TrilaterationWindow
from .widgets.time_axis import TimeAxis


class _Settings:
    """Tiny proxy wrapping :mod:`settings_store` with QSettings-like API."""

    def value(self, key, default=None):
        return settings_store.get(key, default)

    def setValue(self, key, value):
        settings_store.update_field(key, value)


class MainWindow(QtWidgets.QMainWindow):
    """Main application window with spectrum/waterfall and map view."""

    MIN_MHZ = 50.0
    MAX_MHZ = 6000.0

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.settings = _Settings()
        self.setWindowTitle("ПАНОРАМА — версия 0.1 бета")

        center = QtWidgets.QWidget(self)
        self.setCentralWidget(center)
        root = QtWidgets.QHBoxLayout(center)

        self.markers = {}
        self._marker_id_seq = 0

        # ---- left side: controls + plots ----
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 1)

        ctrl = QtWidgets.QHBoxLayout()
        left.addLayout(ctrl)

        sdr_params = cfg.get("sdr", {}).get("params", {})
        defval = lambda k, d: self.settings.value(k, d)
        self.fStart = QtWidgets.QLineEdit(str(defval("f0", f"{sdr_params.get('f_start_mhz', 50.0):.3f}")))
        self.fStop = QtWidgets.QLineEdit(str(defval("f1", f"{sdr_params.get('f_stop_mhz', 6000.0):.3f}")))
        self.binHz = QtWidgets.QLineEdit(str(defval("bin", sdr_params.get('bin_hz', 2000000))))
        self.lnaBox = QtWidgets.QSpinBox(); self.lnaBox.setRange(0, 40); self.lnaBox.setSingleStep(8)
        self.lnaBox.setValue(int(defval("lna", sdr_params.get('lna_db', 24))))
        self.vgaBox = QtWidgets.QSpinBox(); self.vgaBox.setRange(0, 62); self.vgaBox.setSingleStep(2)
        self.vgaBox.setValue(int(defval("vga", sdr_params.get('vga_db', 20))))
        self.btnStart = QtWidgets.QPushButton("Старт")
        self.btnStop = QtWidgets.QPushButton("Стоп"); self.btnStop.setEnabled(False)
        self.btnReset = QtWidgets.QPushButton("Сброс вида")

        def add(w, l):
            lay = QtWidgets.QVBoxLayout(); lay.addWidget(QtWidgets.QLabel(l)); lay.addWidget(w); ctrl.addLayout(lay)

        add(self.fStart, "Начало (МГц)")
        add(self.fStop, "Конец (МГц)")
        add(self.binHz, "Шаг бина (Гц)")
        add(self.lnaBox, "LNA (дБ)")
        add(self.vgaBox, "VGA (дБ)")
        ctrl.addWidget(self.btnStart)
        ctrl.addWidget(self.btnStop)
        ctrl.addWidget(self.btnReset)

        self.fStart.editingFinished.connect(lambda: self._format_line_edit(self.fStart))
        self.fStop.editingFinished.connect(lambda: self._format_line_edit(self.fStop))

        # ---- spectrum plot ----
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Частота (МГц)")
        self.plot.setLabel("left", "Мощность (дБм)")
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.curve_now = self.plot.plot([], [], pen=pg.mkPen('w', width=1))
        self.curve_avg = self.plot.plot([], [], pen=pg.mkPen((0, 255, 0), width=1))
        self.curve_min = self.plot.plot([], [], pen=pg.mkPen((120, 120, 255), width=1))
        self.curve_max = self.plot.plot([], [], pen=pg.mkPen((255, 200, 0), width=1))
        left.addWidget(self.plot, 1)

        self._cursorText = pg.TextItem(color=pg.mkColor(0, 0, 0), anchor=(0, 1))
        self.plot.addItem(self._cursorText)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # ---- waterfall ----
        self.wfLayout = pg.GraphicsLayoutWidget()
        self.timeAxis = TimeAxis(orientation='left')
        self.wfPlot = self.wfLayout.addPlot(row=0, col=0, axisItems={'left': self.timeAxis})
        self.wfPlot.showGrid(x=True, y=False, alpha=0.12)
        self.wfPlot.setLabel("bottom", "Частота (МГц)")
        self.wfPlot.setLabel("left", "Время (с)")
        self.wfPlot.getAxis('bottom').enableAutoSIPrefix(False)
        self.wfImg = pg.ImageItem(axisOrder='row-major')
        self.wfImg.setAutoDownsample(True)
        self.wfPlot.addItem(self.wfImg)
        self.wfPlot.setXLink(self.plot)
        left.addWidget(self.wfLayout, 2)

        visual = cfg.get("visual", {})
        self.level_min = float(defval("lvl_min", visual.get('level_min_dbm', -110.0)))
        self.level_max = float(defval("lvl_max", visual.get('level_max_dbm', -20.0)))
        lut = get_colormap(defval("cmap", visual.get('colormap', 'qsa')))
        self.wfImg.setLookupTable(lut)
        self.wfImg.setLevels([self.level_min, self.level_max])

        self.wfRows = int(defval("wf_rows", visual.get('wf_rows', 600)))
        self.wfBuffer = None
        self._avg_queue = collections.deque(maxlen=int(defval("avg_win", 8)))
        self._avg_line = None
        self._minhold = None
        self._maxhold = None
        self._last_sweep_freqs = None
        self._last_sweep_power = None
        self._last_sweep_t = None
        self._dt_hist = collections.deque(maxlen=32)
        self._axes_ready = False

        # hotkeys
        for vb in (self.plot.getViewBox(), self.wfPlot.getViewBox()):
            vb.enableAutoRange(x=False, y=False)
            vb.setDefaultPadding(0.0)
            vb.setMouseEnabled(x=True, y=False)
        QtWidgets.QShortcut(QtGui.QKeySequence("+"), self, activated=lambda: self.zoom_x(0.6))
        QtWidgets.QShortcut(QtGui.QKeySequence("-"), self, activated=lambda: self.zoom_x(1.6))
        QtWidgets.QShortcut(QtGui.QKeySequence("R"), self, activated=self.reset_view)

        self.plot.scene().sigMouseClicked.connect(lambda ev: self._on_mouse_click_add_marker(ev, self.plot.getViewBox()))
        self.wfPlot.scene().sigMouseClicked.connect(lambda ev: self._on_mouse_click_add_marker(ev, self.wfPlot.getViewBox()))

        # ---- right panel ----
        right = QtWidgets.QVBoxLayout()
        root.addLayout(right)

        self.tabs = QtWidgets.QTabWidget()
        right.addWidget(self.tabs, 1)

        self.tabTools = QtWidgets.QWidget(); self.tabs.addTab(self.tabTools, "Инструменты")
        self.tabMap = pg.PlotWidget(); self.tabMap.showGrid(x=True, y=True, alpha=0.25)
        self.tabs.addTab(self.tabMap, "Карта")

        toolsLay = QtWidgets.QVBoxLayout(self.tabTools)
        self.btnPeaks = QtWidgets.QPushButton("Пики…")
        self.btnTrilat = QtWidgets.QPushButton("Начать трилатерацию")
        toolsLay.addWidget(self.btnPeaks)
        toolsLay.addWidget(self.btnTrilat)
        toolsLay.addWidget(QtWidgets.QLabel("Маркеры:"))
        self.listMarkers = QtWidgets.QListWidget(); toolsLay.addWidget(self.listMarkers, 1)
        btnRow = QtWidgets.QHBoxLayout(); toolsLay.addLayout(btnRow)
        self.btnColorPick = QtWidgets.QPushButton(); btnRow.addWidget(self.btnColorPick)
        self.edMarkerFreq = QtWidgets.QLineEdit(); self.edMarkerFreq.setPlaceholderText("МГц")
        btnRow.addWidget(self.edMarkerFreq)
        self.btnAddMarker = QtWidgets.QPushButton("Добавить"); btnRow.addWidget(self.btnAddMarker)
        self.btnDelMarker = QtWidgets.QPushButton("Удалить"); toolsLay.addWidget(self.btnDelMarker)

        # restore window
        geo = self.settings.value("win_geo", "")
        if geo:
            self.restoreGeometry(QtCore.QByteArray.fromHex(str(geo).encode()))
        state = self.settings.value("win_state", "")
        if state:
            self.restoreState(QtCore.QByteArray.fromHex(str(state).encode()))
        self._load_markers()

        self.btnStart.clicked.connect(self.on_start)
        self.btnStop.clicked.connect(self.on_stop)
        self.btnReset.clicked.connect(self.reset_view)
        self.btnPeaks.clicked.connect(self._open_peaks)
        self.btnTrilat.clicked.connect(self._open_trilat)
        self.btnColorPick.clicked.connect(self._pick_marker_color)
        self.btnAddMarker.clicked.connect(self._add_marker_from_ui)
        self.btnDelMarker.clicked.connect(self._delete_selected_markers)

        self._marker_color = QtGui.QColor(self.settings.value("marker_color", "#ff5252"))
        self._update_color_button()

        self.statusBar().showMessage("Готов")
        self.worker = None

    # ------------------------------------------------------------------
    # settings helpers
    def _format_line_edit(self, le: QtWidgets.QLineEdit):
        try:
            v = float(le.text().replace(",", "."))
        except Exception:
            v = self.MIN_MHZ
        v = max(self.MIN_MHZ, min(self.MAX_MHZ, v))
        le.blockSignals(True); le.setText(f"{v:.3f}"); le.blockSignals(False)

    # ------------------------------------------------------------------
    def closeEvent(self, e):
        self.settings.setValue("f0", self.fStart.text())
        self.settings.setValue("f1", self.fStop.text())
        self.settings.setValue("bin", self.binHz.text())
        self.settings.setValue("lna", self.lnaBox.value())
        self.settings.setValue("vga", self.vgaBox.value())
        self.settings.setValue("wf_rows", self.wfRows)
        self.settings.setValue("avg_win", self._avg_queue.maxlen)
        self.settings.setValue("cmap", self.settings.value("cmap", "qsa"))
        self.settings.setValue("lvl_min", self.level_min)
        self.settings.setValue("lvl_max", self.level_max)
        self.settings.setValue("markers_json", json.dumps([
            {"freq": m["freq"], "color": m["color"], "name": m.get("name", "")} for m in self.markers.values()
        ]))
        self.settings.setValue("marker_color", self._marker_color.name())
        self.settings.setValue("win_geo", self.saveGeometry().toHex().data().decode())
        self.settings.setValue("win_state", self.saveState().toHex().data().decode())
        super().closeEvent(e)

    def _get_next_marker_id(self):
        self._marker_id_seq += 1
        return self._marker_id_seq

    # ------------------------------------------------------------------
    def _load_markers(self):
        s = self.settings.value("markers_json", "")
        if not s:
            return
        try:
            arr = json.loads(s)
        except Exception:
            return
        for m in arr:
            self._add_marker(float(m.get("freq", 0.0)), m.get("color", "#ff5252"), m.get("name", ""))

    def _update_color_button(self):
        pm = QtGui.QPixmap(24, 24)
        pm.fill(self._marker_color)
        self.btnColorPick.setIcon(QtGui.QIcon(pm))

    def _pick_marker_color(self):  # pragma: no cover - GUI interaction
        c = QColorDialog.getColor(self._marker_color, self, "Выбрать цвет маркера")
        if c.isValid():
            self._marker_color = c
            self._update_color_button()

    def _add_marker_from_ui(self):  # pragma: no cover - GUI
        try:
            f = float(self.edMarkerFreq.text())
        except ValueError:
            QtWidgets.QMessageBox.information(self, "Маркеры", "Укажи корректную частоту в МГц.")
            return
        self._add_marker(f, self._marker_color.name())

    def _delete_selected_markers(self):  # pragma: no cover - GUI
        rows = self.listMarkers.selectionModel().selectedRows()
        for r in rows:
            mid = r.data(QtCore.Qt.UserRole)
            self._remove_marker(mid)

    def _on_mouse_click_add_marker(self, ev, vb):  # pragma: no cover - GUI
        if not ev.double() or ev.button() != QtCore.Qt.LeftButton:
            return
        pos = vb.mapSceneToView(ev.scenePos())
        f_mhz = float(pos.x())
        suggested = f"Метка {f_mhz:.3f}"
        name, ok = QtWidgets.QInputDialog.getText(self, "Новая метка",
                                                 f"Частота: {f_mhz:.6f} МГц\nНазвание метки:",
                                                 QtWidgets.QLineEdit.Normal, suggested)
        if not ok or not name.strip():
            return
        self._add_marker(f_mhz, self._marker_color.name(), name.strip())

    def _add_marker(self, f_mhz: float, color_hex: str, name: str = None):
        pen = pg.mkPen(QtGui.QColor(color_hex), width=2, style=QtCore.Qt.DashLine)
        l1 = pg.InfiniteLine(pos=f_mhz, angle=90, movable=True, pen=pen)
        l2 = pg.InfiniteLine(pos=f_mhz, angle=90, movable=True, pen=pen)
        self.plot.addItem(l1); self.wfPlot.addItem(l2)
        mid = self._get_next_marker_id()
        if not name:
            name = f"M{mid}"
        self.markers[mid] = {"freq": f_mhz, "color": color_hex, "name": name, "l_main": l1, "l_wf": l2}
        l1.sigPositionChanged.connect(lambda: self._set_marker_freq(mid, float(l1.value()), "main"))
        l2.sigPositionChanged.connect(lambda: self._set_marker_freq(mid, float(l2.value()), "wf"))
        self._append_marker_to_list(mid)

    def _append_marker_to_list(self, mid: int):
        m = self.markers[mid]
        txt = f"{mid}: {m.get('name','')} — {m['freq']:.6f} MHz  [{m['color']}]"
        item = QtWidgets.QListWidgetItem(txt)
        item.setData(QtCore.Qt.UserRole, mid)
        pm = QtGui.QPixmap(14, 14); pm.fill(QtGui.QColor(m['color']))
        item.setIcon(QtGui.QIcon(pm))
        self.listMarkers.addItem(item)

    def _refresh_marker_in_list(self, mid: int):
        for i in range(self.listMarkers.count()):
            it = self.listMarkers.item(i)
            if it.data(QtCore.Qt.UserRole) == mid:
                m = self.markers[mid]
                it.setText(f"{mid}: {m.get('name','')} — {m['freq']:.6f} MHz  [{m['color']}]")
                pm = QtGui.QPixmap(14, 14); pm.fill(QtGui.QColor(m['color']))
                it.setIcon(QtGui.QIcon(pm))
                break

    def _set_marker_freq(self, mid: int, new_f: float, source: str):
        m = self.markers.get(mid)
        if not m:
            return
        m['freq'] = new_f
        if source != "main":
            m['l_main'].setValue(new_f)
        if source != "wf":
            m['l_wf'].setValue(new_f)
        self._refresh_marker_in_list(mid)

    def _remove_marker(self, mid: int):
        m = self.markers.pop(mid, None)
        if not m:
            return
        self.plot.removeItem(m['l_main'])
        self.wfPlot.removeItem(m['l_wf'])
        for i in range(self.listMarkers.count()):
            it = self.listMarkers.item(i)
            if it.data(QtCore.Qt.UserRole) == mid:
                self.listMarkers.takeItem(i)
                break

    # ------------------------------------------------------------------
    def _open_peaks(self):  # pragma: no cover - GUI
        if not hasattr(self, '_peaks_dlg'):
            self._peaks_dlg = PeaksDialog(self)
            self._peaks_dlg.jumpToFreq.connect(self._jump_to_freq)
        self._update_peaks_dialog()
        self._peaks_dlg.show()

    def _update_peaks_dialog(self):
        if not self._last_sweep_freqs or not self._peaks_dlg.chkAuto.isChecked():
            return
        x_mhz = self._last_sweep_freqs / 1e6
        y_now = self._last_sweep_power
        peaks = find_peaks(x_mhz, y_now, self._peaks_dlg.spnMinDbm.value(), self._peaks_dlg.spnMinSep.value())
        self._peaks_dlg.set_peaks(peaks)

    def _jump_to_freq(self, f_mhz: float):  # pragma: no cover - GUI
        if self._last_sweep_freqs is None:
            return
        x = self._last_sweep_freqs / 1e6
        vb = self.plot.getViewBox()
        span = x[-1] - x[0]
        vb.setXRange(f_mhz - span * 0.1, f_mhz + span * 0.1, padding=0)

    def _open_trilat(self):  # pragma: no cover - GUI
        self.triWin = TrilaterationWindow("(auto)", "(auto)", "(auto)")
        self.triWin.show()

    # ------------------------------------------------------------------
    def on_start(self):  # pragma: no cover - requires hardware
        if self.worker is not None:
            return
        self._format_line_edit(self.fStart)
        self._format_line_edit(self.fStop)
        try:
            self.cfg_f0_mhz = float(self.fStart.text())
            self.cfg_f1_mhz = float(self.fStop.text())
            bw = float(self.binHz.text())
        except ValueError:
            self.statusBar().showMessage("Неверные параметры")
            return
        self._avg_queue.clear(); self._avg_line=None; self._minhold=None; self._maxhold=None
        self.wfBuffer=None; self._axes_ready=False; self._last_sweep_t=None; self._dt_hist.clear()
        method = self.cfg.get("sdr", {}).get("method", "hackrf_sweep")
        if method == "hackrf_sweep":
            self.worker = SweepWorker(self.cfg_f0_mhz, self.cfg_f1_mhz, bw,
                                      self.lnaBox.value(), self.vgaBox.value(), "")
        else:
            self.worker = LibWorker(self.cfg_f0_mhz, self.cfg_f1_mhz, bw,
                                    self.lnaBox.value(), self.vgaBox.value(), "")
        self.worker.spectrumReady.connect(self.on_sweep)
        self.worker.status.connect(lambda s: self.statusBar().showMessage(s))
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.btnStart.setEnabled(False); self.btnStop.setEnabled(True)
        self.worker.start(); self.statusBar().showMessage("Запуск...")

    def on_stop(self):  # pragma: no cover - requires hardware
        if self.worker:
            self.worker.stop()
        self.btnStop.setEnabled(False)

    def on_error(self, msg):  # pragma: no cover - GUI
        QtWidgets.QMessageBox.critical(self, "Ошибка", msg)
        self.statusBar().showMessage(msg)

    def on_worker_finished(self):  # pragma: no cover - GUI
        self.worker = None
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.statusBar().showMessage("Остановлено")

    def on_sweep(self, f_grid_hz: np.ndarray, p_grid_dbm: np.ndarray):  # pragma: no cover - heavy GUI
        if (not self._axes_ready) or self.wfBuffer is None or self.wfBuffer.shape[1] != len(f_grid_hz):
            self.wfBuffer = np.full((self.wfRows, len(f_grid_hz)), self.level_min, dtype=np.float32)
            self._init_axes_once(f_grid_hz)

        y_proc = p_grid_dbm.copy()
        self._last_sweep_freqs = f_grid_hz.copy()
        self._last_sweep_power = y_proc.copy()

        now_t = time.time()
        if self._last_sweep_t is not None:
            self._dt_hist.append(max(1e-3, now_t - self._last_sweep_t))
        self._last_sweep_t = now_t
        self.timeAxis.dt_est = float(np.mean(self._dt_hist) if self._dt_hist else 0.3)

        self.curve_now.setData(f_grid_hz / 1e6, y_proc)

        self._avg_queue.append(y_proc)
        self.curve_avg.setData(f_grid_hz / 1e6, np.mean(self._avg_queue, axis=0))

        if self._minhold is None:
            self._minhold = y_proc.copy(); self._maxhold = y_proc.copy()
        else:
            self._minhold = np.minimum(self._minhold, y_proc)
            self._maxhold = np.maximum(self._maxhold, y_proc)
        self.curve_min.setData(f_grid_hz / 1e6, self._minhold)
        self.curve_max.setData(f_grid_hz / 1e6, self._maxhold)

        self.wfBuffer = np.roll(self.wfBuffer, -1, axis=0)
        self.wfBuffer[-1, :] = y_proc
        self.wfImg.setImage(self.wfBuffer, autoLevels=False)
        self._update_peaks_dialog()

    def _init_axes_once(self, f_grid_hz):
        self.plot.setXRange(f_grid_hz[0] / 1e6, f_grid_hz[-1] / 1e6, padding=0)
        self.plot.setYRange(self.level_min, self.level_max, padding=0)
        self.wfPlot.setXRange(f_grid_hz[0] / 1e6, f_grid_hz[-1] / 1e6, padding=0)
        self._axes_ready = True

    # ------------------------------------------------------------------
    def reset_view(self):  # pragma: no cover - GUI
        if self._last_sweep_freqs is None:
            return
        f = self._last_sweep_freqs / 1e6
        self.plot.getViewBox().setXRange(f[0], f[-1], padding=0)

    def zoom_x(self, factor: float):  # pragma: no cover - GUI
        vb = self.plot.getViewBox()
        r = vb.viewRange()
        c = (r[0][0] + r[0][1]) * 0.5
        w = (r[0][1] - r[0][0]) * factor * 0.5
        vb.setXRange(c - w, c + w, padding=0)

    # ------------------------------------------------------------------
    def _set_marker_freq_main(self, mid, freq):
        self.markers[mid]['l_wf'].setValue(freq)

    def _set_marker_freq_wf(self, mid, freq):
        self.markers[mid]['l_main'].setValue(freq)

    # ------------------------------------------------------------------
    def _on_mouse_moved(self, ev):  # pragma: no cover - GUI
        pos = self.plot.getViewBox().mapSceneToView(ev)
        f_mhz = float(pos.x())
        y_dbm = None
        if self._last_sweep_freqs is not None and self._last_sweep_power is not None and len(self._last_sweep_freqs) > 1:
            x = self._last_sweep_freqs / 1e6
            idx = int(np.clip(np.searchsorted(x, f_mhz), 1, len(x) - 1))
            j = idx if abs(x[idx] - f_mhz) < abs(x[idx - 1] - f_mhz) else idx - 1
            y_dbm = float(self._last_sweep_power[j])
        txt = f"{f_mhz:.3f} МГц"
        if y_dbm is not None:
            txt += f", {y_dbm:.1f} дБм"
        self._cursorText.setText(txt)
        self._cursorText.setPos(f_mhz, self.level_max - 0.2)

