#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PANORAMA — ПАНОРАМА 0.1 бета
HackRF Sweep Analyzer — QSA-like (Инструменты, Детектор активности, Пики, Маркеры, Карта)

Ключевые моменты:
— Меню «Инструменты», вкладка «Карта».
— Точное отображение палитр: plasma/inferno/magma/viridis/turbo берутся из matplotlib (если установлен).
— Безопасные стартовые диапазоны осей (нет NaN при зуме до первого свипа).
— Экспорт свипа — весь текущий спектр.
— На карте: нет метки ЛКМ; трекинг координат мыши x,y сверху.
— Над спектром: подсказка курсора «МГц, дБм».
— «Пики»: порог по умолчанию −80 дБм, поиск ближайших частот в таблице.
— При методе «Разбор (hackrf_sweep)» вкладка «Карта» блокируется.
"""

import sys, os, shutil, subprocess, threading, time, collections, json, random, csv
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QTransform, QColor, QPixmap
from PyQt5.QtWidgets import QColorDialog, QTableWidgetItem
import pyqtgraph as pg
import numpy as np

APP_ORG = "PANORAMA"
APP_NAME = "PANORAMA"

MARKER_PALETTE = [
    "#ff5252", "#40c4ff", "#ffd740", "#69f0ae", "#b388ff",
    "#ffab40", "#18ffff", "#ff6e40", "#64ffda", "#eeff41",
]

# ---------- optional CFFI lib (lazy) ----------
HAVE_HQ = None
HQ_IMPORT_ERR = ""
def _try_import_hq():
    global HAVE_HQ, HQ_IMPORT_ERR
    if HAVE_HQ is not None:
        return HAVE_HQ
    try:
        import hq_cffi  # noqa: F401
        HAVE_HQ = True
    except Exception as e:
        HAVE_HQ = False
        HQ_IMPORT_ERR = str(e)
    return HAVE_HQ

# -------------------- parsing hackrf_sweep --------------------
def parse_sweep_line(line: str):
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 7:
        return None
    try:
        f_low  = float(parts[2])   # Hz
        f_high = float(parts[3])   # Hz
        bin_hz = float(parts[4])   # Hz
        vals = [float(x) for x in parts[6:]]
        if not vals:
            return None
        freqs = f_low + (np.arange(len(vals), dtype=np.float64) + 0.5) * bin_hz
        power = np.array(vals, dtype=np.float32)
        return freqs, power, bin_hz, f_low, f_high
    except Exception:
        return None

# -------------------- color presets --------------------
# Базовые запасные пресеты (если нет matplotlib)
PRESET_GRADIENTS = {
    "qsa": [
        "#0b0030", "#1c006b", "#3c00a8", "#6f02b3", "#a13c93",
        "#d06367", "#ef8d3e", "#f7c545", "#ffffbf"
    ],
    "plasma": ["#0d0887","#5b02a3","#8f0da4","#bb3787","#e16462","#fca636","#f0f921"],
    "inferno": ["#000003","#1f0c48","#550f6d","#88226a","#b73759","#e1642c","#fca50a","#fcffa4"],
    "magma":   ["#000004","#1b0c41","#4f116f","#822681","#b73779","#e35933","#f98e09","#fbfdbf"],
    "turbo":   ["#30123b","#3f1d9a","#2780ff","#1bbbe3","#1edeaa","#5be65f","#c2d923","#ffb000","#ff5800"],
    "gray":    ["#000000","#ffffff"],
}

def _lut_from_hex_gradient(hex_list, n=512):
    cols = []
    for h in hex_list:
        h = h.strip()
        if h.startswith("#") and len(h) == 7:
            r = int(h[1:3],16); g=int(h[3:5],16); b=int(h[5:7],16)
            cols.append((r,g,b))
    if len(cols) < 2:
        cols = [(0,0,0),(255,255,255)]
    cols = np.array(cols, dtype=np.float32)
    stops = np.linspace(0.0, 1.0, len(cols))
    xi = np.linspace(0.0, 1.0, int(n))
    r = np.interp(xi, stops, cols[:,0])
    g = np.interp(xi, stops, cols[:,1])
    b = np.interp(xi, stops, cols[:,2])
    lut = np.stack([r,g,b], axis=1).astype(np.uint8)  # (N,3)
    return lut

def get_colormap(name="qsa", n=512):
    """
    Возвращает LUT (N,3) для pyqtgraph. Пытается взять из matplotlib,
    чтобы plasma/inferno/magma/viridis/turbo выглядели «как в стандарте».
    """
    name = (name or "qsa").lower()
    # Попытка из matplotlib — даёт идеальное совпадение (если установлен)
    if name in ("plasma","inferno","magma","viridis","cividis","turbo"):
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap("turbo" if name=="turbo" else name)
            arr = (cmap(np.linspace(0.0, 1.0, int(n)))[:, :3] * 255.0).astype(np.uint8)
            return arr
        except Exception:
            pass  # fallback ниже
    # Запасной градиент из опорных цветов
    hex_list = PRESET_GRADIENTS.get(name, PRESET_GRADIENTS["qsa"])
    return _lut_from_hex_gradient(hex_list, n)

# ---- ось времени (0 сверху, вниз — отриц.)
class TimeAxis(pg.AxisItem):
    def __init__(self, orientation='left'):
        super().__init__(orientation=orientation)
        self.dt_est = 0.3
    def tickStrings(self, values, scale, spacing):
        return [f"{-v*self.dt_est:.0f}" for v in values]

# -------------------- workers --------------------
class LibWorker(QtCore.QThread):
    spectrumReady = QtCore.pyqtSignal(object, object)
    status = QtCore.pyqtSignal(str)
    error  = QtCore.pyqtSignal(str)
    def __init__(self, f_start_mhz, f_stop_mhz, bin_hz, lna, vga, serial_suffix=""):
        super().__init__()
        self.f0_mhz=float(f_start_mhz); self.f1_mhz=float(f_stop_mhz)
        self.bin_hz_req=float(bin_hz); self.lna=int(lna); self.vga=int(vga)
        self.serial=serial_suffix; self._stop=threading.Event(); self._prev_low=None
        self._grid=None; self._sum=None; self._cnt=None; self._seen=None; self._n=0
    def stop(self):
        self._stop.set()
        try:
            from hq_cffi import lib
            lib.hq_stop()
        except Exception:
            pass
    def _reset_grid(self):
        f0=int(round(self.f0_mhz*1e6)); f1=int(round(self.f1_mhz*1e6)); bw=float(self.bin_hz_req)
        grid=np.arange(f0 + bw*0.5, f1 + bw*0.5, bw, dtype=np.float64)
        self._grid=grid; self._n=len(grid)
        self._sum=np.zeros(self._n,np.float64); self._cnt=np.zeros(self._n,np.int32); self._seen=np.zeros(self._n,bool)
    def _add_segment(self, f_hz, p_dbm):
        if self._grid is None or self._n==0: return
        idx=np.rint((f_hz - self._grid[0]) / self.bin_hz_req).astype(np.int32)
        m=(idx>=0)&(idx<self._n)
        if not np.any(m): return
        idx=idx[m]; p=p_dbm[m].astype(np.float64)
        np.add.at(self._sum, idx, p); np.add.at(self._cnt, idx, 1); self._seen[idx]=True
    def _finish_sweep(self):
        if self._n==0: return
        coverage=float(self._seen.sum())/float(self._n)
        if coverage<0.95: self._reset_grid(); return
        p=np.full(self._n, np.nan, np.float32); valid=self._cnt>0
        p[valid]=(self._sum[valid]/self._cnt[valid]).astype(np.float32)
        if np.isnan(p).any():
            vmask=~np.isnan(p)
            if vmask.any():
                p=np.interp(np.arange(self._n), np.flatnonzero(vmask), p[vmask]).astype(np.float32)
            p[np.isnan(p)]= -120.0
        self.spectrumReady.emit(self._grid.copy(), p); self._reset_grid()
    def run(self):
        if not _try_import_hq():
            self.error.emit(f"Библиотечный режим недоступен: {HQ_IMPORT_ERR}"); return
        from hq_cffi import ffi, lib
        self._reset_grid(); self.status.emit("Открытие HackRF (library)…")
        if lib.hq_open(self.serial.encode('utf-8') if self.serial else ffi.NULL)!=0:
            self.error.emit(ffi.string(lib.hq_last_error()).decode('utf-8')); return
        if lib.hq_configure(self.f0_mhz, self.f1_mhz, self.bin_hz_req, self.lna, self.vga, 0)!=0:
            self.error.emit(ffi.string(lib.hq_last_error()).decode('utf-8')); lib.hq_close(); return
        self.status.emit("Старт свипа (library)…")
        @ffi.callback("void(const double*, const float*, int, double, uint64_t, uint64_t, void*)")
        def on_segment(freqs_ptr, pwr_ptr, count, bin_hz, hz_low, hz_high, user):
            if self._stop.is_set(): return
            f=np.frombuffer(ffi.buffer(freqs_ptr, count*8), dtype=np.float64).copy()
            p=np.frombuffer(ffi.buffer(pwr_ptr,   count*4), dtype=np.float32).copy()
            if self._prev_low is not None and hz_low < (self._prev_low - self.bin_hz_req*10):
                self._finish_sweep()
            self._prev_low=float(hz_low); self._add_segment(f,p)
        if lib.hq_start(on_segment, ffi.NULL)!=0:
            self.error.emit(ffi.string(lib.hq_last_error()).decode('utf-8')); lib.hq_close(); return
        try:
            while not self._stop.is_set(): time.sleep(0.05)
        finally:
            lib.hq_stop(); self._finish_sweep(); lib.hq_close()

class SweepWorker(QtCore.QThread):
    spectrumReady = QtCore.pyqtSignal(object, object)
    status = QtCore.pyqtSignal(str)
    error  = QtCore.pyqtSignal(str)
    def __init__(self, f_start_mhz=2400.0, f_stop_mhz=2483.0, bin_hz=1_000_000, lna=24, vga=20, serial_suffix=""):
        super().__init__()
        self.f0_mhz=float(f_start_mhz); self.f1_mhz=float(f_stop_mhz); self.bin_hz=int(bin_hz)
        self.lna=int(lna); self.vga=int(vga); self._stop=threading.Event(); self._proc=None
        self.serial=serial_suffix; self._grid=None; self._sum=None; self._cnt=None; self._seen=None; self._n=0; self._prev_low=None
    def stop(self):
        self._stop.set()
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                for _ in range(10):
                    if self._proc.poll() is not None: break
                    time.sleep(0.05)
                if self._proc.poll() is None: self._proc.kill()
        except Exception:
            pass
    def _reset_grid(self):
        f0=int(round(self.f0_mhz*1e6)); f1=int(round(self.f1_mhz*1e6)); bw=float(self.bin_hz)
        grid=np.arange(f0 + bw*0.5, f1 + bw*0.5, bw, dtype=np.float64)
        self._grid=grid; self._n=len(grid)
        self._sum=np.zeros(self._n,np.float64); self._cnt=np.zeros(self._n,np.int32); self._seen=np.zeros(self._n,bool)
    def _add_segment(self, freqs, power):
        if self._grid is None or self._n==0: return
        idx=np.rint((freqs - self._grid[0]) / self.bin_hz).astype(np.int32)
        m=(idx>=0)&(idx<self._n)
        if not np.any(m): return
        idx=idx[m]; p=power[m].astype(np.float64)
        np.add.at(self._sum, idx, p); np.add.at(self._cnt, idx, 1); self._seen[idx]=True
    def _finish_sweep(self):
        coverage=float(self._seen.sum())/float(self._n) if self._n else 0.0
        if coverage<0.95: self._reset_grid(); return
        p=np.full(self._n, np.nan, np.float32); valid=self._cnt>0
        p[valid]=(self._sum[valid]/self._cnt[valid]).astype(np.float32)
        if np.isnan(p).any():
            vmask=~np.isnan(p)
            if vmask.any():
                p=np.interp(np.arange(self._n), np.flatnonzero(vmask), p[vmask]).astype(np.float32)
            p[np.isnan(p)]= -120.0
        self.spectrumReady.emit(self._grid.copy(), p); self._reset_grid()
    def run(self):
        if not shutil.which("hackrf_sweep"):
            self.error.emit("Не найден 'hackrf_sweep' в PATH."); return
        f0=int(round(self.f0_mhz)); f1=int(round(self.f1_mhz))
        if f1<=f0: self.error.emit("Fstop (МГц) должен быть > Fstart (МГц)."); return
        self._reset_grid()
        cmd=["hackrf_sweep","-f",f"{f0}:{f1}","-w",str(self.bin_hz),"-l",str(self.lna),"-g",str(self.vga)]
        if self.serial: cmd.extend(["-d", self.serial])
        self.status.emit(" ".join(cmd))
        try:
            self._proc=subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        universal_newlines=True, bufsize=1)
        except Exception as e:
            self.error.emit(f"Не удалось запустить hackrf_sweep: {e}"); return
        try:
            for line in self._proc.stdout:
                if self._stop.is_set(): break
                if not line or line.startswith("Exiting") or "hackrf_" in line: continue
                parsed=parse_sweep_line(line)
                if parsed is None: continue
                fseg,pseg,_,f_low,_=parsed
                if self._prev_low is not None and f_low < self._prev_low - self.bin_hz*10:
                    self._finish_sweep()
                self._add_segment(fseg,pseg); self._prev_low=f_low
            self._finish_sweep()
        finally:
            try:
                if self._proc and self._proc.poll() is None: self._proc.terminate()
            except Exception:
                pass

# -------------------- Peaks dialog --------------------
class PeaksDialog(QtWidgets.QDialog):
    jumpToFreq = QtCore.pyqtSignal(float)  # МГц

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Пики (авто-поиск)")
        self.resize(560, 480)
        v = QtWidgets.QVBoxLayout(self)

        self.chkAuto = QtWidgets.QCheckBox("Автообновление")
        self.chkAuto.setChecked(True)
        v.addWidget(self.chkAuto)

        self.table = QtWidgets.QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["Частота, МГц", "Уровень, дБм", "Ширина, кГц", "Индекс"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        v.addWidget(self.table, 1)

        # Панель управления
        ctl1 = QtWidgets.QHBoxLayout()
        self.spnMinDbm = QtWidgets.QDoubleSpinBox(); self.spnMinDbm.setRange(-200, 100); self.spnMinDbm.setValue(-80.0)
        self.spnMinSep = QtWidgets.QDoubleSpinBox(); self.spnMinSep.setRange(0.0, 5000.0); self.spnMinSep.setDecimals(1); self.spnMinSep.setValue(30.0)
        self.btnExport = QtWidgets.QPushButton("Экспорт CSV…")
        ctl1.addWidget(QtWidgets.QLabel("Порог, дБм:")); ctl1.addWidget(self.spnMinDbm)
        ctl1.addWidget(QtWidgets.QLabel("Мин. разделение, кГц:")); ctl1.addWidget(self.spnMinSep)
        ctl1.addStretch(1); ctl1.addWidget(self.btnExport)
        v.addLayout(ctl1)

        # Поиск по частотам (выделяет ближайшие строки)
        ctl2 = QtWidgets.QHBoxLayout()
        self.edFreqs = QtWidgets.QLineEdit(); self.edFreqs.setPlaceholderText("Частоты, МГц (через запятую)")
        self.btnFind = QtWidgets.QPushButton("Найти в таблице")
        ctl2.addWidget(self.edFreqs); ctl2.addWidget(self.btnFind)
        v.addLayout(ctl2)

        self.table.doubleClicked.connect(self._jump_selected)
        self.btnExport.clicked.connect(self._export_csv)
        self.btnFind.clicked.connect(self._find_freqs)

        self._last_peaks = []  # [(freq_mhz, level_dbm, width_khz, idx)]

    def _jump_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows: return
        r = rows[0].row()
        try:
            f = float(self.table.item(r, 0).text())
        except Exception:
            return
        self.jumpToFreq.emit(f)

    def _export_csv(self):
        if not self._last_peaks:
            QtWidgets.QMessageBox.information(self, "Экспорт CSV", "Нет найденных пиков.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить CSV", "peaks.csv", "CSV (*.csv)")
        if not path: return
        try:
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["freq_mhz","level_dbm","width_khz","index"])
                for row in self._last_peaks:
                    w.writerow(row)
            QtWidgets.QMessageBox.information(self, "Экспорт CSV", "Готово.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Экспорт CSV", f"Ошибка записи: {e}")

    def set_peaks(self, peaks):
        self._last_peaks = peaks
        self.table.setRowCount(0)
        for f, lvl, w, idx in peaks:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, val in enumerate([f"{f:.6f}", f"{lvl:.1f}", f"{w:.1f}", str(int(idx))]):
                self.table.setItem(r, c, QTableWidgetItem(val))

    def _find_freqs(self):
        txt = (self.edFreqs.text() or "").replace(";", ",").replace("|", ",")
        want = []
        for t in txt.split(","):
            t = t.strip().replace(",", ".")
            if not t:
                continue
            try:
                want.append(float(t))
            except:
                pass
        if not want or self.table.rowCount() == 0:
            return
        tab_freqs = []
        for r in range(self.table.rowCount()):
            try:
                tab_freqs.append((r, float(self.table.item(r, 0).text())))
            except:
                pass
        self.table.clearSelection()
        for f0 in want:
            nearest = sorted(((abs(ff - f0), r) for r, ff in tab_freqs), key=lambda x: x[0])[:5]
            for _, r in nearest:
                self.table.selectRow(r)
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.table.scrollTo(rows[0], QtWidgets.QAbstractItemView.PositionAtCenter)

# -------------------- Trilateration window (макет) --------------------
class TrilaterationWindow(QtWidgets.QMainWindow):
    def __init__(self, master_serial, slave1_serial, slave2_serial):
        super().__init__()
        self.setWindowTitle("Трилатерация (макет)")
        self.master_serial = master_serial or "(auto)"
        self.slave1_serial = slave1_serial or "(auto)"
        self.slave2_serial = slave2_serial or "(auto)"

        w = QtWidgets.QWidget(self); self.setCentralWidget(w)
        h = QtWidgets.QHBoxLayout(w)

        # левый — настройки координат
        left = QtWidgets.QVBoxLayout()
        grid = QtWidgets.QFormLayout()
        self.lblM = QtWidgets.QLabel(f"Master: {self.master_serial}")
        self.edMx = QtWidgets.QDoubleSpinBox(); self.edMy = QtWidgets.QDoubleSpinBox(); self.edMz = QtWidgets.QDoubleSpinBox()
        for ed in (self.edMx,self.edMy,self.edMz):
            ed.setRange(-10000,10000); ed.setDecimals(2); ed.setValue(0.0); ed.setEnabled(False)
        grid.addRow(self.lblM); grid.addRow("M X/Y/Z:", self._row3(self.edMx,self.edMy,self.edMz))

        self.lblS1 = QtWidgets.QLabel(f"Slave-1: {self.slave1_serial}")
        self.edS1x = QtWidgets.QDoubleSpinBox(); self.edS1y = QtWidgets.QDoubleSpinBox(); self.edS1z = QtWidgets.QDoubleSpinBox()
        for ed in (self.edS1x,self.edS1y,self.edS1z): ed.setRange(-10000,10000); ed.setDecimals(2)
        grid.addRow(self.lblS1); grid.addRow("S1 X/Y/Z:", self._row3(self.edS1x,self.edS1y,self.edS1z))

        self.lblS2 = QtWidgets.QLabel(f"Slave-2: {self.slave2_serial}")
        self.edS2x = QtWidgets.QDoubleSpinBox(); self.edS2y = QtWidgets.QDoubleSpinBox(); self.edS2z = QtWidgets.QDoubleSpinBox()
        for ed in (self.edS2x,self.edS2y,self.edS2z): ed.setRange(-10000,10000); ed.setDecimals(2)
        grid.addRow(self.lblS2); grid.addRow("S2 X/Y/Z:", self._row3(self.edS2x,self.edS2y,self.edS2z))

        self.btnRedraw = QtWidgets.QPushButton("Обновить схему")
        left.addLayout(grid); left.addWidget(self.btnRedraw); left.addStretch(1)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True,y=True,alpha=0.25)
        self.plot.setLabel("bottom","X"); self.plot.setLabel("left","Y")
        self.plot.setAspectLocked(False)
        self.scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None))
        self.plot.addItem(self.scatter)

        h.addLayout(left,0); h.addWidget(self.plot,1)
        self.btnRedraw.clicked.connect(self._redraw)
        self._redraw()
        self.showMaximized()

    def _row3(self, a,b,c):
        row = QtWidgets.QHBoxLayout(); row.addWidget(a); row.addWidget(b); row.addWidget(c)
        w = QtWidgets.QWidget(); w.setLayout(row); return w

    def _redraw(self):
        self.plot.clear()
        scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None))
        pts=[]; brushes=[]
        pts.append({'pos': (self.edMx.value(), self.edMy.value()), 'data':'M'}); brushes.append(pg.mkBrush('#ff4444'))
        pts.append({'pos': (self.edS1x.value(), self.edS1y.value()), 'data':'S1'}); brushes.append(pg.mkBrush('#44ff44'))
        pts.append({'pos': (self.edS2x.value(), self.edS2y.value()), 'data':'S2'}); brushes.append(pg.mkBrush('#4488ff'))
        scatter.setData(points=pts, brush=brushes)
        self.plot.addItem(scatter)
        for (x,y),label in [((self.edMx.value(), self.edMy.value()), "M(0,0,0)"),
                            ((self.edS1x.value(), self.edS1y.value()), f"S1(z={self.edS1z.value():.2f})"),
                            ((self.edS2x.value(), self.edS2y.value()), f"S2(z={self.edS2z.value():.2f})")]:
            text = pg.TextItem(label, anchor=(0,1)); text.setPos(x,y); self.plot.addItem(text)

# -------------------- UI --------------------
class MainWindow(QtWidgets.QMainWindow):
    MIN_MHZ = 50.000
    MAX_MHZ = 6000.000

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ПАНОРАМА — версия 0.1 бета")
        self.settings = QtCore.QSettings(APP_ORG, APP_NAME)

        center = QtWidgets.QWidget(self); centerLay = QtWidgets.QVBoxLayout(center)
        self.setCentralWidget(center)

        self.markers = {}; self._marker_id_seq = 0

        # верхняя панель
        ctrl = QtWidgets.QHBoxLayout()
        self.fStart = QtWidgets.QLineEdit(self._get("f0","50.000"))
        self.fStop  = QtWidgets.QLineEdit(self._get("f1","6000.000"))
        self.binHz  = QtWidgets.QLineEdit(self._get("bin","2000000"))
        self.lnaBox = QtWidgets.QSpinBox(); self.lnaBox.setRange(0,40); self.lnaBox.setSingleStep(8); self.lnaBox.setValue(int(self._get("lna","24")))
        self.vgaBox = QtWidgets.QSpinBox(); self.vgaBox.setRange(0,62); self.vgaBox.setSingleStep(2); self.vgaBox.setValue(int(self._get("vga","20")))
        self.btnStart = QtWidgets.QPushButton("Старт")
        self.btnStop  = QtWidgets.QPushButton("Стоп"); self.btnStop.setEnabled(False)
        self.btnReset = QtWidgets.QPushButton("Сброс вида")
        def add(w,l): b=QtWidgets.QVBoxLayout(); b.addWidget(QtWidgets.QLabel(l)); b.addWidget(w); ctrl.addLayout(b)
        add(self.fStart,"Начало (МГц)"); add(self.fStop,"Конец (МГц)"); add(self.binHz,"Шаг бина (Гц)")
        add(self.lnaBox,"LNA (дБ)"); add(self.vgaBox,"VGA (дБ)")
        ctrl.addWidget(self.btnStart); ctrl.addWidget(self.btnStop); ctrl.addWidget(self.btnReset)
        centerLay.addLayout(ctrl)

        self.fStart.editingFinished.connect(lambda: self._format_line_edit(self.fStart))
        self.fStop.editingFinished.connect(lambda: self._format_line_edit(self.fStop))

        # график спектра
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Частота (МГц)")
        self.plot.setLabel("left", "Мощность (дБм)")
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.curve_now  = self.plot.plot([], [], name="Текущая", pen=pg.mkPen('w', width=1))
        self.curve_avg  = self.plot.plot([], [], name="Среднее", pen=pg.mkPen((0,255,0), width=1))
        self.curve_min  = self.plot.plot([], [], name="Мин. удерж.", pen=pg.mkPen((120,120,255), width=1))
        self.curve_max  = self.plot.plot([], [], name="Макс. удерж.", pen=pg.mkPen((255,200,0), width=1))
        centerLay.addWidget(self.plot, stretch=1)

        # подсказка курсора (МГц/дБм)
        self._cursorText = pg.TextItem(color=pg.mkColor(0,0,0), anchor=(0,1))
        self.plot.addItem(self._cursorText)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # водопад
        self.wfLayout = pg.GraphicsLayoutWidget()
        g = self.wfLayout
        self.timeAxis = TimeAxis(orientation='left')
        self.wfPlot = g.addPlot(row=0, col=0, axisItems={'left': self.timeAxis})
        self.wfPlot.showGrid(x=True, y=False, alpha=0.12)
        self.wfPlot.setLabel("bottom", "Частота (МГц)")
        self.wfPlot.setLabel("left", "Время (с)")
        self.wfPlot.getAxis('bottom').enableAutoSIPrefix(False)
        self.wfImg = pg.ImageItem(axisOrder='row-major'); self.wfImg.setAutoDownsample(True)
        self.wfPlot.addItem(self.wfImg); self.wfPlot.setXLink(self.plot)
        centerLay.addWidget(self.wfLayout, stretch=2)

        # палитра/уровни
        self.level_min = float(self._get("lvl_min","-110"))
        self.level_max = float(self._get("lvl_max","-20"))
        lut = get_colormap(self._get("cmap","qsa"))
        self.wfImg.setLookupTable(lut)
        self.wfImg.setLevels([self.level_min, self.level_max])

        # служебное
        self.wfRows = int(self._get("wf_rows","600"))
        self.wfBuffer = None; self._axes_ready=False
        self._status_last = 0.0
        self.cfg_f0_mhz=None; self.cfg_f1_mhz=None; self._bin_w_mhz=1.0
        self._avg_queue = collections.deque(maxlen=int(self._get("avg_win","8"))); self._avg_line=None
        self._minhold=None; self._maxhold=None
        self._last_sweep_freqs=None; self._last_sweep_power=None
        self._last_sweep_t=None; self._dt_hist = collections.deque(maxlen=32)

        # хоткеи
        for vb in (self.plot.getViewBox(), self.wfPlot.getViewBox()):
            vb.enableAutoRange(x=False, y=False); vb.setDefaultPadding(0.0); vb.setMouseEnabled(x=True, y=False)
        QtWidgets.QShortcut(QtGui.QKeySequence("+"), self, activated=lambda: self.zoom_x(0.6))
        QtWidgets.QShortcut(QtGui.QKeySequence("-"), self, activated=lambda: self.zoom_x(1.6))
        QtWidgets.QShortcut(QtGui.QKeySequence("R"), self, activated=self.reset_view)

        # даблклик ЛКМ → именованная метка (только спектр/водопад; для карты отключено)
        self.plot.scene().sigMouseClicked.connect(
            lambda ev: self._on_mouse_click_add_marker(ev, self.plot.getViewBox())
        )
        self.wfPlot.scene().sigMouseClicked.connect(
            lambda ev: self._on_mouse_click_add_marker(ev, self.wfPlot.getViewBox())
        )

        # правая панель, меню
        self._build_right_panel()
        self._build_menu()

        self.statusBar().showMessage("Готов")
        self.worker=None
        self.btnStart.clicked.connect(self.on_start)
        self.btnStop.clicked.connect(self.on_stop)
        self.btnReset.clicked.connect(self.reset_view)

        # восстановление окна
        self.restoreGeometry(self.settings.value("win_geo", b""))
        self.restoreState(self.settings.value("win_state", b""))
        self._load_markers()

        # привести источник в актуальное состояние
        self._update_source_state()
        self._refresh_sdr_lists_unique()

        # безопасные стартовые диапазоны (нет NaN до первого свипа)
        self._set_initial_ranges()

    # стартовые диапазоны
    def _set_initial_ranges(self):
        try:
            f0 = float(str(self.fStart.text()).replace(",", "."))
            f1 = float(str(self.fStop.text()).replace(",", "."))
            if not np.isfinite(f0) or not np.isfinite(f1) or f1 <= f0:
                raise ValueError()
        except Exception:
            f0, f1 = self.MIN_MHZ, self.MAX_MHZ
        self.plot.setXRange(f0, f1, padding=0)
        self.plot.setYRange(self.level_min, self.level_max, padding=0)
        self.wfPlot.setXRange(f0, f1, padding=0)
        self.wfPlot.setYRange(0, self.wfRows, padding=0)
        if hasattr(self, 'an_plot'):
            self.an_plot.setXRange(-10, 10, padding=0)
            self.an_plot.setYRange(-10, 10, padding=0)

    # helpers / settings
    def _get(self, key, default): return str(self.settings.value(key, default))

    def _format_line_edit(self, le: QtWidgets.QLineEdit):
        try:
            v = float(le.text().replace(",", "."))
        except Exception:
            v = self.MIN_MHZ
        v = max(self.MIN_MHZ, min(self.MAX_MHZ, v))
        le.blockSignals(True); le.setText(f"{v:.3f}"); le.blockSignals(False)

    def closeEvent(self, e):
        self.settings.setValue("f0", self.fStart.text()); self.settings.setValue("f1", self.fStop.text())
        self.settings.setValue("bin", self.binHz.text())
        self.settings.setValue("lna", self.lnaBox.value()); self.settings.setValue("vga", self.vgaBox.value())
        self.settings.setValue("wf_rows", self.wfRows); self.settings.setValue("avg_win", self.avgSpin.value())
        self.settings.setValue("cmap", self.cmapBox.currentText())
        self.settings.setValue("lvl_min", self.minSpin.value()); self.settings.setValue("lvl_max", self.maxSpin.value())
        self.settings.setValue("chk_main", self.chkMain.isChecked()); self.settings.setValue("chk_max", self.chkMax.isChecked())
        self.settings.setValue("chk_min", self.chkMin.isChecked()); self.settings.setValue("chk_avg", self.chkAvg.isChecked())
        self.settings.setValue("chk_smooth", self.chkSmooth.isChecked()); self.settings.setValue("smooth_win", self.smoothWin.value())
        self.settings.setValue("chk_persist", self.chkPersist.isChecked()); self.settings.setValue("persist_decay", self.persistDecay.value())
        if hasattr(self, "acqBox"): self.settings.setValue("acq_method", self.acqBox.currentIndex())
        if hasattr(self, "masterBox"): self.settings.setValue("master_serial", self.masterBox.currentText())
        if hasattr(self, "slave1Box"): self.settings.setValue("slave1_serial", self.slave1Box.currentText())
        if hasattr(self, "slave2Box"): self.settings.setValue("slave2_serial", self.slave2Box.currentText())
        self.settings.setValue("da_ranges_json", json.dumps(self._ranges_to_json()))
        self.settings.setValue("win_geo", self.saveGeometry()); self.settings.setValue("win_state", self.saveState())
        super().closeEvent(e)

    # -------------------- right panel --------------------
    def _build_right_panel(self):
        dock = QtWidgets.QDockWidget("Панель управления", self); dock.setObjectName("RightPanel")
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        tabs = QtWidgets.QTabWidget(); dock.setWidget(tabs)

        # --- Инструменты (Настройки + Уровни + Источник)
        tabTools = QtWidgets.QWidget(); vTools = QtWidgets.QVBoxLayout(tabTools)

        gbDisp = QtWidgets.QGroupBox("Отображение"); f = QtWidgets.QFormLayout(gbDisp)
        self.chkMain = QtWidgets.QCheckBox("Текущая кривая"); self.chkMain.setChecked(self.settings.value("chk_main","true")=="true")
        self.chkMax  = QtWidgets.QCheckBox("Макс. удержание"); self.chkMax.setChecked(self.settings.value("chk_max","true")=="true")
        self.chkMin  = QtWidgets.QCheckBox("Мин. удержание"); self.chkMin.setChecked(self.settings.value("chk_min","false")=="true")
        self.chkAvg  = QtWidgets.QCheckBox("Среднее");        self.chkAvg.setChecked(self.settings.value("chk_avg","false")=="true")
        self.avgSpin = QtWidgets.QSpinBox(); self.avgSpin.setRange(1,200); self.avgSpin.setValue(int(self._get("avg_win","8"))); self.avgSpin.setSuffix(" свипов")
        f.addRow(self.chkMain); f.addRow(self.chkMax); f.addRow(self.chkMin)
        h = QtWidgets.QHBoxLayout(); h.addWidget(self.chkAvg); h.addWidget(self.avgSpin); w = QtWidgets.QWidget(); w.setLayout(h)
        f.addRow(w); vTools.addWidget(gbDisp)

        gbFX = QtWidgets.QGroupBox("Эффекты"); f2 = QtWidgets.QFormLayout(gbFX)
        self.chkSmooth = QtWidgets.QCheckBox("Сглаживание"); self.chkSmooth.setChecked(self.settings.value("chk_smooth","false")=="true")
        self.smoothWin = QtWidgets.QSpinBox(); self.smoothWin.setRange(1, 101); self.smoothWin.setSingleStep(2); self.smoothWin.setValue(int(self._get("smooth_win","5")))
        self.chkPersist = QtWidgets.QCheckBox("Накопление (Persistence)"); self.chkPersist.setChecked(self.settings.value("chk_persist","false")=="true")
        self.persistDecay = QtWidgets.QDoubleSpinBox(); self.persistDecay.setRange(0.0, 0.05); self.persistDecay.setSingleStep(0.005); self.persistDecay.setValue(float(self._get("persist_decay","0.01")))
        row1 = QtWidgets.QHBoxLayout(); row1.addWidget(self.chkSmooth); row1.addWidget(self.smoothWin)
        row2 = QtWidgets.QHBoxLayout(); row2.addWidget(self.chkPersist); row2.addWidget(self.persistDecay)
        w1, w2 = QtWidgets.QWidget(), QtWidgets.QWidget(); w1.setLayout(row1); w2.setLayout(row2)
        f2.addRow(w1); f2.addRow(w2); vTools.addWidget(gbFX)

        gbVis = QtWidgets.QGroupBox("Визуализация"); f3 = QtWidgets.QFormLayout(gbVis)
        self.cmapBox = QtWidgets.QComboBox()
        self.cmapBox.addItems(list(PRESET_GRADIENTS.keys()))
        self.cmapBox.setCurrentText(self._get("cmap","qsa"))
        self.cmapBox.currentTextChanged.connect(self._apply_cmap)
        self.minSpin = QtWidgets.QDoubleSpinBox(); self.minSpin.setRange(-200,100); self.minSpin.setDecimals(1); self.minSpin.setValue(float(self._get("lvl_min","-110")))
        self.maxSpin = QtWidgets.QDoubleSpinBox(); self.maxSpin.setRange(-200,100); self.maxSpin.setDecimals(1); self.maxSpin.setValue(float(self._get("lvl_max","-20")))
        self.btnApplyLevels = QtWidgets.QPushButton("Применить"); self.btnAutoLevels  = QtWidgets.QPushButton("Авто")
        self.btnApplyLevels.clicked.connect(self._apply_levels_from_spins); self.btnAutoLevels.clicked.connect(self._auto_levels)
        rowS = QtWidgets.QHBoxLayout(); rowS.addWidget(QtWidgets.QLabel("Уровни: мин/макс")); rowS.addWidget(self.minSpin); rowS.addWidget(self.maxSpin)
        rowB = QtWidgets.QHBoxLayout(); rowB.addWidget(self.btnApplyLevels); rowB.addWidget(self.btnAutoLevels)
        f3.addRow(QtWidgets.QLabel("Палитра (пресет):")); f3.addRow(self.cmapBox); f3.addRow(rowS); f3.addRow(rowB)
        vTools.addWidget(gbVis)

        gbSrc = QtWidgets.QGroupBox("Источник данных"); fS = QtWidgets.QFormLayout(gbSrc)
        self.acqBox = QtWidgets.QComboBox()
        self.acqBox.addItems(["Разбор (hackrf_sweep)", "Библиотека (libhackrf)"])
        self.acqBox.setCurrentIndex(int(self._get("acq_method","0")))
        self.acqBox.currentIndexChanged.connect(self._update_source_state)

        self.masterBox = QtWidgets.QComboBox(); self.slave1Box = QtWidgets.QComboBox(); self.slave2Box = QtWidgets.QComboBox()
        for cb in (self.masterBox,self.slave1Box,self.slave2Box):
            cb.setEditable(False); cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            cb.currentTextChanged.connect(self._refresh_sdr_lists_unique)
            cb.currentTextChanged.connect(lambda _=None: self._update_source_state())
        self.btnRefreshDevs = QtWidgets.QPushButton("Обновить устройства")
        self.btnRefreshDevs.clicked.connect(self._fill_devices)
        self._all_devices = []
        self._fill_devices()

        fS.addRow(QtWidgets.QLabel("Метод:"), self.acqBox)
        fS.addRow(QtWidgets.QLabel("Мастер SDR:"), self.masterBox)
        fS.addRow(QtWidgets.QLabel("Slave-1:"), self.slave1Box)
        fS.addRow(QtWidgets.QLabel("Slave-2:"), self.slave2Box)
        vTools.addWidget(gbSrc); vTools.addWidget(self.btnRefreshDevs); vTools.addStretch(1)

        # --- Детектор активности
        tabDA = QtWidgets.QWidget(); vDA = QtWidgets.QVBoxLayout(tabDA)
        self.lblSDRs = QtWidgets.QLabel("")
        vDA.addWidget(self.lblSDRs)
        self.lblNeedSlaves = QtWidgets.QLabel("Для трилатерации подключите Master + 2 уникальных Slave в библиотечном режиме.")
        palette_warn = self.lblNeedSlaves.palette(); palette_warn.setColor(self.lblNeedSlaves.foregroundRole(), QtGui.QColor("#ffa000"))
        self.lblNeedSlaves.setPalette(palette_warn)
        vDA.addWidget(self.lblNeedSlaves)

        gbPresets = QtWidgets.QGroupBox("ROI-пресеты")
        gridP = QtWidgets.QGridLayout(gbPresets)
        preset_rows = [
            ("FM (87–108 МГц)", [(87.5, 108.0)]),
            ("VHF (136–174 МГц)", [(136.0, 174.0)]),
            ("UHF (400–470 МГц)", [(400.0, 470.0)]),
            ("Wi-Fi 2.4 ГГц", [(2400.000, 2483.500)]),
            ("Wi-Fi 5/6 ГГц", [(5170.000, 5895.000)]),
            ("5.8 ГГц FPV", [(5725.000, 5875.000)]),
            ("LTE 700–900", [(703.0, 960.0)]),
            ("ISM 868", [(863.0, 873.0)]),
        ]
        r = 0; c = 0
        for title, rngs in preset_rows:
            b = QtWidgets.QPushButton(title)
            b.clicked.connect(lambda _, rr=rngs: self._add_preset_ranges(rr))
            gridP.addWidget(b, r, c)
            c += 1
            if c >= 2:
                c = 0; r += 1
        vDA.addWidget(gbPresets)

        gbRanges = QtWidgets.QGroupBox("Диапазоны сканирования")
        vR = QtWidgets.QVBoxLayout(gbRanges)
        self.tblRanges = QtWidgets.QTableWidget(0, 2)
        self.tblRanges.setHorizontalHeaderLabels(["Начало, МГц", "Конец, МГц"])
        self.tblRanges.horizontalHeader().setStretchLastSection(True)
        self.tblRanges.itemChanged.connect(self._on_range_item_changed)
        btnRow = QtWidgets.QHBoxLayout()
        self.btnRangeAdd = QtWidgets.QPushButton("Добавить текущий диапазон +")
        self.btnRangeDel = QtWidgets.QPushButton("Удалить выбранные −")
        self.btnMerge    = QtWidgets.QPushButton("Склеить диапазоны")
        btnRow.addWidget(self.btnRangeAdd); btnRow.addWidget(self.btnRangeDel); btnRow.addWidget(self.btnMerge)
        vR.addWidget(self.tblRanges); vR.addLayout(btnRow)
        vDA.addWidget(gbRanges)

        self.btnRangeAdd.clicked.connect(self._on_add_range)
        self.btnRangeDel.clicked.connect(self._on_del_range)
        self.btnMerge.clicked.connect(self._merge_ranges)

        gbDA = QtWidgets.QGroupBox("Параметры детектора (заглушки)")
        fDA = QtWidgets.QFormLayout(gbDA)
        self.daWideN = QtWidgets.QSpinBox(); self.daWideN.setRange(1, 5000); self.daWideN.setValue(200); self.daWideN.setEnabled(False)
        self.daStableN = QtWidgets.QSpinBox(); self.daStableN.setRange(1, 200); self.daStableN.setValue(5); self.daStableN.setEnabled(False)
        self.daBurst = QtWidgets.QCheckBox("Импульсные всплески"); self.daBurst.setEnabled(False)
        self.daHopper = QtWidgets.QCheckBox("Скачущая несущая (FHSS)"); self.daHopper.setEnabled(False)
        fDA.addRow("Широкий спектр, ≥ N бинов:", self.daWideN)
        fDA.addRow("Устойчивый сигнал, ≥ свипов:", self.daStableN)
        fDA.addRow(self.daBurst); fDA.addRow(self.daHopper)
        vDA.addWidget(gbDA)

        rowAct = QtWidgets.QHBoxLayout()
        self.btnDetect = QtWidgets.QPushButton("Начать детект")
        self.btnTrilat = QtWidgets.QPushButton("Начать трилатерацию")
        self.btnDetect.clicked.connect(lambda: QtWidgets.QMessageBox.information(self, "Детект", "Логика детекта будет добавлена позже."))
        self.btnTrilat.clicked.connect(self._start_trilateration)
        rowAct.addWidget(self.btnDetect); rowAct.addWidget(self.btnTrilat); rowAct.addStretch(1)
        vDA.addLayout(rowAct)
        vDA.addStretch(1)

        # --- Маркеры
        tabMarkers = QtWidgets.QWidget(); vM = QtWidgets.QVBoxLayout(tabMarkers)
        rowTop = QtWidgets.QHBoxLayout()
        self.edMarkerFreq = QtWidgets.QLineEdit(); self.edMarkerFreq.setPlaceholderText("Частота, МГц (например 2412)")
        self.btnColorPick = QtWidgets.QPushButton("Цвет")
        self._marker_color = QColor(random.choice(MARKER_PALETTE)); self._update_color_button()
        self.btnColorPick.clicked.connect(self._pick_marker_color)
        rowTop.addWidget(self.edMarkerFreq); rowTop.addWidget(self.btnColorPick)
        self.listMarkers = QtWidgets.QListWidget(); self.listMarkers.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listMarkers.itemDoubleClicked.connect(self._center_on_marker)
        rowBtn = QtWidgets.QHBoxLayout(); self.btnAddMarker = QtWidgets.QPushButton("Добавить"); self.btnDelMarker = QtWidgets.QPushButton("Удалить")
        rowBtn.addWidget(self.btnAddMarker); rowBtn.addWidget(self.btnDelMarker)
        self.btnAddMarker.clicked.connect(self._add_marker_from_ui)
        self.btnDelMarker.clicked.connect(self._remove_selected_markers)
        vM.addLayout(rowTop); vM.addWidget(self.listMarkers, 1); vM.addLayout(rowBtn)

        # --- Карта
        tabAnalysis = self._build_analysis_tab()

        # собрать вкладки
        tabs.addTab(tabTools,   "Инструменты")
        tabs.addTab(tabDA,      "Детектор активности")
        tabs.addTab(tabMarkers, "Маркеры")
        tabs.addTab(tabAnalysis,"Карта")
        self.an_tab = tabAnalysis

        # видимость линий
        self.chkMain.toggled.connect(lambda on: self.curve_now.setVisible(on))
        self.chkMax.toggled.connect( lambda on: self.curve_max.setVisible(on))
        self.chkMin.toggled.connect( lambda on: self.curve_min.setVisible(on))
        self.chkAvg.toggled.connect( lambda on: self.curve_avg.setVisible(on))
        self.curve_now.setVisible(self.chkMain.isChecked())
        self.curve_max.setVisible(self.chkMax.isChecked())
        self.curve_min.setVisible(self.chkMin.isChecked())
        self.curve_avg.setVisible(self.chkAvg.isChecked())

        # загрузить сохранённые диапазоны
        try:
            arr = json.loads(self._get("da_ranges_json","[]"))
            for r in arr:
                self._append_range_row(float(r.get("start_mhz",0)), float(r.get("stop_mhz",0)))
        except Exception:
            pass
        self._refresh_range_overlays()
        self._update_sdr_label()

    # ---- Карта
    def _analysis_on_click(self, ev):
        """ЛКМ на карте: только лог, без постановки метки (как просил)."""
        if ev.button() != QtCore.Qt.LeftButton:
            return
        pos = self.an_plot.getViewBox().mapSceneToView(ev.scenePos())
        x, y = float(pos.x()), float(pos.y())
        self.an_log.append(f"Точка: X={x:.2f}, Y={y:.2f}")
        # Ничего не рисуем на карте, чтобы ЛКМ не ставила метки.
        
    def _build_analysis_tab(self):
        tab = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(tab)

        left = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()

        self.an_edMx = QtWidgets.QDoubleSpinBox(); self.an_edMy = QtWidgets.QDoubleSpinBox(); self.an_edMz = QtWidgets.QDoubleSpinBox()
        for ed in (self.an_edMx, self.an_edMy, self.an_edMz):
            ed.setRange(-10000,10000); ed.setDecimals(2); ed.setValue(0.00); ed.setEnabled(False)
        rowM = QtWidgets.QHBoxLayout(); [rowM.addWidget(w) for w in (self.an_edMx, self.an_edMy, self.an_edMz)]
        wM = QtWidgets.QWidget(); wM.setLayout(rowM)
        form.addRow(QtWidgets.QLabel("Master @ (0,0,0)"))
        form.addRow("M X/Y/Z:", wM)

        self.an_edS1x = QtWidgets.QDoubleSpinBox(); self.an_edS1y = QtWidgets.QDoubleSpinBox(); self.an_edS1z = QtWidgets.QDoubleSpinBox()
        self.an_edS2x = QtWidgets.QDoubleSpinBox(); self.an_edS2y = QtWidgets.QDoubleSpinBox(); self.an_edS2z = QtWidgets.QDoubleSpinBox()
        for ed in (self.an_edS1x,self.an_edS1y,self.an_edS1z,self.an_edS2x,self.an_edS2y,self.an_edS2z):
            ed.setRange(-10000,10000); ed.setDecimals(2); ed.setValue(0.00)
        rowS1 = QtWidgets.QHBoxLayout(); [rowS1.addWidget(w) for w in (self.an_edS1x,self.an_edS1y,self.an_edS1z)]
        rowS2 = QtWidgets.QHBoxLayout(); [rowS2.addWidget(w) for w in (self.an_edS2x,self.an_edS2y,self.an_edS2z)]
        wS1 = QtWidgets.QWidget(); wS1.setLayout(rowS1)
        wS2 = QtWidgets.QWidget(); wS2.setLayout(rowS2)
        form.addRow("S1 X/Y/Z:", wS1)
        form.addRow("S2 X/Y/Z:", wS2)

        self.an_btnRedraw = QtWidgets.QPushButton("Обновить схему")
        form.addRow(self.an_btnRedraw)
        left.addLayout(form)

        self.an_log = QtWidgets.QTextEdit(); self.an_log.setReadOnly(True)
        left.addWidget(QtWidgets.QLabel("Лог координат / событий"))
        left.addWidget(self.an_log, 1)

        right = QtWidgets.QVBoxLayout()
        self.an_cursorLbl = QtWidgets.QLabel("—, —"); fnt = self.an_cursorLbl.font(); fnt.setBold(True); self.an_cursorLbl.setFont(fnt)
        right.addWidget(self.an_cursorLbl)

        self.an_plot = pg.PlotWidget()
        self.an_plot.showGrid(x=True, y=True, alpha=0.25)
        self.an_plot.setLabel("bottom", "X"); self.an_plot.setLabel("left", "Y")
        self.an_plot.setAspectLocked(False)
        self.an_plot.scene().sigMouseClicked.connect(self._analysis_on_click)
        self.an_plot.scene().sigMouseMoved.connect(self._analysis_mouse_moved)

        right.addWidget(self.an_plot, 1)
        h.addLayout(left, 0); h.addLayout(right, 1)

        self.an_btnRedraw.clicked.connect(self._analysis_redraw)
        for ed in (self.an_edS1x,self.an_edS1y,self.an_edS1z,self.an_edS2x,self.an_edS2y,self.an_edS2z):
            ed.valueChanged.connect(self._analysis_redraw)

        self._analysis_redraw()
        return tab

    def _analysis_mouse_moved(self, ev):
        pos = self.an_plot.getViewBox().mapSceneToView(ev)
        x, y = float(pos.x()), float(pos.y())
        self.an_cursorLbl.setText(f"{x:.2f}, {y:.2f}")

    def _analysis_redraw(self):
        self.an_plot.clear()
        sp = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None))
        pts = [
            {'pos': (0.0, 0.0), 'data': 'M', 'brush': pg.mkBrush('#ff4444')},
            {'pos': (self.an_edS1x.value(), self.an_edS1y.value()), 'data': 'S1', 'brush': pg.mkBrush('#44ff44')},
            {'pos': (self.an_edS2x.value(), self.an_edS2y.value()), 'data': 'S2', 'brush': pg.mkBrush('#4488ff')},
        ]
        sp.addPoints(pts); self.an_plot.addItem(sp)
        for (x,y), label in [((0.0,0.0), "M(0,0,0)"),
                             ((self.an_edS1x.value(), self.an_edS1y.value()), f"S1(z={self.an_edS1z.value():.2f})"),
                             ((self.an_edS2x.value(), self.an_edS2y.value()), f"S2(z={self.an_edS2z.value():.2f})")]:
            t = pg.TextItem(label, anchor=(0,1)); t.setPos(x,y); self.an_plot.addItem(t)

    # ---- devices
    def _fill_devices(self):
        ser=[]
        try:
            if _try_import_hq():
                from hq_cffi import list_hackrf_serials
                ser = list_hackrf_serials()
        except Exception:
            ser=[]
        if not ser: ser=["(auto)"]
        self._all_devices = ser
        self._refresh_sdr_lists_unique()

    def _refresh_sdr_lists_unique(self):
        if not hasattr(self, "masterBox"):
            return
        m_cur = self.masterBox.currentText() if self.masterBox.count() else "(auto)"
        s1_cur = self.slave1Box.currentText() if self.slave1Box.count() else "(auto)"
        s2_cur = self.slave2Box.currentText() if self.slave2Box.count() else "(auto)"

        self.masterBox.blockSignals(True)
        self.masterBox.clear(); self.masterBox.addItems(self._all_devices)
        if m_cur in self._all_devices: self.masterBox.setCurrentText(m_cur)
        self.masterBox.blockSignals(False)

        s1_list = [d for d in self._all_devices if d != self.masterBox.currentText()]
        self.slave1Box.blockSignals(True)
        self.slave1Box.clear(); self.slave1Box.addItems(s1_list if s1_list else ["(auto)"])
        if s1_cur in s1_list: self.slave1Box.setCurrentText(s1_cur)
        self.slave1Box.blockSignals(False)

        s2_list = [d for d in self._all_devices if d not in (self.masterBox.currentText(), self.slave1Box.currentText())]
        self.slave2Box.blockSignals(True)
        self.slave2Box.clear(); self.slave2Box.addItems(s2_list if s2_list else ["(auto)"])
        if s2_cur in s2_list: self.slave2Box.setCurrentText(s2_cur)
        self.slave2Box.blockSignals(False)

        if hasattr(self, "lblSDRs"): self._update_sdr_label()
        if hasattr(self, "btnTrilat"): self._update_source_state()

    def _update_source_state(self):
        parsing = (self.acqBox.currentIndex() == 0) if hasattr(self, "acqBox") else True
        sel = []
        for cb in (getattr(self, "masterBox", None), getattr(self, "slave1Box", None), getattr(self, "slave2Box", None)):
            if cb and cb.count():
                s = cb.currentText().strip()
                if s and s != "(auto)":
                    sel.append(s)
        have_three = (len(set(sel)) >= 3)

        if hasattr(self, "btnTrilat"):
            self.btnTrilat.setEnabled((not parsing) and have_three)

        if hasattr(self, "slave1Box"): self.slave1Box.setEnabled(not parsing)
        if hasattr(self, "slave2Box"): self.slave2Box.setEnabled(not parsing)
        if hasattr(self, "lblNeedSlaves"): self.lblNeedSlaves.setVisible(parsing or not have_three)
        if hasattr(self, "lblSDRs"): self._update_sdr_label()

        # блокировка «Карты» при разборе hackrf_sweep
        if hasattr(self, "an_tab"):
            self.an_tab.setEnabled(not parsing)

    def _update_sdr_label(self):
        if not hasattr(self, "lblSDRs"): return
        m = self.masterBox.currentText() if self.masterBox.count() else "(auto)"
        s1 = self.slave1Box.currentText() if self.slave1Box.count() else "(auto)"
        s2 = self.slave2Box.currentText() if self.slave2Box.count() else "(auto)"
        self.lblSDRs.setText(f"Master: {m}    |    Slave-1: {s1}    |    Slave-2: {s2}")

    # ---- DA helpers
    def _add_preset_ranges(self, rngs):
        for a,b in rngs:
            self._append_range_row(a,b)
        self._merge_ranges()

    def _append_range_row(self, start_mhz: float, stop_mhz: float):
        s = max(self.MIN_MHZ, min(self.MAX_MHZ, float(start_mhz)))
        e = max(self.MIN_MHZ, min(self.MAX_MHZ, float(stop_mhz)))
        if e < s: s, e = e, s
        r = self.tblRanges.rowCount(); self.tblRanges.insertRow(r)
        it1 = QTableWidgetItem(f"{s:.3f}"); it2 = QTableWidgetItem(f"{e:.3f}")
        self.tblRanges.setItem(r,0,it1); self.tblRanges.setItem(r,1,it2)
        self._refresh_range_overlays()

    def _on_add_range(self):
        try:
            s = float(self.fStart.text()); e = float(self.fStop.text())
        except Exception:
            s, e = 2400.000, 2483.500
        self._append_range_row(s,e)

    def _on_del_range(self):
        rows = sorted({i.row() for i in self.tblRanges.selectedIndexes()}, reverse=True)
        for r in rows: self.tblRanges.removeRow(r)
        self._refresh_range_overlays()

    def _on_range_item_changed(self, it: QTableWidgetItem):
        try:
            v = float(it.text().replace(",", "."))
        except Exception:
            v = self.MIN_MHZ
        v = max(self.MIN_MHZ, min(self.MAX_MHZ, v))
        self.tblRanges.blockSignals(True)
        it.setText(f"{v:.3f}")
        self.tblRanges.blockSignals(False)
        self._refresh_range_overlays()

    def _ranges_to_json(self):
        out=[]
        for r in range(self.tblRanges.rowCount()):
            try:
                s = float(self.tblRanges.item(r,0).text()); e = float(self.tblRanges.item(r,1).text())
            except Exception:
                continue
            out.append({"start_mhz": s, "stop_mhz": e})
        return out

    def _merge_ranges(self):
        eps = 1e-6
        arr=[]
        for r in range(self.tblRanges.rowCount()):
            try:
                s=float(self.tblRanges.item(r,0).text()); e=float(self.tblRanges.item(r,1).text())
            except Exception:
                continue
            if e<s: s,e=e,s
            arr.sort()
            arr.append((s,e))
        arr.sort()
        merged=[]
        for s,e in arr:
            if not merged or s>merged[-1][1]+eps:
                merged.append([s,e])
            else:
                merged[-1][1]=max(merged[-1][1], e)
        self.tblRanges.blockSignals(True)
        self.tblRanges.setRowCount(0)
        for s,e in merged:
            self._append_range_row(s,e)
        self.tblRanges.blockSignals(False)
        self._refresh_range_overlays()

    def _clear_range_overlays(self):
        if not hasattr(self, "_range_regions"):
            self._range_regions=[]
            return
        for reg_main, reg_wf in self._range_regions:
            try: self.plot.removeItem(reg_main)
            except Exception: pass
            try: self.wfPlot.removeItem(reg_wf)
            except Exception: pass
        self._range_regions.clear()

    def _refresh_range_overlays(self):
        if not hasattr(self, "_range_regions"): self._range_regions=[]
        self._clear_range_overlays()
        brush = pg.mkBrush(255, 255, 0, 60)
        pen = pg.mkPen(255, 220, 0, 120)
        for r in range(self.tblRanges.rowCount()):
            try:
                s=float(self.tblRanges.item(r,0).text()); e=float(self.tblRanges.item(r,1).text())
            except Exception:
                continue
            reg1 = pg.LinearRegionItem(values=(s,e), orientation=pg.LinearRegionItem.Vertical, brush=brush, pen=pen)
            reg2 = pg.LinearRegionItem(values=(s,e), orientation=pg.LinearRegionItem.Vertical, brush=brush, pen=pen)
            reg1.setMovable(False); reg2.setMovable(False); reg1.setZValue(-10); reg2.setZValue(-10)
            self.plot.addItem(reg1); self.wfPlot.addItem(reg2)
            self._range_regions.append((reg1,reg2))

    # -------------------- menu --------------------
    def _build_menu(self):
        m = self.menuBar().addMenu("&Файл")
        act_png = QtWidgets.QAction("Экспорт водопада в PNG…", self)
        act_bmp = QtWidgets.QAction("Экспорт водопада в BMP…", self)
        act_csv = QtWidgets.QAction("Экспорт текущего свипа в CSV…", self)
        act_quit = QtWidgets.QAction("Выход", self); act_quit.setShortcut("Ctrl+Q")
        act_png.triggered.connect(lambda: self._export_waterfall("png"))
        act_bmp.triggered.connect(lambda: self._export_waterfall("bmp"))
        act_csv.triggered.connect(self._export_csv_current_sweep)
        act_quit.triggered.connect(self.close)
        m.addAction(act_png); m.addAction(act_bmp); m.addSeparator()
        m.addAction(act_csv); m.addSeparator(); m.addAction(act_quit)

        tools = self.menuBar().addMenu("&Инструменты")
        act_peaks = QtWidgets.QAction("Пики…", self)
        tools.addAction(act_peaks)
        act_peaks.triggered.connect(self._open_peaks)

        h = self.menuBar().addMenu("&Справка")
        act_doc = QtWidgets.QAction("Документация (пусто)…", self)
        act_doc.triggered.connect(lambda: QtWidgets.QMessageBox.information(self,"Документация","Будет позже."))
        h.addAction(act_doc)

        act_hotkeys = QtWidgets.QAction("Горячие клавиши…", self)
        def _show_hotkeys():
            QtWidgets.QMessageBox.information(
                self, "Горячие клавиши",
                "Горизонтальный зум + : клавиша '+'\n"
                "Горизонтальный зум − : клавиша '-'\n"
                "Сброс диапазона: R\n"
                "Даблклик ЛКМ на спектре/водопаде: добавить метку\n"
                "Даблклик строка в «Пики»: перейти к частоте\n"
            )
        act_hotkeys.triggered.connect(_show_hotkeys)
        h.addAction(act_hotkeys)

    # -------------------- export helpers --------------------
    def _export_waterfall(self, fmt="png"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        default = f"waterfall_{self.fStart.text()}-{self.fStop.text()}MHz_{ts}.{fmt}"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, f"Сохранить {fmt.upper()}",
                                                        default, f"{fmt.upper()} (*.{fmt})")
        if not path: return
        pixmap = self.wfLayout.grab()
        ok = pixmap.save(path, fmt.upper())
        self.statusBar().showMessage(f"Сохранено {fmt.upper()} → {path}" if ok else f"Ошибка сохранения {fmt.upper()}")

    def _export_csv_current_sweep(self):
        if self._last_sweep_freqs is None or self._last_sweep_power is None:
            QtWidgets.QMessageBox.information(self, "Экспорт CSV", "Нет готового свипа для экспорта."); return
        f = self._last_sweep_freqs; p = self._last_sweep_power
        if len(f) < 2:
            QtWidgets.QMessageBox.information(self, "Экспорт CSV", "Свип слишком короткий."); return
        bin_hz = int(round(float(f[1]-f[0])))
        hz_low  = int(round(float(f[0] - 0.5*bin_hz)))
        hz_high = int(round(float(f[-1] + 0.5*bin_hz)))
        date_str = time.strftime("%Y-%m-%d"); time_str = time.strftime("%H:%M:%S")
        vals = ",".join(f"{v:.2f}" for v in p.tolist())
        line = f"{date_str}, {time_str}, {hz_low}, {hz_high}, {bin_hz}, {len(p)}, {vals}\n"
        ts = time.strftime("%Y%m%d_%H%M%S")
        default = f"sweep_{self.fStart.text()}-{self.fStop.text()}MHz_{ts}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить CSV", default, "CSV (*.csv)")
        if not path: return
        try:
            with open(path, "w", newline="") as fh: fh.write(line)
            self.statusBar().showMessage(f"Saved CSV → {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Экспорт CSV", f"Ошибка записи: {e}")

    # -------------------- zoom/pan --------------------
    def reset_view(self):
        if self.cfg_f0_mhz is None: return
        self.plot.setXRange(self.cfg_f0_mhz, self.cfg_f1_mhz, padding=0)
        self.wfPlot.setXRange(self.cfg_f0_mhz, self.cfg_f1_mhz, padding=0)

    def zoom_x(self, factor: float):
        if self.cfg_f0_mhz is None or self.cfg_f1_mhz is None:
            return
        vb = self.plot.getViewBox()
        x0, x1 = vb.viewRange()[0]; cx = 0.5*(x0+x1); half = 0.5*(x1-x0)*factor
        nx0, nx1 = max(self.cfg_f0_mhz, cx-half), min(self.cfg_f1_mhz, cx+half)
        if nx1 - nx0 > 1e-6:
            self.plot.setXRange(nx0, nx1, padding=0)

    # -------------------- axes init --------------------
    def _init_axes_once(self, f_grid_hz: np.ndarray):
        f0_mhz = float(self.cfg_f0_mhz); f1_mhz = float(self.cfg_f1_mhz)
        self._bin_w_mhz = float((f_grid_hz[1]-f_grid_hz[0])/1e6) if len(f_grid_hz)>1 else 1.0
        self.plot.setXRange(f0_mhz, f1_mhz, padding=0); self.plot.setYRange(self.level_min, self.level_max)
        self.wfPlot.setXRange(f0_mhz, f1_mhz, padding=0); self.wfPlot.setYRange(0, self.wfRows)
        self.wfImg.resetTransform()
        tr = QTransform(); tr.scale(self._bin_w_mhz, -1.0)
        self.wfImg.setTransform(tr, False)
        self.wfImg.setPos(f0_mhz, self.wfRows)
        self.wfImg.setLevels([self.level_min, self.level_max])
        self._axes_ready = True

    # -------------------- viz helpers --------------------
    def _apply_cmap(self, name):
        lmin, lmax = self.level_min, self.level_max
        lut = get_colormap(name)
        self.wfImg.setLookupTable(lut)
        self.wfImg.setLevels([lmin, lmax])
        if self.wfBuffer is not None:
            self.wfImg.setImage(self.wfBuffer, autoLevels=False, levels=(lmin, lmax))
        self.settings.setValue("cmap", name)

    def _apply_levels_from_spins(self):
        self.level_min = float(self.minSpin.value()); self.level_max = float(self.maxSpin.value())
        if self.level_max <= self.level_min + 0.1: self.level_max = self.level_min + 0.1; self.maxSpin.setValue(self.level_max)
        self.wfImg.setLevels([self.level_min, self.level_max])
        if self.wfBuffer is not None:
            self.wfImg.setImage(self.wfBuffer, autoLevels=False, levels=(self.level_min, self.level_max))

    def _auto_levels(self):
        arr = self.wfBuffer if self.wfBuffer is not None else None
        if arr is None or arr.size == 0: return
        lo = float(np.percentile(arr, 5)); hi = float(np.percentile(arr, 99))
        self.minSpin.setValue(lo); self.maxSpin.setValue(hi); self._apply_levels_from_spins()

    def _smooth_if_needed(self, y: np.ndarray) -> np.ndarray:
        if not self.chkSmooth.isChecked(): return y
        w = max(1, int(self.smoothWin.value()))
        if w % 2 == 0: w += 1
        if w <= 1: return y
        k = np.ones(w, dtype=np.float32) / float(w)
        return np.convolve(y, k, mode="same").astype(np.float32)

    # ---------- Peaks ----------
    def _open_peaks(self):
        if getattr(self, "_peaks_dlg", None) is None:
            self._peaks_dlg = PeaksDialog(self)
            self._peaks_dlg.jumpToFreq.connect(self._jump_to_freq)
        self._peaks_dlg.show()

    def _jump_to_freq(self, f_mhz: float):
        vb = self.plot.getViewBox()
        x0, x1 = vb.viewRange()[0]; half = 0.5*(x1-x0)
        self.plot.setXRange(f_mhz - half, f_mhz + half, padding=0)
        self.wfPlot.setXRange(f_mhz - half, f_mhz + half, padding=0)

    def _find_peaks_simple(self, x_mhz: np.ndarray, y_dbm: np.ndarray, min_dbm: float, min_sep_khz: float):
        if x_mhz.size < 3: return []
        peaks=[]
        for i in range(1, len(y_dbm)-1):
            if y_dbm[i] > y_dbm[i-1] and y_dbm[i] > y_dbm[i+1] and y_dbm[i] >= min_dbm:
                th = y_dbm[i] - 3.0
                l=i
                while l>0 and y_dbm[l] > th: l-=1
                r=i
                while r<len(y_dbm)-1 and y_dbm[r] > th: r+=1
                width_khz = (x_mhz[max(r, i)] - x_mhz[min(l, i)])*1000.0
                peaks.append((x_mhz[i], y_dbm[i], width_khz, i))
        peaks.sort(key=lambda t: -t[1])
        filtered=[]
        sep = min_sep_khz/1000.0  # МГц
        for p in peaks:
            if not filtered or all(abs(p[0]-q[0]) >= sep for q in filtered):
                filtered.append(p)
        return filtered

    # ---------- markers ----------
    def _on_mouse_click_add_marker(self, ev, vb):
        if not ev.double() or ev.button() != QtCore.Qt.LeftButton:
            return
        pos = vb.mapSceneToView(ev.scenePos())
        f_mhz = float(pos.x())
        if f_mhz < self.MIN_MHZ - 1.0 or f_mhz > self.MAX_MHZ + 1.0:
            return
        suggested = f"Метка {f_mhz:.3f}"
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Новая метка",
            f"Частота: {f_mhz:.6f} МГц\nНазвание метки:",
            QtWidgets.QLineEdit.Normal, suggested
        )
        if not ok or not name.strip():
            return
        self._add_marker(f_mhz, self._marker_color.name(), name.strip())
        self.statusBar().showMessage(f"Добавлена метка “{name.strip()}” @ {f_mhz:.6f} МГц")

    def _update_color_button(self):
        pm = QPixmap(24, 24); pm.fill(self._marker_color)
        self.btnColorPick.setIcon(QtGui.QIcon(pm))
    def _pick_marker_color(self):
        c = QColorDialog.getColor(self._marker_color, self, "Выбрать цвет маркера")
        if c.isValid(): self._marker_color = c; self._update_color_button()
    def _next_marker_id(self): self._marker_id_seq += 1; return self._marker_id_seq
    def _add_marker_from_ui(self):
        try: f_mhz = float(self.edMarkerFreq.text())
        except ValueError:
            QtWidgets.QMessageBox.information(self, "Маркеры", "Укажи корректную частоту в МГц."); return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Название метки",
            f"Частота: {f_mhz:.6f} МГц\nНазвание:",
            QtWidgets.QLineEdit.Normal, f"Метка {f_mhz:.3f}"
        )
        if not ok or not name.strip():
            return
        self._add_marker(f_mhz, self._marker_color.name(), name.strip())

    def _add_marker(self, f_mhz: float, color_hex: str, name: str = None):
        pen = pg.mkPen(QColor(color_hex), width=2, style=QtCore.Qt.DashLine)
        l1 = pg.InfiniteLine(pos=f_mhz, angle=90, movable=True, pen=pen)
        l2 = pg.InfiniteLine(pos=f_mhz, angle=90, movable=True, pen=pen)
        self.plot.addItem(l1); self.wfPlot.addItem(l2)
        mid = self._next_marker_id()
        if not name:
            name = f"M{mid}"
        self.markers[mid] = {
            "freq": float(f_mhz),
            "color": color_hex,
            "name": str(name),
            "l_main": l1,
            "l_wf": l2
        }
        def on_move_main(): self._set_marker_freq(mid, float(l1.value()), source="main")
        def on_move_wf():   self._set_marker_freq(mid, float(l2.value()), source="wf")
        l1.sigPositionChanged.connect(on_move_main); l2.sigPositionChanged.connect(on_move_wf)
        self._append_marker_to_list(mid); self._save_markers()

    def _append_marker_to_list(self, mid: int):
        m = self.markers[mid]; txt = f"{mid}: {m.get('name','')} — {m['freq']:.6f} MHz  [{m['color']}]"
        item = QtWidgets.QListWidgetItem(txt); item.setData(QtCore.Qt.UserRole, mid)
        pm = QPixmap(14, 14); pm.fill(QColor(m['color'])); item.setIcon(QtGui.QIcon(pm))
        self.listMarkers.addItem(item)

    def _refresh_marker_in_list(self, mid: int):
        for i in range(self.listMarkers.count()):
            it = self.listMarkers.item(i)
            if it.data(QtCore.Qt.UserRole) == mid:
                m = self.markers[mid]
                it.setText(f"{mid}: {m.get('name','')} — {m['freq']:.6f} MHz  [{m['color']}]")
                pm = QPixmap(14, 14); pm.fill(QColor(m['color'])); it.setIcon(QtGui.QIcon(pm))
                break

    def _set_marker_freq(self, mid: int, f_mhz: float, source: str):
        m = self.markers.get(mid)
        if m is None: return
        m['freq'] = float(f_mhz)
        if source == "main": m['l_wf'].setValue(f_mhz)
        else:                m['l_main'].setValue(f_mhz)
        self._refresh_marker_in_list(mid); self._save_markers()

    def _remove_selected_markers(self):
        ids = [it.data(QtCore.Qt.UserRole) for it in self.listMarkers.selectedItems()]
        if not ids: return
        for mid in ids:
            m = self.markers.pop(mid, None)
            if not m: continue
            try: self.plot.removeItem(m['l_main']); self.wfPlot.removeItem(m['l_wf'])
            except Exception: pass
        for it in list(self.listMarkers.selectedItems()):
            self.listMarkers.takeItem(self.listMarkers.row(it))
        self._save_markers()

    def _center_on_marker(self, item: QtWidgets.QListWidgetItem):
        mid = item.data(QtCore.Qt.UserRole); m = self.markers.get(mid)
        if not m: return
        f = m['freq']; vb = self.plot.getViewBox()
        x0, x1 = vb.viewRange()[0]; half = 0.5*(x1-x0)
        self.plot.setXRange(f - half, f + half, padding=0)
        self.wfPlot.setXRange(f - half, f + half, padding=0)

    def _save_markers(self):
        arr = [{"freq": m["freq"], "color": m["color"], "name": m.get("name","")} for m in self.markers.values()]
        self.settings.setValue("markers_json", json.dumps(arr))

    def _load_markers(self):
        s = self.settings.value("markers_json", "")
        if not s: return
        try: arr = json.loads(s)
        except Exception: return
        for m in arr:
            self._add_marker(float(m.get("freq", 0.0)),
                             str(m.get("color", "#ff5252")),
                             str(m.get("name", "")))

    # -------------------- slots --------------------
    @QtCore.pyqtSlot()
    def on_start(self):
        if self.worker is not None: return
        self._format_line_edit(self.fStart); self._format_line_edit(self.fStop)
        try:
            self.cfg_f0_mhz = float(self.fStart.text()); self.cfg_f1_mhz = float(self.fStop.text()); bw = float(self.binHz.text())
        except ValueError:
            self.statusBar().showMessage("Неверные параметры"); return

        sel = self.masterBox.currentText().strip() if hasattr(self,"masterBox") and self.masterBox.count() else "(auto)"
        serial = "" if (not sel or sel == "(auto)") else sel

        self._avg_queue.clear(); self._avg_line=None; self._minhold=None; self._maxhold=None
        self.wfBuffer=None; self._axes_ready=False; self._last_sweep_t=None; self._dt_hist.clear()
        self._last_sweep_freqs=None; self._last_sweep_power=None

        method = self.acqBox.currentIndex() if hasattr(self,"acqBox") else 0
        if method==0:
            self.worker = SweepWorker(self.cfg_f0_mhz, self.cfg_f1_mhz, bw, self.lnaBox.value(), self.vgaBox.value(), serial_suffix=serial)
        else:
            self.worker = LibWorker(self.cfg_f0_mhz, self.cfg_f1_mhz, bw, self.lnaBox.value(), self.vgaBox.value(), serial_suffix=serial)

        self.worker.spectrumReady.connect(self.on_sweep)
        self.worker.status.connect(lambda s: self.statusBar().showMessage(s))
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.btnStart.setEnabled(False); self.btnStop.setEnabled(True)
        self.worker.start(); self.statusBar().showMessage("Запуск...")

    @QtCore.pyqtSlot()
    def on_stop(self):
        if self.worker: self.worker.stop()
        self.btnStop.setEnabled(False)

    @QtCore.pyqtSlot(str)
    def on_error(self, msg):
        QtWidgets.QMessageBox.critical(self,"Ошибка",msg)
        self.statusBar().showMessage(msg)

    @QtCore.pyqtSlot()
    def on_worker_finished(self):
        self.worker=None; self.btnStart.setEnabled(True); self.btnStop.setEnabled(False)
        self.statusBar().showMessage("Остановлено")

    @QtCore.pyqtSlot(object, object)
    def on_sweep(self, f_grid_hz: np.ndarray, p_grid_dbm: np.ndarray):
        if (not self._axes_ready) or (self.wfBuffer is None) or (self.wfBuffer.shape[1] != len(f_grid_hz)):
            self.wfBuffer = np.full((self.wfRows, len(f_grid_hz)), self.level_min, dtype=np.float32)
            self._init_axes_once(f_grid_hz)

        y_proc = p_grid_dbm.copy()

        self._last_sweep_freqs = f_grid_hz.copy(); self._last_sweep_power = y_proc.copy()

        now_t = time.time()
        if self._last_sweep_t is not None:
            self._dt_hist.append(max(1e-3, now_t - self._last_sweep_t))
            if self._dt_hist: self.timeAxis.dt_est = float(np.median(self._dt_hist))
        self._last_sweep_t = now_t

        x_mhz = f_grid_hz / 1e6
        self._avg_queue.append(y_proc.copy())
        if len(self._avg_queue) > self.avgSpin.value(): self._avg_queue.popleft()

        y_now = self._smooth_if_needed(y_proc); self.curve_now.setData(x_mhz, y_now)

        if self.chkAvg.isChecked() and self._avg_queue:
            y_avg = np.mean(self._avg_queue, axis=0).astype(np.float32)
            y_avg = self._smooth_if_needed(y_avg)
            self.curve_avg.setData(x_mhz, y_avg)

        if self._maxhold is None: self._maxhold = y_proc.copy()
        else: self._maxhold = np.maximum(self._maxhold, y_proc)
        if self._minhold is None: self._minhold = y_proc.copy()
        else: self._minhold = np.minimum(self._minhold, y_proc)

        self.curve_max.setData(x_mhz, self._maxhold); self.curve_min.setData(x_mhz, self._minhold)
        self.curve_now.setVisible(self.chkMain.isChecked()); self.curve_max.setVisible(self.chkMax.isChecked())
        self.curve_min.setVisible(self.chkMin.isChecked()); self.curve_avg.setVisible(self.chkAvg.isChecked())

        if self.chkPersist.isChecked():
            decay = max(0.0, min(0.05, float(self.persistDecay.value())))
            self.wfBuffer *= (1.0 - decay)

        self.wfBuffer = np.roll(self.wfBuffer, 1, axis=0)
        self.wfBuffer[0, :] = self._smooth_if_needed(y_proc)
        self.wfImg.setImage(self.wfBuffer, autoLevels=False, levels=(self.level_min, self.level_max))

        if getattr(self, "_peaks_dlg", None) and self._peaks_dlg.isVisible() and self._peaks_dlg.chkAuto.isChecked():
            peaks = self._find_peaks_simple(x_mhz, y_now, self._peaks_dlg.spnMinDbm.value(), self._peaks_dlg.spnMinSep.value())
            self._peaks_dlg.set_peaks([(float(f), float(lv), float(w), int(idx)) for (f,lv,w,idx) in peaks])

        now = time.time()
        if now - self._status_last > 0.8:
            self._status_last = now
            bin_mhz = (f_grid_hz[1]-f_grid_hz[0])/1e6 if len(f_grid_hz)>1 else 0.0
            fps = 1.0 / max(1e-9, self.timeAxis.dt_est)
            self.statusBar().showMessage(
                f"Свип: {len(f_grid_hz)} бинов, {self.cfg_f0_mhz:.3f}–{self.cfg_f1_mhz:.3f} МГц; bin {bin_mhz:.3f} МГц; ~{fps:.2f} свип/с"
            )

    # ---- trilateration
    def _start_trilateration(self):
        if not hasattr(self, "acqBox") or self.acqBox.currentIndex()==0:
            QtWidgets.QMessageBox.information(self, "Трилатерация", "Для трилатерации нужен библиотечный режим (libhackrf) и 2 slave устройства.")
            return
        sel = []
        for cb in (self.masterBox, self.slave1Box, self.slave2Box):
            if cb and cb.count():
                s = cb.currentText().strip()
                if s and s != "(auto)":
                    sel.append(s)
        if len(set(sel)) < 3:
            QtWidgets.QMessageBox.information(self, "Трилатерация", "Нужно выбрать Master и 2 уникальных Slave.")
            return
        m = self.masterBox.currentText() if self.masterBox.count() else "(auto)"
        s1 = self.slave1Box.currentText() if self.slave1Box.count() else "(auto)"
        s2 = self.slave2Box.currentText() if self.slave2Box.count() else "(auto)"
        self.triWin = TrilaterationWindow(m, s1, s2)
        self.triWin.show()

    # ---- курсор подпись ----
    def _on_mouse_moved(self, ev):
        pos = self.plot.getViewBox().mapSceneToView(ev)
        f_mhz = float(pos.x())
        y_dbm = None
        if self._last_sweep_freqs is not None and self._last_sweep_power is not None and len(self._last_sweep_freqs) > 1:
            x = self._last_sweep_freqs / 1e6
            idx = int(np.clip(np.searchsorted(x, f_mhz), 1, len(x)-1))
            j = idx if abs(x[idx] - f_mhz) < abs(x[idx-1] - f_mhz) else idx-1
            y_dbm = float(self._last_sweep_power[j])
        txt = f"{f_mhz:.3f} МГц"
        if y_dbm is not None:
            txt += f", {y_dbm:.1f} дБм"
        self._cursorText.setText(txt)
        self._cursorText.setPos(f_mhz, self.level_max - 0.2)

# -------------------- main --------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName(APP_ORG); app.setApplicationName(APP_NAME)
    win = MainWindow(); win.resize(1380, 860); win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    print("hackrf_sweep:", shutil.which("hackrf_sweep"), flush=True)
    main()
