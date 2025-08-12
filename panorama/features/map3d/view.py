from __future__ import annotations
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from .model import MapModel

class MapView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = MapModel()

        # ---- левая часть: граф (карта) ----
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "X")
        self.plot.setLabel("left", "Y")
        self.plot.setAspectLocked(lock=False)

        # Курсор и подпись координат
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((80,80,80,120)))
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((80,80,80,120)))
        self.plot.addItem(self._vline, ignoreBounds=True)
        self.plot.addItem(self._hline, ignoreBounds=True)
        self.cursor_lbl = QtWidgets.QLabel("X: —, Y: —")
        self.cursor_lbl.setStyleSheet("color:#333;")
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # Точки узлов
        self._sc_master = pg.ScatterPlotItem(size=12, brush=self.model.master.color, pen="k")
        self._sc_s1     = pg.ScatterPlotItem(size=12, brush=self.model.slave1.color, pen="k")
        self._sc_s2     = pg.ScatterPlotItem(size=12, brush=self.model.slave2.color, pen="k")
        for sc in (self._sc_master, self._sc_s1, self._sc_s2):
            self.plot.addItem(sc)

        # Подписи
        self._txt_master = pg.TextItem("M", color=self.model.master.color, anchor=(0.5,1.5))
        self._txt_s1     = pg.TextItem("S1", color=self.model.slave1.color, anchor=(0.5,1.5))
        self._txt_s2     = pg.TextItem("S2", color=self.model.slave2.color, anchor=(0.5,1.5))
        for t in (self._txt_master, self._txt_s1, self._txt_s2):
            self.plot.addItem(t)

        # ---- правая панель: координаты ----
        right = QtWidgets.QVBoxLayout()

        grid = QtWidgets.QGridLayout()
        # Master (фиксированный)
        grid.addWidget(QtWidgets.QLabel("<b>Master</b> (фикс.)"), 0, 0, 1, 2)
        self.m_x = QtWidgets.QDoubleSpinBox(); self.m_x.setRange(-1e6, 1e6); self.m_x.setValue(0.0); self.m_x.setEnabled(False)
        self.m_y = QtWidgets.QDoubleSpinBox(); self.m_y.setRange(-1e6, 1e6); self.m_y.setValue(0.0); self.m_y.setEnabled(False)
        grid.addWidget(QtWidgets.QLabel("X"), 1, 0); grid.addWidget(self.m_x, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Y"), 2, 0); grid.addWidget(self.m_y, 2, 1)

        # Slave1
        grid.addWidget(QtWidgets.QLabel("<b>Slave 1</b>"), 3, 0, 1, 2)
        self.s1_x = QtWidgets.QDoubleSpinBox(); self.s1_x.setRange(-1e6, 1e6); self.s1_x.setValue(self.model.slave1.x)
        self.s1_y = QtWidgets.QDoubleSpinBox(); self.s1_y.setRange(-1e6, 1e6); self.s1_y.setValue(self.model.slave1.y)
        grid.addWidget(QtWidgets.QLabel("X"), 4, 0); grid.addWidget(self.s1_x, 4, 1)
        grid.addWidget(QtWidgets.QLabel("Y"), 5, 0); grid.addWidget(self.s1_y, 5, 1)

        # Slave2
        grid.addWidget(QtWidgets.QLabel("<b>Slave 2</b>"), 6, 0, 1, 2)
        self.s2_x = QtWidgets.QDoubleSpinBox(); self.s2_x.setRange(-1e6, 1e6); self.s2_x.setValue(self.model.slave2.x)
        self.s2_y = QtWidgets.QDoubleSpinBox(); self.s2_y.setRange(-1e6, 1e6); self.s2_y.setValue(self.model.slave2.y)
        grid.addWidget(QtWidgets.QLabel("X"), 7, 0); grid.addWidget(self.s2_x, 7, 1)
        grid.addWidget(QtWidgets.QLabel("Y"), 8, 0); grid.addWidget(self.s2_y, 8, 1)

        # Кнопка (пока заглушка)
        self.btn_trilat = QtWidgets.QPushButton("Начать трилатерацию")
        self.btn_trilat.setEnabled(False)
        grid.addWidget(self.btn_trilat, 9, 0, 1, 2)

        right.addLayout(grid)
        right.addSpacing(10)

        # ---- финальная раскладка ----
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.cursor_lbl)
        left.addWidget(self.plot, stretch=1)

        root = QtWidgets.QHBoxLayout(self)
        root.addLayout(left, stretch=3)
        root.addLayout(right, stretch=2)

        # Сигналы изменения координат
        self.s1_x.valueChanged.connect(self._on_coords_changed)
        self.s1_y.valueChanged.connect(self._on_coords_changed)
        self.s2_x.valueChanged.connect(self._on_coords_changed)
        self.s2_y.valueChanged.connect(self._on_coords_changed)

        self._refresh_nodes()
        self._auto_view()

    # --- helpers ---
    def _on_coords_changed(self, _v):
        self.model.slave1.x = float(self.s1_x.value())
        self.model.slave1.y = float(self.s1_y.value())
        self.model.slave2.x = float(self.s2_x.value())
        self.model.slave2.y = float(self.s2_y.value())
        self._refresh_nodes()

    def _refresh_nodes(self):
        m = self.model.master; s1 = self.model.slave1; s2 = self.model.slave2
        self._sc_master.setData([m.x], [m.y])
        self._sc_s1.setData([s1.x], [s1.y])
        self._sc_s2.setData([s2.x], [s2.y])
        self._txt_master.setPos(m.x, m.y)
        self._txt_s1.setPos(s1.x, s1.y)
        self._txt_s2.setPos(s2.x, s2.y)

    def _auto_view(self):
        xs = [self.model.master.x, self.model.slave1.x, self.model.slave2.x]
        ys = [self.model.master.y, self.model.slave1.y, self.model.slave2.y]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        pad_x = max(1.0, (xmax - xmin) * 0.2)
        pad_y = max(1.0, (ymax - ymin) * 0.2)
        vb = self.plot.getViewBox()
        vb.setXRange(xmin - pad_x, xmax + pad_x, padding=0)
        vb.setYRange(ymin - pad_y, ymax + pad_y, padding=0)

    def _on_mouse_moved(self, pos):
        vb = self.plot.getViewBox()
        if not vb:
            return
        p = vb.mapSceneToView(pos)
        self._vline.setPos(p.x())
        self._hline.setPos(p.y())
        self.cursor_lbl.setText(f"X: {p.x():.2f}, Y: {p.y():.2f}")
