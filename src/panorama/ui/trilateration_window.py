from PyQt5 import QtWidgets
import pyqtgraph as pg


class TrilaterationWindow(QtWidgets.QMainWindow):
    def __init__(self, master_serial, slave1_serial, slave2_serial):
        super().__init__()
        self.setWindowTitle("Трилатерация (макет)")
        self.master_serial = master_serial or "(auto)"
        self.slave1_serial = slave1_serial or "(auto)"
        self.slave2_serial = slave2_serial or "(auto)"

        w = QtWidgets.QWidget(self)
        self.setCentralWidget(w)
        h = QtWidgets.QHBoxLayout(w)

        left = QtWidgets.QVBoxLayout(); h.addLayout(left)
        grid = QtWidgets.QFormLayout(); left.addLayout(grid)
        self.lblM = QtWidgets.QLabel(f"Master: {self.master_serial}")
        self.edMx = QtWidgets.QDoubleSpinBox(); self.edMy = QtWidgets.QDoubleSpinBox(); self.edMz = QtWidgets.QDoubleSpinBox()
        for ed in (self.edMx, self.edMy, self.edMz):
            ed.setRange(-10000, 10000); ed.setDecimals(2); ed.setValue(0.0); ed.setEnabled(False)
        grid.addRow(self.lblM); grid.addRow("M X/Y/Z:", self._row3(self.edMx, self.edMy, self.edMz))

        self.lblS1 = QtWidgets.QLabel(f"Slave-1: {self.slave1_serial}")
        self.edS1x = QtWidgets.QDoubleSpinBox(); self.edS1y = QtWidgets.QDoubleSpinBox(); self.edS1z = QtWidgets.QDoubleSpinBox()
        for ed in (self.edS1x, self.edS1y, self.edS1z):
            ed.setRange(-10000, 10000); ed.setDecimals(2)
        grid.addRow(self.lblS1); grid.addRow("S1 X/Y/Z:", self._row3(self.edS1x, self.edS1y, self.edS1z))

        self.lblS2 = QtWidgets.QLabel(f"Slave-2: {self.slave2_serial}")
        self.edS2x = QtWidgets.QDoubleSpinBox(); self.edS2y = QtWidgets.QDoubleSpinBox(); self.edS2z = QtWidgets.QDoubleSpinBox()
        for ed in (self.edS2x, self.edS2y, self.edS2z):
            ed.setRange(-10000, 10000); ed.setDecimals(2)
        grid.addRow(self.lblS2); grid.addRow("S2 X/Y/Z:", self._row3(self.edS2x, self.edS2y, self.edS2z))

        view = pg.PlotWidget(); h.addWidget(view, 1)
        view.setLabel("bottom", "X")
        view.setLabel("left", "Y")
        self.plot = view.getPlotItem()

    def _row3(self, a, b, c):
        w = QtWidgets.QWidget(); l = QtWidgets.QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(a); l.addWidget(b); l.addWidget(c)
        return w
