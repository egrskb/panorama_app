import csv
from typing import List, Tuple

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem

Peak = Tuple[float, float, float, int]


class PeaksDialog(QtWidgets.QDialog):
    jumpToFreq = QtCore.pyqtSignal(float)

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

        ctl1 = QtWidgets.QHBoxLayout()
        self.spnMinDbm = QtWidgets.QDoubleSpinBox(); self.spnMinDbm.setRange(-200, 100); self.spnMinDbm.setValue(-80.0)
        self.spnMinSep = QtWidgets.QDoubleSpinBox(); self.spnMinSep.setRange(0.0, 5000.0); self.spnMinSep.setDecimals(1); self.spnMinSep.setValue(30.0)
        self.btnExport = QtWidgets.QPushButton("Экспорт CSV…")
        ctl1.addWidget(QtWidgets.QLabel("Порог, дБм:")); ctl1.addWidget(self.spnMinDbm)
        ctl1.addWidget(QtWidgets.QLabel("Мин. разделение, кГц:")); ctl1.addWidget(self.spnMinSep)
        ctl1.addStretch(1); ctl1.addWidget(self.btnExport)
        v.addLayout(ctl1)

        ctl2 = QtWidgets.QHBoxLayout()
        self.edFreqs = QtWidgets.QLineEdit(); self.edFreqs.setPlaceholderText("Частоты, МГц (через запятую)")
        self.btnFind = QtWidgets.QPushButton("Найти в таблице")
        ctl2.addWidget(self.edFreqs); ctl2.addWidget(self.btnFind)
        v.addLayout(ctl2)

        self.table.doubleClicked.connect(self._jump_selected)
        self.btnExport.clicked.connect(self._export_csv)
        self.btnFind.clicked.connect(self._find_freqs)

        self._last_peaks: List[Peak] = []

    def _jump_selected(self):  # pragma: no cover - GUI
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        try:
            f = float(self.table.item(r, 0).text())
        except Exception:
            return
        self.jumpToFreq.emit(f)

    def _export_csv(self):  # pragma: no cover - GUI
        if not self._last_peaks:
            QtWidgets.QMessageBox.information(self, "Экспорт CSV", "Нет найденных пиков.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить CSV", "peaks.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["freq_mhz", "level_dbm", "width_khz", "index"])
                for row in self._last_peaks:
                    w.writerow(row)
            QtWidgets.QMessageBox.information(self, "Экспорт CSV", "Готово.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Экспорт CSV", f"Ошибка записи: {e}")

    def set_peaks(self, peaks: List[Peak]):
        self._last_peaks = peaks
        self.table.setRowCount(0)
        for f, lvl, w, idx in peaks:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, val in enumerate([f"{f:.6f}", f"{lvl:.1f}", f"{w:.1f}", str(int(idx))]):
                self.table.setItem(r, c, QTableWidgetItem(val))

    def _find_freqs(self):  # pragma: no cover - GUI
        txt = (self.edFreqs.text() or "").replace(";", ",").replace("|", ",")
        want = []
        for t in txt.split(","):
            t = t.strip().replace(",", ".")
            if not t:
                continue
            try:
                want.append(float(t))
            except Exception:
                pass
        if not want or self.table.rowCount() == 0:
            return
        tab_freqs = []
        for r in range(self.table.rowCount()):
            try:
                tab_freqs.append((r, float(self.table.item(r, 0).text())))
            except Exception:
                pass
        self.table.clearSelection()
        for f0 in want:
            nearest = sorted(((abs(ff - f0), r) for r, ff in tab_freqs), key=lambda x: x[0])[:5]
            for _, r in nearest:
                self.table.selectRow(r)
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.table.scrollTo(rows[0], QtWidgets.QAbstractItemView.PositionAtCenter)
