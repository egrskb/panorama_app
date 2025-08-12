from __future__ import annotations
from typing import Optional, List
from PyQt5 import QtWidgets, QtCore
import numpy as np
from .logic import find_peak_indices, summarize


class PeaksWidget(QtWidgets.QWidget):
    goToFreq = QtCore.pyqtSignal(float)  # freq_hz

    def __init__(self, parent=None):
        super().__init__(parent)
        self._freqs: Optional[np.ndarray] = None
        self._row: Optional[np.ndarray] = None

        # Панель параметров
        top = QtWidgets.QHBoxLayout()
        self.th = QtWidgets.QDoubleSpinBox(); self.th.setRange(-160.0, 30.0); self.th.setDecimals(1); self.th.setValue(-80.0); self.th.setSuffix(" дБм")
        self.min_dist = QtWidgets.QDoubleSpinBox(); self.min_dist.setRange(0.0, 5000.0); self.min_dist.setDecimals(0); self.min_dist.setSingleStep(50.0); self.min_dist.setValue(400.0); self.min_dist.setSuffix(" кГц")
        self.auto = QtWidgets.QCheckBox("Авто"); self.auto.setChecked(True)
        self.btn = QtWidgets.QPushButton("Найти пики")
        for w, lab in [(self.th, "Порог"), (self.min_dist, "Мин. разнесение")]:
            box = QtWidgets.QVBoxLayout(); box.addWidget(QtWidgets.QLabel(lab)); box.addWidget(w); top.addLayout(box)
        top.addStretch(1); top.addWidget(self.auto); top.addWidget(self.btn)

        # Таблица
        self.table = QtWidgets.QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Частота (МГц)", "Уровень (дБм)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.table)

        self.btn.clicked.connect(self._run_find)
        self.table.doubleClicked.connect(self._on_double_clicked)

    # --- внешнее API ---
    def update_from_row(self, freqs_hz, row_dbm):
        self._freqs = np.asarray(freqs_hz, dtype=float)
        self._row = np.asarray(row_dbm, dtype=float)
        if self.auto.isChecked():
            self._run_find()

    # --- внутреннее ---
    def _run_find(self):
        if self._freqs is None or self._row is None:
            return
        bins = max(1, int(round(self.min_dist.value() * 1e3 / (self._freqs[1] - self._freqs[0]))))
        idx = find_peak_indices(self._row, min_distance_bins=bins, threshold_dbm=self.th.value())
        rows = summarize(self._freqs, self._row, idx)
        self._fill_table(rows)

    def _fill_table(self, rows: List[dict]):
        self.table.setRowCount(len(rows))
        for r, it in enumerate(rows):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{it['freq_mhz']:.6f}"))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{it['level_dbm']:.1f}"))

    def _on_double_clicked(self, mi):
        r = mi.row()
        try:
            f_mhz = float(self.table.item(r, 0).text())
            self.goToFreq.emit(f_mhz * 1e6)
        except Exception:
            pass
