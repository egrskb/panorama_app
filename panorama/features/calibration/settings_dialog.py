s#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List
from dataclasses import dataclass

from PyQt5 import QtWidgets, QtCore


@dataclass
class CalibrationDialogResult:
    targets_mhz: List[float]
    dwell_ms: int
    search_span_khz: int
    amplitude_tolerance_db: float
    sync_timeout_sec: float


class CalibrationSettingsDialog(QtWidgets.QDialog):
    settingsApplied = QtCore.pyqtSignal(object)  # CalibrationDialogResult

    def __init__(self, parent=None, initial_targets_mhz: List[float] = None,
                 dwell_ms: int = 500, search_span_khz: int = 50,
                 amplitude_tolerance_db: float = 3.0,
                 sync_timeout_sec: float = 30.0):
        super().__init__(parent)
        self.setWindowTitle("Настройки калибровки HackRF")
        self.resize(520, 260)

        self._targets_mhz = initial_targets_mhz or [100.0, 935.0, 1850.0, 2450.0]
        self._build_ui(dwell_ms, search_span_khz, amplitude_tolerance_db, sync_timeout_sec)

    def _build_ui(self, dwell_ms: int, search_span_khz: int, amplitude_tolerance_db: float, sync_timeout_sec: float):
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()

        self.targets_edit = QtWidgets.QLineEdit(
            ", ".join(f"{v:.3f}" for v in self._targets_mhz)
        )
        self.targets_edit.setToolTip("Список частот (МГц), через запятую")
        form.addRow("Частоты (МГц):", self.targets_edit)

        self.dwell_spin = QtWidgets.QSpinBox()
        self.dwell_spin.setRange(50, 5000)
        self.dwell_spin.setValue(int(dwell_ms))
        self.dwell_spin.setSuffix(" мс")
        form.addRow("Время накопления:", self.dwell_spin)

        self.span_spin = QtWidgets.QSpinBox()
        self.span_spin.setRange(5, 1000)
        self.span_spin.setValue(int(search_span_khz))
        self.span_spin.setSuffix(" кГц")
        form.addRow("Диапазон поиска:", self.span_spin)

        self.amp_tol_spin = QtWidgets.QDoubleSpinBox()
        self.amp_tol_spin.setRange(0.0, 20.0)
        self.amp_tol_spin.setSingleStep(0.1)
        self.amp_tol_spin.setValue(float(amplitude_tolerance_db))
        self.amp_tol_spin.setSuffix(" дБ")
        form.addRow("Допуск амплитуды:", self.amp_tol_spin)

        self.sync_timeout_spin = QtWidgets.QDoubleSpinBox()
        self.sync_timeout_spin.setRange(1.0, 120.0)
        self.sync_timeout_spin.setSingleStep(1.0)
        self.sync_timeout_spin.setValue(float(sync_timeout_sec))
        self.sync_timeout_spin.setSuffix(" с")
        form.addRow("Таймаут синхронизации:", self.sync_timeout_spin)

        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self):
        try:
            # Парс частот
            text = self.targets_edit.text().strip()
            parts = [p.strip() for p in text.split(',') if p.strip()]
            targets_mhz = []
            for p in parts:
                try:
                    targets_mhz.append(float(p))
                except Exception:
                    continue
            if not targets_mhz:
                targets_mhz = [100.0, 935.0, 1850.0, 2450.0]

            result = CalibrationDialogResult(
                targets_mhz=targets_mhz,
                dwell_ms=int(self.dwell_spin.value()),
                search_span_khz=int(self.span_spin.value()),
                amplitude_tolerance_db=float(self.amp_tol_spin.value()),
                sync_timeout_sec=float(self.sync_timeout_spin.value()),
            )
            self.settingsApplied.emit(result)
            self.accept()
        except Exception:
            self.reject()


