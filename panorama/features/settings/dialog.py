from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
from PyQt5 import QtWidgets, QtCore


@dataclass
class SlaveEntry:
    nickname: str
    uri: str


class SettingsDialog(QtWidgets.QDialog):
    saved = QtCore.pyqtSignal(object)  # payload: dict {'master': {...}, 'slaves': [...]}
    next_step = QtCore.pyqtSignal(object)  # payload: dict {'master': {...}}

    def __init__(self, parent=None, current: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Настройки — Устройства")
        self.resize(600, 400)
        self._current = current or {'master': {'enabled': True, 'nickname': 'Master', 'uri': ''}, 'slaves': []}

        self.tabs = QtWidgets.QTabWidget(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)

        # Master tab
        master_tab = QtWidgets.QWidget()
        mt = QtWidgets.QFormLayout(master_tab)
        self.master_enable = QtWidgets.QCheckBox("Включить Master")
        self.master_enable.setChecked(bool(self._current.get('master', {}).get('enabled', True)))
        self.master_nick = QtWidgets.QLineEdit(self._current.get('master', {}).get('nickname', 'Master'))
        # Выпадающий список устройств для Master
        self.master_uri = QtWidgets.QComboBox()
        self.master_uri.setEditable(True)
        self.master_uri.addItem(self._current.get('master', {}).get('uri', 'driver=hackrf'))
        mt.addRow(self.master_enable)
        mt.addRow("Никнейм:", self.master_nick)
        mt.addRow("Serial (HackRF):", self.master_uri)
        self.tabs.addTab(master_tab, "Master")

        # Slaves tab (список обнаруженных устройств; не добавляем/удаляем вручную)
        slaves_tab = QtWidgets.QWidget()
        svl = QtWidgets.QVBoxLayout(slaves_tab)
        self.slave_table = QtWidgets.QTableWidget(0, 8)
        self.slave_table.setHorizontalHeaderLabels(["Никнейм", "URI/Args", "Driver", "Serial", "Label", "X", "Y", "Z"])
        self.slave_table.horizontalHeader().setStretchLastSection(True)
        svl.addWidget(self.slave_table)
        self.tabs.addTab(slaves_tab, "Slaves")
        # Изначально блокируем вкладку Slaves — доступна только после "Далее"
        self.tabs.setTabEnabled(1, False)

        # Footer
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        ok_btn = QtWidgets.QPushButton("Сохранить")
        cancel_btn = QtWidgets.QPushButton("Отмена")
        self.btn_next = QtWidgets.QPushButton("Далее")
        footer.addWidget(self.btn_next)
        footer.addWidget(ok_btn)
        footer.addWidget(cancel_btn)
        layout.addLayout(footer)

        self.btn_next.clicked.connect(self._on_next)
        ok_btn.clicked.connect(self._on_save)
        cancel_btn.clicked.connect(self.reject)

        # Не заполняем из сохранённых сразу, чтобы не дублировать — применим через apply_saved_overrides

    def _append_row(self, nickname: str, uri: str, driver: str = '', serial: str = '', label: str = '', pos: list | None = None):
        r = self.slave_table.rowCount()
        self.slave_table.insertRow(r)
        # Никнейм редактируемый
        self.slave_table.setItem(r, 0, QtWidgets.QTableWidgetItem(nickname))
        # Остальные поля только для чтения
        for c, val in enumerate([uri, driver, serial, label], start=1):
            it = QtWidgets.QTableWidgetItem(val)
            it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.slave_table.setItem(r, c, it)
        # Координаты редактируемые
        px, py, pz = (pos or [0.0, 0.0, 0.0])
        for col, v in zip([5, 6, 7], [px, py, pz]):
            it = QtWidgets.QTableWidgetItem(str(float(v)))
            self.slave_table.setItem(r, col, it)

    def populate_slave_devices(self, devices: list):
        """Принимает список dict: driver, serial, label, uri. Заполняет таблицу."""
        self.slave_table.setRowCount(0)
        for dev in devices:
            self._append_row(dev.get('label') or dev.get('serial') or dev.get('driver') or 'slave',
                             dev.get('uri', ''),
                             dev.get('driver', ''),
                             dev.get('serial', ''),
                             dev.get('label', ''),
                             [0.0, 0.0, 0.0])

    def apply_saved_overrides(self, saved_slaves: list):
        """Совмещает сохранённые никнеймы и координаты с обнаруженными устройствами (по serial или uri)."""
        if not saved_slaves:
            return
        # Быстрый индекс по serial/uri
        by_serial = {s.get('serial', ''): s for s in saved_slaves if s.get('serial')}
        by_uri = {s.get('uri', ''): s for s in saved_slaves if s.get('uri')}
        for r in range(self.slave_table.rowCount()):
            serial_it = self.slave_table.item(r, 3)
            uri_it = self.slave_table.item(r, 1)
            key_serial = serial_it.text().strip() if serial_it else ''
            key_uri = uri_it.text().strip() if uri_it else ''
            saved = by_serial.get(key_serial) or by_uri.get(key_uri)
            if saved:
                # Никнейм
                nick_it = self.slave_table.item(r, 0)
                if nick_it:
                    nick_it.setText(saved.get('nickname', nick_it.text()))
                # Координаты
                pos = saved.get('pos', [0.0, 0.0, 0.0])
                for col, v in zip([5, 6, 7], pos):
                    it = self.slave_table.item(r, col)
                    if it:
                        it.setText(str(float(v)))

    def _on_save(self):
        data = {
            'master': {
                'enabled': self.master_enable.isChecked(),
                'nickname': self.master_nick.text().strip(),
                'uri': self.master_uri.currentText().strip() if hasattr(self.master_uri, 'currentText') else self.master_uri.text().strip(),
            },
            'slaves': []
        }
        for r in range(self.slave_table.rowCount()):
            nick = self.slave_table.item(r, 0)
            uri = self.slave_table.item(r, 1)
            drv = self.slave_table.item(r, 2)
            ser = self.slave_table.item(r, 3)
            lbl = self.slave_table.item(r, 4)
            # Координаты
            xit = self.slave_table.item(r, 5)
            yit = self.slave_table.item(r, 6)
            zit = self.slave_table.item(r, 7)
            def f2f(it):
                try:
                    return float(it.text()) if it else 0.0
                except Exception:
                    return 0.0
            data['slaves'].append({
                'nickname': nick.text().strip() if nick else '',
                'uri': uri.text().strip() if uri else '',
                'driver': drv.text().strip() if drv else '',
                'serial': ser.text().strip() if ser else '',
                'label': lbl.text().strip() if lbl else '',
                'pos': [f2f(xit), f2f(yit), f2f(zit)],
            })
        self.saved.emit(data)
        self.accept()

    def _on_next(self):
        """Шаг 1: передаём выбранный Master наружу и ждём указаний перейти к Slaves."""
        payload = {
            'master': {
                'enabled': self.master_enable.isChecked(),
                'nickname': self.master_nick.text().strip(),
                'uri': self.master_uri.currentText().strip() if hasattr(self.master_uri, 'currentText') else self.master_uri.text().strip(),
            }
        }
        self.next_step.emit(payload)

    def proceed_to_slaves(self):
        """Блокирует вкладку Master и переключается на Slaves."""
        # Запретить переключение на Master
        self.tabs.setTabEnabled(0, False)
        # Переключиться на вкладку Slaves
        self.tabs.setCurrentIndex(1)
