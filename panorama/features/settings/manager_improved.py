from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import time

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal

# Reuse existing helpers from the project
from panorama.features.settings.storage import save_sdr_settings


class DeviceRole(Enum):
    MASTER = "master"
    SLAVE = "slave"
    NONE = "none"


@dataclass
class SDRDeviceInfo:
    driver: str
    serial: str
    label: str = ""
    uri: str = ""
    role: DeviceRole = DeviceRole.NONE
    nickname: str = ""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_available: bool = False
    is_configured: bool = False
    capabilities: Dict[str, Any] = None
    last_seen: float = 0.0

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = {}
        if not self.nickname:
            base = self.label or (self.driver.upper())
            tail = (self.serial[-4:] if self.serial else "")
            self.nickname = f"{base}{('-' + tail) if tail else ''}"

    def to_dict(self):
        data = asdict(self)
        data['role'] = self.role.value
        data['position'] = list(self.position)
        return data


class DeviceDiscoveryThread(QThread):
    devicesFound = pyqtSignal(list)  # List[SDRDeviceInfo]
    progressUpdate = pyqtSignal(int, str)
    discoveryError = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            devices: List[SDRDeviceInfo] = []
            # Step 1: HackRF via C wrapper
            self.progressUpdate.emit(20, "Поиск HackRF…")
            try:
                from panorama.drivers.hackrf_master.hackrf_master_wrapper import HackRFMaster
                hw = HackRFMaster()
                serials = hw.enumerate_devices() or []
                for s in serials:
                    serial = s or "default"
                    info = SDRDeviceInfo(
                        driver="hackrf",
                        serial=serial,
                        label=f"HackRF {serial[-4:]}" if serial != "default" else "HackRF (default)",
                        uri=(f"driver=hackrf,serial={serial}" if serial != "default" else "driver=hackrf"),
                        capabilities={
                            "frequency_range": (24e6, 6e9),
                            "bandwidth": 20e6,
                        },
                    )
                    try:
                        hw.set_serial(None if serial == "default" else serial)
                        info.is_available = bool(hw.probe())
                    except Exception:
                        info.is_available = False
                    info.last_seen = time.time()
                    devices.append(info)
                try:
                    hw.cleanup()
                except Exception:
                    pass
            except Exception:
                # ignore, no HackRF
                pass

            if self._stop:
                return

            # Step 2: Soapy devices (reuse project enumeration to avoid heavy opens)
            self.progressUpdate.emit(60, "Поиск SoapySDR…")
            try:
                # Lazy import to avoid circulars
                from panorama.features.slave_sdr.slave import SlaveManager
                dummy_log = None
                sm = SlaveManager(dummy_log)
                soapy = sm.enumerate_soapy_devices() or []
                for d in soapy:
                    drv = (d.get('driver') or '').strip()
                    ser = (d.get('serial') or '').strip()
                    label = d.get('label', '')
                    uri = (d.get('uri') or '').strip()
                    # Skip HackRF seen as Soapy, we already list HackRF above
                    if drv == 'hackrf':
                        continue
                    # If uri is empty, construct minimal args from driver/serial to allow visibility
                    if not uri and drv:
                        uri = f"driver={drv}" + (f",serial={ser}" if ser else "")
                    if not drv:
                        continue
                    info = SDRDeviceInfo(driver=drv, serial=ser, label=label or '', uri=uri)
                    # We don't hard-probe Soapy here; visible regardless, adding as slave will validate
                    info.is_available = True
                    info.last_seen = time.time()
                    devices.append(info)
            except Exception:
                pass

            # Deduplicate by serial/uri
            uniq: List[SDRDeviceInfo] = []
            seen = set()
            for d in devices:
                key = (d.driver, d.serial or d.uri or d.label)
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(d)

            self.progressUpdate.emit(100, f"Найдено устройств: {len(uniq)}")
            self.devicesFound.emit(uniq)
        except Exception as e:
            self.discoveryError.emit(str(e))


class ImprovedDeviceManagerDialog(QtWidgets.QDialog):
    devicesConfigured = pyqtSignal(dict)  # {'master': {...}, 'slaves': [...]}

    def __init__(self, parent=None, current_config: Dict[str, Any] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Настройка SDR устройств")
        self.resize(1000, 600)
        self._current = current_config or {"master": {"serial": "", "nickname": "Master", "pos": [0,0,0]}, "slaves": []}
        self.devices: List[SDRDeviceInfo] = []
        self.master_device: Optional[SDRDeviceInfo] = None
        self.slave_devices: List[SDRDeviceInfo] = []

        self._build_ui()
        self._wire_signals()

        # Start discovery shortly after show
        QtCore.QTimer.singleShot(50, self._start_discovery)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.btn_refresh = QtWidgets.QPushButton("🔄 Поиск устройств")
        top.addWidget(QtWidgets.QLabel("<b>Обнаружение и конфигурация SDR</b>"))
        top.addStretch(1)
        top.addWidget(self.btn_refresh)
        layout.addLayout(top)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.progress_label = QtWidgets.QLabel()
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter)

        # Left: devices table
        left = QtWidgets.QWidget()
        lyt = QtWidgets.QVBoxLayout(left)
        self.tbl_devices = QtWidgets.QTableWidget(0, 6)
        self.tbl_devices.setHorizontalHeaderLabels(["Тип", "Serial", "Название", "Статус", "Роль", "Действие"])
        hdr = self.tbl_devices.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)
        self.tbl_devices.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        lyt.addWidget(self.tbl_devices)
        splitter.addWidget(left)

        # Right: master info + slaves table
        right = QtWidgets.QWidget()
        ryt = QtWidgets.QVBoxLayout(right)
        grp_master = QtWidgets.QGroupBox("Master SDR")
        mlyt = QtWidgets.QVBoxLayout(grp_master)
        self.txt_master = QtWidgets.QTextEdit()
        self.txt_master.setReadOnly(True)
        self.txt_master.setMaximumHeight(100)
        self.txt_master.setPlainText("Master SDR не выбран")
        mlyt.addWidget(self.txt_master)
        ryt.addWidget(grp_master)

        grp_slaves = QtWidgets.QGroupBox("Slave SDR")
        slyt = QtWidgets.QVBoxLayout(grp_slaves)
        self.tbl_slaves = QtWidgets.QTableWidget(0, 6)
        self.tbl_slaves.setHorizontalHeaderLabels(["Никнейм", "X", "Y", "Z", "URI", "Удалить"])
        sh = self.tbl_slaves.horizontalHeader()
        sh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        sh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        sh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        sh.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        sh.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        sh.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)
        slyt.addWidget(self.tbl_slaves)
        ryt.addWidget(grp_slaves)

        splitter.addWidget(right)
        splitter.setSizes([550, 450])

        # Bottom buttons
        bottom = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Готово")
        bottom.addWidget(self.lbl_status)
        bottom.addStretch(1)
        self.btn_save = QtWidgets.QPushButton("💾 Сохранить")
        self.btn_close = QtWidgets.QPushButton("Закрыть")
        bottom.addWidget(self.btn_save)
        bottom.addWidget(self.btn_close)
        layout.addLayout(bottom)

    def _wire_signals(self):
        self.btn_refresh.clicked.connect(self._start_discovery)
        self.btn_close.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._save)

    # Discovery
    def _start_discovery(self):
        if hasattr(self, '_disc') and self._disc.isRunning():
            return
        self.progress.setVisible(True)
        self.progress_label.setVisible(True)
        self._disc = DeviceDiscoveryThread(self)
        self._disc.devicesFound.connect(self._on_devices_found)
        self._disc.progressUpdate.connect(self._on_progress)
        self._disc.discoveryError.connect(self._on_discovery_error)
        self._disc.finished.connect(self._on_discovery_finished)
        self._disc.start()

    def _on_progress(self, pc: int, msg: str):
        self.progress.setValue(pc)
        self.progress_label.setText(msg)

    def _on_discovery_error(self, msg: str):
        QtWidgets.QMessageBox.warning(self, "Поиск устройств", msg)

    def _on_discovery_finished(self):
        self.progress.setVisible(False)
        self.progress_label.setVisible(False)

    def _on_devices_found(self, devices: List[SDRDeviceInfo]):
        # Apply current roles if any
        self.devices = devices
        self._apply_existing_config()
        self._refresh_devices_table()
        self._refresh_master_info()
        self._refresh_slaves_table()

    # UI updates
    def _refresh_devices_table(self):
        self.tbl_devices.setRowCount(len(self.devices))
        for row, d in enumerate(self.devices):
            # Type
            it_type = QtWidgets.QTableWidgetItem(d.driver.upper())
            it_type.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tbl_devices.setItem(row, 0, it_type)
            # Serial
            self.tbl_devices.setItem(row, 1, QtWidgets.QTableWidgetItem(d.serial or "N/A"))
            # Name
            self.tbl_devices.setItem(row, 2, QtWidgets.QTableWidgetItem(d.nickname))
            # Status
            st = QtWidgets.QTableWidgetItem("✅" if d.is_available else "❌")
            st.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tbl_devices.setItem(row, 3, st)
            # Role
            role = QtWidgets.QComboBox()
            role.addItems(["Не используется", "Master", "Slave"])
            if d.role == DeviceRole.MASTER:
                role.setCurrentIndex(1)
            elif d.role == DeviceRole.SLAVE:
                role.setCurrentIndex(2)
            else:
                role.setCurrentIndex(0)
            role.currentIndexChanged.connect(lambda idx, dev=d: self._on_role_changed(dev, idx))
            self.tbl_devices.setCellWidget(row, 4, role)
            # Action
            btn = QtWidgets.QPushButton("Добавить" if d.role == DeviceRole.SLAVE else ("Настроить" if d.role == DeviceRole.MASTER else "—"))
            btn.setEnabled(d.is_available and d.role != DeviceRole.NONE)
            if d.role == DeviceRole.SLAVE:
                btn.clicked.connect(lambda _=False, dev=d: self._add_slave(dev))
            elif d.role == DeviceRole.MASTER:
                btn.clicked.connect(lambda _=False, dev=d: self._set_master(dev))
            self.tbl_devices.setCellWidget(row, 5, btn)

    def _refresh_master_info(self):
        if self.master_device:
            info = f"Устройство: {self.master_device.nickname}\nТип: {self.master_device.driver.upper()}\nСерийный номер: {self.master_device.serial}"
            rng = self.master_device.capabilities.get('frequency_range') if self.master_device.capabilities else None
            if rng:
                info += f"\nДиапазон: {rng[0]/1e6:.0f} - {rng[1]/1e9:.1f} ГГц"
            self.txt_master.setPlainText(info)
        else:
            self.txt_master.setPlainText("Master SDR не выбран")

    def _refresh_slaves_table(self):
        self.tbl_slaves.setRowCount(len(self.slave_devices))
        for r, dev in enumerate(self.slave_devices):
            # Nickname
            self.tbl_slaves.setItem(r, 0, QtWidgets.QTableWidgetItem(dev.nickname))
            # XYZ
            x, y, z = dev.position
            self.tbl_slaves.setItem(r, 1, QtWidgets.QTableWidgetItem(str(float(x))))
            self.tbl_slaves.setItem(r, 2, QtWidgets.QTableWidgetItem(str(float(y))))
            self.tbl_slaves.setItem(r, 3, QtWidgets.QTableWidgetItem(str(float(z))))
            # URI (read-only)
            it = QtWidgets.QTableWidgetItem(dev.uri)
            it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.tbl_slaves.setItem(r, 4, it)
            # Remove button
            btn = QtWidgets.QPushButton("Удалить")
            btn.clicked.connect(lambda _=False, idx=r: self._remove_slave_at(idx))
            self.tbl_slaves.setCellWidget(r, 5, btn)

    # Role handlers
    def _on_role_changed(self, dev: SDRDeviceInfo, index: int):
        new_role = DeviceRole.NONE if index == 0 else (DeviceRole.MASTER if index == 1 else DeviceRole.SLAVE)
        if new_role == DeviceRole.MASTER:
            # ensure single master
            for d in self.devices:
                if d is not dev and d.role == DeviceRole.MASTER:
                    d.role = DeviceRole.NONE
            self.master_device = dev
        elif dev.role == DeviceRole.MASTER and new_role != DeviceRole.MASTER:
            self.master_device = None
        dev.role = new_role
        self._refresh_devices_table()
        self._refresh_master_info()

    def _set_master(self, dev: SDRDeviceInfo):
        self.master_device = dev
        dev.role = DeviceRole.MASTER
        self._refresh_devices_table()
        self._refresh_master_info()

    def _add_slave(self, dev: SDRDeviceInfo):
        if dev not in self.slave_devices:
            dev.role = DeviceRole.SLAVE
            self.slave_devices.append(dev)
            self._refresh_devices_table()
            self._refresh_slaves_table()

    def _remove_slave_at(self, idx: int):
        if 0 <= idx < len(self.slave_devices):
            self.slave_devices.pop(idx)
            self._refresh_slaves_table()

    # Save/apply
    def _save(self):
        # Gather from tables: update nicknames and positions
        for r, dev in enumerate(self.slave_devices):
            try:
                nick = self.tbl_slaves.item(r, 0).text().strip()
            except Exception:
                nick = dev.nickname
            def parsef(c):
                try:
                    it = self.tbl_slaves.item(r, c)
                    return float(it.text()) if it else 0.0
                except Exception:
                    return 0.0
            dev.nickname = nick or dev.nickname
            dev.position = (parsef(1), parsef(2), parsef(3))

        config = {
            'master': {
                'nickname': (self.master_device.nickname if self.master_device else 'Master'),
                'serial': (self.master_device.serial if self.master_device else ''),
                'pos': [0.0, 0.0, 0.0],
            },
            'slaves': [
                {
                    'nickname': dev.nickname,
                    'uri': dev.uri,
                    'driver': dev.driver,
                    'serial': dev.serial,
                    'label': dev.label,
                    'pos': list(dev.position),
                }
                for dev in self.slave_devices
            ]
        }

        save_sdr_settings(config)
        self.devicesConfigured.emit(config)
        self.accept()

    # helpers
    def _apply_existing_config(self):
        master_serial = (self._current or {}).get('master', {}).get('serial', '')
        saved_slaves = {s.get('serial') or s.get('uri'): s for s in (self._current or {}).get('slaves', [])}
        for d in self.devices:
            key = d.serial or d.uri
            if master_serial and d.driver == 'hackrf' and d.serial == master_serial:
                d.role = DeviceRole.MASTER
                self.master_device = d
            s = saved_slaves.get(key)
            if s:
                d.role = DeviceRole.SLAVE
                d.nickname = s.get('nickname') or d.nickname
                pos = s.get('pos') or [0.0, 0.0, 0.0]
                d.position = (float(pos[0]), float(pos[1]), float(pos[2]))
                if d not in self.slave_devices:
                    self.slave_devices.append(d)


