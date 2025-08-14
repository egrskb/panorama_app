# panorama/features/devices/manager.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from PyQt5 import QtWidgets, QtCore, QtGui
import json
import os


@dataclass
class SDRDevice:
    """Информация об SDR устройстве."""
    serial: str
    nickname: str = ""
    role: str = "none"  # "master", "slave1", "slave2", "none"
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    last_seen: float = 0.0
    is_online: bool = False
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class DeviceManager:
    """Менеджер SDR устройств с персистентностью."""
    
    CONFIG_FILE = "sdr_devices.json"
    
    def __init__(self):
        self.devices: Dict[str, SDRDevice] = {}  # serial -> SDRDevice
        self.master: Optional[str] = None
        self.slave1: Optional[str] = None
        self.slave2: Optional[str] = None
        self.load_config()
    
    def load_config(self):
        """Загружает конфигурацию устройств из файла."""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Загружаем устройства
                    for serial, device_data in data.get('devices', {}).items():
                        self.devices[serial] = SDRDevice.from_dict(device_data)
                    
                    # Загружаем роли
                    self.master = data.get('master')
                    self.slave1 = data.get('slave1')
                    self.slave2 = data.get('slave2')
                    
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {e}")
    
    def save_config(self):
        """Сохраняет конфигурацию устройств в файл."""
        try:
            data = {
                'devices': {serial: device.to_dict() for serial, device in self.devices.items()},
                'master': self.master,
                'slave1': self.slave1,
                'slave2': self.slave2
            }
            
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")
    
    def update_device_list(self, serials: List[str]):
        """Обновляет список доступных устройств."""
        import time
        current_time = time.time()
        
        # Помечаем все как оффлайн
        for device in self.devices.values():
            device.is_online = False
        
        # Обновляем или добавляем новые
        for serial in serials:
            if serial in self.devices:
                self.devices[serial].is_online = True
                self.devices[serial].last_seen = current_time
            else:
                # Новое устройство
                self.devices[serial] = SDRDevice(
                    serial=serial,
                    nickname=f"SDR-{serial[-4:]}",  # Последние 4 символа серийника
                    last_seen=current_time,
                    is_online=True
                )
        
        self.save_config()
    
    def set_device_role(self, serial: str, role: str):
        """Устанавливает роль устройства."""
        if serial not in self.devices:
            return
        
        # Сбрасываем старые роли
        for device in self.devices.values():
            if device.role == role:
                device.role = "none"
        
        # Устанавливаем новую роль
        self.devices[serial].role = role
        
        # Обновляем быстрый доступ
        if role == "master":
            self.master = serial
        elif role == "slave1":
            self.slave1 = serial
        elif role == "slave2":
            self.slave2 = serial
        
        self.save_config()
    
    def set_device_nickname(self, serial: str, nickname: str):
        """Устанавливает никнейм устройства."""
        if serial in self.devices:
            self.devices[serial].nickname = nickname
            self.save_config()
    
    def set_device_position(self, serial: str, x: float, y: float, z: float = 0.0):
        """Устанавливает позицию устройства."""
        if serial in self.devices:
            self.devices[serial].position_x = x
            self.devices[serial].position_y = y
            self.devices[serial].position_z = z
            self.save_config()
    
    def get_trilateration_devices(self) -> Tuple[Optional[SDRDevice], Optional[SDRDevice], Optional[SDRDevice]]:
        """Возвращает устройства для трилатерации."""
        master_dev = self.devices.get(self.master) if self.master else None
        slave1_dev = self.devices.get(self.slave1) if self.slave1 else None
        slave2_dev = self.devices.get(self.slave2) if self.slave2 else None
        
        return master_dev, slave1_dev, slave2_dev


class DeviceConfigDialog(QtWidgets.QDialog):
    """Диалог настройки SDR устройств."""
    
    devicesConfigured = QtCore.pyqtSignal()
    
    def __init__(self, manager: DeviceManager, available_serials: List[str], parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Настройка SDR устройств")
        self.resize(800, 600)
        
        # Обновляем список устройств
        self.manager.update_device_list(available_serials)
        
        self._build_ui()
        self._refresh_table()
    
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Заголовок
        header = QtWidgets.QLabel("Настройка SDR устройств для трилатерации")
        header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(header)
        
        # Таблица устройств
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Серийный номер", "Никнейм", "Роль", "X (м)", "Y (м)", "Z (м)", "Статус"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        
        # Панель быстрого выбора ролей
        roles_group = QtWidgets.QGroupBox("Быстрая настройка ролей для трилатерации")
        roles_layout = QtWidgets.QFormLayout(roles_group)
        
        self.master_combo = QtWidgets.QComboBox()
        self.slave1_combo = QtWidgets.QComboBox()
        self.slave2_combo = QtWidgets.QComboBox()
        
        roles_layout.addRow("Master SDR:", self.master_combo)
        roles_layout.addRow("Slave 1:", self.slave1_combo)
        roles_layout.addRow("Slave 2:", self.slave2_combo)
        
        layout.addWidget(roles_group)
        
        # Кнопки
        buttons_layout = QtWidgets.QHBoxLayout()
        
        btn_refresh = QtWidgets.QPushButton("🔄 Обновить")
        btn_refresh.clicked.connect(self._refresh_devices)
        
        btn_apply = QtWidgets.QPushButton("✓ Применить")
        btn_apply.clicked.connect(self._apply_roles)
        btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
        """)
        
        btn_close = QtWidgets.QPushButton("Закрыть")
        btn_close.clicked.connect(self.accept)
        
        buttons_layout.addWidget(btn_refresh)
        buttons_layout.addStretch()
        buttons_layout.addWidget(btn_apply)
        buttons_layout.addWidget(btn_close)
        
        layout.addLayout(buttons_layout)
        
        # Подключаем сигналы
        self.master_combo.currentTextChanged.connect(self._on_role_changed)
        self.slave1_combo.currentTextChanged.connect(self._on_role_changed)
        self.slave2_combo.currentTextChanged.connect(self._on_role_changed)
    
    def _refresh_table(self):
        """Обновляет таблицу устройств."""
        self.table.setRowCount(len(self.manager.devices))
        
        for row, (serial, device) in enumerate(self.manager.devices.items()):
            # Серийный номер
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(serial))
            
            # Никнейм (редактируемый)
            nickname_item = QtWidgets.QTableWidgetItem(device.nickname)
            self.table.setItem(row, 1, nickname_item)
            
            # Роль
            role_combo = QtWidgets.QComboBox()
            role_combo.addItems(["none", "master", "slave1", "slave2"])
            role_combo.setCurrentText(device.role)
            role_combo.currentTextChanged.connect(
                lambda role, s=serial: self.manager.set_device_role(s, role)
            )
            self.table.setCellWidget(row, 2, role_combo)
            
            # Позиция X
            x_spin = QtWidgets.QDoubleSpinBox()
            x_spin.setRange(-1000, 1000)
            x_spin.setValue(device.position_x)
            x_spin.setSuffix(" м")
            x_spin.valueChanged.connect(
                lambda v, s=serial: self._update_position(s, x=v)
            )
            self.table.setCellWidget(row, 3, x_spin)
            
            # Позиция Y
            y_spin = QtWidgets.QDoubleSpinBox()
            y_spin.setRange(-1000, 1000)
            y_spin.setValue(device.position_y)
            y_spin.setSuffix(" м")
            y_spin.valueChanged.connect(
                lambda v, s=serial: self._update_position(s, y=v)
            )
            self.table.setCellWidget(row, 4, y_spin)
            
            # Позиция Z
            z_spin = QtWidgets.QDoubleSpinBox()
            z_spin.setRange(-100, 100)
            z_spin.setValue(device.position_z)
            z_spin.setSuffix(" м")
            z_spin.valueChanged.connect(
                lambda v, s=serial: self._update_position(s, z=v)
            )
            self.table.setCellWidget(row, 5, z_spin)
            
            # Статус
            status = "🟢 Онлайн" if device.is_online else "⚪ Оффлайн"
            status_item = QtWidgets.QTableWidgetItem(status)
            if device.is_online:
                status_item.setBackground(QtGui.QBrush(QtGui.QColor(200, 255, 200)))
            else:
                status_item.setBackground(QtGui.QBrush(QtGui.QColor(240, 240, 240)))
            self.table.setItem(row, 6, status_item)
        
        # Обновляем комбобоксы ролей
        self._update_role_combos()
    
    def _update_role_combos(self):
        """Обновляет выпадающие списки ролей."""
        # Сохраняем текущий выбор
        current_master = self.master_combo.currentText()
        current_slave1 = self.slave1_combo.currentText()
        current_slave2 = self.slave2_combo.currentText()
        
        # Очищаем
        self.master_combo.clear()
        self.slave1_combo.clear()
        self.slave2_combo.clear()
        
        # Добавляем опцию "не выбрано"
        self.master_combo.addItem("(не выбрано)")
        self.slave1_combo.addItem("(не выбрано)")
        self.slave2_combo.addItem("(не выбрано)")
        
        # Добавляем устройства
        for serial, device in self.manager.devices.items():
            if device.is_online:
                display_name = f"{device.nickname} ({serial})"
                self.master_combo.addItem(display_name, serial)
                self.slave1_combo.addItem(display_name, serial)
                self.slave2_combo.addItem(display_name, serial)
        
        # Восстанавливаем выбор по ролям
        if self.manager.master:
            for i in range(self.master_combo.count()):
                if self.master_combo.itemData(i) == self.manager.master:
                    self.master_combo.setCurrentIndex(i)
                    break
        
        if self.manager.slave1:
            for i in range(self.slave1_combo.count()):
                if self.slave1_combo.itemData(i) == self.manager.slave1:
                    self.slave1_combo.setCurrentIndex(i)
                    break
        
        if self.manager.slave2:
            for i in range(self.slave2_combo.count()):
                if self.slave2_combo.itemData(i) == self.manager.slave2:
                    self.slave2_combo.setCurrentIndex(i)
                    break
    
    def _update_position(self, serial: str, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None):
        """Обновляет позицию устройства."""
        if serial in self.manager.devices:
            device = self.manager.devices[serial]
            if x is not None:
                device.position_x = x
            if y is not None:
                device.position_y = y
            if z is not None:
                device.position_z = z
            self.manager.save_config()
    
    def _on_role_changed(self):
        """Проверяет уникальность выбранных ролей."""
        master = self.master_combo.currentData()
        slave1 = self.slave1_combo.currentData()
        slave2 = self.slave2_combo.currentData()
        
        # Проверяем конфликты
        devices = [d for d in [master, slave1, slave2] if d]
        if len(devices) != len(set(devices)):
            # Есть дубликаты
            self.master_combo.setStyleSheet("QComboBox { background-color: #ffcccc; }")
            self.slave1_combo.setStyleSheet("QComboBox { background-color: #ffcccc; }")
            self.slave2_combo.setStyleSheet("QComboBox { background-color: #ffcccc; }")
        else:
            # Все уникальные
            self.master_combo.setStyleSheet("")
            self.slave1_combo.setStyleSheet("")
            self.slave2_combo.setStyleSheet("")
    
    def _apply_roles(self):
        """Применяет выбранные роли."""
        master = self.master_combo.currentData()
        slave1 = self.slave1_combo.currentData()
        slave2 = self.slave2_combo.currentData()
        
        # Проверяем уникальность
        devices = [d for d in [master, slave1, slave2] if d]
        if len(devices) != len(set(devices)):
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите разные устройства для каждой роли!")
            return
        
        # Сбрасываем все роли
        for device in self.manager.devices.values():
            device.role = "none"
        
        # Устанавливаем новые роли
        if master:
            self.manager.set_device_role(master, "master")
        if slave1:
            self.manager.set_device_role(slave1, "slave1")
        if slave2:
            self.manager.set_device_role(slave2, "slave2")
        
        # Обновляем никнеймы из таблицы
        for row in range(self.table.rowCount()):
            serial = self.table.item(row, 0).text()
            nickname = self.table.item(row, 1).text()
            self.manager.set_device_nickname(serial, nickname)
        
        self.manager.save_config()
        self._refresh_table()
        self.devicesConfigured.emit()
        
        QtWidgets.QMessageBox.information(self, "Успех", 
            f"Настройки сохранены!\n"
            f"Master: {self.manager.devices[master].nickname if master else 'не выбран'}\n"
            f"Slave 1: {self.manager.devices[slave1].nickname if slave1 else 'не выбран'}\n"
            f"Slave 2: {self.manager.devices[slave2].nickname if slave2 else 'не выбран'}"
        )
    
    def _refresh_devices(self):
        """Обновляет список устройств."""
        # Здесь должен быть код получения списка устройств
        # Пока заглушка
        self._refresh_table()