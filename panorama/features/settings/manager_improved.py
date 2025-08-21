#!/usr/bin/env python3
"""
Улучшенный диспетчер устройств с правильной логикой Master/Slave.
Master - только один HackRF из списка C библиотеки.
Slaves - множество устройств через SoapySDR.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import time
import json

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal

# Импорт C библиотеки для HackRF
try:
    from panorama.drivers.hrf_backend import HackRFMaster
    HACKRF_AVAILABLE = True
except ImportError:
    HackRFMaster = None
    HACKRF_AVAILABLE = False

# Импорт SoapySDR для Slave устройств
try:
    import SoapySDR
    SOAPY_AVAILABLE = True
except ImportError:
    SoapySDR = None
    SOAPY_AVAILABLE = False

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
            if self.driver == "hackrf":
                self.nickname = f"HackRF-{self.serial[-4:]}" if self.serial and self.serial != "default" else "HackRF"
            else:
                self.nickname = f"{self.driver.upper()}-{self.serial[-4:]}" if self.serial else self.driver.upper()

    def to_dict(self):
        data = asdict(self)
        data['role'] = self.role.value
        data['position'] = list(self.position)
        return data


class DeviceDiscoveryThread(QThread):
    """Поток для поиска устройств."""
    devicesFound = pyqtSignal(list)  # List[SDRDeviceInfo]
    progressUpdate = pyqtSignal(int, str)
    discoveryError = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False
        self.master_controller = None  # Master контроллер для сканирования
        self.selected_master_serial = None  # Добавляем хранение выбранного Master

    def set_master_controller(self, master_controller):
        """Устанавливает Master контроллер для сканирования."""
        self.master_controller = master_controller

    def stop(self):
        self._stop = True

    def run(self):
        try:
            all_devices: List[SDRDeviceInfo] = []
            
            # Шаг 1: Ищем HackRF устройства через C библиотеку (для Master)
            self.progressUpdate.emit(20, "Поиск HackRF устройств...")
            hackrf_devices = self._find_hackrf_devices()
            all_devices.extend(hackrf_devices)
            
            if self._stop:
                return
            
            # Шаг 2: Ищем устройства через SoapySDR (для Slaves)
            self.progressUpdate.emit(60, "Поиск SoapySDR устройств...")
            soapy_devices = self._find_soapy_devices()
            all_devices.extend(soapy_devices)
            
            # Удаляем дубликаты
            unique_devices = self._deduplicate_devices(all_devices)
            
            self.progressUpdate.emit(100, f"Найдено устройств: {len(unique_devices)}")
            self.devicesFound.emit(unique_devices)
            
        except Exception as e:
            self.discoveryError.emit(str(e))
    
    def _find_hackrf_devices(self) -> List[SDRDeviceInfo]:
        """Поиск HackRF устройств через C библиотеку."""
        print("DEBUG: _find_hackrf_devices called")
        devices = []
        
        if not HACKRF_AVAILABLE:
            print("DEBUG: HACKRF_AVAILABLE is False")
            return devices
        
        try:
            # Используем Master контроллер если есть
            if self.master_controller and hasattr(self.master_controller, 'enumerate_devices'):
                print("DEBUG: Using master_controller.enumerate_devices")
                serials = self.master_controller.enumerate_devices()
                print(f"DEBUG: Got {len(serials) if serials else 0} serials from master_controller")
                
                if not serials:
                    print("DEBUG: No serials returned")
                    return []
                    
                for serial in serials:
                    print(f"DEBUG: Processing serial: '{serial}'")
                    # Еще более строгая проверка
                    if not serial or len(serial) != 32:
                        print(f"DEBUG: Skipping invalid serial: '{serial}'")
                        continue
                    
                    try:
                        # Проверяем что это hex
                        int(serial, 16)
                    except:
                        print(f"DEBUG: Skipping non-hex serial: '{serial}'")
                        continue
                    
                    info = SDRDeviceInfo(
                        driver="hackrf",
                        serial=serial,
                        label=f"HackRF {serial[-4:]}",
                        uri=f"driver=hackrf,serial={serial}",
                        capabilities={
                            "frequency_range": (24e6, 6e9),
                            "bandwidth": 20e6,
                            "sample_rates": [2e6, 4e6, 8e6, 10e6, 12.5e6, 16e6, 20e6]
                        }
                    )
                    info.is_available = True
                    info.last_seen = time.time()
                    devices.append(info)
                    print(f"DEBUG: Added device: {serial[-4:]}")
                
                return devices
                
            else:
                print("DEBUG: No master_controller, trying direct approach")
                # Если нет контроллера - НЕ создаем фиктивные устройства
                return []
                
        except Exception as e:
            print(f"DEBUG: Exception in _find_hackrf_devices: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _find_soapy_devices(self) -> List[SDRDeviceInfo]:
        """Поиск устройств через SoapySDR (исключая выбранный Master)."""
        devices = []
        
        if not SOAPY_AVAILABLE:
            return devices
        
        try:
            results = SoapySDR.Device.enumerate()
            
            for info_dict in results:
                # SoapyKwargs у разных сборок может не иметь .get — используем безопасные извлечения
                if hasattr(info_dict, 'get'):
                    driver = info_dict.get('driver', '')
                    serial = info_dict.get('serial', '')
                    label = info_dict.get('label', '')
                else:
                    try:
                        driver = str(info_dict)
                    except Exception:
                        driver = ''
                    serial = ''
                    label = ''
                
                # ВАЖНО: Исключаем устройство, выбранное как Master
                if self.selected_master_serial and serial == self.selected_master_serial:
                    print(f"Skipping Master device: {serial}")
                    continue
                
                # Также пропускаем HackRF устройства (они только для Master)
                if driver.lower() == 'hackrf':
                    print(f"Skipping HackRF for Slave: {serial}")
                    continue
                
                # Формируем URI для Slave
                uri_parts = []
                if driver:
                    uri_parts.append(f"driver={driver}")
                if serial:
                    uri_parts.append(f"serial={serial}")
                uri = ",".join(uri_parts) if uri_parts else ""
                
                if not driver:
                    continue
                
                device_info = SDRDeviceInfo(
                    driver=driver,
                    serial=serial,
                    label=label,
                    uri=uri,
                    is_available=True,  # SoapySDR уже проверил доступность
                    last_seen=time.time()
                )
                
                devices.append(device_info)
                
        except Exception as e:
            print(f"Error finding SoapySDR devices: {e}")
        
        return devices
    
    def _deduplicate_devices(self, devices: List[SDRDeviceInfo]) -> List[SDRDeviceInfo]:
        """Удаляет дубликаты устройств."""
        unique = []
        seen = set()
        
        for device in devices:
            key = (device.driver, device.serial or device.uri)
            if key not in seen:
                seen.add(key)
                unique.append(device)
        
        return unique


class ImprovedDeviceManagerDialog(QtWidgets.QDialog):
    """Улучшенный диалог управления устройствами."""
    
    devicesConfigured = pyqtSignal(dict)  # Конфигурация устройств

    def __init__(self, parent=None, current_config: Dict[str, Any] = None):
        super().__init__(parent)
        self.setWindowTitle("Диспетчер SDR устройств")
        self.resize(1200, 700)
        
        self._current_config = current_config or {}
        self.all_devices: List[SDRDeviceInfo] = []
        self.master_device: Optional[SDRDeviceInfo] = None
        self.slave_devices: List[SDRDeviceInfo] = []
        self.master_controller = None # Добавляем атрибут для хранения контроллера Master

        self._build_ui()
        self._connect_signals()

        # Начинаем поиск устройств
        QtCore.QTimer.singleShot(100, self._start_discovery)

    def set_master_controller(self, master_controller):
        """Устанавливает Master контроллер для сканирования устройств."""
        self.master_controller = master_controller
    
    def _build_ui(self):
        """Создает интерфейс."""
        layout = QtWidgets.QVBoxLayout(self)

        # Заголовок и кнопка обновления
        header_layout = QtWidgets.QHBoxLayout()
        header_label = QtWidgets.QLabel("<b>Настройка SDR устройств для трилатерации</b>")
        header_label.setStyleSheet("font-size: 14px; padding: 5px;")
        self.btn_refresh = QtWidgets.QPushButton("🔄 Обновить список")
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_refresh)
        layout.addLayout(header_layout)
        
        # Прогресс-бар
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QtWidgets.QLabel()
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)

        # Основной сплиттер
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # === Левая панель: Master устройство ===
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        
        master_group = QtWidgets.QGroupBox("Master устройство (HackRF)")
        master_layout = QtWidgets.QVBoxLayout(master_group)
        
        # Список HackRF устройств
        self.master_list = QtWidgets.QListWidget()
        self.master_list.setMaximumHeight(150)
        master_layout.addWidget(QtWidgets.QLabel("Выберите HackRF для Master sweep:"))
        master_layout.addWidget(self.master_list)
        
        # Информация о выбранном Master
        self.master_info = QtWidgets.QTextEdit()
        self.master_info.setReadOnly(True)
        self.master_info.setMaximumHeight(100)
        master_layout.addWidget(QtWidgets.QLabel("Информация:"))
        master_layout.addWidget(self.master_info)
        
        left_layout.addWidget(master_group)
        
        # === Центральная панель: Доступные устройства для Slave ===
        center_widget = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center_widget)
        
        available_group = QtWidgets.QGroupBox("Доступные устройства для Slave")
        available_layout = QtWidgets.QVBoxLayout(available_group)
        
        self.available_table = QtWidgets.QTableWidget()
        self.available_table.setColumnCount(5)
        self.available_table.setHorizontalHeaderLabels(["Тип", "Серийный", "Название", "Статус", "Действие"])
        self.available_table.horizontalHeader().setStretchLastSection(False)
        self.available_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        available_layout.addWidget(self.available_table)
        
        center_layout.addWidget(available_group)
        
        # === Правая панель: Настроенные Slave устройства ===
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        
        slaves_group = QtWidgets.QGroupBox("Slave устройства (для трилатерации)")
        slaves_layout = QtWidgets.QVBoxLayout(slaves_group)
        
        self.slaves_table = QtWidgets.QTableWidget()
        self.slaves_table.setColumnCount(7)
        self.slaves_table.setHorizontalHeaderLabels(["Никнейм", "X (м)", "Y (м)", "Z (м)", "URI", "Статус", "Удалить"])
        self.slaves_table.horizontalHeader().setStretchLastSection(False)
        slaves_layout.addWidget(self.slaves_table)
        
        # Кнопки для предустановленных позиций
        preset_layout = QtWidgets.QHBoxLayout()
        btn_triangle = QtWidgets.QPushButton("📐 Треугольник")
        btn_triangle.clicked.connect(self._preset_triangle)
        btn_square = QtWidgets.QPushButton("⬜ Квадрат")
        btn_square.clicked.connect(self._preset_square)
        btn_line = QtWidgets.QPushButton("📏 Линия")
        btn_line.clicked.connect(self._preset_line)
        preset_layout.addWidget(btn_triangle)
        preset_layout.addWidget(btn_square)
        preset_layout.addWidget(btn_line)
        slaves_layout.addLayout(preset_layout)
        
        right_layout.addWidget(slaves_group)
        
        # Добавляем панели в сплиттер
        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 400, 500])
        
        layout.addWidget(splitter)

        # Нижняя панель с кнопками
        bottom_layout = QtWidgets.QHBoxLayout()
        
        self.status_label = QtWidgets.QLabel("Готово")
        bottom_layout.addWidget(self.status_label)
        bottom_layout.addStretch()
        
        self.btn_save = QtWidgets.QPushButton("💾 Сохранить конфигурацию")
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 15px;
            }
        """)
        self.btn_cancel = QtWidgets.QPushButton("Отмена")
        
        bottom_layout.addWidget(self.btn_save)
        bottom_layout.addWidget(self.btn_cancel)
        
        layout.addLayout(bottom_layout)
    
    def _connect_signals(self):
        """Подключает сигналы."""
        self.btn_refresh.clicked.connect(self._start_discovery)
        self.btn_save.clicked.connect(self._save_configuration)
        self.btn_cancel.clicked.connect(self.reject)
        self.master_list.itemSelectionChanged.connect(self._on_master_selected)

    def _start_discovery(self):
        """Запускает поиск устройств."""
        if hasattr(self, '_discovery_thread') and self._discovery_thread.isRunning():
            return
        
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.status_label.setText("Поиск устройств...")
        
        self._discovery_thread = DeviceDiscoveryThread(self)
        
        # Передаем Master контроллер если доступен
        if hasattr(self, 'master_controller') and self.master_controller:
            self._discovery_thread.set_master_controller(self.master_controller)
        
        self._discovery_thread.devicesFound.connect(self._on_devices_found)
        self._discovery_thread.progressUpdate.connect(self._on_progress_update)
        self._discovery_thread.discoveryError.connect(self._on_discovery_error)
        self._discovery_thread.finished.connect(self._on_discovery_finished)
        self._discovery_thread.start()
    
    def _on_progress_update(self, value: int, message: str):
        """Обновляет прогресс."""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
    
    def _on_discovery_error(self, error: str):
        """Обрабатывает ошибку поиска."""
        QtWidgets.QMessageBox.warning(self, "Ошибка поиска", f"Ошибка при поиске устройств:\n{error}")

    def _on_discovery_finished(self):
        """Завершение поиска."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.status_label.setText(f"Найдено устройств: {len(self.all_devices)}")

    def _on_devices_found(self, devices: List[SDRDeviceInfo]):
        """Обрабатывает найденные устройства."""
        self.all_devices = devices
        
        # Если устройств нет, показываем сообщение
        if not devices:
            self.master_list.clear()
            item = QtWidgets.QListWidgetItem("Нет доступных HackRF устройств")
            item.setForeground(QtGui.QBrush(QtGui.QColor(150, 150, 150)))
            item.setFlags(QtCore.Qt.NoItemFlags)  # Делаем неселектируемым
            self.master_list.addItem(item)
            
            self.available_table.setRowCount(0)
            self.status_label.setText("Устройства не найдены")
            return
        
        # Применяем сохраненную конфигурацию
        self._apply_saved_config()
        
        # Обновляем UI
        self._update_master_list()
        self._update_available_table()
        self._update_slaves_table()
    
    def _update_master_list(self):
        """Обновляет список HackRF для Master."""
        self.master_list.clear()
        
        hackrf_devices = [d for d in self.all_devices if d.driver == "hackrf"]
        
        for device in hackrf_devices:
            item_text = f"{device.nickname} ({device.serial})"
            if device.is_available:
                item_text += " ✅"
            else:
                item_text += " ❌"
            
            item = QtWidgets.QListWidgetItem(item_text)
            item.setData(QtCore.Qt.UserRole, device)
            
            if device.is_available:
                item.setForeground(QtGui.QBrush(QtGui.QColor(0, 150, 0)))
            else:
                item.setForeground(QtGui.QBrush(QtGui.QColor(150, 0, 0)))
            
            self.master_list.addItem(item)
            
            # Выбираем, если это текущий Master
            if self.master_device and device.serial == self.master_device.serial:
                item.setSelected(True)
    
    def _update_available_table(self):
        """Обновляет таблицу доступных устройств для Slave."""
        # Фильтруем устройства для Slave:
        # 1. Исключаем все HackRF (они только для Master)
        # 2. Исключаем выбранный Master
        available = []
        
        for d in self.all_devices:
            # Пропускаем HackRF устройства
            if d.driver.lower() == "hackrf":
                continue
            
            # Пропускаем выбранный Master
            if self.master_device and d.serial == self.master_device.serial:
                continue
            
            available.append(d)
        
        self.available_table.setRowCount(len(available))
        
        for row, device in enumerate(available):
            # Тип
            self.available_table.setItem(row, 0, QtWidgets.QTableWidgetItem(device.driver.upper()))
            
            # Серийный номер
            self.available_table.setItem(row, 1, QtWidgets.QTableWidgetItem(device.serial or "N/A"))
            
            # Название
            self.available_table.setItem(row, 2, QtWidgets.QTableWidgetItem(device.label or device.nickname))
            
            # Статус
            status = "✅ Доступен" if device.is_available else "❌ Недоступен"
            status_item = QtWidgets.QTableWidgetItem(status)
            if device.is_available:
                status_item.setForeground(QtGui.QBrush(QtGui.QColor(0, 150, 0)))
            else:
                status_item.setForeground(QtGui.QBrush(QtGui.QColor(150, 0, 0)))
            self.available_table.setItem(row, 3, status_item)
            
            # Кнопка действия
            if device not in self.slave_devices:
                btn = QtWidgets.QPushButton("➕ Добавить")
                btn.clicked.connect(lambda checked, d=device: self._add_slave(d))
                self.available_table.setCellWidget(row, 4, btn)
            else:
                label = QtWidgets.QLabel("Уже добавлен")
                label.setAlignment(QtCore.Qt.AlignCenter)
                self.available_table.setCellWidget(row, 4, label)
    
    def _update_slaves_table(self):
        """Обновляет таблицу Slave устройств."""
        self.slaves_table.setRowCount(len(self.slave_devices))
        
        for row, device in enumerate(self.slave_devices):
            # Никнейм (редактируемый)
            nickname_item = QtWidgets.QTableWidgetItem(device.nickname)
            self.slaves_table.setItem(row, 0, nickname_item)
            
            # Позиция X, Y, Z (редактируемые)
            for col, value in enumerate(device.position, start=1):
                pos_item = QtWidgets.QTableWidgetItem(str(value))
                self.slaves_table.setItem(row, col, pos_item)
            
            # URI (только чтение)
            uri_item = QtWidgets.QTableWidgetItem(device.uri)
            uri_item.setFlags(uri_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.slaves_table.setItem(row, 4, uri_item)
            
            # Статус
            status = "✅" if device.is_available else "❌"
            status_item = QtWidgets.QTableWidgetItem(status)
            status_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.slaves_table.setItem(row, 5, status_item)
            
            # Кнопка удаления
            btn_remove = QtWidgets.QPushButton("❌ Удалить")
            btn_remove.clicked.connect(lambda checked, d=device: self._remove_slave(d))
            self.slaves_table.setCellWidget(row, 6, btn_remove)
    
    def _on_master_selected(self):
        """Обрабатывает выбор Master устройства."""
        selected = self.master_list.selectedItems()
        
        if selected:
            device = selected[0].data(QtCore.Qt.UserRole)
            self.master_device = device
            
            # ВАЖНО: Обновляем список доступных Slave устройств
            # исключая выбранный Master
            if hasattr(self, '_discovery_thread'):
                self._discovery_thread.selected_master_serial = device.serial
            
            # Удаляем Master из списка Slaves если он там есть
            self._remove_master_from_slaves(device.serial)
            
            # Обновляем информацию
            info_text = f"Устройство: {device.nickname}\n"
            info_text += f"Серийный номер: {device.serial}\n"
            info_text += f"Драйвер: {device.driver}\n"
            
            if device.capabilities:
                freq_range = device.capabilities.get('frequency_range')
                if freq_range:
                    info_text += f"Диапазон: {freq_range[0]/1e6:.0f} - {freq_range[1]/1e9:.1f} ГГц\n"
                
                bandwidth = device.capabilities.get('bandwidth')
                if bandwidth:
                    info_text += f"Полоса: {bandwidth/1e6:.0f} МГц\n"
            
            self.master_info.setPlainText(info_text)
        else:
            self.master_device = None
            if hasattr(self, '_discovery_thread'):
                self._discovery_thread.selected_master_serial = None
            self.master_info.clear()
        
        # Обновляем таблицу доступных устройств
        self._update_available_table()
    
    def _add_slave(self, device: SDRDeviceInfo):
        """Добавляет устройство как Slave."""
        if device not in self.slave_devices:
            device.role = DeviceRole.SLAVE
            self.slave_devices.append(device)
            self._update_available_table()
            self._update_slaves_table()
    
    def _remove_slave(self, device: SDRDeviceInfo):
        """Удаляет устройство из Slave."""
        if device in self.slave_devices:
            device.role = DeviceRole.NONE
            self.slave_devices.remove(device)
            self._update_available_table()
            self._update_slaves_table()
    
    def _remove_master_from_slaves(self, master_serial: str):
        """Удаляет Master из списка Slave устройств."""
        # Удаляем из slave_devices если есть
        self.slave_devices = [
            dev for dev in self.slave_devices 
            if dev.serial != master_serial
        ]
        
        # Обновляем таблицу Slaves
        self._update_slaves_table()
    
    def _preset_triangle(self):
        """Устанавливает позиции Slave в виде треугольника."""
        if len(self.slave_devices) < 3:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", 
                                         "Нужно минимум 3 Slave устройства для треугольника")
            return
        
        # Равносторонний треугольник с Master в центре
        # Master в (0, 0), Slaves на вершинах
        positions = [
            (0.0, 10.0, 0.0),      # Верх
            (-8.66, -5.0, 0.0),    # Левый нижний
            (8.66, -5.0, 0.0),     # Правый нижний
        ]
        
        for i, device in enumerate(self.slave_devices[:3]):
            device.position = positions[i]
        
        self._update_slaves_table()
    
    def _preset_square(self):
        """Устанавливает позиции Slave в виде квадрата."""
        if len(self.slave_devices) < 4:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", 
                                         "Нужно минимум 4 Slave устройства для квадрата")
            return
        
        # Квадрат с Master в центре
        positions = [
            (-5.0, 5.0, 0.0),    # Левый верхний
            (5.0, 5.0, 0.0),     # Правый верхний
            (5.0, -5.0, 0.0),    # Правый нижний
            (-5.0, -5.0, 0.0),   # Левый нижний
        ]
        
        for i, device in enumerate(self.slave_devices[:4]):
            device.position = positions[i]
        
        self._update_slaves_table()
    
    def _preset_line(self):
        """Устанавливает позиции Slave в линию."""
        if len(self.slave_devices) < 2:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", 
                                         "Нужно минимум 2 Slave устройства для линии")
            return
        
        # Линия вдоль оси X
        spacing = 10.0
        for i, device in enumerate(self.slave_devices):
            device.position = ((i + 1) * spacing, 0.0, 0.0)
        
        self._update_slaves_table()
    
    def _apply_saved_config(self):
        """Применяет сохраненную конфигурацию."""
        if not self._current_config:
            return
        
        # Восстанавливаем Master
        master_serial = self._current_config.get('master', {}).get('serial')
        if master_serial:
            for device in self.all_devices:
                if device.driver == "hackrf" and device.serial == master_serial:
                    self.master_device = device
                    device.role = DeviceRole.MASTER
                    break
        
        # Восстанавливаем Slaves
        saved_slaves = self._current_config.get('slaves', [])
        for saved_slave in saved_slaves:
            # Ищем по URI или серийному номеру
            for device in self.all_devices:
                if (device.uri == saved_slave.get('uri') or 
                    device.serial == saved_slave.get('serial')):
                    device.role = DeviceRole.SLAVE
                    device.nickname = saved_slave.get('nickname', device.nickname)
                    pos = saved_slave.get('pos', [0.0, 0.0, 0.0])
                    device.position = (float(pos[0]), float(pos[1]), float(pos[2]))
                    if device not in self.slave_devices:
                        self.slave_devices.append(device)
                    break
    
    def _gather_table_data(self):
        """Собирает данные из таблиц."""
        # Обновляем никнеймы и позиции Slave из таблицы
        for row, device in enumerate(self.slave_devices):
            # Никнейм
            nickname_item = self.slaves_table.item(row, 0)
            if nickname_item:
                device.nickname = nickname_item.text()
            
            # Позиции
            try:
                x = float(self.slaves_table.item(row, 1).text())
                y = float(self.slaves_table.item(row, 2).text())
                z = float(self.slaves_table.item(row, 3).text())
                device.position = (x, y, z)
            except (AttributeError, ValueError):
                pass
    
    def _save_configuration(self):
        """Сохраняет конфигурацию."""
        # Собираем данные из таблиц
        self._gather_table_data()
        
        # Проверяем конфигурацию
        if not self.master_device:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", 
                                         "Выберите Master устройство (HackRF)")
            return
        
        if len(self.slave_devices) < 2:
            reply = QtWidgets.QMessageBox.question(self, "Подтверждение",
                "Для трилатерации рекомендуется минимум 3 устройства (1 Master + 2 Slave).\n"
                "Сейчас настроено меньше. Продолжить?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            
            if reply != QtWidgets.QMessageBox.Yes:
                return
        
        # Формируем конфигурацию (ключи pos совместимы с основной логикой)
        config = {
            'master': {
                'driver': self.master_device.driver,
                'serial': self.master_device.serial,
                'nickname': self.master_device.nickname,
                'uri': self.master_device.uri,
                'pos': [0.0, 0.0, 0.0]
            },
            'slaves': [
                {
                    'driver': device.driver,
                    'serial': device.serial,
                    'nickname': device.nickname,
                    'uri': device.uri,
                    'pos': list(device.position),
                    'label': device.label
                }
                for device in self.slave_devices
            ]
        }
        
        # Эмитим сигнал с конфигурацией
        self.devicesConfigured.emit(config)
        
        # Сохраняем через проектный сторедж и в файл пользователя
        try:
            save_sdr_settings(config)
        except Exception:
            pass
        self._save_to_file(config)
        
        QtWidgets.QMessageBox.information(self, "Успех", 
            f"Конфигурация сохранена:\n"
            f"Master: {self.master_device.nickname}\n"
            f"Slaves: {len(self.slave_devices)} устройств")
        
        self.accept()

    def _save_to_file(self, config: dict):
        """Сохраняет конфигурацию в файл."""
        import os
        from pathlib import Path
        
        config_dir = Path.home() / ".panorama"
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / "device_config.json"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")


def load_device_config() -> dict:
    """Загружает конфигурацию устройств из файла."""
    from pathlib import Path
    
    config_file = Path.home() / ".panorama" / "device_config.json"
    
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    
    return {}


