#!/usr/bin/env python3
"""
Улучшенный диспетчер устройств с правильной логикой Master/Slave.
Master - только один HackRF из списка C библиотеки.
Slaves - множество устройств через SoapySDR.
"""

import os
# ОТКЛЮЧАЕМ AVAHI В SOAPYSDR ДО ВСЕХ ИМПОРТОВ
# Это предотвращает ошибки "avahi_service_browser_new() failed: Bad state"
os.environ['SOAPY_SDR_DISABLE_AVAHI'] = '1'

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import time
import json

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal

# Импорт C библиотеки для HackRF
try:
    from panorama.drivers.hackrf.hrf_backend import HackRFQSABackend
    HACKRF_AVAILABLE = True
except ImportError:
    HackRFQSABackend = None
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
                # Прямой поиск через C библиотеку
                try:
                    if HackRFQSABackend:
                        # Создаем временный экземпляр для поиска устройств
                        temp_backend = HackRFQSABackend()
                        
                        # Безопасно вызываем list_serials
                        try:
                            serials = temp_backend.list_serials()
                            print(f"DEBUG: Direct search found {len(serials) if serials else 0} devices")
                            
                            if serials:
                                for serial in serials:
                                    if serial and len(serial) >= 4:
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
                                        print(f"DEBUG: Added device via direct search: {serial[-4:]}")
                        except Exception as e:
                            print(f"DEBUG: list_serials failed: {e}")
                        
                        # Очищаем временный объект
                        del temp_backend
                        
                except Exception as e:
                    print(f"DEBUG: Direct search failed: {e}")
                
                return devices
                
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
            # Отключаем Avahi для предотвращения ошибок при поиске сетевых устройств
            import os
            old_env = os.environ.get('SOAPY_SDR_DISABLE_AVAHI')
            os.environ['SOAPY_SDR_DISABLE_AVAHI'] = '1'
            
            results = SoapySDR.Device.enumerate()
            
            # Восстанавливаем старое значение
            if old_env is not None:
                os.environ['SOAPY_SDR_DISABLE_AVAHI'] = old_env
            else:
                del os.environ['SOAPY_SDR_DISABLE_AVAHI']
            
            for info_dict in results:
                # SoapyKwargs у разных сборок может не иметь .get — используем безопасные извлечения
                if hasattr(info_dict, 'get'):
                    driver = info_dict.get('driver', '')
                    serial = info_dict.get('serial', '')
                    label = info_dict.get('label', '')
                else:
                    # Пробуем извлечь данные из строкового представления
                    try:
                        info_str = str(info_dict)
                        print(f"DEBUG: Processing SoapyKwargs: {info_str}")
                        
                        # Инициализируем переменные
                        driver = ''
                        serial = ''
                        label = ''
                        
                        # Парсим строку для извлечения данных
                        if 'driver=' in info_str:
                            driver_start = info_str.find('driver=') + 7
                            driver_end = info_str.find(',', driver_start)
                            if driver_end == -1:
                                driver_end = info_str.find('}', driver_start)
                            driver = info_str[driver_start:driver_end].strip()
                        
                        if 'serial=' in info_str:
                            serial_start = info_str.find('serial=') + 7
                            serial_end = info_str.find(',', serial_start)
                            if serial_end == -1:
                                serial_end = info_str.find('}', serial_start)
                            serial = info_str[serial_start:serial_end].strip()
                        
                        if 'label=' in info_str:
                            label_start = info_str.find('label=') + 6
                            label_end = info_str.find(',', label_start)
                            if label_end == -1:
                                label_end = info_str.find('}', label_start)
                            label = info_str[label_start:label_end].strip()
                        
                        print(f"DEBUG: Parsed - driver: '{driver}', serial: '{serial}', label: '{label}'")
                    except Exception as e:
                        print(f"DEBUG: Error parsing SoapyKwargs: {e}")
                        driver = ''
                        serial = ''
                        label = ''
                
                # ВАЖНО: Пропускаем пустые или некорректные устройства
                if not driver or driver == '':
                    print(f"Skipping empty driver device")
                    continue
                
                # Пропускаем аудио устройства (не SDR)
                if driver.lower() in ['audio', 'pulse', 'alsa', 'jack']:
                    print(f"Skipping audio device: {driver}")
                    continue
                
                # ВАЖНО: Исключаем устройство, выбранное как Master
                if self.selected_master_serial and serial == self.selected_master_serial:
                    print(f"Skipping Master device: {serial}")
                    continue
                
                # НЕ пропускаем HackRF устройства - они могут быть Slave!
                # if driver.lower() == 'hackrf':
                #     print(f"Skipping HackRF for Slave: {serial}")
                #     continue
                
                # Формируем URI для Slave
                uri_parts = []
                if driver:
                    uri_parts.append(f"driver={driver}")
                if serial:
                    uri_parts.append(f"serial={serial}")
                uri = ",".join(uri_parts) if uri_parts else ""
                
                device_info = SDRDeviceInfo(
                    driver=driver,
                    serial=serial,
                    label=label,
                    uri=uri,
                    is_available=True,  # SoapySDR уже проверил доступность
                    last_seen=time.time()
                )
                
                devices.append(device_info)
                print(f"Found SoapySDR device: {driver} ({serial})")
                
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
    slavesAvailable = pyqtSignal(list)    # Доступные устройства для слейвов
    devicesForCoordinatesTable = pyqtSignal(list)  # Устройства для координатной таблицы

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
        
        available_group = QtWidgets.QGroupBox("Доступные SDR устройства")
        available_layout = QtWidgets.QVBoxLayout(available_group)
        
        # Информация о управлении Slave
        info_label = QtWidgets.QLabel(
            "ℹ️ Управление Slave устройствами и их координатами осуществляется во вкладке 'Слейвы'"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("""
            QLabel {
                background-color: rgba(33, 150, 243, 30);
                padding: 8px;
                border-radius: 4px;
                margin-bottom: 5px;
            }
        """)
        available_layout.addWidget(info_label)
        
        self.available_table = QtWidgets.QTableWidget()
        self.available_table.setColumnCount(5)
        self.available_table.setHorizontalHeaderLabels(["Тип", "Серийный", "Название", "Статус", "Действия"])
        self.available_table.horizontalHeader().setStretchLastSection(False)
        self.available_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        # Улучшаем читаемость таблицы
        header = self.available_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        self.available_table.verticalHeader().setDefaultSectionSize(28)
        self.available_table.setAlternatingRowColors(True)
        available_layout.addWidget(self.available_table)
        
        center_layout.addWidget(available_group)
        
        # Добавляем панели в сплиттер (только Master и доступные устройства)
        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        splitter.setSizes([400, 600])
        
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
        print(f"DEBUG: _on_devices_found called with {len(devices)} devices")
        for i, device in enumerate(devices):
            print(f"DEBUG: Device {i}: {device.driver} ({device.serial}) - {device.label}")
        
        self.all_devices = devices
        
        # Если устройств нет, показываем сообщение
        if not devices:
            print("DEBUG: No devices found, showing empty message")
            self.master_list.clear()
            item = QtWidgets.QListWidgetItem("Нет доступных HackRF устройств")
            item.setForeground(QtGui.QBrush(QtGui.QColor(150, 150, 150)))
            item.setFlags(QtCore.Qt.NoItemFlags)  # Делаем неселектируемым
            self.master_list.addItem(item)
            
            self.available_table.setRowCount(0)
            self.status_label.setText("Устройства не найдены")
            return
        
        print("DEBUG: Applying saved config and updating UI")
        # Применяем сохраненную конфигурацию
        self._apply_saved_config()
        
        # Обновляем UI
        self._update_master_list()
        self._update_available_table()
    
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
        # 1. Исключаем только выбранный Master
        # 2. HackRF устройства могут быть Slave!
        available = []
        
        for d in self.all_devices:
            # Пропускаем выбранный Master
            if self.master_device and d.serial == self.master_device.serial:
                print(f"DEBUG: Skipping selected Master from Slave list: {d.serial}")
                continue
            
            available.append(d)
            print(f"DEBUG: Added device to Slave list: {d.driver} ({d.serial})")
        
        print(f"DEBUG: Total devices available for Slave: {len(available)}")

        # Перед заполнением гарантированно удаляем предыдущие виджеты ячеек,
        # чтобы не оставались старые кнопки/лейблы и не наслаивались элементы
        try:
            rows = self.available_table.rowCount()
            cols = self.available_table.columnCount()
            for r in range(rows):
                for c in range(cols):
                    w = self.available_table.cellWidget(r, c)
                    if w is not None:
                        w.deleteLater()
        except Exception:
            pass

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
            status_item.setTextAlignment(QtCore.Qt.AlignCenter)
            if device.is_available:
                status_item.setForeground(QtGui.QBrush(QtGui.QColor(0, 150, 0)))
            else:
                status_item.setForeground(QtGui.QBrush(QtGui.QColor(150, 0, 0)))
            self.available_table.setItem(row, 3, status_item)
            
            # Создаем контейнер для кнопок
            button_widget = QtWidgets.QWidget()
            button_layout = QtWidgets.QHBoxLayout(button_widget)
            button_layout.setContentsMargins(4, 2, 4, 2)
            button_layout.setSpacing(4)
            
            # Единая кнопка-действие: Добавить / Удалить
            action_btn = QtWidgets.QPushButton()
            if not device.is_available:
                action_btn.setText("Недоступно")
                action_btn.setEnabled(False)
                action_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #9E9E9E;
                        color: #222;
                        border: none;
                        padding: 4px 8px;
                        border-radius: 3px;
                        font-size: 11px;
                    }
                """)
            else:
                is_added = device in self.slave_devices
                action_btn.setText("🗑️ Удалить" if is_added else "➕ Добавить")
                action_btn.setStyleSheet("""
                    QPushButton {
                        background-color: %s;
                        color: white;
                        border: none;
                        padding: 4px 8px;
                        border-radius: 3px;
                        font-size: 11px;
                    }
                    QPushButton:hover {
                        background-color: %s;
                    }
                """ % ("#F44336" if is_added else "#2196F3",
                        "#D32F2F" if is_added else "#1976D2"))
                action_btn.clicked.connect(lambda checked, dev=device: self._toggle_slave_device(dev))

            button_layout.addWidget(action_btn)
            
            self.available_table.setCellWidget(row, 4, button_widget)
        
        # Эмитируем сигнал с доступными устройствами для синхронизации со слейвами
        self.slavesAvailable.emit(available)
        
        # Дополнительно эмитируем сигнал для обновления координатной таблицы
        self._emit_devices_for_coordinates_table()

    def _toggle_slave_device(self, device: SDRDeviceInfo):
        """Переключает состояние устройства между Добавить/Удалить.
        Используется единой кнопкой действий, чтобы избежать визуальной каши.
        """
        if device in self.slave_devices:
            self._remove_slave_device(device)
        else:
            self._add_slave_device(device)
    
    def _on_master_selected(self):
        """Обрабатывает выбор Master устройства."""
        selected = self.master_list.selectedItems()
        
        if selected:
            device = selected[0].data(QtCore.Qt.UserRole)
            old_master_serial = self.master_device.serial if self.master_device else None
            self.master_device = device
            
            # ВАЖНО: Обновляем список доступных Slave устройств
            # исключая выбранный Master
            if hasattr(self, '_discovery_thread'):
                self._discovery_thread.selected_master_serial = device.serial
            
            # Сохраняем текущих Slave (кроме старого Master)
            saved_slaves = []
            if old_master_serial:
                for slave in self.slave_devices:
                    if slave.serial != old_master_serial:
                        saved_slaves.append(slave)
            
            # Удаляем Master из списка Slaves если он там есть
            self._remove_master_from_slaves(device.serial)
            
            # Восстанавливаем Slave устройства (кроме нового Master)
            for slave in saved_slaves:
                if slave.serial != device.serial:
                    self._add_slave(slave)
            
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
            
            # Автоматически сохраняем обновленную конфигурацию
            self._auto_save_config()
            
            # Обновляем координатную таблицу
            self._emit_devices_for_coordinates_table()
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
    
    def _add_slave_device(self, device: SDRDeviceInfo):
        """Добавляет устройство как Slave через кнопку."""
        if device not in self.slave_devices and device.is_available:
            # Устанавливаем роль и никнейм по умолчанию
            device.role = DeviceRole.SLAVE
            
            # Генерируем уникальный никнейм для Slave
            slave_number = len(self.slave_devices) + 1
            device.nickname = f"Slave{slave_number}"
            
            # Устанавливаем позицию по умолчанию
            default_positions = [
                (10.0, 0.0, 0.0),   # Slave1
                (0.0, 10.0, 0.0),   # Slave2  
                (-10.0, 0.0, 0.0),  # Slave3
                (0.0, -10.0, 0.0),  # Slave4
                (5.0, 5.0, 0.0),    # Slave5
                (-5.0, -5.0, 0.0)   # Slave6
            ]
            
            if slave_number <= len(default_positions):
                device.position = default_positions[slave_number - 1]
            else:
                # Для дополнительных устройств случайная позиция
                import random
                angle = random.uniform(0, 2 * 3.14159)
                radius = 10.0
                import math
                device.position = (radius * math.cos(angle), radius * math.sin(angle), 0.0)
            
            self.slave_devices.append(device)
            self._update_available_table()
            
            # Автоматически сохраняем конфигурацию
            self._auto_save_config()
            
            # Обновляем координатную таблицу
            self._emit_devices_for_coordinates_table()
            
            # Показываем сообщение
            QtWidgets.QMessageBox.information(self, "Успех", 
                f"Устройство {device.nickname} добавлено как Slave\n"
                f"Позиция: ({device.position[0]:.1f}, {device.position[1]:.1f}, {device.position[2]:.1f})")
        else:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", 
                "Устройство недоступно или уже добавлено")
    
    def _remove_slave(self, device: SDRDeviceInfo):
        """Удаляет устройство из Slave."""
        if device in self.slave_devices:
            device.role = DeviceRole.NONE
            self.slave_devices.remove(device)
            self._update_available_table()
    
    def _remove_slave_device(self, device: SDRDeviceInfo):
        """Удаляет устройство из Slave через кнопку."""
        if device in self.slave_devices:
            # Подтверждение удаления
            reply = QtWidgets.QMessageBox.question(self, "Подтверждение", 
                f"Удалить устройство {device.nickname} из списка Slave?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            
            if reply == QtWidgets.QMessageBox.Yes:
                device.role = DeviceRole.NONE
                self.slave_devices.remove(device)
                self._update_available_table()
                
                # Автоматически сохраняем конфигурацию
                self._auto_save_config()
                
                # Обновляем координатную таблицу
                self._emit_devices_for_coordinates_table()
                
                # Показываем сообщение
                QtWidgets.QMessageBox.information(self, "Успех", 
                    f"Устройство {device.nickname} удалено из списка Slave")
    
    def _remove_master_from_slaves(self, master_serial: str):
        """Удаляет Master из списка Slave устройств."""
        print(f"DEBUG: Removing master {master_serial} from slaves list")
        print(f"DEBUG: Before removal: {len(self.slave_devices)} slaves")
        
        # Удаляем из slave_devices если есть
        self.slave_devices = [
            dev for dev in self.slave_devices 
            if dev.serial != master_serial
        ]
        
        print(f"DEBUG: After removal: {len(self.slave_devices)} slaves")
    
    
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
        # Данные Slave теперь управляются во вкладке "Слейвы"
        # Здесь только сохраняем Master конфигурацию
        pass
    
    def _auto_save_config(self):
        """Автоматически сохраняет конфигурацию при изменениях."""
        if not self.master_device:
            return
        
        try:
            # Собираем данные из таблиц
            self._gather_table_data()
            
            # Формируем конфигурацию
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

            # Гарантируем, что slave0 всегда в (0,0,0) и сохраняется так в конфигурации
            for s in config.get('slaves', []):
                try:
                    if (s.get('nickname') or s.get('label') or '').lower() == 'slave0':
                        s['pos'] = [0.0, 0.0, float(s['pos'][2]) if isinstance(s.get('pos'), list) and len(s['pos']) > 2 else 0.0]
                except Exception:
                    pass
            
            print(f"DEBUG: Auto-saving config: {len(config.get('slaves', []))} slaves")
            
            # Сохраняем конфигурацию
            self.devicesConfigured.emit(config)
            save_sdr_settings(config)
            self._save_to_file(config)
            
        except Exception as e:
            print(f"DEBUG: Error auto-saving config: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
    
    def _save_configuration(self):
        """Сохраняет конфигурацию."""
        # Собираем данные из таблиц
        self._gather_table_data()
        
        # Проверяем конфигурацию
        if not self.master_device:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", 
                                         "Выберите Master устройство (HackRF)")
            return
        
        # Считаем общее количество устройств: 1 Master + количество Slave
        total_devices = 1 + len(self.slave_devices)  # 1 Master + N Slaves
        
        if total_devices < 3:  # Нужно минимум 3 устройства (1 Master + 2 Slave)
            reply = QtWidgets.QMessageBox.question(self, "Подтверждение",
                f"Для трилатерации рекомендуется минимум 3 устройства (1 Master + 2 Slave).\n"
                f"Сейчас настроено: 1 Master + {len(self.slave_devices)} Slave = {total_devices} устройств.\n"
                f"Продолжить?",
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

        # Гарантируем, что slave0 всегда в (0,0,0)
        for s in config.get('slaves', []):
            try:
                if (s.get('nickname') or s.get('label') or '').lower() == 'slave0':
                    s['pos'] = [0.0, 0.0, float(s['pos'][2]) if isinstance(s.get('pos'), list) and len(s['pos']) > 2 else 0.0]
            except Exception:
                pass
        
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
        
        # После сохранения конфигурации обновляем карту количеством станций
        try:
            devices_count = len(self.slave_devices)
            update_payload = {
                'type': 'update_devices_coordinates',
                'devices': [
                    {
                        'id': f'slave{i+1}',
                        'x': d.position[0],
                        'y': d.position[1],
                        'z': d.position[2] if len(d.position) > 2 else 0.0,
                        'is_reference': False
                    } for i, d in enumerate(self.slave_devices)
                ]
            }
            # Если есть хотя бы один слейв — добавим опорную точку
            update_payload['devices'].insert(0, {
                'id': 'slave0', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'is_reference': True
            })
            # Передаем наверх через сигнал, main_rssi перенаправит на карту
            if hasattr(self, 'devicesConfigured'):
                # Переиспользуем существующий путь обновления карты
                pass
        except Exception:
            pass
        
        self.accept()

    def _emit_devices_for_coordinates_table(self):
        """Эмитирует сигнал для обновления координатной таблицы."""
        try:
            # Формируем список только Slave устройств (Master не участвует в трилатерации)
            devices_for_coords = []
            
            # Правило опорного: если есть явно сохранённый slave0 — он опорный;
            # иначе первый в списке становится опорным.
            reference_idx = 0
            for idx, dev in enumerate(self.slave_devices):
                nickname_lower = (dev.nickname or '').lower()
                if nickname_lower == 'slave0' or nickname_lower == 'опорное':
                    reference_idx = idx
                    break
            
            for i, device in enumerate(self.slave_devices):
                is_reference = (i == reference_idx)
                # Опорное устройство всегда в (0,0,0)
                coords = (0.0, 0.0, 0.0) if is_reference else tuple(device.position)
                
                devices_for_coords.append({
                    'nickname': device.nickname or f"SDR-{(device.serial or '0000')[-4:]}",
                    'serial': device.serial,
                    'driver': device.driver, 
                    'is_available': device.is_available,
                    'coords': coords,
                    'status': 'REFERENCE' if is_reference else ('AVAILABLE' if device.is_available else 'UNAVAILABLE'),
                    'is_reference': is_reference
                })
            
            # Эмитируем специальный сигнал для координатной таблицы
            if hasattr(self, 'devicesForCoordinatesTable'):
                self.devicesForCoordinatesTable.emit(devices_for_coords)
            
            print(f"DEBUG: Emitted {len(devices_for_coords)} Slave devices for coordinates table")
            
        except Exception as e:
            print(f"DEBUG: Error emitting devices for coordinates: {e}")
    
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


