#!/usr/bin/env python3
"""
ПАНОРАМА RSSI - Система трилатерации по RSSI в реальном времени.
Основное приложение для координации Master sweep и Slave SDR операций.
"""

import sys
import os
import logging
import time
from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QSettings, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QTabWidget, QGroupBox, QLabel, QPushButton, QSpinBox, QDoubleSpinBox
from PyQt5.QtWidgets import QTextEdit, QTableWidget, QTableWidgetItem, QComboBox, QCheckBox
from PyQt5.QtWidgets import QSplitter, QFrame, QMessageBox, QFileDialog, QFormLayout
import numpy as np

# Импортируем наши модули
from panorama.features.spectrum.master_adapter import MasterSourceAdapter
from panorama.features.slave_controller.slave import SlaveManager
from panorama.features.trilateration import RSSITrilaterationEngine
from panorama.features.slave_controller.orchestrator import Orchestrator
from panorama.features.calibration.manager import CalibrationManager

# Импортируем существующие модули для совместимости
from panorama.features.map import OpenLayersMapWidget
from panorama.features.spectrum import SpectrumView
from panorama.features.settings.dialog import SettingsDialog
from panorama.features.settings.manager_improved import ImprovedDeviceManagerDialog
from panorama.features.settings.storage import load_sdr_settings, save_sdr_settings
from panorama.features.spectrum.master_adapter import MasterSourceAdapter
from panorama.features.detector.settings_dialog import (
    DetectorSettingsDialog, DetectorSettings, load_detector_settings, apply_settings_to_watchlist_manager
)
from panorama.features.watchlist.view import ImprovedSlavesView
from panorama.features.detector.peak_watchlist_manager import PeakWatchlistManager
from panorama.features.trilateration.coordinator import TrilaterationCoordinator


class RSSIPanoramaMainWindow(QMainWindow):
    """Главное окно приложения ПАНОРАМА RSSI."""
    
    def __init__(self):
        super().__init__()
        
        # Настройка логирования
        self._setup_logging()
        
        # Инициализация компонентов
        self._init_components()
        
        # Настройка UI
        self._setup_ui()
        
        # Подключение сигналов
        self._connect_signals()
        
        # Загрузка калибровки
        self._load_calibration()
        
        # Проверяем наличие конфигурации Master (БЕЗ инициализации SDR)
        self.master_ready = self._check_and_init_master()
        
        # Обновляем состояние кнопок в зависимости от наличия конфигурации Master
        self._update_ui_for_master_status()
        
        # Статус системы
        self.system_status = {
            'master_running': False,
            'orchestrator_running': False,
            'n_slaves': 0,
            'n_targets': 0
        }
        
        # Таймер обновления статуса
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(1000)  # Каждую секунду
        
        self.log.info("ПАНОРАМА RSSI initialized")
    
    def _setup_logging(self):
        """Настраивает систему логирования."""
        self.log = logging.getLogger("panorama_rssi")
        self.log.setLevel(logging.INFO)
        
        # Создаем handler для вывода в консоль
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Форматтер
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        self.log.addHandler(console_handler)
    
    def _init_components(self):
        """Инициализирует основные компоненты системы."""
        try:
            # Менеджер калибровки
            self.calibration_manager = CalibrationManager(self.log)
            
            # Движок трилатерации
            self.trilateration_engine = RSSITrilaterationEngine()
            
            # Менеджер slave SDR
            self.slave_manager = SlaveManager(self.log)
            
            # Контроллер Master sweep больше не используется напрямую
            self.master_controller = None
            
            # Оркестратор
            self.orchestrator = Orchestrator(self.log)
            
            # Создаем менеджер пиков и координатор трилатерации
            self.peak_watchlist_manager = PeakWatchlistManager()
            self.trilateration_coordinator = TrilaterationCoordinator()
            
            # Устанавливаем span из UI
            self.trilateration_coordinator.set_user_span(5.0)  # По умолчанию 5 МГц
            
            # Подключение к спектру будет выполнено после создания UI
            
            # Подключаем компоненты к оркестратору
            self.orchestrator.set_master_controller(self.master_controller)
            self.orchestrator.set_slave_manager(self.slave_manager)
            self.orchestrator.set_trilateration_engine(self.trilateration_engine)
            
            # Подключаем координатор трилатерации к оркестратору
            self.orchestrator.set_trilateration_coordinator(self.trilateration_coordinator)

            # Применяем сохранённые настройки детектора к менеджеру пиков и оркестратору
            try:
                det_settings = load_detector_settings()
                if det_settings:
                    apply_settings_to_watchlist_manager(det_settings, self.trilateration_coordinator.peak_manager)
                    self.orchestrator.set_global_parameters(
                        span_hz=det_settings.watchlist_span_mhz * 1e6,
                        dwell_ms=int(det_settings.watchlist_dwell_ms)
                    )
                    self.trilateration_coordinator.set_user_span(float(det_settings.watchlist_span_mhz))
            except Exception:
                pass
            
            # Загружаем настройки SDR (master/slaves) из JSON
            self.sdr_settings = load_sdr_settings()

            # Инициализируем слейвы из настроек сразу при старте
            try:
                self._init_slaves_from_settings()
            except Exception as e:
                self.log.error(f"Failed to init slaves from settings: {e}")

            # Настраиваем трилатерацию
            self._setup_trilateration()
            
            self.log.info("All components initialized successfully")
            
        except Exception as e:
            self.log.error(f"Error initializing components: {e}")
            QMessageBox.critical(self, "Ошибка инициализации", 
                               f"Не удалось инициализировать компоненты: {e}")

    def _init_slaves_from_settings(self):
        """Читает конфиг и инициализирует слейвы в SlaveManager."""
        if not self.slave_manager or not self.sdr_settings:
            return
        # Очистим текущих
        for sid in list(self.slave_manager.slaves.keys()):
            self.slave_manager.remove_slave(sid)
        # Добавим из конфига
        for idx, s in enumerate(self.sdr_settings.get('slaves', []), start=1):
            sid = s.get('nickname') or (s.get('label') or s.get('serial') or f"slave{idx:02d}")
            uri = s.get('uri') or s.get('soapy') or (f"driver={s.get('driver')}" if s.get('driver') else '')
            if (not uri) and s.get('serial'):
                uri = f"serial={s.get('serial')}"
            if uri:
                ok = self.slave_manager.add_slave(sid, uri)
                if not ok:
                    self.log.error(f"Failed to init slave {sid} with uri={uri}")
    
    def _check_and_init_master(self):
        """Проверяет наличие конфигурации Master устройства при старте (БЕЗ инициализации SDR)."""
        try:
            if not self.sdr_settings or 'master' not in self.sdr_settings:
                self.log.warning("No SDR settings found - Master device not configured")
                return False
            
            master_config = self.sdr_settings['master']
            if not master_config or 'serial' not in master_config:
                self.log.warning("Invalid master configuration - no serial number")
                return False
            
            master_serial = master_config['serial']
            if not master_serial or len(master_serial) < 16:
                self.log.warning(f"Invalid master serial: {master_serial}")
                return False
            
            self.log.info(f"Found master configuration: {master_serial}")
            self.log.info("Master device configured but NOT initialized (will be initialized when needed)")
            return True
            
        except Exception as e:
            self.log.error(f"Error checking master configuration: {e}")
            return False
    
    def _update_ui_for_master_status(self):
        """Обновляет состояние UI в зависимости от наличия конфигурации Master устройства."""
        try:
            if hasattr(self, 'toolbar_orch_action'):
                if self.master_ready:
                    self.toolbar_orch_action.setEnabled(True)
                    self.toolbar_orch_action.setToolTip("Master устройство настроено. Можно запускать спектр.")
                else:
                    self.toolbar_orch_action.setEnabled(False)
                    self.toolbar_orch_action.setToolTip("Master устройство не настроено")
            
            # Показываем сообщение о статусе Master
            if not self.master_ready:
                QMessageBox.information(self, "Настройка Master устройства", 
                    "Master устройство не настроено.\n\n"
                    "Для работы спектра необходимо:\n"
                    "1. Перейти в Настройки → Диспетчер устройств\n"
                    "2. Выбрать HackRF устройство как Master\n"
                    "3. Сохранить конфигурацию\n\n"
                    "После настройки перезапустите приложение.")
            
        except Exception as e:
            self.log.error(f"Error updating UI for master status: {e}")
    
    def _setup_trilateration(self):
        """Настраивает движок трилатерации."""
        try:
            # Получаем параметры калибровки
            cal_params = self.calibration_manager.get_calibration_parameters()
            
            if cal_params:
                self.trilateration_engine.set_path_loss_exponent(cal_params['path_loss_exponent'])
                self.trilateration_engine.set_reference_parameters(
                    cal_params['reference_distance'],
                    cal_params['reference_power']
                )
            
            # Очищаем все станции
            self.trilateration_engine.stations.clear()
            
            # НЕ добавляем Master в трилатерацию - он отвечает только за спектр
            # Master добавляем только если он настроен
            # master_config = self.sdr_settings.get('master', {})
            # if master_config.get('serial') or master_config.get('uri'):
            #     self.trilateration_engine.add_station("master", 0.0, 0.0, 0.0, 0.0)
            
            # Slaves добавляем только те, которые реально настроены
            for s in self.sdr_settings.get('slaves', []):
                # Проверяем, что устройство реально настроено
                ser = s.get('serial', '')
                uri = s.get('uri') or s.get('soapy') or ''
                if ser or uri:  # Если есть хотя бы один идентификатор
                    x, y, z = s.get('pos', [0.0, 0.0, 0.0])
                    sid = s.get('nickname') or (s.get('label') or s.get('serial') or 'slave')
                    self.trilateration_engine.add_station(sid, float(x), float(y), float(z), 0.0)
            
            stations_count = len(self.trilateration_engine.get_station_positions())
            self.log.info(f"Trilateration engine configured with {stations_count} stations (Master excluded - spectrum only)")
            
        except Exception as e:
            self.log.error(f"Error setting up trilateration: {e}")
    
    def _setup_ui(self):
        """Настраивает пользовательский интерфейс."""
        self.setWindowTitle("ПАНОРАМА RSSI - Система трилатерации по RSSI")
        self.setGeometry(100, 100, 1400, 900)
        # Современная тёмная тема: qdarkstyle (если доступен), иначе fallback на палитру
        try:
            import qdarkstyle  # type: ignore
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        except Exception:
            dark = QtGui.QPalette()
            dark.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
            dark.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
            dark.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
            dark.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(35, 35, 35))
            dark.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
            dark.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
            dark.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
            dark.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
            dark.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
            dark.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
            dark.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
            dark.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
            dark.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
            self.setPalette(dark)
            self.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout - теперь только правая панель
        main_layout = QVBoxLayout(central_widget)
        
        # Правая панель - карта, спектр и управление слейвами
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel)
        
        # Создаем меню
        self._create_menu()
        
        # Создаем панель инструментов
        self._create_toolbar()
        
        # Подключаем трилатерацию к спектру (после создания UI)
        self._connect_trilateration()
    

    
    def _show_detector_settings(self):
        """Показывает диалог настроек детектора."""
        try:
            dlg = DetectorSettingsDialog(self)
            def _on_changed(s: DetectorSettings):
                # Применяем некоторые глобальные параметры к оркестратору
                try:
                    if self.orchestrator:
                        self.orchestrator.set_global_parameters(span_hz=s.watchlist_span_mhz * 1e6,
                                                                dwell_ms=int(s.watchlist_dwell_ms))
                    if self.trilateration_coordinator:
                        self.trilateration_coordinator.set_user_span(float(s.watchlist_span_mhz))
                    # Применяем параметры детектора к менеджеру пиков
                    if self.trilateration_coordinator:
                        apply_settings_to_watchlist_manager(s, self.trilateration_coordinator.peak_manager)
                except Exception:
                    pass
            dlg.settingsChanged.connect(_on_changed)
            dlg.exec_()
        except Exception as e:
            self.log.error(f"Detector settings dialog error: {e}")
    
    def _create_right_panel(self):
        """Создает правую панель с картой, спектром и управлением слейвами."""
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        

        
        # Создаем вкладки
        tab_widget = QTabWidget()
        
        # Вкладка карты
        self.map_view = OpenLayersMapWidget()
        try:
            if hasattr(self, 'sdr_settings') and self.sdr_settings:
                self.map_view.update_stations_from_config(self.sdr_settings)
        except Exception:
            pass
        tab_widget.addTab(self.map_view, "🗺️ Карта")
        
        # Вкладка спектра
        self.spectrum_view = SpectrumView(orchestrator=self.orchestrator)
        # Привязываем источник через адаптер, чтобы старт сразу запускал C-свип
        try:
            self.spectrum_view.set_source(MasterSourceAdapter(self.log))
        except Exception:
            pass
        tab_widget.addTab(self.spectrum_view, "📊 Спектр")
        
        # Вкладка управления слейвами (объединяет watchlist, результаты и контроль)
        self.slaves_view = ImprovedSlavesView(orchestrator=self.orchestrator)
        tab_widget.addTab(self.slaves_view, "🎯 Слейвы")
        
        layout.addWidget(tab_widget)
        
        return right_widget
    

    
    def _create_menu(self):
        """Создает главное меню."""
        menubar = self.menuBar()
        
        # Меню Файл
        file_menu = menubar.addMenu('Файл')
        
        # Действие загрузки калибровки
        load_cal_action = file_menu.addAction('Загрузить калибровку...')
        load_cal_action.triggered.connect(self._load_calibration_dialog)
        
        # Действие сохранения калибровки
        save_cal_action = file_menu.addAction('Сохранить калибровку...')
        save_cal_action.triggered.connect(self._save_calibration_dialog)
        
        file_menu.addSeparator()
        
        # Действие выхода
        exit_action = file_menu.addAction('Выход')
        exit_action.triggered.connect(self.close)
        
        # Меню Настройки
        settings_menu = menubar.addMenu('Настройки')
        
        # Действие настройки калибровки
        cal_settings_action = settings_menu.addAction('Настройки калибровки...')
        cal_settings_action.triggered.connect(self._show_calibration_settings)

        # Действие настройки детектора
        det_settings_action = settings_menu.addAction('Настройки детектора...')
        det_settings_action.triggered.connect(self._show_detector_settings)
        
        # Меню Слейвы
        slaves_menu = menubar.addMenu('🎯 Слейвы')
        
        # Действие обновления данных слейвов
        refresh_slaves_action = slaves_menu.addAction('🔄 Обновить данные')
        refresh_slaves_action.triggered.connect(self._refresh_slaves_data)
        
        # Действие экспорта состояния слейвов
        export_slaves_action = slaves_menu.addAction('💾 Экспорт состояния...')
        export_slaves_action.triggered.connect(self._export_slaves_state)
        
        slaves_menu.addSeparator()
        
        # Действие очистки данных слейвов
        clear_slaves_action = slaves_menu.addAction('🗑️ Очистить данные')
        clear_slaves_action.triggered.connect(self._clear_slaves_data)
        
        # Меню Справка
        help_menu = menubar.addMenu('Справка')
        
        # Действие о программе
        about_action = help_menu.addAction('О программе...')
        about_action.triggered.connect(self._show_about)
    
    def _create_toolbar(self):
        """Создает панель инструментов."""
        toolbar = self.addToolBar('Основная панель')
        # Убираем кнопку управления оркестратором — управление автоматически при наличии 1 master + >=2 slaves
        
        # Только новый диспетчер устройств
        toolbar.addAction('🧭 Диспетчер устройств', self._open_device_manager)
        
        
        # Убираем дублирование кнопки "Измерить" в тулбаре — остаётся кнопка в параметрах
    
    def _connect_signals(self):
        """Подключает сигналы компонентов."""
        try:
            # Сигналы Master
            if self.master_controller:
                self.master_controller.peak_detected.connect(self._on_peak_detected)
                self.master_controller.sweep_error.connect(self._on_sweep_error)
            # Сигналы оркестратора
            if self.orchestrator:
                self.orchestrator.task_created.connect(self._on_task_created)
                self.orchestrator.task_completed.connect(self._on_task_completed)
                self.orchestrator.task_failed.connect(self._on_task_failed)
                self.orchestrator.target_detected.connect(self._on_target_detected)
            # Сигналы трилатерации
            if self.trilateration_engine:
                self.trilateration_engine.target_update.connect(self._on_target_update)
                self.trilateration_engine.trilateration_error.connect(self._on_trilateration_error)
            # Сигналы slave
            if self.slave_manager:
                self.slave_manager.measurement_error.connect(self._on_measurement_error)
                
            # Сигналы WatchlistView
            if hasattr(self, 'watchlist_view') and self.watchlist_view:
                self.watchlist_view.task_cancelled.connect(self._on_task_cancelled)
                self.watchlist_view.task_retried.connect(self._on_task_retried)
            
            # Сигналы SlavesView
            if hasattr(self, 'slaves_view') and self.slaves_view:
                self.slaves_view.send_to_map.connect(self._on_slave_target_to_map)
                self.slaves_view.task_selected.connect(self._on_slave_task_selected)
                self.slaves_view.watchlist_updated.connect(self._on_slave_watchlist_updated)
                
            self.log.info("All signals connected successfully")
        except Exception as e:
            self.log.error(f"Error connecting signals: {e}")

    def _open_settings(self):
        # Устаревший диалог — переадресация на новый диспетчер
        self._open_device_manager()

    def _open_device_manager(self):
        current = {
            'master': {
                'nickname': self.sdr_settings.get('master', {}).get('nickname', 'Master'),
                'serial': self.sdr_settings.get('master', {}).get('serial', ''),
                'pos': [0.0, 0.0, 0.0],
            },
            'slaves': self.sdr_settings.get('slaves', [])
        }
        dlg = ImprovedDeviceManagerDialog(self, current)
        # Передаем Master контроллер для сканирования устройств
        if hasattr(dlg, 'set_master_controller'):
            dlg.set_master_controller(self.master_controller)
        def _on_conf(data: dict):
            # Apply to runtime
            self.sdr_settings = data
            # Master
            try:
                serial = data.get('master', {}).get('serial')
                if self.master_controller and getattr(self.master_controller, 'sweep_source', None):
                    if not serial:
                        self.master_controller.stop_sweep()
                        self.master_controller.sweep_source.set_serial(None)
                    else:
                        self.master_controller.sweep_source.set_serial(serial)
            except Exception:
                pass
            # Rebuild slaves
            try:
                if self.slave_manager:
                    for sid in list(self.slave_manager.slaves.keys()):
                        self.slave_manager.remove_slave(sid)
                    for idx, s in enumerate(data.get('slaves', []), start=1):
                        sid = s.get('nickname') or (s.get('label') or s.get('serial') or f"slave{idx:02d}")
                        # поддержка форматов: uri, soapy, driver, serial
                        uri = s.get('uri') or s.get('soapy') or (f"driver={s.get('driver')}" if s.get('driver') else '')
                        if (not uri) and s.get('serial'):
                            uri = f"serial={s.get('serial')}"
                        if uri:
                            ok = self.slave_manager.add_slave(sid, uri)
                            if not ok:
                                self.log.error(f"Failed to init slave {sid} with uri={uri}")
                try:
                    self._update_slave_table()
                except Exception:
                    pass
            except Exception:
                pass
            # Re-configure trilateration stations
            try:
                self.trilateration_engine.stations.clear()
                self._setup_trilateration()
            except Exception:
                pass
            # Update map stations from new config
            try:
                if hasattr(self, 'map_view') and self.map_view:
                    self.map_view.update_stations_from_config(self.sdr_settings)
            except Exception:
                pass
        dlg.devicesConfigured.connect(_on_conf)
        dlg.exec_()
    
    def _load_calibration(self):
        """Загружает калибровку по умолчанию."""
        try:
            # Калибровка уже загружена в _init_components
            self.log.info("Calibration loaded successfully")
        except Exception as e:
            self.log.error(f"Error loading calibration: {e}")
    
    def _add_slave(self):
        """Добавляет новый slave SDR."""
        try:
            uri = self.slave_uri_edit.currentText()
            if not uri:
                QMessageBox.warning(self, "Предупреждение", "Введите URI для SDR")
                return
            
            # Генерируем ID для slave
            slave_id = f"slave{len(self.slave_manager.slaves) + 1:02d}"
            
            # Добавляем slave
            success = self.slave_manager.add_slave(slave_id, uri)
            
            if success:
                self._update_slave_table()
                self.log.info(f"Added slave: {slave_id} ({uri})")
            else:
                QMessageBox.critical(self, "Ошибка", f"Не удалось добавить slave: {uri}")
            
        except Exception as e:
            self.log.error(f"Error adding slave: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при добавлении slave: {e}")
    
    def _update_slave_table(self):
        """Обновляет таблицу slave."""
        try:
            # Если таблицы нет (упрощённый UI), выходим без ошибок
            if not hasattr(self, 'slave_table') or self.slave_table is None:
                return
            slaves = self.slave_manager.get_slave_status()
            
            self.slave_table.setRowCount(len(slaves))
            
            for row, (slave_id, status) in enumerate(slaves.items()):
                # ID
                self.slave_table.setItem(row, 0, QTableWidgetItem(slave_id))
                
                # URI
                self.slave_table.setItem(row, 1, QTableWidgetItem(status.get('uri', 'N/A')))
                
                # Статус
                status_text = "READY" if status.get('is_initialized') else "ERROR"
                self.slave_table.setItem(row, 2, QTableWidgetItem(status_text))
                
                # Кнопка удаления
                remove_btn = QPushButton("Удалить")
                remove_btn.clicked.connect(lambda checked, sid=slave_id: self._remove_slave(sid))
                self.slave_table.setCellWidget(row, 3, remove_btn)
            
            self.system_status['n_slaves'] = len(slaves)
            
        except Exception as e:
            self.log.error(f"Error updating slave table: {e}")
    
    def _remove_slave(self, slave_id: str):
        """Удаляет slave SDR."""
        try:
            self.slave_manager.remove_slave(slave_id)
            self._update_slave_table()
            self.log.info(f"Removed slave: {slave_id}")
            
        except Exception as e:
            self.log.error(f"Error removing slave {slave_id}: {e}")
    

    
    def _start_orchestrator(self):
        """Запускает оркестратор."""
        try:
            # Проверяем наличие конфигурации Master устройства
            if not self.master_ready:
                QMessageBox.warning(self, "Master устройство не настроено", 
                    "Для запуска спектра необходимо настроить Master устройство.\n\n"
                    "Перейдите в Настройки → Диспетчер устройств")
                return
            
            # Инициализируем SDR если нужно
            if self.master_controller and not self.master_controller.is_sdr_initialized():
                self.log.info("Initializing SDR for orchestrator...")
                if not self.master_controller.initialize_sdr():
                    raise RuntimeError("Failed to initialize SDR")
                self.log.info("SDR initialized successfully")
            
            # Устанавливаем глобальные параметры по умолчанию
            # (конкретные значения настраиваются через диалог настроек детектора)
            span_hz = 2.0 * 1e6  # 2 MHz по умолчанию
            dwell_ms = 150  # 150 ms по умолчанию
            
            self.orchestrator.set_global_parameters(span_hz, dwell_ms)
            
            # Запускаем оркестратор
            self.orchestrator.start()
            
            # Кнопки управления оркестратором находятся в тулбаре
            self.toolbar_orch_action.setText('⏹ Контроль')
            
            self.system_status['orchestrator_running'] = True
            self.log.info("Orchestrator started")
            
        except Exception as e:
            self.log.error(f"Error starting orchestrator: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить оркестратор: {e}")
    
    def _stop_orchestrator(self):
        """Останавливает оркестратор."""
        try:
            # Безопасно останавливаем мастер, чтобы не осталось потоков/коллбеков
            try:
                if self.master_controller:
                    self.master_controller.stop_sweep()
            except Exception:
                pass
            self.orchestrator.stop()
            
            # Деинициализируем SDR после остановки
            if self.master_controller and self.master_controller.is_sdr_initialized():
                self.log.info("Deinitializing SDR after orchestrator stop...")
                self.master_controller.deinitialize_sdr()
                self.log.info("SDR deinitialized successfully")
            
            # Кнопки управления оркестратором находятся в тулбаре
            self.toolbar_orch_action.setText('▶ Контроль')
            
            self.system_status['orchestrator_running'] = False
            self.log.info("Orchestrator stopped")
            
        except Exception as e:
            self.log.error(f"Error stopping orchestrator: {e}")
    
    def _toggle_orchestrator(self):
        """Переключает состояние оркестратора."""
        if self.system_status['orchestrator_running']:
            self._stop_orchestrator()
        else:
            self._start_orchestrator()
    
    def _on_peak_detected(self, peak):
        """Обрабатывает обнаружение пика."""
        try:
            self.log.info(f"Пик: {peak.f_peak/1e6:.1f} MHz, SNR: {peak.snr_db:.1f} dB")
            
            # Обновляем спектр
            if hasattr(self.spectrum_view, 'add_peak'):
                self.spectrum_view.add_peak(peak.f_peak, peak.snr_db)
                
        except Exception as e:
            self.log.error(f"Error handling peak: {e}")
    
    def _on_sweep_error(self, error_msg):
        """Обрабатывает ошибку sweep."""
        self.log.info(f"Ошибка sweep: {error_msg}")
    
    def _on_task_created(self, task):
        """Обрабатывает создание задачи."""
        try:
            self.log.info(f"Задача создана: {task.id}")
            self._update_tasks_table()
            
        except Exception as e:
            self.log.error(f"Error handling task created: {e}")
    
    def _on_task_completed(self, task):
        """Обрабатывает завершение задачи."""
        try:
            self.log.info(f"Задача завершена: {task.id}")
            self._update_tasks_table()
            
        except Exception as e:
            self.log.error(f"Error handling task completed: {e}")
    
    def _on_task_failed(self, task):
        """Обрабатывает ошибку задачи."""
        try:
            self.log.info(f"Задача провалена: {task.id}")
            self._update_tasks_table()
            
        except Exception as e:
            self.log.error(f"Error handling task failed: {e}")
    
    def _on_target_detected(self, target):
        """Обрабатывает обнаружение цели."""
        try:
            self.log.info(f"Цель: {target.center_hz/1e6:.1f} MHz, "
                               f"({target.x:.1f}, {target.y:.1f}), доверие: {target.confidence:.2f}")
            
            # Обновляем карту
            if hasattr(self.map_view, 'add_target'):
                self.map_view.add_target(target.x, target.y, target.confidence)
            
            # Обновляем таблицу целей
            self._update_targets_table()
            
            self.system_status['n_targets'] += 1
            
        except Exception as e:
            self.log.error(f"Error handling target: {e}")
    
    def _on_target_update(self, target):
        """Обрабатывает обновление цели."""
        try:
            # Обновляем карту
            if hasattr(self.map_view, 'update_target'):
                self.map_view.update_target(target.x, target.y, target.confidence)
            
            # Обновляем таблицу целей
            self._update_targets_table()
            
        except Exception as e:
            self.log.error(f"Error handling target update: {e}")
    
    def _on_trilateration_error(self, error_msg):
        """Обрабатывает ошибку трилатерации."""
        self.log.info(f"Ошибка трилатерации: {error_msg}")
    
    def _on_measurement_error(self, error_msg):
        """Обрабатывает ошибку измерения."""
        self.log.info(f"Ошибка измерения: {error_msg}")
    
    def _on_task_cancelled(self, task_id: str):
        """Обрабатывает отмену задачи."""
        try:
            self.log.info(f"Задача отменена: {task_id}")
            # Обновляем статус задачи в оркестраторе
            if self.orchestrator and task_id in self.orchestrator.tasks:
                self.orchestrator.tasks[task_id].status = "CANCELLED"
                self._update_tasks_table()
                
        except Exception as e:
            self.log.error(f"Error handling task cancelled: {e}")
    
    def _on_task_retried(self, task_id: str):
        """Обрабатывает повторение задачи."""
        try:
            self.log.info(f"Задача повторяется: {task_id}")
            # Создаем новую задачу на основе старой
            if self.orchestrator and task_id in self.orchestrator.tasks:
                old_task = self.orchestrator.tasks[task_id]
                # Создаем новую задачу с теми же параметрами
                self.orchestrator._enqueue_task(
                    old_task.peak, 
                    old_task.window.span, 
                    old_task.window.dwell_ms
                )
                self._update_tasks_table()
                
        except Exception as e:
            self.log.error(f"Error handling task retry: {e}")
    
    def _update_tasks_table(self):
        """Обновляет таблицу задач."""
        try:
            if not hasattr(self, 'tasks_table') or self.tasks_table is None:
                return
            tasks = self.orchestrator.get_active_tasks()
            
            self.tasks_table.setRowCount(len(tasks))
            
            for row, task in enumerate(tasks):
                # ID
                self.tasks_table.setItem(row, 0, QTableWidgetItem(task.id))
                
                # Частота
                freq_mhz = task.peak.f_peak / 1e6
                self.tasks_table.setItem(row, 1, QTableWidgetItem(f"{freq_mhz:.1f}"))
                
                # Статус
                self.tasks_table.setItem(row, 2, QTableWidgetItem(task.status))
                
                # Время создания
                created_time = time.strftime("%H:%M:%S", time.localtime(task.created_at))
                self.tasks_table.setItem(row, 3, QTableWidgetItem(created_time))
                
                # Время завершения
                if task.completed_at:
                    completed_time = time.strftime("%H:%M:%S", time.localtime(task.completed_at))
                    self.tasks_table.setItem(row, 4, QTableWidgetItem(completed_time))
                else:
                    self.tasks_table.setItem(row, 4, QTableWidgetItem(""))
            
        except Exception as e:
            self.log.error(f"Error updating tasks table: {e}")
    
    def _update_targets_table(self):
        """Обновляет таблицу целей."""
        try:
            if not hasattr(self, 'targets_table') or self.targets_table is None:
                return
            targets = self.trilateration_engine.get_latest_results()
            
            self.targets_table.setRowCount(len(targets))
            
            for row, target in enumerate(targets):
                # Частота
                freq_mhz = target.center_hz / 1e6
                self.targets_table.setItem(row, 0, QTableWidgetItem(f"{freq_mhz:.1f}"))
                
                # Координаты
                self.targets_table.setItem(row, 1, QTableWidgetItem(f"{target.x:.1f}"))
                self.targets_table.setItem(row, 2, QTableWidgetItem(f"{target.y:.1f}"))
                
                # Доверие
                self.targets_table.setItem(row, 3, QTableWidgetItem(f"{target.confidence:.2f}"))
                
                # Возраст
                age_sec = target.age_ms / 1000
                self.targets_table.setItem(row, 4, QTableWidgetItem(f"{age_sec:.1f}"))
                
                # Количество станций
                self.targets_table.setItem(row, 5, QTableWidgetItem(str(target.n_stations)))
            
        except Exception as e:
            self.log.error(f"Error updating targets table: {e}")
    
    def _update_status(self):
        """Обновляет статус системы."""
        try:
            # Обновляем статус мастера по факту работы контроллера
            if self.master_controller is not None:
                self.system_status['master_running'] = bool(getattr(self.master_controller, 'is_running', False))
            # Обновляем статус оркестратора (без отображения в UI)
            if self.orchestrator:
                orch_status = self.orchestrator.get_system_status()
                # Статус отображается только в заголовке окна
            
            # Обновляем заголовок окна
            title = f"ПАНОРАМА RSSI - Master: {'ON' if self.system_status['master_running'] else 'OFF'}, "
            title += f"Orch: {'ON' if self.system_status['orchestrator_running'] else 'OFF'}, "
            title += f"Slaves: {self.system_status['n_slaves']}, Targets: {self.system_status['n_targets']}"
            
            self.setWindowTitle(title)
            
        except Exception as e:
            self.log.error(f"Error updating status: {e}")
    
    def _load_calibration_dialog(self):
        """Показывает диалог загрузки калибровки."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Загрузить калибровку", "", "JSON files (*.json)"
            )
            
            if file_path:
                success = self.calibration_manager.import_profile(file_path)
                if success:
                    QMessageBox.information(self, "Успех", "Калибровка загружена успешно")
                    self._setup_trilateration()
                else:
                    QMessageBox.critical(self, "Ошибка", "Не удалось загрузить калибровку")
                    
        except Exception as e:
            self.log.error(f"Error in load calibration dialog: {e}")
    
    def _save_calibration_dialog(self):
        """Показывает диалог сохранения калибровки."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить калибровку", "", "JSON files (*.json)"
            )
            
            if file_path:
                if self.calibration_manager.current_profile:
                    success = self.calibration_manager.export_profile(
                        self.calibration_manager.current_profile.name, file_path
                    )
                    if success:
                        QMessageBox.information(self, "Успех", "Калибровка сохранена успешно")
                    else:
                        QMessageBox.critical(self, "Ошибка", "Не удалось сохранить калибровку")
                        
        except Exception as e:
            self.log.error(f"Error in save calibration dialog: {e}")
    
    def _show_calibration_settings(self):
        """Показывает настройки калибровки."""
        # TODO: Реализовать диалог настроек калибровки
        QMessageBox.information(self, "Информация", "Настройки калибровки будут добавлены в следующей версии")

    def _show_detector_settings(self):
        """Показывает диалог настроек детектора."""
        try:
            dlg = DetectorSettingsDialog(self)
            def _on_changed(s: DetectorSettings):
                # Применяем некоторые глобальные параметры к оркестратору
                try:
                    if self.orchestrator:
                        self.orchestrator.set_global_parameters(span_hz=s.watchlist_span_mhz * 1e6,
                                                                dwell_ms=int(s.watchlist_dwell_ms))
                except Exception:
                    pass
            dlg.settingsChanged.connect(_on_changed)
            dlg.exec_()
        except Exception as e:
            self.log.error(f"Detector settings dialog error: {e}")
    
    def _on_slave_target_to_map(self, target_data: dict):
        """Обрабатывает сигнал отправки цели на карту от slaves_view."""
        try:
            if hasattr(self, 'map_view') and self.map_view:
                # Отправляем цель на карту
                if hasattr(self.map_view, 'add_target_from_detector'):
                    self.map_view.add_target_from_detector(target_data)
                    self.log.info(f"Цель от слейва отправлена на карту: {target_data.get('id', 'Unknown')}")
                else:
                    self.log.warning("Метод add_target_from_detector не доступен в map_view")
            else:
                self.log.warning("Map view не доступен")
        except Exception as e:
            self.log.error(f"Ошибка отправки цели на карту: {e}")
    
    def _on_slave_task_selected(self, task_id: str):
        """Обрабатывает сигнал выбора задачи от slaves_view."""
        try:
            self.log.info(f"Выбрана задача от слейва: {task_id}")
            # TODO: Реализовать логику выбора задачи
        except Exception as e:
            self.log.error(f"Ошибка обработки выбора задачи: {e}")
    
    def _on_slave_watchlist_updated(self, watchlist_data: list):
        """Обрабатывает сигнал обновления watchlist от slaves_view."""
        try:
            # Логируем только при реальных изменениях, а не каждые 2 секунды
            if not hasattr(self, '_last_watchlist_count') or self._last_watchlist_count != len(watchlist_data):
                self.log.info(f"Watchlist обновлен от слейва: {len(watchlist_data)} элементов")
                self._last_watchlist_count = len(watchlist_data)
            # TODO: Реализовать логику обновления watchlist
        except Exception as e:
            self.log.error(f"Ошибка обработки обновления watchlist: {e}")
    
    def _refresh_slaves_data(self):
        """Обновляет данные слейвов."""
        try:
            if hasattr(self, 'slaves_view') and self.slaves_view:
                # Вызываем метод обновления данных в slaves_view
                if hasattr(self.slaves_view, 'manual_refresh'):
                    self.slaves_view.manual_refresh()
                self.log.info("Данные слейвов обновлены")
            else:
                self.log.warning("Slaves view не доступен")
        except Exception as e:
            self.log.error(f"Ошибка обновления данных слейвов: {e}")
    
    def _export_slaves_state(self):
        """Экспортирует состояние слейвов в файл."""
        try:
            if hasattr(self, 'slaves_view') and self.slaves_view:
                # Вызываем метод экспорта в slaves_view
                if hasattr(self.slaves_view, 'export_current_state'):
                    self.slaves_view.export_current_state()
                    self.log.info("Состояние слейвов экспортировано")
                else:
                    self.log.warning("Метод экспорта не доступен в slaves_view")
            else:
                self.log.warning("Slaves view не доступен")
        except Exception as e:
            self.log.error(f"Ошибка экспорта состояния слейвов: {e}")
    
    def _clear_slaves_data(self):
        """Очищает данные слейвов."""
        try:
            if hasattr(self, 'slaves_view') and self.slaves_view:
                # Вызываем метод очистки в slaves_view
                if hasattr(self.slaves_view, 'clear_all_data'):
                    self.slaves_view.clear_all_data()
                    self.log.info("Данные слейвов очищены")
                else:
                    self.log.warning("Метод очистки не доступен в slaves_view")
            else:
                self.log.warning("Slaves view не доступен")
        except Exception as e:
            self.log.error(f"Ошибка очистки данных слейвов: {e}")
    
    def _show_about(self):
        """Показывает информацию о программе."""
        QMessageBox.about(self, "О программе", 
                         "ПАНОРАМА RSSI v1.0\n\n"
                         "Система трилатерации по RSSI в реальном времени\n"
                         "Поддерживает множественные SDR станции\n\n"
                         "© 2024 ПАНОРАМА Team")
    
    def _connect_trilateration(self):
        """Подключает систему трилатерации к спектру."""
        # Когда приходят данные от Master
        self.spectrum_view.newRowReady.connect(
            self.trilateration_coordinator.process_master_spectrum
        )
        # Пересылка задач watchlist в оркестратор для измерений слейвами
        try:
            self.trilateration_coordinator.peak_manager.watchlist_task_ready.connect(
                self.orchestrator.enqueue_watchlist_task
            )
        except Exception as e:
            self.log.error(f"Failed to connect watchlist tasks: {e}")
        
        # Обновления позиций slave из настроек
        self.trilateration_coordinator.set_slave_positions({
            'slave1': (10.0, 0.0, 0.0),
            'slave2': (0.0, 10.0, 0.0),
            'slave3': (-10.0, 0.0, 0.0)
        })
        
        # Подключаем к карте
        self.trilateration_coordinator.target_detected.connect(
            self.map_view.add_target_from_detector
        )
        # И в UI слейвов — добавлять передатчики
        try:
            if hasattr(self, 'slaves_view') and self.slaves_view:
                self.trilateration_coordinator.target_detected.connect(
                    self.slaves_view.add_transmitter
                )
                # Обновление трекинга — позиция и уверенность
                if hasattr(self.trilateration_coordinator, 'target_updated'):
                    self.trilateration_coordinator.target_updated.connect(
                        self.slaves_view.update_transmitter_position
                    )
        except Exception:
            pass
        # TODO: Добавить метод update_target_position в OpenLayersMapWidget
        # self.trilateration_coordinator.target_updated.connect(
        #     self.map_view.update_target_position
        # )
        # Подписка карты на живой список слейвов
        try:
            if self.slave_manager and hasattr(self.slave_manager, 'slaves_updated'):
                def _on_slaves_updated(status: dict):
                    # Преобразуем в список станций
                    stations = []
                    for sid, st in status.items():
                        pos = (0.0, 0.0, 0.0)
                        for cfg in self.sdr_settings.get('slaves', []):
                            cid = cfg.get('nickname') or cfg.get('label') or cfg.get('serial')
                            if cid == sid:
                                p = cfg.get('pos', [0.0, 0.0, 0.0])
                                if len(p) >= 2:
                                    pos = (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0)
                                break
                        stations.append({'id': sid, 'x': pos[0], 'y': pos[1], 'z': pos[2]})
                    # Обновляем карту
                    try:
                        self.map_view.update_stations_from_config({'slaves': [{'nickname': s['id'], 'pos': [s['x'], s['y'], s['z']]} for s in stations]})
                    except Exception:
                        pass
                self.slave_manager.slaves_updated.connect(_on_slaves_updated)
                # Инициализация: отразить отсутствие слейвов
                self.slave_manager.slaves_updated.emit(self.slave_manager.get_slave_status())
        except Exception:
            pass
        # Запуск координатора
        try:
            self.trilateration_coordinator.start()
        except Exception as e:
            self.log.error(f"Failed to start TrilaterationCoordinator: {e}")
    
    def closeEvent(self, event):
        """Обрабатывает закрытие приложения."""
        try:
            # Останавливаем все компоненты в правильном порядке
            if hasattr(self, 'trilateration_coordinator') and self.trilateration_coordinator:
                try:
                    self.trilateration_coordinator.stop()
                except Exception:
                    pass
            if self.master_controller:
                self.master_controller.stop_sweep()
                self.master_controller.cleanup()
            
            if self.orchestrator:
                self.orchestrator.stop()
            
            if self.slave_manager:
                self.slave_manager.close_all()
            
            self.log.info("Application closed")
            event.accept()
            
        except Exception as e:
            self.log.error(f"Error during application close: {e}")
            event.accept()


def main():
    """Главная функция приложения."""
    # WSL/без GPU: отключаем аппаратное ускорение для QtWebEngine/Qt OpenGL
    try:
        import os
        os.environ.setdefault("QT_OPENGL", "software")
        os.environ.setdefault("QTWEBENGINE_DISABLE_GPU", "1")
        os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-software-rasterizer --in-process-gpu")
        os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
        from PyQt5 import QtCore
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL)
    except Exception:
        pass

    app = QApplication(sys.argv)
    
    # Настройка приложения
    app.setApplicationName("ПАНОРАМА RSSI")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("ПАНОРАМА Team")
    
    # Создание главного окна
    window = RSSIPanoramaMainWindow()
    window.show()
    
    # Запуск приложения
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
