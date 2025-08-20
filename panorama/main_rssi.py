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
from PyQt5.QtWidgets import QSplitter, QFrame, QMessageBox, QFileDialog
import numpy as np

# Импортируем наши модули
from panorama.features.master_sweep.master import MasterSweepController
from panorama.features.slave_sdr.slave import SlaveManager
from panorama.features.trilateration.engine import RSSITrilaterationEngine
from panorama.features.orchestrator.core import Orchestrator
from panorama.features.calibration.manager import CalibrationManager

# Импортируем существующие модули для совместимости
from panorama.features.map3d import MapView
from panorama.features.spectrum import SpectrumView
from panorama.features.settings.dialog import SettingsDialog
from panorama.features.settings.storage import load_sdr_settings, save_sdr_settings


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
            self.trilateration_engine = RSSITrilaterationEngine(self.log)
            
            # Менеджер slave SDR
            self.slave_manager = SlaveManager(self.log)
            
            # Контроллер Master sweep
            self.master_controller = MasterSweepController(self.log)
            
            # Оркестратор
            self.orchestrator = Orchestrator(self.log)
            
            # Подключаем компоненты к оркестратору
            self.orchestrator.set_master_controller(self.master_controller)
            self.orchestrator.set_slave_manager(self.slave_manager)
            self.orchestrator.set_trilateration_engine(self.trilateration_engine)
            
            # Загружаем настройки SDR (master/slaves) из JSON
            self.sdr_settings = load_sdr_settings()

            # Настраиваем трилатерацию
            self._setup_trilateration()
            
            self.log.info("All components initialized successfully")
            
        except Exception as e:
            self.log.error(f"Error initializing components: {e}")
            QMessageBox.critical(self, "Ошибка инициализации", 
                               f"Не удалось инициализировать компоненты: {e}")
    
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
            
            # Добавляем станции: Master по умолчанию (0,0,0), Slaves — из настроек, но только доступные
            self.trilateration_engine.add_station("master", 0.0, 0.0, 0.0, 0.0)
            # Фильтруем по реально доступным Soapy устройствам
            available = []
            try:
                if self.slave_manager:
                    available = self.slave_manager.enumerate_soapy_devices()
            except Exception:
                available = []
            avail_serials = {d.get('serial', '') for d in (available or []) if d.get('serial')}
            avail_uris = {d.get('uri', '') for d in (available or []) if d.get('uri')}
            for s in self.sdr_settings.get('slaves', []):
                ser = s.get('serial', '')
                uri = s.get('uri', '')
                if (ser and ser in avail_serials) or (uri and uri in avail_uris):
                    x, y, z = s.get('pos', [0.0, 0.0, 0.0])
                    sid = s.get('nickname') or (s.get('label') or s.get('serial') or 'slave')
                    self.trilateration_engine.add_station(sid, float(x), float(y), float(z), 0.0)
            
            stations_count = len(self.trilateration_engine.get_station_positions())
            self.log.info(f"Trilateration engine configured with {stations_count} stations")
            
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
        
        # Основной layout
        main_layout = QHBoxLayout(central_widget)
        
        # Создаем сплиттер для разделения панелей
        splitter = QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Левая панель — только спектр/водопад и результаты, без Master/Slave настроек
        left_panel = self._create_left_panel_min()
        splitter.addWidget(left_panel)
        
        # Правая панель - карта и спектр
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Устанавливаем пропорции сплиттера
        splitter.setSizes([400, 1000])
        
        # Создаем меню
        self._create_menu()
        
        # Создаем панель инструментов
        self._create_toolbar()
    
    def _create_left_panel_min(self):
        """Левая панель: параметры измерений + управление оркестратором, без настроек SDR."""
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)
        # Панель параметров измерений
        
        # Панель параметров
        params_group = QGroupBox("Параметры измерений")
        params_layout = QVBoxLayout(params_group)
        
        # Глобальные параметры
        global_params_layout = QHBoxLayout()
        global_params_layout.addWidget(QLabel("Span (MHz):"))
        self.span_spin = QDoubleSpinBox()
        self.span_spin.setRange(0.1, 10.0)
        self.span_spin.setValue(2.0)
        self.span_spin.setDecimals(1)
        global_params_layout.addWidget(self.span_spin)
        
        global_params_layout.addWidget(QLabel("Dwell (ms):"))
        self.global_dwell_spin = QSpinBox()
        self.global_dwell_spin.setRange(50, 500)
        self.global_dwell_spin.setValue(150)
        global_params_layout.addWidget(self.global_dwell_spin)
        params_layout.addLayout(global_params_layout)
        
        # Режим работы
        mode_layout = QHBoxLayout()
        self.auto_mode_check = QCheckBox("Автоматический режим")
        self.auto_mode_check.setChecked(True)
        self.auto_mode_check.toggled.connect(self._toggle_mode)
        mode_layout.addWidget(self.auto_mode_check)
        
        self.manual_mode_check = QCheckBox("Ручной режим")
        self.manual_mode_check.toggled.connect(self._toggle_mode)
        mode_layout.addWidget(self.manual_mode_check)
        params_layout.addLayout(mode_layout)
        
        # Ручное измерение
        manual_layout = QHBoxLayout()
        manual_layout.addWidget(QLabel("Частота (MHz):"))
        self.manual_freq_spin = QDoubleSpinBox()
        self.manual_freq_spin.setRange(24.0, 6000.0)
        self.manual_freq_spin.setValue(2400.0)
        self.manual_freq_spin.setDecimals(1)
        manual_layout.addWidget(self.manual_freq_spin)
        
        self.manual_measure_btn = QPushButton("Измерить")
        self.manual_measure_btn.clicked.connect(self._manual_measure)
        manual_layout.addWidget(self.manual_measure_btn)
        params_layout.addLayout(manual_layout)
        
        layout.addWidget(params_group)
        
        # Панель оркестратора
        orchestrator_group = QGroupBox("Оркестратор")
        orchestrator_layout = QVBoxLayout(orchestrator_group)
        
        # Кнопки управления
        orch_buttons_layout = QHBoxLayout()
        self.start_orch_btn = QPushButton("Старт")
        self.start_orch_btn.clicked.connect(self._start_orchestrator)
        orch_buttons_layout.addWidget(self.start_orch_btn)
        
        self.stop_orch_btn = QPushButton("Стоп")
        self.stop_orch_btn.clicked.connect(self._stop_orchestrator)
        self.stop_orch_btn.setEnabled(False)
        orch_buttons_layout.addWidget(self.stop_orch_btn)
        orchestrator_layout.addLayout(orch_buttons_layout)
        
        # Статус
        self.orchestrator_status_label = QLabel("Статус: Остановлен")
        orchestrator_layout.addWidget(self.orchestrator_status_label)
        
        layout.addWidget(orchestrator_group)
        
        # Лог
        log_group = QGroupBox("Лог")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        return left_widget
    
    def _create_right_panel(self):
        """Создает правую панель с картой и спектром."""
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        
        # Создаем вкладки
        tab_widget = QTabWidget()
        
        # Вкладка карты
        self.map_view = MapView()
        tab_widget.addTab(self.map_view, "Карта")
        
        # Вкладка спектра
        self.spectrum_view = SpectrumView()
        tab_widget.addTab(self.spectrum_view, "Спектр")
        
        # Вкладка результатов
        results_widget = self._create_results_widget()
        tab_widget.addTab(results_widget, "Результаты")
        
        layout.addWidget(tab_widget)
        
        return right_widget
    
    def _create_results_widget(self):
        """Создает виджет для отображения результатов."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Таблица целей
        targets_group = QGroupBox("Обнаруженные цели")
        targets_layout = QVBoxLayout(targets_group)
        
        self.targets_table = QTableWidget()
        self.targets_table.setColumnCount(6)
        self.targets_table.setHorizontalHeaderLabels([
            "Частота (МГц)", "X (м)", "Y (м)", "Доверие", "Возраст (с)", "Станции"
        ])
        targets_layout.addWidget(self.targets_table)
        
        layout.addWidget(targets_group)
        
        # Таблица задач
        tasks_group = QGroupBox("Задачи измерений")
        tasks_layout = QVBoxLayout(tasks_group)
        
        self.tasks_table = QTableWidget()
        self.tasks_table.setColumnCount(5)
        self.tasks_table.setHorizontalHeaderLabels([
            "ID", "Частота (МГц)", "Статус", "Создана", "Завершена"
        ])
        tasks_layout.addWidget(self.tasks_table)
        
        layout.addWidget(tasks_group)
        
        return widget
    
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
        
        # Меню Справка
        help_menu = menubar.addMenu('Справка')
        
        # Действие о программе
        about_action = help_menu.addAction('О программе...')
        about_action.triggered.connect(self._show_about)
    
    def _create_toolbar(self):
        """Создает панель инструментов."""
        toolbar = self.addToolBar('Основная панель')
        
        # Кнопка старт/стоп Master
        self.toolbar_master_action = toolbar.addAction('▶ Master')
        self.toolbar_master_action.triggered.connect(self._toggle_master)
        
        # Кнопка старт/стоп оркестратора
        self.toolbar_orch_action = toolbar.addAction('▶ Оркестратор')
        self.toolbar_orch_action.triggered.connect(self._toggle_orchestrator)
        
        toolbar.addSeparator()
        
        # Кнопка настройки устройств (Master/Slaves)
        toolbar.addAction('⚙ Устройства', self._open_settings)
        
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
            self.log.info("All signals connected successfully")
        except Exception as e:
            self.log.error(f"Error connecting signals: {e}")

    def _open_settings(self):
        """Открывает диалог настроек устройств (Master/Slaves)."""
        current = {
            'master': {
                'enabled': True,
                'nickname': self.sdr_settings.get('master', {}).get('nickname', 'Master'),
                'uri': self.sdr_settings.get('master', {}).get('serial', ''),
                'pos': self.sdr_settings.get('master', {}).get('pos', [0.0, 0.0, 0.0]),
            },
            'slaves': self.sdr_settings.get('slaves', [])
        }
        dlg = SettingsDialog(self, current)
        # Попробуем заполнить список доступных HackRF через C-библиотеку
        try:
            if self.master_controller and getattr(self.master_controller, 'sweep_source', None):
                devs = self.master_controller.sweep_source.enumerate_devices()
                dlg.master_uri.clear()
                dlg.master_uri.addItem("(нет)")
                for s in devs or []:
                    dlg.master_uri.addItem(s if s else "(по умолчанию)")
        except Exception:
            pass
        # Заполним список Slaves через SoapySDR, если доступен
        try:
            if self.slave_manager:
                devices = self.slave_manager.enumerate_soapy_devices()
                if devices and hasattr(dlg, 'populate_slave_devices'):
                    dlg.populate_slave_devices(devices)
        except Exception:
            pass
        def _save(data: dict):
            # Применяем Master (пока просто храним URI; запуск/стоп — через кнопки)
            # Применяем Slaves: пересоздаём список ТОЛЬКО из сохранённых строк, не автодобавляем найденные
            if self.slave_manager:
                # Удаляем текущие
                for sid in list(self.slave_manager.slaves.keys()):
                    self.slave_manager.remove_slave(sid)
                # Добавляем устройства из таблицы диалога
                for idx, entry in enumerate(data.get('slaves', []), start=1):
                    sid = entry.get('nickname') or (entry.get('label') or entry.get('serial') or f"slave{idx:02d}")
                    uri = entry.get('uri') or (f"driver={entry.get('driver')}" if entry.get('driver') else '')
                    if uri:
                        self.slave_manager.add_slave(sid, uri)
            # Обновим настройки в памяти и сохраним на диск
            self.sdr_settings = {
                'master': {
                    'nickname': data.get('master', {}).get('nickname') or 'Master',
                    'serial': '' if data.get('master', {}).get('uri') in ('', '(нет)') else data.get('master', {}).get('uri'),
                    'pos': [0.0, 0.0, 0.0],  # Master всегда (0,0,0)
                },
                'slaves': []
            }
            for entry in data.get('slaves', []):
                pos = entry.get('pos') or [0.0, 0.0, 0.0]
                self.sdr_settings['slaves'].append({
                    'nickname': entry.get('nickname') or '',
                    'uri': entry.get('uri') or '',
                    'driver': entry.get('driver') or '',
                    'serial': entry.get('serial') or '',
                    'label': entry.get('label') or '',
                    'pos': pos,
                })
            save_sdr_settings(self.sdr_settings)
            # Применим выбранный serial для Master
            try:
                master_serial = data.get('master', {}).get('uri')
                if master_serial in ('(по умолчанию)',):
                    master_serial = ''
                if self.master_controller and getattr(self.master_controller, 'sweep_source', None):
                    if master_serial in ('', '(нет)'):
                        # Отключаем мастер
                        self.master_controller.stop_sweep()
                        self.master_controller.sweep_source.set_serial(None)
                    else:
                        self.master_controller.sweep_source.set_serial(master_serial)
                        # Пробуем заранее проинициализировать/проверить — если не доступен, не падаем
                        self.master_controller.sweep_source.probe()
            except Exception:
                pass
            # Обновим таблицу слейвов, если она присутствует
            try:
                self._update_slave_table()
            except Exception:
                pass
        dlg.saved.connect(_save)
        def _next(payload: dict):
            # Внешняя логика "Далее": валидируем мастер, блокируем вкладку Master и подгружаем Slaves
            try:
                master_serial = payload.get('master', {}).get('uri')
                if master_serial in ('(по умолчанию)',):
                    master_serial = ''
                if self.master_controller and getattr(self.master_controller, 'sweep_source', None):
                    if master_serial in ('', '(нет)'):
                        # Ручной режим без мастера допустим
                        self.master_controller.stop_sweep()
                        self.master_controller.sweep_source.set_serial(None)
                        # Обновим список Slaves
                        if self.slave_manager and hasattr(dlg, 'populate_slave_devices'):
                            devices = self.slave_manager.enumerate_soapy_devices()
                            # Дедуп
                            uniq, seen = [], set()
                            for d in devices or []:
                                key = (d.get('serial') or '').strip() or (d.get('uri') or '').strip() or d.get('label','')
                                if key in seen:
                                    continue
                                seen.add(key)
                                uniq.append(d)
                            dlg.populate_slave_devices(uniq)
                            if hasattr(dlg, 'apply_saved_overrides'):
                                dlg.apply_saved_overrides(self.sdr_settings.get('slaves', []))
                        if hasattr(dlg, 'proceed_to_slaves'):
                            dlg.proceed_to_slaves()
                    else:
                        self.master_controller.sweep_source.set_serial(master_serial)
                        if self.master_controller.sweep_source.probe():
                            # Исключаем мастер HackRF из списка Slaves (если Soapy его показывает)
                            if self.slave_manager and hasattr(dlg, 'populate_slave_devices'):
                                devices = self.slave_manager.enumerate_soapy_devices()
                                filtered = []
                                for d in devices or []:
                                    if (d.get('driver') == 'hackrf') and d.get('serial') == master_serial:
                                        continue
                                    filtered.append(d)
                                # Дедуп
                                uniq, seen = [], set()
                                for d in filtered:
                                    key = (d.get('serial') or '').strip() or (d.get('uri') or '').strip() or d.get('label','')
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    uniq.append(d)
                                dlg.populate_slave_devices(uniq)
                                if hasattr(dlg, 'apply_saved_overrides'):
                                    dlg.apply_saved_overrides(self.sdr_settings.get('slaves', []))
                            if hasattr(dlg, 'proceed_to_slaves'):
                                dlg.proceed_to_slaves()
                        else:
                            QMessageBox.critical(self, "Master", "Выбранный HackRF недоступен")
                else:
                    # Нет контроллера — просто показываем Slaves
                    if self.slave_manager and hasattr(dlg, 'populate_slave_devices'):
                        devices = self.slave_manager.enumerate_soapy_devices()
                        dlg.populate_slave_devices(devices)
                        if hasattr(dlg, 'apply_saved_overrides'):
                            dlg.apply_saved_overrides(self.sdr_settings.get('slaves', []))
                    if hasattr(dlg, 'proceed_to_slaves'):
                        dlg.proceed_to_slaves()
            except Exception as e:
                QMessageBox.critical(self, "Master", f"Ошибка при проверке Master: {e}")
        dlg.next_step.connect(_next)

        # Пополняем Slaves обнаруженными устройствами и применяем сохранённые правки (с дедупликацией)
        try:
            if self.slave_manager:
                devices = self.slave_manager.enumerate_soapy_devices()
                uniq = []
                seen = set()
                for d in devices or []:
                    key = (d.get('serial') or '').strip() or (d.get('uri') or '').strip() or d.get('label','')
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(d)
                if uniq and hasattr(dlg, 'populate_slave_devices'):
                    dlg.populate_slave_devices(uniq)
                    if hasattr(dlg, 'apply_saved_overrides'):
                        dlg.apply_saved_overrides(self.sdr_settings.get('slaves', []))
        except Exception:
            pass
        dlg.exec_()
    
    def _load_calibration(self):
        """Загружает калибровку по умолчанию."""
        try:
            # Калибровка уже загружена в _init_components
            self.log.info("Calibration loaded successfully")
        except Exception as e:
            self.log.error(f"Error loading calibration: {e}")
    
    def _start_master(self):
        """Запускает Master sweep."""
        try:
            # Используем дефолтные параметры, т.к. элементы управления убраны в упрощённом UI
            start_hz = 24e6
            stop_hz = 6e9
            bin_hz = 200e3
            dwell_ms = self.global_dwell_spin.value()
            
            self.master_controller.start_sweep(
                start_hz=start_hz,
                stop_hz=stop_hz,
                bin_hz=bin_hz,
                dwell_ms=dwell_ms
            )
            
            self.toolbar_master_action.setText('⏹ Master')
            
            self.system_status['master_running'] = True
            self.log.info("Master sweep started")
            
        except Exception as e:
            self.log.error(f"Error starting master: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить Master: {e}")
    
    def _stop_master(self):
        """Останавливает Master sweep."""
        try:
            self.master_controller.stop_sweep()
            
            self.start_master_btn.setEnabled(True)
            self.stop_master_btn.setEnabled(False)
            self.toolbar_master_action.setText('▶ Master')
            
            self.system_status['master_running'] = False
            self.log.info("Master sweep stopped")
            
        except Exception as e:
            self.log.error(f"Error stopping master: {e}")
    
    def _toggle_master(self):
        """Переключает состояние Master."""
        if self.system_status['master_running']:
            self._stop_master()
        else:
            self._start_master()
    
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
    
    def _toggle_mode(self):
        """Переключает режим работы."""
        try:
            auto_mode = self.auto_mode_check.isChecked()
            self.orchestrator.set_auto_mode(auto_mode)
            
            if auto_mode:
                self.log.info("Auto mode enabled")
            else:
                self.log.info("Manual mode enabled")
                
        except Exception as e:
            self.log.error(f"Error toggling mode: {e}")
    
    def _manual_measure(self):
        """Выполняет ручное измерение."""
        try:
            freq_hz = self.manual_freq_spin.value() * 1e6
            span_hz = self.span_spin.value() * 1e6
            dwell_ms = self.global_dwell_spin.value()
            
            self.orchestrator.create_manual_measurement(freq_hz, span_hz, dwell_ms)
            
            self.log.info(f"Manual measurement started: {freq_hz/1e6:.1f} MHz")
            
        except Exception as e:
            self.log.error(f"Error starting manual measurement: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить измерение: {e}")
    
    def _start_orchestrator(self):
        """Запускает оркестратор."""
        try:
            # Устанавливаем глобальные параметры
            span_hz = self.span_spin.value() * 1e6
            dwell_ms = self.global_dwell_spin.value()
            
            self.orchestrator.set_global_parameters(span_hz, dwell_ms)
            
            # Запускаем оркестратор
            self.orchestrator.start()
            
            self.start_orch_btn.setEnabled(False)
            self.stop_orch_btn.setEnabled(True)
            self.toolbar_orch_action.setText('⏹ Оркестратор')
            
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
            
            self.start_orch_btn.setEnabled(True)
            self.stop_orch_btn.setEnabled(False)
            self.toolbar_orch_action.setText('▶ Оркестратор')
            
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
            self.log_text.append(f"Пик: {peak.f_peak/1e6:.1f} MHz, SNR: {peak.snr_db:.1f} dB")
            
            # Обновляем спектр
            if hasattr(self.spectrum_view, 'add_peak'):
                self.spectrum_view.add_peak(peak.f_peak, peak.snr_db)
                
        except Exception as e:
            self.log.error(f"Error handling peak: {e}")
    
    def _on_sweep_error(self, error_msg):
        """Обрабатывает ошибку sweep."""
        self.log_text.append(f"Ошибка sweep: {error_msg}")
    
    def _on_task_created(self, task):
        """Обрабатывает создание задачи."""
        try:
            self.log_text.append(f"Задача создана: {task.id}")
            self._update_tasks_table()
            
        except Exception as e:
            self.log.error(f"Error handling task created: {e}")
    
    def _on_task_completed(self, task):
        """Обрабатывает завершение задачи."""
        try:
            self.log_text.append(f"Задача завершена: {task.id}")
            self._update_tasks_table()
            
        except Exception as e:
            self.log.error(f"Error handling task completed: {e}")
    
    def _on_task_failed(self, task):
        """Обрабатывает ошибку задачи."""
        try:
            self.log_text.append(f"Задача провалена: {task.id}")
            self._update_tasks_table()
            
        except Exception as e:
            self.log.error(f"Error handling task failed: {e}")
    
    def _on_target_detected(self, target):
        """Обрабатывает обнаружение цели."""
        try:
            self.log_text.append(f"Цель: {target.center_hz/1e6:.1f} MHz, "
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
        self.log_text.append(f"Ошибка трилатерации: {error_msg}")
    
    def _on_measurement_error(self, error_msg):
        """Обрабатывает ошибку измерения."""
        self.log_text.append(f"Ошибка измерения: {error_msg}")
    
    def _update_tasks_table(self):
        """Обновляет таблицу задач."""
        try:
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
            # Обновляем статус оркестратора
            if self.orchestrator:
                orch_status = self.orchestrator.get_system_status()
                if orch_status['is_running']:
                    self.orchestrator_status_label.setText("Статус: Работает")
                else:
                    self.orchestrator_status_label.setText("Статус: Остановлен")
            
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
    
    def _show_about(self):
        """Показывает информацию о программе."""
        QMessageBox.about(self, "О программе", 
                         "ПАНОРАМА RSSI v1.0\n\n"
                         "Система трилатерации по RSSI в реальном времени\n"
                         "Поддерживает множественные SDR станции\n\n"
                         "© 2024 ПАНОРАМА Team")
    
    def closeEvent(self, event):
        """Обрабатывает закрытие приложения."""
        try:
            # Останавливаем все компоненты
            if self.master_controller:
                self.master_controller.stop_sweep()
            
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
