"""
Менеджер пользовательского интерфейса для ПАНОРАМА RSSI.
"""

import logging
from typing import Optional
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QTabWidget, QGroupBox, QLabel, QPushButton, 
                            QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget, 
                            QTableWidgetItem, QComboBox, QCheckBox, QSplitter, 
                            QFrame, QMessageBox, QFileDialog, QFormLayout)

from panorama.ui import SpectrumView, ImprovedSlavesView, OpenLayersMapWidget
from panorama.features.spectrum.master_adapter import MasterSourceAdapter


class PanoramaUI(QObject):
    """Менеджер для создания и управления главным UI."""
    
    # Сигналы для подключения к главному окну
    load_calibration_requested = pyqtSignal()
    save_calibration_requested = pyqtSignal()
    calibration_settings_requested = pyqtSignal()
    detector_settings_requested = pyqtSignal()
    refresh_slaves_requested = pyqtSignal()
    export_slaves_requested = pyqtSignal()
    clear_slaves_requested = pyqtSignal()
    about_requested = pyqtSignal()
    device_manager_requested = pyqtSignal()
    theme_toggle_requested = pyqtSignal()
    
    def __init__(self, main_window: QMainWindow, orchestrator=None, logger: Optional[logging.Logger] = None):
        super().__init__()
        
        self.main_window = main_window
        self.orchestrator = orchestrator
        self.log = logger or logging.getLogger(__name__)
        
        # Ссылки на основные виджеты
        self.map_view: Optional[OpenLayersMapWidget] = None
        self.spectrum_view: Optional[SpectrumView] = None
        self.slaves_view: Optional[ImprovedSlavesView] = None
    
    def setup_main_ui(self):
        """Настраивает главный пользовательский интерфейс."""
        self._setup_window_properties()
        self._setup_theme()
        
        # Центральный виджет
        central_widget = QWidget()
        self.main_window.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QVBoxLayout(central_widget)
        
        # Создаем основную панель
        main_panel = self._create_main_panel()
        main_layout.addWidget(main_panel)
        
        # Создаем меню и панель инструментов
        self._create_menu()
        self._create_toolbar()
        
        self.log.info("Panorama UI setup completed")
    
    def _setup_window_properties(self):
        """Настраивает свойства окна."""
        self.main_window.setWindowTitle("ПАНОРАМА RSSI - Система трилатерации по RSSI")
        self.main_window.setGeometry(100, 100, 1400, 900)
    
    def _setup_theme(self):
        """Инициализация темы через PyQtDarkTheme (qdarktheme), с поддержкой старых версий (<1.0)."""
        try:
            import qdarktheme as _qdt
            app = QtWidgets.QApplication.instance()
            if hasattr(_qdt, 'setup_theme'):
                _qdt.setup_theme('light')
            else:
                # Старые версии: применяем палитру и стиль вручную
                if app is not None:
                    pal = _qdt.load_palette(theme='light') if hasattr(_qdt, 'load_palette') else None
                    ss = _qdt.load_stylesheet(theme='light') if hasattr(_qdt, 'load_stylesheet') else ''
                    if pal is not None:
                        app.setPalette(pal)
                    if ss:
                        app.setStyleSheet(ss)
        except Exception:
            # Если библиотеки нет — оставляем стандартную тему
            pass
    
    def _setup_dark_palette(self):
        """Настраивает тёмную палитру как fallback."""
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

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setPalette(dark)
            app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")
    
    def _create_main_panel(self) -> QWidget:
        """Создает основную панель с вкладками."""
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        
        # Создаем вкладки
        tab_widget = QTabWidget()
        
        # Вкладка карты
        self.map_view = OpenLayersMapWidget()
        tab_widget.addTab(self.map_view, "🗺️ Карта")
        
        # Вкладка спектра
        self.spectrum_view = SpectrumView(orchestrator=self.orchestrator)
        # Привязываем источник через адаптер, чтобы старт сразу запускал C-свип
        try:
            self.spectrum_view.set_source(MasterSourceAdapter(self.log))
        except Exception as e:
            self.log.error(f"Failed to set spectrum source: {e}")
        tab_widget.addTab(self.spectrum_view, "📊 Спектр")
        
        # Вкладка управления слейвами (объединяет watchlist, результаты и контроль)
        self.slaves_view = ImprovedSlavesView(orchestrator=self.orchestrator)
        tab_widget.addTab(self.slaves_view, "🎯 Слейвы")
        
        layout.addWidget(tab_widget)
        
        return main_widget
    
    def _create_menu(self):
        """Создает главное меню."""
        menubar = self.main_window.menuBar()
        
        # Меню Файл
        file_menu = menubar.addMenu('Файл')
        file_menu.addAction('Загрузить калибровку...').triggered.connect(self.load_calibration_requested.emit)
        file_menu.addAction('Сохранить калибровку...').triggered.connect(self.save_calibration_requested.emit)
        file_menu.addSeparator()
        file_menu.addAction('Выход').triggered.connect(self.main_window.close)
        
        # Меню Настройки
        settings_menu = menubar.addMenu('Настройки')
        settings_menu.addAction('Настройки калибровки...').triggered.connect(self.calibration_settings_requested.emit)
        settings_menu.addAction('Настройки детектора...').triggered.connect(self.detector_settings_requested.emit)
        
        # Меню Слейвы
        slaves_menu = menubar.addMenu('🎯 Слейвы')
        slaves_menu.addAction('🔄 Обновить данные').triggered.connect(self.refresh_slaves_requested.emit)
        slaves_menu.addAction('💾 Экспорт состояния...').triggered.connect(self.export_slaves_requested.emit)
        slaves_menu.addSeparator()
        slaves_menu.addAction('🗑️ Очистить данные').triggered.connect(self.clear_slaves_requested.emit)
        
        # Меню Справка
        help_menu = menubar.addMenu('Справка')
        help_menu.addAction('О программе...').triggered.connect(self.about_requested.emit)
    
    def _create_toolbar(self):
        """Создает панель инструментов."""
        toolbar = self.main_window.addToolBar('Основная панель')
        toolbar.addAction('🧭 Диспетчер устройств').triggered.connect(self.device_manager_requested.emit)
        # Выпадающий список темы (Light/Dark)
        try:
            theme_label = QLabel('Тема: ')
            toolbar.addWidget(theme_label)
            self.theme_combo = QComboBox()
            self.theme_combo.addItems(["Light", "Dark"]) 
            self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
            toolbar.addWidget(self.theme_combo)
        except Exception:
            pass

    def _on_theme_changed(self, name: str):
        target = 'dark' if name.lower() == 'dark' else 'light'
        try:
            import qdarktheme as _qdt
            app = QtWidgets.QApplication.instance()
            if hasattr(_qdt, 'setup_theme'):
                _qdt.setup_theme(target)
            else:
                # Старые версии: вручную
                if app is not None:
                    pal = _qdt.load_palette(theme=target) if hasattr(_qdt, 'load_palette') else None
                    ss = _qdt.load_stylesheet(theme=target) if hasattr(_qdt, 'load_stylesheet') else ''
                    if pal is not None:
                        app.setPalette(pal)
                    if ss:
                        app.setStyleSheet(ss)
        except Exception:
            pass
        # done
    
    
    def update_stations_from_config(self, config: dict):
        """Обновляет станции на карте из конфигурации."""
        try:
            if self.map_view:
                self.map_view.update_stations_from_config(config)
        except Exception as e:
            self.log.error(f"Error updating stations from config: {e}")
    
    def get_map_view(self) -> Optional[OpenLayersMapWidget]:
        """Возвращает виджет карты."""
        return self.map_view
    
    def get_spectrum_view(self) -> Optional[SpectrumView]:
        """Возвращает виджет спектра."""
        return self.spectrum_view
    
    def get_slaves_view(self) -> Optional[ImprovedSlavesView]:
        """Возвращает виджет слейвов."""
        return self.slaves_view

# Backward compatibility alias (after full class definition)
MainUIManager = PanoramaUI