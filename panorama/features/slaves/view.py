# -*- coding: utf-8 -*-
"""
Современный UI для управления слейвами - объединяет watchlist, результаты и контроль оркестратора.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget, 
    QTableWidgetItem, QGroupBox, QLabel, QPushButton, QProgressBar,
    QHeaderView, QFrame, QSplitter, QTextEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QFormLayout, QGridLayout, QScrollArea
)
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap

import numpy as np
import pyqtgraph as pg

from panorama.features.orchestrator.core import Orchestrator, MeasurementTask
from panorama.features.trilateration import TrilaterationResult
from panorama.features.slave_sdr.slave import RSSIMeasurement


@dataclass
class SlaveStatus:
    """Статус слейва."""
    id: str
    name: str
    connected: bool
    last_seen: float
    rssi_rms: Dict[str, float]  # диапазон -> RSSI RMS
    active_ranges: List[str]
    error_count: int
    status: str  # ONLINE, OFFLINE, ERROR


class SlavesView(QWidget):
    """Современный виджет для управления слейвами."""
    
    # Сигналы
    slave_selected = pyqtSignal(str)  # ID слейва
    range_selected = pyqtSignal(str, str)  # slave_id, range_id
    task_action = pyqtSignal(str, str)  # task_id, action
    
    def __init__(self, orchestrator: Optional[Orchestrator] = None, parent=None):
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.slave_statuses: Dict[str, SlaveStatus] = {}
        
        # Настройка стилей
        self._setup_styles()
        
        # Создание UI
        self._create_ui()
        self._setup_connections()
        
        # Таймер обновления
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_data)
        self.update_timer.start(1000)  # Обновление каждую секунду
        
        # Инициализация данных
        self._update_data()
    
    def _setup_styles(self):
        """Настройка современных стилей с использованием qdarkstyle."""
        try:
            import qdarkstyle
            # Применяем qdarkstyle как основную тему
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
            
            # Дополнительные кастомные стили для специфичных элементов
            custom_styles = """
                /* Кастомные стили для заголовков */
                QLabel[class="header"] {
                    color: #0078d4;
                    font-size: 16px;
                    font-weight: bold;
                    margin: 10px;
                }
                
                /* Кастомные стили для статусных индикаторов */
                QLabel[class="status-success"] {
                    color: #00ff00;
                    font-weight: bold;
                    padding: 5px;
                }
                
                QLabel[class="status-error"] {
                    color: #ff0000;
                    font-weight: bold;
                    padding: 5px;
                }
                
                QLabel[class="status-warning"] {
                    color: #ffaa00;
                    font-weight: bold;
                    padding: 5px;
                }
                
                /* Кастомные стили для кнопок действий */
                QPushButton[class="action-view"] {
                    background-color: #0078d4;
                    border-radius: 4px;
                    padding: 4px;
                    min-width: 30px;
                    max-width: 30px;
                }
                
                QPushButton[class="action-activate"] {
                    background-color: #00aa00;
                    border-radius: 4px;
                    padding: 4px;
                    min-width: 30px;
                    max-width: 30px;
                }
                
                QPushButton[class="action-deactivate"] {
                    background-color: #ffaa00;
                    border-radius: 4px;
                    padding: 4px;
                    min-width: 30px;
                    max-width: 30px;
                }
                
                /* Кастомные стили для кнопок управления */
                QPushButton[class="control-start"] {
                    background-color: #00aa00;
                }
                
                QPushButton[class="control-cancel"] {
                    background-color: #aa0000;
                }
                
                /* Кастомные стили для прогресс-баров */
                QProgressBar[class="progress-success"]::chunk {
                    background-color: #00aa00;
                    border-radius: 3px;
                }
                
                QProgressBar[class="progress-running"]::chunk {
                    background-color: #0078d4;
                    border-radius: 3px;
                }
                
                QProgressBar[class="progress-pending"]::chunk {
                    background-color: #555555;
                    border-radius: 3px;
                }
            """
            
            # Применяем кастомные стили поверх qdarkstyle
            self.setStyleSheet(self.styleSheet() + custom_styles)
            
        except ImportError:
            # Fallback на встроенные стили, если qdarkstyle недоступен
            self.setStyleSheet("""
                QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 9pt;
                }
                
                QTabWidget::pane {
                    border: 1px solid #555555;
                    background-color: #2b2b2b;
                }
                
                QTabBar::tab {
                    background-color: #404040;
                    color: #ffffff;
                    padding: 8px 16px;
                    margin-right: 2px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                
                QTabBar::tab:selected {
                    background-color: #0078d4;
                    color: #ffffff;
                }
                
                QTabBar::tab:hover {
                    background-color: #505050;
                }
                
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #555555;
                    border-radius: 6px;
                    margin-top: 12px;
                    padding-top: 8px;
                }
                
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 8px 0 8px;
                    color: #ffffff;
                }
                
                QTableWidget {
                    background-color: #1e1e1e;
                    alternate-background-color: #2d2d2d;
                    gridline-color: #555555;
                    border: 1px solid #555555;
                }
                
                QTableWidget::item {
                    padding: 4px;
                    border: none;
                }
                
                QTableWidget::item:selected {
                    background-color: #0078d4;
                    color: #ffffff;
                }
                
                QHeaderView::section {
                    background-color: #404040;
                    color: #ffffff;
                    padding: 8px;
                    border: 1px solid #555555;
                    font-weight: bold;
                }
                
                QPushButton {
                    background-color: #0078d4;
                    color: #ffffff;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                
                QPushButton:hover {
                    background-color: #106ebe;
                }
                
                QPushButton:pressed {
                    background-color: #005a9e;
                }
                
                QPushButton:disabled {
                    background-color: #555555;
                    color: #888888;
                }
                
                QProgressBar {
                    border: 1px solid #555555;
                    border-radius: 4px;
                    text-align: center;
                    background-color: #1e1e1e;
                }
                
                QProgressBar::chunk {
                    background-color: #0078d4;
                    border-radius: 3px;
                }
                
                QLabel {
                    color: #ffffff;
                }
                
                QComboBox {
                    background-color: #404040;
                    border: 1px solid #555555;
                    border-radius: 4px;
                    padding: 4px;
                    color: #ffffff;
                }
                
                QComboBox::drop-down {
                    border: none;
                    width: 20px;
                }
                
                QComboBox::down-arrow {
                    image: none;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 5px solid #ffffff;
                }
            """)
    
    def _create_ui(self):
        """Создание пользовательского интерфейса."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Заголовок
        header = QLabel("🎯 Управление слейвами")
        header.setProperty("class", "header")
        header.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Вкладки
        self.tab_widget = QTabWidget()
        
        # Вкладка Watchlist
        self.watchlist_tab = self._create_watchlist_tab()
        self.tab_widget.addTab(self.watchlist_tab, "📊 Активные диапазоны")
        
        # Вкладка Результаты
        self.results_tab = self._create_results_tab()
        self.tab_widget.addTab(self.results_tab, "📈 Результаты измерений")
        
        # Вкладка Контроль
        self.control_tab = self._create_control_tab()
        self.tab_widget.addTab(self.control_tab, "🎮 Контроль оркестратора")
        
        layout.addWidget(self.tab_widget)
        
        # Статусная строка
        self.status_bar = QLabel("Готов к работе")
        self.status_bar.setProperty("class", "status-success")
        layout.addWidget(self.status_bar)
    
    def _create_watchlist_tab(self) -> QWidget:
        """Создание вкладки Watchlist."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Панель управления
        control_panel = QGroupBox("Управление диапазонами")
        control_layout = QHBoxLayout(control_panel)
        
        self.add_range_btn = QPushButton("➕ Добавить диапазон")
        self.add_range_btn.clicked.connect(self._add_range)
        control_layout.addWidget(self.add_range_btn)
        
        self.remove_range_btn = QPushButton("➖ Удалить диапазон")
        self.remove_range_btn.clicked.connect(self._remove_range)
        control_layout.addWidget(self.remove_range_btn)
        
        self.refresh_btn = QPushButton("🔄 Обновить")
        self.refresh_btn.clicked.connect(self._refresh_watchlist)
        control_layout.addWidget(self.refresh_btn)
        
        control_layout.addStretch()
        
        # Статистика
        self.ranges_count_label = QLabel("Диапазонов: 0")
        self.ranges_count_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        control_layout.addWidget(self.ranges_count_label)
        
        layout.addWidget(control_panel)
        
        # Таблица диапазонов
        self.ranges_table = QTableWidget()
        self.ranges_table.setColumnCount(6)
        self.ranges_table.setHorizontalHeaderLabels([
            "Диапазон (МГц)", "Статус", "Активные слейвы", "RSSI RMS (дБм)", "Последнее обновление", "Действия"
        ])
        
        # Настройка таблицы
        header = self.ranges_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Диапазон
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Статус
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Слейвы
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # RSSI
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Время
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Действия
        
        self.ranges_table.setAlternatingRowColors(True)
        self.ranges_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.ranges_table.itemSelectionChanged.connect(self._on_range_selected)
        
        layout.addWidget(self.ranges_table)
        
        return widget
    
    def _create_results_tab(self) -> QWidget:
        """Создание вкладки результатов."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Панель фильтров
        filter_panel = QGroupBox("Фильтры")
        filter_layout = QFormLayout(filter_panel)
        
        self.slave_filter = QComboBox()
        self.slave_filter.addItem("Все слейвы")
        self.slave_filter.currentTextChanged.connect(self._filter_results)
        filter_layout.addRow("Слейв:", self.slave_filter)
        
        self.range_filter = QComboBox()
        self.range_filter.addItem("Все диапазоны")
        self.range_filter.currentTextChanged.connect(self._filter_results)
        filter_layout.addRow("Диапазон:", self.range_filter)
        
        self.time_filter = QComboBox()
        self.time_filter.addItems(["Последний час", "Последние 24 часа", "Последняя неделя"])
        self.time_filter.currentTextChanged.connect(self._filter_results)
        filter_layout.addRow("Временной период:", self.time_filter)
        
        layout.addWidget(filter_panel)
        
        # Таблица результатов
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "Время", "Слейв", "Диапазон (МГц)", "RSSI (дБм)", "SNR (дБ)", "Частота (МГц)", "Статус"
        ])
        
        # Настройка таблицы
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Время
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Слейв
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Диапазон
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # RSSI
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # SNR
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Частота
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Статус
        
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        layout.addWidget(self.results_table)
        
        # График RSSI
        graph_group = QGroupBox("График RSSI по времени")
        graph_layout = QVBoxLayout(graph_group)
        
        self.rssi_plot = pg.PlotWidget()
        self.rssi_plot.setBackground('w')
        self.rssi_plot.setLabel('left', 'RSSI (дБм)')
        self.rssi_plot.setLabel('bottom', 'Время')
        self.rssi_plot.showGrid(x=True, y=True, alpha=0.3)
        graph_layout.addWidget(self.rssi_plot)
        
        layout.addWidget(graph_group)
        
        return widget
    
    def _create_control_tab(self) -> QWidget:
        """Создание вкладки контроля оркестратора."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Панель управления
        control_panel = QGroupBox("Управление системой")
        control_layout = QGridLayout(control_panel)
        
        # Кнопки управления
        self.start_btn = QPushButton("▶ Старт")
        self.start_btn.setStyleSheet("background-color: #00aa00;")
        self.start_btn.clicked.connect(self._start_orchestrator)
        control_layout.addWidget(self.start_btn, 0, 0)
        
        self.stop_btn = QPushButton("⏹ Стоп")
        self.stop_btn.setStyleSheet("background-color: #aa0000;")
        self.stop_btn.clicked.connect(self._stop_orchestrator)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn, 0, 1)
        
        self.reset_btn = QPushButton("🔄 Сброс")
        self.reset_btn.clicked.connect(self._reset_orchestrator)
        control_layout.addWidget(self.reset_btn, 0, 2)
        
        # Режимы работы
        self.auto_mode_cb = QCheckBox("Автоматический режим")
        self.auto_mode_cb.setChecked(True)
        self.auto_mode_cb.toggled.connect(self._toggle_auto_mode)
        control_layout.addWidget(self.auto_mode_cb, 1, 0)
        
        self.manual_mode_cb = QCheckBox("Ручной режим")
        self.manual_mode_cb.toggled.connect(self._toggle_manual_mode)
        control_layout.addWidget(self.manual_mode_cb, 1, 1)
        
        layout.addWidget(control_panel)
        
        # Статус системы
        status_panel = QGroupBox("Статус системы")
        status_layout = QFormLayout(status_panel)
        
        self.system_status_label = QLabel("Остановлен")
        self.system_status_label.setStyleSheet("color: #ff0000; font-weight: bold;")
        status_layout.addRow("Статус:", self.system_status_label)
        
        self.slaves_count_label = QLabel("0")
        status_layout.addRow("Подключенных слейвов:", self.slaves_count_label)
        
        self.tasks_count_label = QLabel("0")
        status_layout.addRow("Активных задач:", self.tasks_count_label)
        
        self.targets_count_label = QLabel("0")
        status_layout.addRow("Обнаруженных целей:", self.targets_count_label)
        
        layout.addWidget(status_panel)
        
        # Таблица задач
        tasks_group = QGroupBox("Активные задачи")
        tasks_layout = QVBoxLayout(tasks_group)
        
        self.tasks_table = QTableWidget()
        self.tasks_table.setColumnCount(6)
        self.tasks_table.setHorizontalHeaderLabels([
            "ID", "Частота (МГц)", "Span (МГц)", "Статус", "Прогресс", "Действия"
        ])
        
        # Настройка таблицы
        header = self.tasks_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # ID
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Частота
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Span
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Статус
        header.setSectionResizeMode(4, QHeaderView.Stretch)  # Прогресс
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Действия
        
        self.tasks_table.setAlternatingRowColors(True)
        self.tasks_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        tasks_layout.addWidget(self.tasks_table)
        layout.addWidget(tasks_group)
        
        return widget
    
    def _setup_connections(self):
        """Настройка соединений сигналов."""
        if self.orchestrator:
            self.orchestrator.status_changed.connect(self._on_orchestrator_status)
            self.orchestrator.task_created.connect(self._on_task_created)
            self.orchestrator.task_completed.connect(self._on_task_completed)
            self.orchestrator.task_failed.connect(self._on_task_failed)
            self.orchestrator.target_update.connect(self._on_target_update)
    
    def _update_data(self):
        """Обновление данных."""
        self._update_watchlist()
        self._update_results()
        self._update_tasks()
        self._update_system_status()
    
    def _update_watchlist(self):
        """Обновление списка диапазонов."""
        # Заглушка для демонстрации
        ranges_data = [
            ("50-100", "Активен", "3", "-45.2", "2 мин назад", ""),
            ("100-200", "Активен", "2", "-52.1", "1 мин назад", ""),
            ("200-500", "Неактивен", "0", "N/A", "5 мин назад", ""),
            ("500-1000", "Активен", "4", "-38.7", "30 сек назад", ""),
            ("1000-2000", "Активен", "3", "-41.3", "1 мин назад", ""),
            ("2000-6000", "Неактивен", "0", "N/A", "10 мин назад", "")
        ]
        
        self.ranges_table.setRowCount(len(ranges_data))
        for row, (range_name, status, slaves, rssi, time, actions) in enumerate(ranges_data):
            self.ranges_table.setItem(row, 0, QTableWidgetItem(range_name))
            
            status_item = QTableWidgetItem(status)
            if status == "Активен":
                status_item.setBackground(QColor(0, 170, 0, 100))
            else:
                status_item.setBackground(QColor(170, 0, 0, 100))
            self.ranges_table.setItem(row, 1, status_item)
            
            self.ranges_table.setItem(row, 2, QTableWidgetItem(slaves))
            self.ranges_table.setItem(row, 3, QTableWidgetItem(rssi))
            self.ranges_table.setItem(row, 4, QTableWidgetItem(time))
            
            # Кнопки действий
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            
            view_btn = QPushButton("👁")
            view_btn.setProperty("class", "action-view")
            view_btn.setToolTip("Просмотр диапазона")
            view_btn.clicked.connect(lambda checked, r=range_name: self._view_range(r))
            actions_layout.addWidget(view_btn)
            
            if status == "Неактивен":
                activate_btn = QPushButton("✅")
                activate_btn.setProperty("class", "action-activate")
                activate_btn.setToolTip("Активировать диапазон")
                activate_btn.clicked.connect(lambda checked, r=range_name: self._activate_range(r))
                actions_layout.addWidget(activate_btn)
            else:
                deactivate_btn = QPushButton("❌")
                deactivate_btn.setProperty("class", "action-deactivate")
                deactivate_btn.setToolTip("Деактивировать диапазон")
                deactivate_btn.clicked.connect(lambda checked, r=range_name: self._deactivate_range(r))
                actions_layout.addWidget(deactivate_btn)
            
            actions_layout.addStretch()
            self.ranges_table.setCellWidget(row, 5, actions_widget)
        
        self.ranges_count_label.setText(f"Диапазонов: {len(ranges_data)}")
    
    def _update_results(self):
        """Обновление результатов измерений."""
        # Заглушка для демонстрации
        results_data = [
            ("14:30:15", "Slave-1", "50-100", "-45.2", "12.3", "75.5", "✅"),
            ("14:30:12", "Slave-2", "50-100", "-48.7", "8.9", "78.2", "✅"),
            ("14:30:10", "Slave-3", "50-100", "-52.1", "5.2", "76.8", "✅"),
            ("14:29:58", "Slave-1", "100-200", "-51.3", "9.8", "125.4", "✅"),
            ("14:29:55", "Slave-2", "100-200", "-54.2", "6.1", "128.7", "⚠️"),
            ("14:29:52", "Slave-4", "500-1000", "-38.7", "15.2", "750.3", "✅"),
            ("14:29:48", "Slave-3", "500-1000", "-41.3", "11.7", "745.8", "✅")
        ]
        
        self.results_table.setRowCount(len(results_data))
        for row, (time, slave, range_name, rssi, snr, freq, status) in enumerate(results_data):
            self.results_table.setItem(row, 0, QTableWidgetItem(time))
            self.results_table.setItem(row, 1, QTableWidgetItem(slave))
            self.results_table.setItem(row, 2, QTableWidgetItem(range_name))
            self.results_table.setItem(row, 3, QTableWidgetItem(rssi))
            self.results_table.setItem(row, 4, QTableWidgetItem(snr))
            self.results_table.setItem(row, 5, QTableWidgetItem(freq))
            
            status_item = QTableWidgetItem(status)
            if status == "✅":
                status_item.setBackground(QColor(0, 170, 0, 100))
            elif status == "⚠️":
                status_item.setBackground(QColor(255, 170, 0, 100))
            else:
                status_item.setBackground(QColor(170, 0, 0, 100))
            self.results_table.setItem(row, 6, status_item)
    
    def _update_tasks(self):
        """Обновление списка задач."""
        if not self.orchestrator:
            return
        
        tasks = self.orchestrator.get_active_tasks()
        self.tasks_table.setRowCount(len(tasks))
        
        for row, task in enumerate(tasks):
            self.tasks_table.setItem(row, 0, QTableWidgetItem(task.id))
            self.tasks_table.setItem(row, 1, QTableWidgetItem(f"{task.peak.f_peak/1e6:.1f}"))
            self.tasks_table.setItem(row, 2, QTableWidgetItem(f"{task.window.span/1e6:.1f}"))
            
            status_item = QTableWidgetItem(task.status)
            if task.status == "COMPLETED":
                status_item.setBackground(QColor(0, 170, 0, 100))
            elif task.status == "RUNNING":
                status_item.setBackground(QColor(0, 100, 170, 100))
            elif task.status == "FAILED":
                status_item.setBackground(QColor(170, 0, 0, 100))
            self.tasks_table.setItem(row, 3, status_item)
            
            # Прогресс-бар
            progress_widget = QWidget()
            progress_layout = QHBoxLayout(progress_widget)
            progress_layout.setContentsMargins(2, 2, 2, 2)
            
            progress_bar = QProgressBar()
            if task.status == "COMPLETED":
                progress_bar.setValue(100)
                progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #00aa00; }")
            elif task.status == "RUNNING":
                progress_bar.setValue(50)
                progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #0078d4; }")
            else:
                progress_bar.setValue(0)
                progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #555555; }")
            
            progress_layout.addWidget(progress_bar)
            self.tasks_table.setCellWidget(row, 4, progress_widget)
            
            # Кнопки действий
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            
            if task.status == "PENDING":
                start_btn = QPushButton("▶")
                start_btn.setMaximumSize(30, 25)
                start_btn.clicked.connect(lambda checked, t=task.id: self._start_task(t))
                actions_layout.addWidget(start_btn)
            
            if task.status in ["PENDING", "RUNNING"]:
                cancel_btn = QPushButton("❌")
                cancel_btn.setMaximumSize(30, 25)
                cancel_btn.setStyleSheet("background-color: #aa0000;")
                cancel_btn.clicked.connect(lambda checked, t=task.id: self._cancel_task(t))
                actions_layout.addWidget(cancel_btn)
            
            actions_layout.addStretch()
            self.tasks_table.setCellWidget(row, 5, actions_widget)
        
        self.tasks_count_label.setText(str(len(tasks)))
    
    def _update_system_status(self):
        """Обновление статуса системы."""
        if not self.orchestrator:
            return
        
        status = self.orchestrator.get_system_status()
        
        if status.get('is_running', False):
            self.system_status_label.setText("Работает")
            self.system_status_label.setStyleSheet("color: #00ff00; font-weight: bold;")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        else:
            self.system_status_label.setText("Остановлен")
            self.system_status_label.setStyleSheet("color: #ff0000; font-weight: bold;")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
        
        self.slaves_count_label.setText(str(status.get('slave_connected', 0)))
        self.targets_count_label.setText(str(status.get('n_targets', 0)))
    
    # Обработчики событий
    def _on_range_selected(self):
        """Обработка выбора диапазона."""
        current_row = self.ranges_table.currentRow()
        if current_row >= 0:
            range_name = self.ranges_table.item(current_row, 0).text()
            self.range_selected.emit("", range_name)
    
    def _add_range(self):
        """Добавление нового диапазона."""
        # TODO: Реализовать диалог добавления диапазона
        pass
    
    def _remove_range(self):
        """Удаление выбранного диапазона."""
        current_row = self.ranges_table.currentRow()
        if current_row >= 0:
            self.ranges_table.removeRow(current_row)
    
    def _refresh_watchlist(self):
        """Обновление списка диапазонов."""
        self._update_watchlist()
    
    def _filter_results(self):
        """Фильтрация результатов."""
        # TODO: Реализовать фильтрацию
        pass
    
    def _start_orchestrator(self):
        """Запуск оркестратора."""
        if self.orchestrator:
            self.orchestrator.start()
    
    def _stop_orchestrator(self):
        """Остановка оркестратора."""
        if self.orchestrator:
            self.orchestrator.stop()
    
    def _reset_orchestrator(self):
        """Сброс оркестратора."""
        if self.orchestrator:
            self.orchestrator.shutdown()
    
    def _toggle_auto_mode(self, checked: bool):
        """Переключение автоматического режима."""
        if self.orchestrator:
            self.orchestrator.set_auto_mode(checked)
        if checked:
            self.manual_mode_cb.setChecked(False)
    
    def _toggle_manual_mode(self, checked: bool):
        """Переключение ручного режима."""
        if self.orchestrator:
            self.orchestrator.set_auto_mode(not checked)
        if checked:
            self.auto_mode_cb.setChecked(False)
    
    def _view_range(self, range_name: str):
        """Просмотр диапазона."""
        self.range_selected.emit("", range_name)
    
    def _activate_range(self, range_name: str):
        """Активация диапазона."""
        # TODO: Реализовать активацию
        pass
    
    def _deactivate_range(self, range_name: str):
        """Деактивация диапазона."""
        # TODO: Реализовать деактивацию
        pass
    
    def _start_task(self, task_id: str):
        """Запуск задачи."""
        self.task_action.emit(task_id, "start")
    
    def _cancel_task(self, task_id: str):
        """Отмена задачи."""
        self.task_action.emit(task_id, "cancel")
    
    # Обработчики сигналов оркестратора
    def _on_orchestrator_status(self, status: Dict):
        """Обработка изменения статуса оркестратора."""
        self._update_system_status()
    
    def _on_task_created(self, task: MeasurementTask):
        """Обработка создания задачи."""
        self._update_tasks()
    
    def _on_task_completed(self, task: MeasurementTask):
        """Обработка завершения задачи."""
        self._update_tasks()
    
    def _on_task_failed(self, task: MeasurementTask):
        """Обработка ошибки задачи."""
        self._update_tasks()
    
    def _on_target_update(self, result: TrilaterationResult):
        """Обработка обновления цели."""
        # TODO: Обновить отображение целей
        pass
