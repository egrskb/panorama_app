# panorama/features/slaves/view.py
"""
Улучшенный UI для управления слейвами с таблицей RSSI_rms.
- Убрана кнопка "Добавить диапазон" (теперь через детектор автоматически)
- Исправлена матрица RSSI RMS с правильными заголовками
- Полное использование qdarkstyle
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
import json
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QGroupBox, QLabel, QPushButton, QHeaderView,
    QSplitter, QTextEdit, QComboBox, QSpinBox, QCheckBox,
    QGridLayout, QProgressBar, QFrame, QMessageBox, QFileDialog
)
from PyQt5.QtGui import QFont, QColor, QBrush, QPalette

import numpy as np

# Импорт qdarkstyle
try:
    import qdarkstyle
    QDARKSTYLE_AVAILABLE = True
except ImportError:
    QDARKSTYLE_AVAILABLE = False
    print("[SlavesView] qdarkstyle not installed. Install it with: pip install qdarkstyle")


class ImprovedSlavesView(QWidget):
    """
    Современный виджет управления слейвами с RSSI матрицей.
    - Автоматическое добавление диапазонов через детектор
    - Правильная матрица RSSI RMS
    - Полная поддержка qdarkstyle
    """

    # Сигналы
    send_to_map = pyqtSignal(dict)       # Отправка цели на карту
    task_selected = pyqtSignal(str)      # Выбрана задача
    watchlist_updated = pyqtSignal(list) # Обновлен watchlist

    def __init__(self, orchestrator: Any = None, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.orchestrator = orchestrator

        # Данные
        self.rssi_matrix: Dict[str, Dict[str, float]] = {}  # {range_id: {slave_id: rssi_rms}}
        self.watchlist: List[Dict[str, Any]] = []
        self.tasks_data: List[Dict[str, Any]] = []
        self.slave_statuses: Dict[str, Any] = {}
        
        # Создаем UI
        self._create_ui()

        # Применяем темный стиль qdarkstyle
        self._apply_dark_style()

        # Таймер обновления
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_data)
        self.update_timer.start(2000)  # Обновление каждые 2 секунды

    def _apply_dark_style(self):
        """Применяет темный стиль qdarkstyle к виджету."""
        if QDARKSTYLE_AVAILABLE:
            # Применяем полный стиль qdarkstyle
            self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Заголовок
        header = self._create_header()
        layout.addWidget(header)

        # Основной сплиттер
        splitter = QSplitter(Qt.Horizontal)

        # Левая панель - RSSI матрица
        left_panel = self._create_rssi_panel()
        splitter.addWidget(left_panel)

        # Правая панель - Watchlist и задачи
        right_panel = self._create_watchlist_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([700, 500])
        layout.addWidget(splitter)

        # Статус бар
        self.status_bar = self._create_status_bar()
        layout.addWidget(self.status_bar)

    def _create_header(self) -> QWidget:
        """Создает заголовок с кнопками управления."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Заголовок
        title = QLabel("🎯 Система управления Slave SDR")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        layout.addStretch()

        # Кнопки управления
        self.btn_refresh = QPushButton("🔄 Обновить")
        self.btn_refresh.clicked.connect(self._refresh_data)
        layout.addWidget(self.btn_refresh)

        self.btn_clear = QPushButton("🗑️ Очистить")
        self.btn_clear.clicked.connect(self._clear_data)
        layout.addWidget(self.btn_clear)

        self.btn_export = QPushButton("💾 Экспорт")
        self.btn_export.clicked.connect(self._export_data)
        layout.addWidget(self.btn_export)

        return widget

    def _create_rssi_panel(self) -> QWidget:
        """Создает панель с матрицей RSSI RMS."""
        group = QGroupBox("📊 Матрица RSSI RMS (дБм)")
        layout = QVBoxLayout(group)

        # Контролы фильтрации
        filter_layout = QHBoxLayout()

        self.range_filter = QComboBox()
        self.range_filter.addItem("Все диапазоны")
        self.range_filter.currentTextChanged.connect(self._filter_rssi_table)
        filter_layout.addWidget(QLabel("Диапазон:"))
        filter_layout.addWidget(self.range_filter)

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(-120, 0)
        self.threshold_spin.setValue(-70)
        self.threshold_spin.setSuffix(" дБм")
        self.threshold_spin.setToolTip("Порог подсветки RSSI")
        self.threshold_spin.valueChanged.connect(self._update_rssi_colors)
        filter_layout.addWidget(QLabel("Порог:"))
        filter_layout.addWidget(self.threshold_spin)

        filter_layout.addStretch()

        self.auto_scroll = QCheckBox("Автопрокрутка")
        self.auto_scroll.setChecked(True)
        filter_layout.addWidget(self.auto_scroll)

        layout.addLayout(filter_layout)

        # Таблица RSSI с правильными заголовками
        self.rssi_table = QTableWidget()
        self.rssi_table.setAlternatingRowColors(True)

        # Инициализация таблицы с правильной структурой
        self._setup_rssi_table()

        layout.addWidget(self.rssi_table)

        # Статистика RSSI
        stats_layout = QGridLayout()

        self.lbl_min_rssi = QLabel("Мин: — дБм")
        self.lbl_max_rssi = QLabel("Макс: — дБм")
        self.lbl_avg_rssi = QLabel("Сред: — дБм")
        self.lbl_active_slaves = QLabel("Активных Slave: 0")

        stats_layout.addWidget(self.lbl_min_rssi, 0, 0)
        stats_layout.addWidget(self.lbl_max_rssi, 0, 1)
        stats_layout.addWidget(self.lbl_avg_rssi, 0, 2)
        stats_layout.addWidget(self.lbl_active_slaves, 0, 3)

        layout.addLayout(stats_layout)

        return group

    def _create_watchlist_panel(self) -> QWidget:
        """Создает панель watchlist и задач."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Вкладки
        tabs = QTabWidget()

        # Вкладка Watchlist (БЕЗ кнопки добавления диапазона)
        watchlist_tab = self._create_watchlist_tab()
        tabs.addTab(watchlist_tab, "📡 Watchlist")

        # Вкладка задач
        tasks_tab = self._create_tasks_tab()
        tabs.addTab(tasks_tab, "📋 Задачи")

        # Вкладка передатчиков
        transmitters_tab = self._create_transmitters_tab()
        tabs.addTab(transmitters_tab, "📻 Передатчики")

        layout.addWidget(tabs)
        return widget

    def _create_watchlist_tab(self) -> QWidget:
        """Создает вкладку watchlist (без ручного добавления)."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Информационная панель
        info_panel = QLabel(
            "📍 Диапазоны добавляются автоматически через детектор при обнаружении сигналов"
        )
        info_panel.setWordWrap(True)
        info_panel.setStyleSheet("""
            QLabel {
                background-color: rgba(100, 100, 255, 30);
                padding: 10px;
                border-radius: 5px;
                border: 1px solid rgba(100, 100, 255, 100);
            }
        """)
        layout.addWidget(info_panel)

        # Таблица watchlist
        self.watchlist_table = QTableWidget()
        self.watchlist_table.setColumnCount(8)
        self.watchlist_table.setHorizontalHeaderLabels([
            "ID", "Частота (МГц)", "Ширина (МГц)", "RSSI_1", "RSSI_2", "RSSI_3",
            "Обновлено", "Действия"
        ])

        header = self.watchlist_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.Fixed)
        header.resizeSection(7, 150)

        self.watchlist_table.setAlternatingRowColors(True)

        layout.addWidget(self.watchlist_table)

        # Панель управления (только очистка)
        control_panel = QHBoxLayout()

        self.btn_clear_watchlist = QPushButton("🗑️ Очистить watchlist")
        self.btn_clear_watchlist.clicked.connect(self._clear_watchlist)
        control_panel.addWidget(self.btn_clear_watchlist)

        control_panel.addStretch()

        self.lbl_watchlist_count = QLabel("Записей: 0")
        control_panel.addWidget(self.lbl_watchlist_count)

        layout.addLayout(control_panel)

        return widget

    def _create_tasks_tab(self) -> QWidget:
        """Создает вкладку задач."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Лог задач
        self.tasks_log_view = QTextEdit()
        self.tasks_log_view.setReadOnly(True)
        self.tasks_log_view.setMaximumHeight(150)
        layout.addWidget(QLabel("Лог задач:"))
        layout.addWidget(self.tasks_log_view)

        # Таблица активных задач
        self.tasks_table = QTableWidget()
        self.tasks_table.setColumnCount(6)
        self.tasks_table.setHorizontalHeaderLabels([
            "ID задачи", "Диапазон", "Статус", "Прогресс", "Время", "Приоритет"
        ])

        header = self.tasks_table.horizontalHeader()
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.resizeSection(3, 150)

        self.tasks_table.setAlternatingRowColors(True)
        layout.addWidget(QLabel("Активные задачи:"))
        layout.addWidget(self.tasks_table)

        # Статистика задач
        stats_layout = QHBoxLayout()

        self.lbl_total_tasks = QLabel("Всего: 0")
        self.lbl_pending_tasks = QLabel("Ожидает: 0")
        self.lbl_running_tasks = QLabel("Выполняется: 0")
        self.lbl_completed_tasks = QLabel("Завершено: 0")

        for lbl in [self.lbl_total_tasks, self.lbl_pending_tasks,
                    self.lbl_running_tasks, self.lbl_completed_tasks]:
            stats_layout.addWidget(lbl)

        layout.addLayout(stats_layout)

        return widget

    def _create_transmitters_tab(self) -> QWidget:
        """Создает вкладку передатчиков."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Таблица передатчиков
        self.transmitters_table = QTableWidget()
        self.transmitters_table.setColumnCount(9)
        self.transmitters_table.setHorizontalHeaderLabels([
            "ID", "Частота (МГц)", "Мощность (дБм)", "Тип", "X", "Y",
            "Уверенность", "Время", "На карту"
        ])

        header = self.transmitters_table.horizontalHeader()
        for i in range(8):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.Fixed)
        header.resizeSection(8, 100)

        self.transmitters_table.setAlternatingRowColors(True)

        layout.addWidget(self.transmitters_table)

        # Кнопки управления
        filter_layout = QHBoxLayout()

        self.btn_send_all_to_map = QPushButton("📍 Все на карту")
        self.btn_send_all_to_map.clicked.connect(self._send_all_to_map)
        filter_layout.addWidget(self.btn_send_all_to_map)

        filter_layout.addStretch()

        layout.addLayout(filter_layout)

        return widget

    def _create_status_bar(self) -> QWidget:
        """Создает статус бар."""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)

        layout = QHBoxLayout(widget)

        # Статусы
        self.status_labels: Dict[str, Tuple[QLabel, QLabel]] = {}

        statuses = [
            ("system", "Система", "#4ade80"),
            ("slaves", "Slaves", "#60a5fa"),
            ("watchlist", "Watchlist", "#fbbf24"),
            ("trilateration", "Трилатерация", "#a78bfa")
        ]

        for key, label, color in statuses:
            status_widget = QWidget()
            status_layout = QHBoxLayout(status_widget)
            status_layout.setContentsMargins(0, 0, 0, 0)

            indicator = QLabel("●")
            indicator.setStyleSheet(f"color: {color}; font-size: 16px;")
            status_layout.addWidget(indicator)

            text = QLabel(f"{label}: OK")
            status_layout.addWidget(text)

            self.status_labels[key] = (indicator, text)
            layout.addWidget(status_widget)

        layout.addStretch()

        # Время последнего обновления
        self.lbl_last_update = QLabel("Обновлено: —")
        layout.addWidget(self.lbl_last_update)

        return widget

    def _setup_rssi_table(self):
        """Инициализирует таблицу RSSI с правильной структурой."""
        # Создаем заголовки: Диапазон (МГц) | Slave0 (RSSI-rms) | Slave1 (RSSI-rms) | Slave2 (RSSI-rms)
        headers = ["Диапазон (МГц)", "Slave0 (RSSI-rms)", "Slave1 (RSSI-rms)", "Slave2 (RSSI-rms)"]
        
        self.rssi_table.setColumnCount(len(headers))
        self.rssi_table.setHorizontalHeaderLabels(headers)
        
        # Примеры диапазонов (будут заполняться автоматически из детектора)
        sample_ranges = [
            "433.0-435.0",
            "868.0-870.0",
            "2400.0-2450.0",
            "2450.0-2500.0",
            "5725.0-5825.0",
            "5825.0-5875.0"
        ]
        
        self.rssi_table.setRowCount(len(sample_ranges))
        
        for row, range_str in enumerate(sample_ranges):
            # Диапазон
            range_item = QTableWidgetItem(range_str)
            range_item.setFont(QFont("Arial", 10, QFont.Bold))
            self.rssi_table.setItem(row, 0, range_item)
            
            # RSSI для каждого Slave (изначально пустые)
            for col in range(1, 4):  # Slave0, Slave1, Slave2
                rssi_item = QTableWidgetItem("—")
                rssi_item.setTextAlignment(Qt.AlignCenter)
                self.rssi_table.setItem(row, col, rssi_item)
        
        # Настройка ширины колонок
        header = self.rssi_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.resizeSection(0, 150)  # Диапазон
        for i in range(1, 4):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

    def add_range_from_detector(self, freq_start_mhz: float, freq_stop_mhz: float):
        """Добавляет новый диапазон из детектора (вызывается автоматически)."""
        range_str = f"{freq_start_mhz:.1f}-{freq_stop_mhz:.1f}"
        
        # Проверяем, нет ли уже такого диапазона
        for row in range(self.rssi_table.rowCount()):
            if self.rssi_table.item(row, 0).text() == range_str:
                return  # Уже есть
        
        # Добавляем новую строку
        row = self.rssi_table.rowCount()
        self.rssi_table.insertRow(row)
        
        # Диапазон
        range_item = QTableWidgetItem(range_str)
        range_item.setFont(QFont("Arial", 10, QFont.Bold))
        self.rssi_table.setItem(row, 0, range_item)
        
        # RSSI для каждого Slave (изначально пустые)
        for col in range(1, 4):
            rssi_item = QTableWidgetItem("—")
            rssi_item.setTextAlignment(Qt.AlignCenter)
            self.rssi_table.setItem(row, col, rssi_item)
        
        # Обновляем фильтр
        self.range_filter.addItem(range_str)
        
        # Автопрокрутка если включена
        if self.auto_scroll.isChecked():
            self.rssi_table.scrollToBottom()

    def update_rssi_value(self, range_str: str, slave_id: str, rssi_rms: float):
        """Обновляет значение RSSI RMS для конкретного диапазона и слейва."""
        # Находим строку с диапазоном
        row_idx = -1
        for row in range(self.rssi_table.rowCount()):
            if self.rssi_table.item(row, 0).text() == range_str:
                row_idx = row
                break
        
        if row_idx == -1:
            return  # Диапазон не найден
        
        # Определяем колонку по slave_id
        col_idx = -1
        if slave_id.lower() == "slave0":
            col_idx = 1
        elif slave_id.lower() == "slave1":
            col_idx = 2
        elif slave_id.lower() == "slave2":
            col_idx = 3
        
        if col_idx == -1:
            return  # Неизвестный slave
        
        # Обновляем значение
        rssi_item = QTableWidgetItem(f"{rssi_rms:.1f}")
        rssi_item.setTextAlignment(Qt.AlignCenter)
        
        # Цветовая индикация по уровню
        rssi_item.setBackground(QBrush(self._get_rssi_color(rssi_rms)))
        
        self.rssi_table.setItem(row_idx, col_idx, rssi_item)
        
        # Обновляем статистику
        self._update_rssi_stats()

    def _get_rssi_color(self, rssi: float) -> QColor:
        """Возвращает цвет для RSSI значения."""
        threshold = self.threshold_spin.value()
        
        if rssi >= threshold + 20:
            return QColor(74, 222, 128, 100)   # Зеленый - сильный сигнал
        elif rssi >= threshold + 10:
            return QColor(134, 239, 172, 100)  # Светло-зеленый
        elif rssi >= threshold:
            return QColor(251, 191, 36, 100)   # Желтый - средний
        elif rssi >= threshold - 10:
            return QColor(251, 146, 60, 100)   # Оранжевый
        else:
            return QColor(248, 113, 113, 100)  # Красный - слабый

    def _update_rssi_stats(self):
        """Обновляет статистику RSSI."""
        all_rssi = []
        active_slaves = set()
        
        for row in range(self.rssi_table.rowCount()):
            for col in range(1, 4):  # Slave0, Slave1, Slave2
                item = self.rssi_table.item(row, col)
                if item and item.text() != "—":
                    try:
                        rssi = float(item.text())
                        all_rssi.append(rssi)
                        active_slaves.add(f"Slave{col-1}")
                    except ValueError:
                        pass
        
        if all_rssi:
            self.lbl_min_rssi.setText(f"Мин: {min(all_rssi):.1f} дБм")
            self.lbl_max_rssi.setText(f"Макс: {max(all_rssi):.1f} дБм")
            self.lbl_avg_rssi.setText(f"Сред: {np.mean(all_rssi):.1f} дБм")
        else:
            self.lbl_min_rssi.setText("Мин: — дБм")
            self.lbl_max_rssi.setText("Макс: — дБм")
            self.lbl_avg_rssi.setText("Сред: — дБм")
        
        self.lbl_active_slaves.setText(f"Активных Slave: {len(active_slaves)}")

    def _update_rssi_colors(self):
        """Обновляет цвета при изменении порога."""
        for row in range(self.rssi_table.rowCount()):
            for col in range(1, 4):
                item = self.rssi_table.item(row, col)
                if item and item.text() != "—":
                    try:
                        rssi = float(item.text())
                        item.setBackground(QBrush(self._get_rssi_color(rssi)))
                    except ValueError:
                        pass

    def _filter_rssi_table(self):
        """Фильтрует таблицу по выбранному диапазону."""
        filter_text = self.range_filter.currentText()
        
        for row in range(self.rssi_table.rowCount()):
            range_item = self.rssi_table.item(row, 0)
            if range_item:
                if filter_text == "Все диапазоны":
                    self.rssi_table.setRowHidden(row, False)
                else:
                    self.rssi_table.setRowHidden(row, range_item.text() != filter_text)

    def _update_data(self):
        """Периодическое обновление данных."""
        if self.orchestrator and hasattr(self.orchestrator, "get_ui_snapshot"):
            try:
                snapshot = self.orchestrator.get_ui_snapshot()
                if snapshot:
                    self.update_from_orchestrator(snapshot)
            except Exception as e:
                print(f"[SlavesView] Error getting orchestrator snapshot: {e}")
        
        # Обновляем время
        from datetime import datetime
        self.lbl_last_update.setText(f"Обновлено: {datetime.now().strftime('%H:%M:%S')}")

    def update_from_orchestrator(self, data: Dict[str, Any]):
        """Обновляет данные из оркестратора."""
        if 'watchlist' in data:
            self._render_watchlist(data['watchlist'])
        
        if 'tasks' in data:
            self._render_tasks(data['tasks'])
        
        if 'rssi_measurements' in data:
            for measurement in data['rssi_measurements']:
                self.update_rssi_value(
                    measurement['range'],
                    measurement['slave_id'],
                    measurement['rssi_rms']
                )

    def _render_watchlist(self, watchlist_data: List[Dict[str, Any]]):
        """Отрисовывает watchlist."""
        self.watchlist_table.setRowCount(len(watchlist_data))
        
        for row, data in enumerate(watchlist_data):
            # Заполняем колонки
            self.watchlist_table.setItem(row, 0, QTableWidgetItem(data.get('id', '')))
            self.watchlist_table.setItem(row, 1, QTableWidgetItem(f"{data.get('freq', 0):.1f}"))
            self.watchlist_table.setItem(row, 2, QTableWidgetItem(f"{data.get('span', 0):.1f}"))
            
            # RSSI для каждого slave
            for i in range(3):
                rssi_key = f'rssi_{i+1}'
                if rssi_key in data:
                    rssi_item = QTableWidgetItem(f"{data[rssi_key]:.1f}")
                    rssi_item.setTextAlignment(Qt.AlignCenter)
                    rssi_item.setBackground(QBrush(self._get_rssi_color(data[rssi_key])))
                    self.watchlist_table.setItem(row, 3 + i, rssi_item)
            
            # Время обновления
            self.watchlist_table.setItem(row, 6, QTableWidgetItem(data.get('updated', '')))
            
            # Кнопка на карту
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(5, 2, 5, 2)
            
            btn_to_map = QPushButton("📍 На карту")
            btn_to_map.clicked.connect(lambda _, d=data: self._send_watchlist_to_map(d))
            btn_layout.addWidget(btn_to_map)
            
            self.watchlist_table.setCellWidget(row, 7, btn_widget)
        
        self.lbl_watchlist_count.setText(f"Записей: {len(watchlist_data)}")
        self.watchlist_updated.emit(watchlist_data)

    def _render_tasks(self, tasks_data: List[Dict[str, Any]]):
        """Отрисовывает задачи."""
        # Обновляем лог
        log_lines = []
        for task in tasks_data[-20:]:  # Последние 20 записей
            timestamp = time.strftime('%H:%M:%S', time.localtime(task.get('timestamp', time.time())))
            status =