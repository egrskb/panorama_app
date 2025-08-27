# panorama/features/slaves/improved_view.py
"""
Улучшенный UI для управления слейвами с таблицей RSSI_rms и watchlist.

Особенности:
- Матрица RSSI по диапазонам с подсветкой по порогу
- Watchlist с быстрым выводом на карту
- Лог и таблица задач со статусами/прогрессом
- Таблица обнаруженных передатчиков с отправкой на карту
- Экспорт текущего состояния в JSON

Интеграция:
- При необходимости подключите ваш оркестратор через set_orchestrator(orchestrator)
  и периодически вызывайте update_from_orchestrator(snapshot_dict).
- Сигналы:
    send_to_map(dict) — отправка цели на карту (watchlist/передатчики)
    task_selected(str) — выбранная задача (зарезервировано)
    watchlist_updated(list) — обновлён watchlist
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass  # noqa: F401 (на будущее)
import time
import json
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal, QTimer, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QGroupBox, QLabel, QPushButton, QHeaderView,
    QSplitter, QTextEdit, QComboBox, QSpinBox, QCheckBox,
    QGridLayout, QProgressBar, QFrame
)
from PyQt5.QtGui import QFont, QColor, QBrush

import numpy as np

# Импорт qdarkstyle для темного стиля
try:
    import qdarkstyle
    QDARKSTYLE_AVAILABLE = True
except ImportError:
    QDARKSTYLE_AVAILABLE = False


class ImprovedSlavesView(QWidget):
    """
    Современный виджет управления слейвами с RSSI матрицей и watchlist.
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
        
        # Настройки темы
        self._is_dark_theme = True  # По умолчанию темная тема

        # Создаем UI
        self._create_ui()

        # Применяем темный стиль qdarkstyle
        self._apply_dark_style()

        # Таймер обновления
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_data)
        self.update_timer.start(2000)  # Обновление каждые 2 секунды (было 500мс - слишком часто)

    # -----------------------------
    # UI
    # -----------------------------
    def _create_ui(self):
        layout = QVBoxLayout(self)

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

        splitter.setSizes([600, 400])
        layout.addWidget(splitter)

        # Статус бар
        self.status_bar = self._create_status_bar()
        layout.addWidget(self.status_bar)

    def _apply_dark_style(self):
        """Применяет темный стиль qdarkstyle к виджету."""
        try:
            if QDARKSTYLE_AVAILABLE:
                # Применяем темный стиль qdarkstyle
                self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
                print("[ImprovedSlavesView] QDarkStyle applied successfully")
            else:
                # Если qdarkstyle недоступен, применяем базовый темный стиль
                self.setStyleSheet("""
                    QWidget {
                        background-color: #2b2b2b;
                        color: #ffffff;
                    }
                    QGroupBox {
                        border: 2px solid #555555;
                        border-radius: 5px;
                        margin-top: 10px;
                        padding-top: 10px;
                        font-weight: bold;
                    }
                    QGroupBox::title {
                        color: #ffffff;
                        subcontrol-origin: margin;
                        left: 10px;
                        padding: 0 5px;
                    }
                    QTableWidget {
                        background-color: #3c3c3c;
                        alternate-background-color: #4a4a4a;
                        gridline-color: #555555;
                        color: #ffffff;
                    }
                    QHeaderView::section {
                        background-color: #555555;
                        color: #ffffff;
                        padding: 5px;
                        border: 1px solid #666666;
                    }
                    QPushButton {
                        background-color: #555555;
                        border: 1px solid #666666;
                        border-radius: 3px;
                        padding: 5px 10px;
                        color: #ffffff;
                    }
                    QPushButton:hover {
                        background-color: #666666;
                    }
                    QPushButton:pressed {
                        background-color: #444444;
                    }
                    QComboBox, QSpinBox {
                        background-color: #3c3c3c;
                        border: 1px solid #555555;
                        border-radius: 3px;
                        padding: 3px;
                        color: #ffffff;
                    }
                    QTextEdit {
                        background-color: #3c3c3c;
                        border: 1px solid #555555;
                        border-radius: 3px;
                        color: #ffffff;
                    }
                """)
                print("[ImprovedSlavesView] Basic dark style applied")
        except Exception as e:
            print(f"[ImprovedSlavesView] Error applying dark style: {e}")
            # В случае ошибки применяем минимальный темный стиль
            self.setStyleSheet("""
                QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
            """)

    def toggle_theme(self):
        """Переключает между темным и светлым стилем."""
        try:
            if QDARKSTYLE_AVAILABLE:
                # Переключаем между темным и светлым стилем qdarkstyle
                if hasattr(self, '_is_dark_theme'):
                    self._is_dark_theme = not self._is_dark_theme
                else:
                    self._is_dark_theme = False
                
                if self._is_dark_theme:
                    self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
                    print("[ImprovedSlavesView] Switched to dark theme")
                else:
                    self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', variant='light'))
                    print("[ImprovedSlavesView] Switched to light theme")
            else:
                # Переключаем между базовыми стилями
                if hasattr(self, '_is_dark_theme'):
                    self._is_dark_theme = not self._is_dark_theme
                else:
                    self._is_dark_theme = False
                
                if self._is_dark_theme:
                    self._apply_dark_style()
                else:
                    self.setStyleSheet("")  # Сброс к системному стилю
                    print("[ImprovedSlavesView] Switched to system theme")
        except Exception as e:
            print(f"[ImprovedSlavesView] Error toggling theme: {e}")

    def get_theme_info(self) -> dict:
        """Возвращает информацию о текущем стиле."""
        return {
            'qdarkstyle_available': QDARKSTYLE_AVAILABLE,
            'current_theme': 'dark' if self._is_dark_theme else 'light',
            'style_engine': 'QDarkStyle' if QDARKSTYLE_AVAILABLE else 'Basic',
            'can_toggle': True
        }

    def _create_header(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 10px;
                padding: 10px;
            }
            QLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton {
                background: rgba(255, 255, 255, 0.2);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.3);
            }
        """)

        layout = QHBoxLayout(widget)

        title = QLabel("🎯 Система управления Slave SDR")
        title.setFont(QFont("Arial", 16, QFont.Bold))
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

        # Кнопка переключения темы
        self.btn_theme = QPushButton("🌙 Тема")
        self.btn_theme.clicked.connect(self.toggle_theme)
        layout.addWidget(self.btn_theme)

        return widget

    def _create_rssi_panel(self) -> QWidget:
        group = QGroupBox("📊 Матрица RSSI RMS (дБм)")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #667eea;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #667eea;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

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
        self.threshold_spin.valueChanged.connect(self._update_rssi_colors)
        filter_layout.addWidget(QLabel("Порог:"))
        filter_layout.addWidget(self.threshold_spin)

        filter_layout.addStretch()

        self.auto_scroll = QCheckBox("Автопрокрутка")
        self.auto_scroll.setChecked(True)
        filter_layout.addWidget(self.auto_scroll)

        layout.addLayout(filter_layout)

        # Таблица RSSI
        self.rssi_table = QTableWidget()
        self.rssi_table.setAlternatingRowColors(True)
        self.rssi_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #e0e0e0;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #667eea;
                color: white;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                font-weight: bold;
                padding: 8px;
                border: none;
            }
        """)

        # Инициализация таблицы
        self._setup_rssi_table()
        self._refresh_range_filter_options()

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
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Вкладки
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                background: white;
            }
            QTabBar::tab {
                padding: 8px 15px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #667eea;
                color: white;
            }
        """)

        # Вкладка Watchlist
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
        widget = QWidget()
        layout = QVBoxLayout(widget)

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
        self.watchlist_table.setStyleSheet("QTableWidget::item { padding: 5px; }")

        layout.addWidget(self.watchlist_table)

        # Панель управления
        control_panel = QHBoxLayout()

        self.btn_add_to_watchlist = QPushButton("➕ Добавить диапазон")
        self.btn_add_to_watchlist.setStyleSheet("""
            QPushButton {
                background-color: #4ade80;
                color: white;
                font-weight: bold;
                padding: 8px 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #22c55e;
            }
        """)
        # Здесь можно привязать диалог добавления диапазона
        control_panel.addWidget(self.btn_add_to_watchlist)

        self.btn_clear_watchlist = QPushButton("🗑️ Очистить")
        self.btn_clear_watchlist.clicked.connect(self._clear_watchlist)
        control_panel.addWidget(self.btn_clear_watchlist)

        control_panel.addStretch()

        self.lbl_watchlist_count = QLabel("Записей: 0")
        self.lbl_watchlist_count.setStyleSheet("font-weight: bold; color: #667eea;")
        control_panel.addWidget(self.lbl_watchlist_count)

        layout.addLayout(control_panel)

        return widget

    def _create_tasks_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Лог задач (виджет)
        self.tasks_log_view = QTextEdit()
        self.tasks_log_view.setReadOnly(True)
        self.tasks_log_view.setMaximumHeight(200)
        self.tasks_log_view.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #333;
                border-radius: 5px;
                padding: 10px;
            }
        """)
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
            lbl.setStyleSheet("padding: 5px; background-color: #f5f5f5; border-radius: 3px;")
            stats_layout.addWidget(lbl)

        layout.addLayout(stats_layout)

        return widget

    def _create_transmitters_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Таблица передатчиков
        self.transmitters_table = QTableWidget()
        self.transmitters_table.setColumnCount(9)
        self.transmitters_table.setHorizontalHeaderLabels([
            "ID", "Частота (МГц)", "Мощность (дБм)", "Тип", "Позиция X", "Позиция Y",
            "Уверенность", "Время обнаружения", "На карту"
        ])

        header = self.transmitters_table.horizontalHeader()
        for i in range(8):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.Fixed)
        header.resizeSection(8, 100)

        self.transmitters_table.setAlternatingRowColors(True)
        self.transmitters_table.setStyleSheet("""
            QTableWidget::item { padding: 5px; }
            QTableWidget::item:selected { background-color: #667eea; color: white; }
        """)

        layout.addWidget(self.transmitters_table)

        # Фильтры
        filter_layout = QHBoxLayout()

        self.freq_filter = QComboBox()
        self.freq_filter.addItems(["Все частоты", "433 МГц", "868 МГц", "2.4 ГГц", "5.8 ГГц"])
        filter_layout.addWidget(QLabel("Частота:"))
        filter_layout.addWidget(self.freq_filter)

        self.type_filter = QComboBox()
        self.type_filter.addItems(["Все типы", "Дрон", "Видео", "Телеметрия", "RC", "Неизвестно"])
        filter_layout.addWidget(QLabel("Тип:"))
        filter_layout.addWidget(self.type_filter)

        filter_layout.addStretch()

        self.btn_send_all_to_map = QPushButton("📍 Все на карту")
        self.btn_send_all_to_map.setStyleSheet("""
            QPushButton {
                background-color: #f59e0b;
                color: white;
                font-weight: bold;
                padding: 8px 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d97706;
            }
        """)
        self.btn_send_all_to_map.clicked.connect(self._send_all_to_map)
        filter_layout.addWidget(self.btn_send_all_to_map)

        layout.addLayout(filter_layout)

        return widget

    def _create_status_bar(self) -> QWidget:
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        widget.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                padding: 5px;
            }
        """)

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
            text.setStyleSheet("font-size: 12px;")
            status_layout.addWidget(text)

            self.status_labels[key] = (indicator, text)
            layout.addWidget(status_widget)

        layout.addStretch()

        # Информация о стиле
        style_info = QLabel(f"Стиль: {'QDarkStyle' if QDARKSTYLE_AVAILABLE else 'Базовый'}")
        style_info.setStyleSheet("font-size: 11px; color: #666; font-style: italic;")
        layout.addWidget(style_info)

        # Время последнего обновления
        self.lbl_last_update = QLabel("Обновлено: —")
        self.lbl_last_update.setStyleSheet("font-size: 12px; color: #666;")
        layout.addWidget(self.lbl_last_update)

        return widget

    # -----------------------------
    # Логика обновлений
    # -----------------------------
    def _setup_rssi_table(self):
        """Инициализирует таблицу RSSI."""
        # Начальная конфигурация: 3 slave, несколько диапазонов
        slave_ids = ["Slave-1", "Slave-2", "Slave-3"]
        ranges = [
            "50-100 МГц",
            "100-200 МГц",
            "433-435 МГц",
            "868-870 МГц",
            "2400-2500 МГц",
            "5725-5875 МГц"
        ]

        self.rssi_table.setRowCount(len(ranges))
        self.rssi_table.setColumnCount(len(slave_ids) + 2)  # +2 для диапазона и среднего

        headers = ["Диапазон"] + [f"RSSI_{s}" for s in slave_ids] + ["Среднее"]
        self.rssi_table.setHorizontalHeaderLabels(headers)

        for row, range_name in enumerate(ranges):
            item = QTableWidgetItem(range_name)
            item.setFont(QFont("Arial", 10, QFont.Bold))
            self.rssi_table.setItem(row, 0, item)

            for col in range(1, len(slave_ids) + 1):
                rssi_item = QTableWidgetItem("—")
                rssi_item.setTextAlignment(Qt.AlignCenter)
                self.rssi_table.setItem(row, col, rssi_item)

            avg_item = QTableWidgetItem("—")
            avg_item.setTextAlignment(Qt.AlignCenter)
            avg_item.setFont(QFont("Arial", 10, QFont.Bold))
            self.rssi_table.setItem(row, len(slave_ids) + 1, avg_item)

        header = self.rssi_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.resizeSection(0, 150)
        for i in range(1, self.rssi_table.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

    def _refresh_range_filter_options(self):
        """Обновляет список диапазонов в комбобоксе фильтра."""
        existing = set(self._iter_combo_items(self.range_filter))
        for row in range(self.rssi_table.rowCount()):
            txt = self.rssi_table.item(row, 0).text()
            if txt not in existing:
                self.range_filter.addItem(txt)

    @staticmethod
    def _iter_combo_items(combo: QComboBox):
        for i in range(combo.count()):
            yield combo.itemText(i)

    def _update_data(self):
        """Обновляет данные (пулл из оркестратора)."""
        # Если есть оркестратор и он умеет давать снимок — используем
        snapshot = None
        if self.orchestrator and hasattr(self.orchestrator, "get_ui_snapshot"):
            try:
                snapshot = self.orchestrator.get_ui_snapshot()
            except Exception as e:
                print(f"[ImprovedSlavesView] orchestrator.get_ui_snapshot() error: {e}")

        if snapshot:
            self.update_from_orchestrator(snapshot)
        else:
            # Без оркестратора показываем пустые таблицы
            self._clear_all_tables()

        from datetime import datetime
        self.lbl_last_update.setText(f"Обновлено: {datetime.now().strftime('%H:%M:%S')}")

    def _update_rssi_matrix(self):
        """Очищает матрицу RSSI (демо-данные удалены)."""
        # Очищаем все ячейки
        for row in range(self.rssi_table.rowCount()):
            for col in range(1, self.rssi_table.columnCount() - 1):
                item = self.rssi_table.item(row, col)
                if item:
                    item.setText("—")
                    item.setBackground(QBrush(QColor(240, 240, 240)))

            # Очищаем среднее
            avg_item = self.rssi_table.item(row, self.rssi_table.columnCount() - 1)
            if avg_item:
                avg_item.setText("—")
                avg_item.setBackground(QBrush(QColor(240, 240, 240)))

        self._update_rssi_stats()

    def _update_watchlist(self):
        """Очищает таблицу watchlist (демо-данные удалены)."""
        watchlist_data = []  # Пустой список вместо демо-данных
        self._render_watchlist(watchlist_data)

    def _render_watchlist(self, watchlist_data: List[Dict[str, Any]]):
        self.watchlist_table.setRowCount(len(watchlist_data))

        for row, data in enumerate(watchlist_data):
            # ID
            self.watchlist_table.setItem(row, 0, QTableWidgetItem(data['id']))

            # Частота
            freq_item = QTableWidgetItem(f"{data['freq']:.1f}")
            freq_item.setTextAlignment(Qt.AlignCenter)
            self.watchlist_table.setItem(row, 1, freq_item)

            # Ширина
            span_item = QTableWidgetItem(f"{data['span']:.1f}")
            span_item.setTextAlignment(Qt.AlignCenter)
            self.watchlist_table.setItem(row, 2, span_item)

            # RSSI от каждого slave
            for i, rssi in enumerate([data['rssi_1'], data['rssi_2'], data['rssi_3']]):
                rssi_item = QTableWidgetItem(f"{rssi:.1f}")
                rssi_item.setTextAlignment(Qt.AlignCenter)
                rssi_item.setBackground(QBrush(self._get_rssi_color(rssi)))
                self.watchlist_table.setItem(row, 3 + i, rssi_item)

            # Время обновления
            time_item = QTableWidgetItem(data['updated'])
            time_item.setTextAlignment(Qt.AlignCenter)
            self.watchlist_table.setItem(row, 6, time_item)

            # Кнопки
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(5, 2, 5, 2)

            btn_to_map = QPushButton("📍 На карту")
            btn_to_map.setStyleSheet("""
                QPushButton {
                    background-color: #667eea;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 5px 10px;
                    font-size: 12px;
                }
                QPushButton:hover { background-color: #5a67d8; }
            """)
            btn_to_map.clicked.connect(lambda _=False, d=data: self._send_watchlist_to_map(d))
            action_layout.addWidget(btn_to_map)

            self.watchlist_table.setCellWidget(row, 7, action_widget)

        self.lbl_watchlist_count.setText(f"Записей: {len(watchlist_data)}")
        self.watchlist_updated.emit(watchlist_data)

    def _update_tasks(self):
        """Очищает информацию о задачах (демо-данные удалены)."""
        # Очищаем лог
        self.tasks_log_view.clear()

        # Пустой список задач
        tasks_data = []
        self._render_tasks(tasks_data)

    def _render_tasks(self, tasks_data: List[Dict[str, Any]]):
        self.tasks_table.setRowCount(len(tasks_data))

        running = 0
        pending = 0
        completed = 0

        for row, task in enumerate(tasks_data):
            # ID
            self.tasks_table.setItem(row, 0, QTableWidgetItem(task['id']))
            # Диапазон
            self.tasks_table.setItem(row, 1, QTableWidgetItem(task['range']))
            # Статус
            status_item = QTableWidgetItem(task['status'])
            st = task['status']
            if st == 'RUNNING':
                status_item.setBackground(QBrush(QColor(74, 222, 128, 100)))
                running += 1
            elif st == 'PENDING':
                status_item.setBackground(QBrush(QColor(251, 191, 36, 100)))
                pending += 1
            elif st == 'COMPLETED':
                completed += 1
            self.tasks_table.setItem(row, 2, status_item)

            # Прогресс бар
            progress_widget = QProgressBar()
            progress_widget.setValue(int(task.get('progress', 0)))
            progress_widget.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #d0d0d0;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #667eea;
                    border-radius: 2px;
                }
            """)
            self.tasks_table.setCellWidget(row, 3, progress_widget)

            # Время
            self.tasks_table.setItem(row, 4, QTableWidgetItem(task.get('time', '—')))

            # Приоритет
            priority_item = QTableWidgetItem(task.get('priority', 'NORMAL'))
            if task.get('priority') == 'HIGH':
                priority_item.setForeground(QBrush(QColor(239, 68, 68)))
            elif task.get('priority') == 'NORMAL':
                priority_item.setForeground(QBrush(QColor(59, 130, 246)))
            self.tasks_table.setItem(row, 5, priority_item)

        self.lbl_total_tasks.setText(f"Всего: {len(tasks_data)}")
        self.lbl_pending_tasks.setText(f"Ожидает: {pending}")
        self.lbl_running_tasks.setText(f"Выполняется: {running}")
        self.lbl_completed_tasks.setText(f"Завершено: {completed}")

    def _update_transmitters(self):
        """Очищает таблицу передатчиков (демо-данные удалены)."""
        # Пустой список передатчиков
        transmitters_data = []
        self.transmitters_table.setRowCount(0)

    def _update_statistics(self):
        """Очищает общую статистику (демо-статусы удалены)."""
        self._update_status("system", "Ожидание данных", "#9ca3af")
        self._update_status("slaves", "Нет активных", "#9ca3af")
        self._update_status("watchlist", "Пусто", "#9ca3af")
        self._update_status("trilateration", "Неактивна", "#9ca3af")

    def _update_status(self, key: str, text: str, color: str):
        if key in self.status_labels:
            indicator, label = self.status_labels[key]
            indicator.setStyleSheet(f"color: {color}; font-size: 16px;")
            label.setText(text)

    def _get_rssi_color(self, rssi: float) -> QColor:
        """Возвращает цвет для RSSI значения относительно порога."""
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
        """Обновляет агрегированную статистику по RSSI."""
        all_rssi: List[float] = []
        active_slaves = set()

        for row in range(self.rssi_table.rowCount()):
            for col in range(1, self.rssi_table.columnCount() - 1):
                item = self.rssi_table.item(row, col)
                if item and item.text() != "—":
                    try:
                        rssi = float(item.text())
                        all_rssi.append(rssi)
                        active_slaves.add(self.rssi_table.horizontalHeaderItem(col).text())
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

    def _clear_all_tables(self):
        """Очищает все таблицы (демо-данные удалены)."""
        # Очищаем RSSI матрицу
        self._update_rssi_matrix()
        
        # Очищаем watchlist
        self._update_watchlist()
        
        # Очищаем задачи
        self._update_tasks()
        
        # Очищаем передатчики
        self._update_transmitters()
        
        # Очищаем статистику
        self._update_statistics()

    # -----------------------------
    # Действия (карта/кнопки/фильтры)
    # -----------------------------
    def _send_watchlist_to_map(self, watchlist_data: dict):
        """Отправляет данные из watchlist на карту (упрощённая позиция)."""
        rssi_values = [watchlist_data['rssi_1'], watchlist_data['rssi_2'], watchlist_data['rssi_3']]

        # Простейшая псевдопозиция (здесь должна быть настоящая трилатерация)
        x = (rssi_values[0] + rssi_values[1]) * 0.5
        y = (rssi_values[1] + rssi_values[2]) * 0.5

        map_data = {
            'id': watchlist_data['id'],
            'freq': watchlist_data['freq'],
            'x': x,
            'y': y,
            'rssi_avg': float(np.mean(rssi_values))
        }

        self.send_to_map.emit(map_data)
        print(f"[Watchlist] Sent to map: {watchlist_data['id']}")

    def _send_transmitter_to_map(self, transmitter_data: dict):
        """Отправляет данные передатчика на карту."""
        map_data = {
            'id': transmitter_data['id'],
            'freq': transmitter_data['freq'],
            'power': transmitter_data['power'],
            'type': transmitter_data['type'],
            'x': transmitter_data['x'],
            'y': transmitter_data['y'],
            'confidence': transmitter_data['confidence']
        }

        self.send_to_map.emit(map_data)
        print(f"[Transmitter] Sent to map: {transmitter_data['id']}")

    def _send_all_to_map(self):
        """Отправляет все передатчики на карту."""
        for row in range(self.transmitters_table.rowCount()):
            tx = {
                'id': self.transmitters_table.item(row, 0).text(),
                'freq': float(self.transmitters_table.item(row, 1).text()),
                'power': float(self.transmitters_table.item(row, 2).text()),
                'type': self.transmitters_table.item(row, 3).text(),
                'x': float(self.transmitters_table.item(row, 4).text()),
                'y': float(self.transmitters_table.item(row, 5).text()),
                'confidence': float(self.transmitters_table.item(row, 6).text().replace('%', '')) / 100.0
            }
            self._send_transmitter_to_map(tx)

    def _filter_rssi_table(self):
        """Фильтрует таблицу RSSI по выбранному диапазону."""
        filter_text = self.range_filter.currentText()

        for row in range(self.rssi_table.rowCount()):
            range_item = self.rssi_table.item(row, 0)
            if range_item:
                self.rssi_table.setRowHidden(
                    row,
                    not (filter_text == "Все диапазоны" or filter_text in range_item.text())
                )

    def _update_rssi_colors(self):
        """Обновляет цвета RSSI при изменении порога."""
        for row in range(self.rssi_table.rowCount()):
            for col in range(1, self.rssi_table.columnCount()):
                item = self.rssi_table.item(row, col)
                if item and item.text() != "—":
                    try:
                        rssi = float(item.text())
                        item.setBackground(QBrush(self._get_rssi_color(rssi)))
                    except ValueError:
                        pass

    def _refresh_data(self):
        """Ручное обновление данных."""
        self._update_data()
        print("[ImprovedSlavesView] Manual refresh")

    def _clear_data(self):
        """Очищает таблицы и логи (UI-сброс, не трогает оркестратор)."""
        # RSSI
        for row in range(self.rssi_table.rowCount()):
            for col in range(1, self.rssi_table.columnCount()):
                it = self.rssi_table.item(row, col)
                if it:
                    it.setText("—")
                    it.setBackground(QBrush(QColor(240, 240, 240)))
        self._update_rssi_stats()

        # Watchlist
        self.watchlist_table.setRowCount(0)
        self.lbl_watchlist_count.setText("Записей: 0")

        # Tasks
        self.tasks_table.setRowCount(0)
        self.tasks_log_view.clear()
        self.lbl_total_tasks.setText("Всего: 0")
        self.lbl_pending_tasks.setText("Ожидает: 0")
        self.lbl_running_tasks.setText("Выполняется: 0")
        self.lbl_completed_tasks.setText("Завершено: 0")

        # Transmitters
        self.transmitters_table.setRowCount(0)

        print("[ImprovedSlavesView] UI cleared")

    def _export_data(self):
        """Экспорт текущего UI-состояния в JSON."""
        snapshot = {
            "rssi": self._collect_rssi_snapshot(),
            "watchlist": self._collect_table(self.watchlist_table),
            "tasks": self._collect_tasks_snapshot(),
            "transmitters": self._collect_table(self.transmitters_table)
        }
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path.home() / f"panorama_ui_export_{ts}.json"
        try:
            out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
            print(f"[ImprovedSlavesView] Exported to {out_path}")
        except Exception as e:
            print(f"[ImprovedSlavesView] Export error: {e}")

    def _collect_rssi_snapshot(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for row in range(self.rssi_table.rowCount()):
            rng = self.rssi_table.item(row, 0).text()
            row_data: Dict[str, Any] = {"range": rng}
            for col in range(1, self.rssi_table.columnCount()):
                header = self.rssi_table.horizontalHeaderItem(col).text()
                txt = self.rssi_table.item(row, col).text()
                row_data[header] = txt
            rows.append(row_data)
        return rows

    def _collect_tasks_snapshot(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        for row in range(self.tasks_table.rowCount()):
            rec = {
                "id": self._safe_item(self.tasks_table, row, 0),
                "range": self._safe_item(self.tasks_table, row, 1),
                "status": self._safe_item(self.tasks_table, row, 2),
                "progress": self._safe_progress(row),
                "time": self._safe_item(self.tasks_table, row, 4),
                "priority": self._safe_item(self.tasks_table, row, 5),
            }
            data.append(rec)
        return data

    def _safe_item(self, table: QTableWidget, row: int, col: int) -> str:
        it = table.item(row, col)
        return it.text() if it else ""

    def _safe_progress(self, row: int) -> int:
        w = self.tasks_table.cellWidget(row, 3)
        if isinstance(w, QProgressBar):
            return w.value()
        elif w and w.findChild(QProgressBar):
            return w.findChild(QProgressBar).value()
        return 0

    def _collect_table(self, table: QTableWidget) -> List[List[str]]:
        data: List[List[str]] = []
        for row in range(table.rowCount()):
            row_vals: List[str] = []
            for col in range(table.columnCount()):
                it = table.item(row, col)
                row_vals.append(it.text() if it else "")
            data.append(row_vals)
        return data

    def _clear_watchlist(self):
        self.watchlist_table.setRowCount(0)
        self.lbl_watchlist_count.setText("Записей: 0")
        self.watchlist = []
        self.watchlist_updated.emit(self.watchlist)

    # -----------------------------
    # Публичные методы интеграции
    # -----------------------------
    def update_from_orchestrator(self, orchestrator_data: Dict[str, Any]):
        """Обновляет данные из оркестратора."""
        if 'slaves' in orchestrator_data:
            self._update_slave_statuses(orchestrator_data['slaves'])

        if 'watchlist' in orchestrator_data:
            self._update_watchlist_from_orchestrator(orchestrator_data['watchlist'])

        if 'tasks' in orchestrator_data:
            self._update_tasks_from_orchestrator(orchestrator_data['tasks'])

        if 'transmitters' in orchestrator_data:
            self._render_transmitters_from_orchestrator(orchestrator_data['transmitters'])

    def _update_slave_statuses(self, slaves_data: List[Dict[str, Any]]):
        """Обновляет статусы слейвов из оркестратора."""
        for slave_data in slaves_data:
            slave_id = slave_data.get('id', 'unknown')

            if slave_id not in self.slave_statuses:
                self.slave_statuses[slave_id] = {
                    'id': slave_id,
                    'name': slave_data.get('name', slave_id),
                    'is_online': slave_data.get('is_online', False),
                    'last_rssi': {}
                }

            status = self.slave_statuses[slave_id]
            status['is_online'] = slave_data.get('is_online', False)

            if 'rssi_measurements' in slave_data:
                for range_id, rssi in slave_data['rssi_measurements'].items():
                    status['last_rssi'][range_id] = rssi

    def _update_watchlist_from_orchestrator(self, watchlist_data: List[Dict[str, Any]]):
        """Обновляет watchlist из оркестратора."""
        self.watchlist = watchlist_data
        self._render_watchlist(self.watchlist)

    def _update_tasks_from_orchestrator(self, tasks_data: List[Dict[str, Any]]):
        """Обновляет задачи из оркестратора."""
        self.tasks_data = tasks_data

        # Лог — последние 20 записей
        log_lines: List[str] = []
        for task in tasks_data[-20:]:
            timestamp = time.strftime('%H:%M:%S', time.localtime(task.get('timestamp', time.time())))
            status = task.get('status', 'UNKNOWN')
            task_id = task.get('id', 'N/A')
            log_lines.append(f"[{timestamp}] Task {task_id}: {status}")
        self.tasks_log_view.setPlainText("\n".join(log_lines))

        # Активные задачи
        active_tasks = [t for t in tasks_data if t.get('status') in ['PENDING', 'RUNNING', 'IN_PROGRESS']]
        # Нормализуем структуру для отрисовки
        for t in active_tasks:
            t.setdefault('range', t.get('freq_range', 'N/A'))
            t.setdefault('progress', t.get('progress', 0))
            t.setdefault('time', time.strftime('%H:%M:%S', time.localtime(t.get('timestamp', time.time()))))
            t.setdefault('priority', t.get('priority', 'NORMAL'))
        self._render_tasks(active_tasks)

        # Итоги
        self.lbl_total_tasks.setText(f"Всего: {len(tasks_data)}")
        self.lbl_pending_tasks.setText(f"Ожидает: {len([t for t in tasks_data if t.get('status') == 'PENDING'])}")
        self.lbl_running_tasks.setText(f"Выполняется: {len([t for t in tasks_data if t.get('status') in ['RUNNING', 'IN_PROGRESS']])}")
        self.lbl_completed_tasks.setText(f"Завершено: {len([t for t in tasks_data if t.get('status') == 'COMPLETED'])}")

    def _render_transmitters_from_orchestrator(self, txs: List[Dict[str, Any]]):
        """Отрисовывает передатчики из данных оркестратора."""
        self.transmitters_table.setRowCount(len(txs))
        for row, tx in enumerate(txs):
            self.transmitters_table.setItem(row, 0, QTableWidgetItem(str(tx.get('id', 'TX'))))
            self.transmitters_table.setItem(row, 1, QTableWidgetItem(f"{float(tx.get('freq', 0.0)):.2f}"))
            self.transmitters_table.setItem(row, 2, QTableWidgetItem(f"{float(tx.get('power', 0.0)):.1f}"))
            self.transmitters_table.setItem(row, 3, QTableWidgetItem(str(tx.get('type', ''))))
            self.transmitters_table.setItem(row, 4, QTableWidgetItem(f"{float(tx.get('x', 0.0)):.1f}"))
            self.transmitters_table.setItem(row, 5, QTableWidgetItem(f"{float(tx.get('y', 0.0)):.1f}"))

            conf = float(tx.get('confidence', 0.0))
            conf_item = QTableWidgetItem(f"{conf*100:.0f}%")
            if conf > 0.8:
                conf_item.setBackground(QBrush(QColor(74, 222, 128, 100)))
            elif conf > 0.6:
                conf_item.setBackground(QBrush(QColor(251, 191, 36, 100)))
            else:
                conf_item.setBackground(QBrush(QColor(248, 113, 113, 100)))
            self.transmitters_table.setItem(row, 6, conf_item)

            self.transmitters_table.setItem(row, 7, QTableWidgetItem(str(tx.get('time', '—'))))

            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(5, 2, 5, 2)
            btn_map = QPushButton("📍")
            btn_map.clicked.connect(lambda _=False, t=tx: self._send_transmitter_to_map(t))
            btn_layout.addWidget(btn_map, alignment=Qt.AlignCenter)
            self.transmitters_table.setCellWidget(row, 8, btn_widget)

    def get_selected_range(self) -> Optional[Tuple[float, float]]:
        """Возвращает выбранный диапазон частот (начало, конец) из таблицы."""
        current_row = self.rssi_table.currentRow()
        if current_row >= 0:
            range_text = self.rssi_table.item(current_row, 0).text()
            parts = range_text.replace(' МГц', '').split('-')
            if len(parts) == 2:
                try:
                    start = float(parts[0])
                    stop = float(parts[1])
                    return (start, stop)
                except ValueError:
                    pass
        return None

    def set_orchestrator(self, orchestrator: Any):
        """Устанавливает ссылку на оркестратор."""
        self.orchestrator = orchestrator
        print("[ImprovedSlavesView] Orchestrator connected")

    def _update_rssi_table(self):
        """Обновляет таблицу RSSI."""
        try:
            if hasattr(self, 'rssi_table'):
                # Очищаем таблицу
                self.rssi_table.setRowCount(0)
                
                # Заполняем данными из матрицы
                for range_id, slave_data in self.rssi_matrix.items():
                    row = self.rssi_table.rowCount()
                    self.rssi_table.insertRow(row)
                    
                    # Диапазон
                    self.rssi_table.setItem(row, 0, QTableWidgetItem(str(range_id)))
                    
                    # RSSI для каждого слейва
                    for col, (slave_id, rssi) in enumerate(slave_data.items(), 1):
                        item = QTableWidgetItem(f"{rssi:.1f}")
                        # Подсветка по порогу
                        if rssi > -50:
                            item.setBackground(QColor(255, 200, 200))  # Красный
                        elif rssi > -60:
                            item.setBackground(QColor(255, 255, 200))  # Желтый
                        self.rssi_table.setItem(row, col, item)
        except Exception as e:
            print(f"[ImprovedSlavesView] Error updating RSSI table: {e}")
    
    def _update_watchlist_table(self):
        """Обновляет таблицу watchlist."""
        try:
            if hasattr(self, 'watchlist_table'):
                # Очищаем таблицу
                self.watchlist_table.setRowCount(0)
                
                # Заполняем данными
                for item in self.watchlist:
                    row = self.watchlist_table.rowCount()
                    self.watchlist_table.insertRow(row)
                    
                    # ID
                    self.watchlist_table.setItem(row, 0, QTableWidgetItem(str(item.get('id', ''))))
                    # Частота
                    self.watchlist_table.setItem(row, 1, QTableWidgetItem(f"{item.get('freq', 0):.2f}"))
                    # Span
                    self.watchlist_table.setItem(row, 2, QTableWidgetItem(f"{item.get('span', 0):.1f}"))
                    # RSSI для каждого слейва
                    for col, slave_id in enumerate(['rssi_1', 'rssi_2', 'rssi_3'], 3):
                        rssi = item.get(slave_id, 0)
                        self.watchlist_table.setItem(row, col, QTableWidgetItem(f"{rssi:.1f}"))
                    # Время обновления
                    self.watchlist_table.setItem(row, 6, QTableWidgetItem(str(item.get('updated', ''))))
        except Exception as e:
            print(f"[ImprovedSlavesView] Error updating watchlist table: {e}")
    
    def _update_tasks_table(self):
        """Обновляет таблицу задач."""
        try:
            if hasattr(self, 'tasks_table'):
                # Очищаем таблицу
                self.tasks_table.setRowCount(0)
                
                # Заполняем данными
                for task in self.tasks_data:
                    row = self.tasks_table.rowCount()
                    self.tasks_table.insertRow(row)
                    
                    # ID задачи
                    self.tasks_table.setItem(row, 0, QTableWidgetItem(str(task.get('id', ''))))
                    # Диапазон
                    self.tasks_table.setItem(row, 1, QTableWidgetItem(str(task.get('range', ''))))
                    # Статус
                    status_item = QTableWidgetItem(str(task.get('status', '')))
                    # Подсветка статуса
                    status = task.get('status', '')
                    if status == 'COMPLETED':
                        status_item.setBackground(QColor(200, 255, 200))  # Зеленый
                    elif status == 'RUNNING':
                        status_item.setBackground(QColor(200, 200, 255))  # Синий
                    elif status == 'FAILED':
                        status_item.setBackground(QColor(255, 200, 200))  # Красный
                    self.tasks_table.setItem(row, 2, status_item)
                    # Прогресс
                    progress = task.get('progress', 0)
                    self.tasks_table.setItem(row, 3, QTableWidgetItem(f"{progress}%"))
                    # Приоритет
                    self.tasks_table.setItem(row, 4, QTableWidgetItem(str(task.get('priority', ''))))
        except Exception as e:
            print(f"[ImprovedSlavesView] Error updating tasks table: {e}")
    
    def _clear_transmitters_table(self):
        """Очищает таблицу передатчиков."""
        try:
            if hasattr(self, 'transmitters_table'):
                self.transmitters_table.setRowCount(0)
        except Exception as e:
            print(f"[ImprovedSlavesView] Error clearing transmitters table: {e}")
    
    def clear_all_data(self):
        """Очищает все данные в интерфейсе."""
        try:
            # Очищаем RSSI матрицу
            self.rssi_matrix.clear()
            self._update_rssi_table()
            
            # Очищаем watchlist
            self.watchlist.clear()
            self._update_watchlist_table()
            
            # Очищаем задачи
            self.tasks_data.clear()
            self._update_tasks_table()
            
            # Очищаем передатчики
            self._clear_transmitters_table()
            
            print("[ImprovedSlavesView] All data cleared")
        except Exception as e:
            print(f"[ImprovedSlavesView] Error clearing data: {e}")
    
    def cleanup(self):
        """Очистка ресурсов при закрытии."""
        self.update_timer.stop()
        print("[ImprovedSlavesView] Cleanup completed")

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)


# Демо-режим удален - все демо-данные убраны


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Применяем темный стиль к приложению
    try:
        if QDARKSTYLE_AVAILABLE:
            app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
            print("[Demo] QDarkStyle applied to application")
        else:
            print("[Demo] QDarkStyle not available, using basic dark style")
    except Exception as e:
        print(f"[Demo] Error applying QDarkStyle: {e}")

    widget = ImprovedSlavesView()
    widget.resize(1400, 900)
    widget.show()

    # Демо-данные удалены - виджет покажет пустые таблицы

    sys.exit(app.exec_())
