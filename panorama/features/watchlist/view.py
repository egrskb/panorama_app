# panorama/features/watchlist/view_fixed.py
"""
Исправленная версия UI для управления слейвами
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import time
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QGroupBox, QLabel, QPushButton, QHeaderView, QSplitter, 
    QTextEdit, QComboBox, QSpinBox, QCheckBox, QGridLayout, 
    QProgressBar, QFrame, QTabWidget
)
from PyQt5.QtGui import QFont, QColor, QBrush


class ImprovedSlavesView(QWidget):
    """Исправленный виджет управления слейвами с RSSI матрицей."""
    
    send_to_map = pyqtSignal(dict)
    task_selected = pyqtSignal(str)
    watchlist_updated = pyqtSignal(list)

    def __init__(self, orchestrator=None, parent=None):
        super().__init__(parent)
        self.orchestrator = orchestrator
        
        # Инициализация данных
        self.rssi_matrix = {}
        self.watchlist = []
        self.tasks_data = []
        
        # Создаем UI
        self._create_ui()
        
        # Таймер обновления
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_data)
        self.update_timer.start(2000)

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
        
        splitter.setSizes([800, 600])
        layout.addWidget(splitter)
        
        # Статус бар
        self.status_bar = self._create_status_bar()
        layout.addWidget(self.status_bar)

    def _create_header(self) -> QWidget:
        """Создает заголовок."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        title = QLabel("🎯 Система управления Slave SDR")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        layout.addStretch()
        
        self.btn_refresh = QPushButton("🔄 Обновить")
        self.btn_refresh.clicked.connect(self._refresh_data)
        layout.addWidget(self.btn_refresh)
        
        self.btn_clear = QPushButton("🗑️ Очистить")
        self.btn_clear.clicked.connect(self._clear_data)
        layout.addWidget(self.btn_clear)
        
        return widget

    def _create_rssi_panel(self) -> QWidget:
        """Создает панель с матрицей RSSI RMS."""
        group = QGroupBox("📊 Матрица RSSI RMS (дБм)")
        layout = QVBoxLayout(group)
        
        # Добавляем недостающие элементы управления
        controls = QHBoxLayout()
        
        # Фильтр диапазонов
        self.range_filter = QComboBox()
        self.range_filter.addItem("Все диапазоны")
        self.range_filter.currentTextChanged.connect(self._filter_rssi_table)
        controls.addWidget(QLabel("Фильтр:"))
        controls.addWidget(self.range_filter)
        
        # Порог RSSI
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(-120, 0)
        self.threshold_spin.setValue(-80)
        self.threshold_spin.setSuffix(" дБм")
        self.threshold_spin.valueChanged.connect(self._update_rssi_colors)
        controls.addWidget(QLabel("Порог:"))
        controls.addWidget(self.threshold_spin)
        
        # Автопрокрутка
        self.auto_scroll = QCheckBox("Автопрокрутка")
        self.auto_scroll.setChecked(True)
        controls.addWidget(self.auto_scroll)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Таблица RSSI
        self.rssi_table = QTableWidget()
        self.rssi_table.setAlternatingRowColors(True)
        self._setup_rssi_table()
        layout.addWidget(self.rssi_table)
        
        # Статистика
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
        
        tabs = QTabWidget()
        
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
        """Создает вкладку watchlist."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Информация
        info = QLabel(
            "📍 Диапазоны добавляются автоматически при обнаружении сигналов.\n"
            "RSSI_1..3 - измеренные уровни от каждого Slave SDR."
        )
        info.setWordWrap(True)
        info.setStyleSheet("""
            QLabel {
                background-color: rgba(100, 100, 255, 30);
                padding: 10px;
                border-radius: 5px;
            }
        """)
        layout.addWidget(info)
        
        # Таблица watchlist
        self.watchlist_table = QTableWidget()
        self.watchlist_table.setColumnCount(10)
        self.watchlist_table.setHorizontalHeaderLabels([
            "ID", "Частота (МГц)", "Ширина (МГц)", "Halfspan (МГц)",
            "RMS_1 (дБм)", "RMS_2 (дБм)", "RMS_3 (дБм)", "Бинов", "Обновлено", "Действия"
        ])
        self.watchlist_table.setAlternatingRowColors(True)
        layout.addWidget(self.watchlist_table)
        
        # Управление
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
        
        # Таблица задач
        self.tasks_table = QTableWidget()
        self.tasks_table.setColumnCount(6)
        self.tasks_table.setHorizontalHeaderLabels([
            "ID", "Диапазон", "Статус", "Прогресс", "Время", "Приоритет"
        ])
        layout.addWidget(self.tasks_table)
        
        # Статистика
        stats = QHBoxLayout()
        self.lbl_total_tasks = QLabel("Всего: 0")
        self.lbl_pending_tasks = QLabel("Ожидает: 0")
        self.lbl_running_tasks = QLabel("Выполняется: 0")
        self.lbl_completed_tasks = QLabel("Завершено: 0")
        
        for lbl in [self.lbl_total_tasks, self.lbl_pending_tasks,
                   self.lbl_running_tasks, self.lbl_completed_tasks]:
            stats.addWidget(lbl)
        
        layout.addLayout(stats)
        return widget

    def _create_transmitters_tab(self) -> QWidget:
        """Создает вкладку передатчиков."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Таблица передатчиков
        self.transmitters_table = QTableWidget()
        self.transmitters_table.setColumnCount(9)
        self.transmitters_table.setHorizontalHeaderLabels([
            "ID", "Частота (МГц)", "Мощность (дБм)", "Тип",
            "X", "Y", "Уверенность", "Время", "На карту"
        ])
        layout.addWidget(self.transmitters_table)
        
        return widget

    def _create_status_bar(self) -> QWidget:
        """Создает статус бар."""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        layout = QHBoxLayout(widget)
        
        self.lbl_last_update = QLabel("Обновлено: —")
        layout.addWidget(self.lbl_last_update)
        
        return widget

    def _setup_rssi_table(self):
        """Инициализирует таблицу RSSI."""
        headers = ["Диапазон (МГц)", "Slave0", "Slave1", "Slave2"]
        self.rssi_table.setColumnCount(len(headers))
        self.rssi_table.setHorizontalHeaderLabels(headers)
        self.rssi_table.setRowCount(0)

    def add_range_from_detector(self, freq_start_mhz: float, freq_stop_mhz: float):
        """Добавляет диапазон из детектора."""
        range_str = f"{freq_start_mhz:.1f}-{freq_stop_mhz:.1f}"
        
        # Проверяем дубликаты
        for row in range(self.rssi_table.rowCount()):
            if self.rssi_table.item(row, 0) and \
               self.rssi_table.item(row, 0).text() == range_str:
                return
        
        # Добавляем строку
        row = self.rssi_table.rowCount()
        self.rssi_table.insertRow(row)
        
        # Диапазон
        self.rssi_table.setItem(row, 0, QTableWidgetItem(range_str))
        
        # RSSI для каждого Slave
        for col in range(1, 4):
            self.rssi_table.setItem(row, col, QTableWidgetItem("—"))
        
        # Обновляем фильтр
        self.range_filter.addItem(range_str)
        
        # Автопрокрутка
        if self.auto_scroll.isChecked():
            self.rssi_table.scrollToBottom()

    def update_rssi_value(self, range_str: str, slave_id: str, rssi_rms: float):
        """Обновляет значение RSSI."""
        # Находим строку
        row_idx = -1
        for row in range(self.rssi_table.rowCount()):
            item = self.rssi_table.item(row, 0)
            if item and item.text() == range_str:
                row_idx = row
                break
        
        if row_idx == -1:
            # Создаем новую строку если не найдена
            self.add_range_from_detector(
                float(range_str.split('-')[0]),
                float(range_str.split('-')[1])
            )
            row_idx = self.rssi_table.rowCount() - 1
        
        # Определяем колонку
        col_map = {"slave0": 1, "slave1": 2, "slave2": 3}
        col_idx = col_map.get(slave_id.lower(), -1)
        
        if col_idx > 0:
            item = QTableWidgetItem(f"{rssi_rms:.1f}")
            item.setTextAlignment(Qt.AlignCenter)
            item.setBackground(QBrush(self._get_rssi_color(rssi_rms)))
            self.rssi_table.setItem(row_idx, col_idx, item)
            
            self._update_rssi_stats()

    def _get_rssi_color(self, rssi: float) -> QColor:
        """Цвет для RSSI."""
        threshold = self.threshold_spin.value()
        
        if rssi >= threshold + 20:
            return QColor(74, 222, 128, 100)  # Зеленый
        elif rssi >= threshold + 10:
            return QColor(251, 191, 36, 100)  # Желтый
        else:
            return QColor(248, 113, 113, 100) # Красный

    def _update_rssi_stats(self):
        """Обновляет статистику RSSI."""
        all_rssi = []
        
        for row in range(self.rssi_table.rowCount()):
            for col in range(1, 4):
                item = self.rssi_table.item(row, col)
                if item and item.text() != "—":
                    try:
                        all_rssi.append(float(item.text()))
                    except:
                        pass
        
        if all_rssi:
            self.lbl_min_rssi.setText(f"Мин: {min(all_rssi):.1f} дБм")
            self.lbl_max_rssi.setText(f"Макс: {max(all_rssi):.1f} дБм")
            self.lbl_avg_rssi.setText(f"Сред: {np.mean(all_rssi):.1f} дБм")

    def _update_rssi_colors(self):
        """Обновляет цвета при изменении порога."""
        for row in range(self.rssi_table.rowCount()):
            for col in range(1, 4):
                item = self.rssi_table.item(row, col)
                if item and item.text() != "—":
                    try:
                        rssi = float(item.text())
                        item.setBackground(QBrush(self._get_rssi_color(rssi)))
                    except:
                        pass

    def _filter_rssi_table(self):
        """Фильтрует таблицу."""
        filter_text = self.range_filter.currentText()
        
        for row in range(self.rssi_table.rowCount()):
            item = self.rssi_table.item(row, 0)
            if item:
                if filter_text == "Все диапазоны":
                    self.rssi_table.setRowHidden(row, False)
                else:
                    self.rssi_table.setRowHidden(row, item.text() != filter_text)

    def _update_data(self):
        """Периодическое обновление данных."""
        if self.orchestrator and hasattr(self.orchestrator, "get_ui_snapshot"):
            try:
                snapshot = self.orchestrator.get_ui_snapshot()
                if snapshot:
                    self.update_from_orchestrator(snapshot)
            except Exception as e:
                print(f"[SlavesView] Error: {e}")
        
        # Время обновления
        self.lbl_last_update.setText(f"Обновлено: {time.strftime('%H:%M:%S')}")

    def update_from_orchestrator(self, data: Dict[str, Any]):
        """Обновляет данные из оркестратора."""
        # Обновляем watchlist
        if 'watchlist' in data:
            self._render_watchlist(data['watchlist'])
        
        # Обновляем задачи
        if 'tasks' in data:
            self._render_tasks(data['tasks'])
        
        # Обновляем RSSI измерения
        if 'rssi_measurements' in data:
            for m in data['rssi_measurements']:
                self.update_rssi_value(
                    m['range'],
                    m['slave_id'],
                    m['rssi_rms']
                )

    def _render_watchlist(self, watchlist_data: List[Dict]):
        """Отрисовывает watchlist."""
        self.watchlist_table.setRowCount(len(watchlist_data))
        
        for row, data in enumerate(watchlist_data):
            # ID
            self.watchlist_table.setItem(row, 0, 
                QTableWidgetItem(str(data.get('id', ''))))
            
            # Частота и ширина
            freq = float(data.get('freq', 0))
            span = float(data.get('span', 0))
            self.watchlist_table.setItem(row, 1, 
                QTableWidgetItem(f"{freq:.1f}"))
            self.watchlist_table.setItem(row, 2, 
                QTableWidgetItem(f"{span:.1f}"))
            
            # Halfspan для RMS
            halfspan = float(data.get('halfspan', 2.5))
            self.watchlist_table.setItem(row, 3, 
                QTableWidgetItem(f"{halfspan:.1f}"))
            
            # RMS для каждого slave
            for i in range(3):
                rms_key = f'rms_{i+1}'
                val = data.get(rms_key)
                if val is not None:
                    item = QTableWidgetItem(f"{float(val):.1f}")
                    item.setBackground(QBrush(self._get_rssi_color(float(val))))
                    # Добавляем tooltip с информацией
                    bins_used = data.get(f'bins_used_{i+1}', 'N/A')
                    timestamp = data.get(f'timestamp_{i+1}', '')
                    item.setToolTip(f"Бинов использовано: {bins_used}\nПоследнее обновление: {timestamp}")
                else:
                    item = QTableWidgetItem("—")
                item.setTextAlignment(Qt.AlignCenter)
                self.watchlist_table.setItem(row, 4 + i, item)
            
            # Общее количество бинов
            total_bins = data.get('total_bins', 0)
            self.watchlist_table.setItem(row, 7, 
                QTableWidgetItem(str(total_bins)))
            
            # Время обновления
            self.watchlist_table.setItem(row, 8, 
                QTableWidgetItem(data.get('updated', '')))
            
            # Кнопка
            btn = QPushButton("📍 На карту")
            btn.clicked.connect(lambda _, d=data: self._send_to_map(d))
            self.watchlist_table.setCellWidget(row, 9, btn)
        
        self.lbl_watchlist_count.setText(f"Записей: {len(watchlist_data)}")

    def _render_tasks(self, tasks_data: List[Dict]):
        """Отрисовывает задачи."""
        # Лог
        log_lines = []
        for task in tasks_data[-20:]:
            timestamp = time.strftime('%H:%M:%S', 
                time.localtime(task.get('timestamp', time.time())))
            status = task.get('status', 'UNKNOWN')
            task_id = task.get('id', 'N/A')
            log_lines.append(f"[{timestamp}] Task {task_id}: {status}")
        
        self.tasks_log_view.setPlainText("\n".join(log_lines))
        
        # Таблица активных задач
        active = [t for t in tasks_data 
                 if t.get('status') in ['PENDING', 'RUNNING']]
        self.tasks_table.setRowCount(len(active))
        
        stats = {'pending': 0, 'running': 0, 'completed': 0}
        
        for row, task in enumerate(active):
            # Заполняем колонки
            self.tasks_table.setItem(row, 0, 
                QTableWidgetItem(task.get('id', '')))
            self.tasks_table.setItem(row, 1, 
                QTableWidgetItem(task.get('range', '')))
            
            # Статус
            status = task.get('status', '')
            status_item = QTableWidgetItem(status)
            if status == 'RUNNING':
                status_item.setBackground(QBrush(QColor(74, 222, 128, 100)))
                stats['running'] += 1
            elif status == 'PENDING':
                status_item.setBackground(QBrush(QColor(251, 191, 36, 100)))
                stats['pending'] += 1
            self.tasks_table.setItem(row, 2, status_item)
            
            # Прогресс
            progress = QProgressBar()
            progress.setValue(task.get('progress', 0))
            self.tasks_table.setCellWidget(row, 3, progress)
            
            # Время и приоритет
            self.tasks_table.setItem(row, 4, 
                QTableWidgetItem(task.get('time', '')))
            self.tasks_table.setItem(row, 5, 
                QTableWidgetItem(task.get('priority', 'NORMAL')))
        
        # Обновляем статистику
        self.lbl_total_tasks.setText(f"Всего: {len(tasks_data)}")
        self.lbl_pending_tasks.setText(f"Ожидает: {stats['pending']}")
        self.lbl_running_tasks.setText(f"Выполняется: {stats['running']}")
        self.lbl_completed_tasks.setText(f"Завершено: {stats['completed']}")

    def add_transmitter(self, result):
        """Добавляет передатчик в таблицу."""
        try:
            row = self.transmitters_table.rowCount()
            self.transmitters_table.insertRow(row)
            
            # Заполняем данные
            peak_id = getattr(result, 'peak_id', 'unknown')
            freq = getattr(result, 'freq_mhz', 0.0)
            x = getattr(result, 'x', 0.0)
            y = getattr(result, 'y', 0.0)
            confidence = getattr(result, 'confidence', 0.0)
            
            self.transmitters_table.setItem(row, 0, QTableWidgetItem(str(peak_id)))
            self.transmitters_table.setItem(row, 1, QTableWidgetItem(f"{freq:.1f}"))
            self.transmitters_table.setItem(row, 2, QTableWidgetItem("-"))
            self.transmitters_table.setItem(row, 3, QTableWidgetItem("Video"))
            self.transmitters_table.setItem(row, 4, QTableWidgetItem(f"{x:.1f}"))
            self.transmitters_table.setItem(row, 5, QTableWidgetItem(f"{y:.1f}"))
            self.transmitters_table.setItem(row, 6, QTableWidgetItem(f"{confidence*100:.0f}%"))
            self.transmitters_table.setItem(row, 7, QTableWidgetItem(time.strftime("%H:%M:%S")))
            
            # Кнопка на карту
            btn = QPushButton("📍")
            btn.clicked.connect(lambda: self.send_to_map.emit({
                'id': peak_id, 'freq': freq, 'x': x, 'y': y
            }))
            self.transmitters_table.setCellWidget(row, 8, btn)
            
        except Exception as e:
            print(f"[SlavesView] Error adding transmitter: {e}")

    def _send_to_map(self, data):
        """Отправляет на карту."""
        self.send_to_map.emit(data)

    def _clear_watchlist(self):
        """Очищает watchlist."""
        self.watchlist_table.setRowCount(0)
        self.lbl_watchlist_count.setText("Записей: 0")

    def _refresh_data(self):
        """Ручное обновление."""
        self._update_data()

    def _clear_data(self):
        """Очищает все данные."""
        self.rssi_table.setRowCount(0)
        self.watchlist_table.setRowCount(0)
        self.tasks_table.setRowCount(0)
        self.transmitters_table.setRowCount(0)
        self.tasks_log_view.clear()
        self._update_rssi_stats()