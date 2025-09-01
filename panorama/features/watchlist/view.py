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
        
        # Заглушки для обратной совместимости
        self.watchlist_table = None
        self.lbl_watchlist_count = None
        self.rssi_table = None
        
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
        
        # Основная панель - только объединенный интерфейс
        main_panel = self._create_watchlist_panel()
        layout.addWidget(main_panel)
        
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


    def _create_watchlist_panel(self) -> QWidget:
        """Создает панель watchlist и задач."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        tabs = QTabWidget()
        
        # Объединенная вкладка Watchlist + RSSI
        combined_tab = self._create_combined_watchlist_tab()
        tabs.addTab(combined_tab, "📡 Измерения")
        
        # Вкладка задач
        tasks_tab = self._create_tasks_tab()
        tabs.addTab(tasks_tab, "📋 Задачи")
        
        # Вкладка координат слейвов
        coordinates_tab = self._create_coordinates_tab()
        tabs.addTab(coordinates_tab, "📍 Координаты")
        
        layout.addWidget(tabs)
        return widget

    def _create_combined_watchlist_tab(self) -> QWidget:
        """Создает объединенную вкладку watchlist с RSSI."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Информационная панель
        info = QLabel(
            "📍 Диапазоны добавляются автоматически при обнаружении сигналов. "
            "Таблица показывает RSSI от каждого слейва и результаты трилатерации."
        )
        info.setWordWrap(True)
        info.setStyleSheet("""
            QLabel {
                background-color: rgba(100, 100, 255, 30);
                padding: 8px;
                border-radius: 4px;
                margin-bottom: 5px;
            }
        """)
        layout.addWidget(info)
        
        # Элементы управления
        controls = QHBoxLayout()
        
        # Фильтр диапазонов  
        self.range_filter = QComboBox()
        self.range_filter.addItem("Все диапазоны")
        self.range_filter.currentTextChanged.connect(self._filter_combined_table)
        controls.addWidget(QLabel("Фильтр:"))
        controls.addWidget(self.range_filter)
        
        # Порог RSSI
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(-120, 0)
        self.threshold_spin.setValue(-80)
        self.threshold_spin.setSuffix(" дБм")
        self.threshold_spin.valueChanged.connect(self._update_combined_colors)
        controls.addWidget(QLabel("Порог:"))
        controls.addWidget(self.threshold_spin)
        
        # Автообновление
        self.auto_update_cb = QCheckBox("Автообновление")
        self.auto_update_cb.setChecked(True)
        controls.addWidget(self.auto_update_cb)
        
        controls.addStretch()
        
        # Кнопки управления
        self.btn_clear_combined = QPushButton("🗑️ Очистить")
        self.btn_clear_combined.clicked.connect(self._clear_combined_data)
        controls.addWidget(self.btn_clear_combined)
        
        layout.addLayout(controls)
        
        # Объединенная таблица
        self.combined_table = QTableWidget()
        self.combined_table.setColumnCount(12)
        self.combined_table.setHorizontalHeaderLabels([
            "Диапазон (МГц)", "Центр (МГц)", "Ширина",
            "Slave0 (дБм)", "Slave1 (дБм)", "Slave2 (дБм)", 
            "X", "Y", "Доверие", "Время", "На карту", "Статус"
        ])
        self.combined_table.setAlternatingRowColors(True)
        
        # Настройка столбцов
        header = self.combined_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Диапазон
        
        layout.addWidget(self.combined_table)
        
        # Статистика
        stats_layout = QHBoxLayout()
        self.lbl_combined_count = QLabel("Записей: 0")
        self.lbl_avg_rssi_combined = QLabel("Сред. RSSI: — дБм")
        self.lbl_active_ranges = QLabel("Активных: 0")
        
        for lbl in [self.lbl_combined_count, self.lbl_avg_rssi_combined, self.lbl_active_ranges]:
            stats_layout.addWidget(lbl)
        
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
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

    def _create_coordinates_tab(self) -> QWidget:
        """Создает вкладку настройки координат слейвов."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Информация
        info = QLabel(
            "⚙️ Управление координатами и ролями SDR устройств для трилатерации.\n"
            "Первое устройство в списке автоматически становится опорным (0, 0, 0).\n"
            "Если устройства не настроены в диспетчере, таблица будет пустой."
        )
        info.setWordWrap(True)
        info.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 165, 0, 30);
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(info)
        
        # Таблица координат
        self.coordinates_table = QTableWidget()
        self.coordinates_table.setColumnCount(6)
        self.coordinates_table.setHorizontalHeaderLabels([
            "Никнейм", "Роль", "X (метры)", "Y (метры)", "Z (метры)", "Статус"
        ])
        self.coordinates_table.setAlternatingRowColors(True)
        
        # Настройка столбцов
        header = self.coordinates_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(self.coordinates_table)
        
        # Инициализация координат
        self._initialize_coordinates_table()
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        
        self.btn_save_coords = QPushButton("💾 Сохранить координаты")
        self.btn_save_coords.clicked.connect(self._save_coordinates)
        buttons_layout.addWidget(self.btn_save_coords)
        
        self.btn_reset_coords = QPushButton("🔄 Сброс по умолчанию")
        self.btn_reset_coords.clicked.connect(self._reset_coordinates)
        buttons_layout.addWidget(self.btn_reset_coords)
        
        buttons_layout.addStretch()
        
        # Визуализация расположения
        self.btn_show_layout = QPushButton("👁️ Показать на карте")
        self.btn_show_layout.clicked.connect(self._show_slaves_on_map)
        buttons_layout.addWidget(self.btn_show_layout)
        
        layout.addLayout(buttons_layout)
        
        return widget

    def _create_status_bar(self) -> QWidget:
        """Создает статус бар."""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        layout = QHBoxLayout(widget)
        
        self.lbl_last_update = QLabel("Обновлено: —")
        layout.addWidget(self.lbl_last_update)
        
        return widget


    def update_rssi_value(self, range_str: str, slave_id: str, rssi_rms: float):
        """Обновляет значение RSSI в объединенной таблице."""
        # Теперь работаем только с объединенной таблицей
        self.update_combined_rssi(range_str, slave_id, rssi_rms)

    def _get_rssi_color(self, rssi: float) -> QColor:
        """Цвет для RSSI."""
        # Используем фиксированные пороги, так как threshold_spin теперь в объединенной таблице
        threshold = getattr(self, 'threshold_spin', None)
        if threshold:
            threshold_val = threshold.value()
        else:
            threshold_val = -80  # Значение по умолчанию
        
        if rssi >= threshold_val + 20:
            return QColor(74, 222, 128, 100)  # Зеленый
        elif rssi >= threshold_val + 10:
            return QColor(251, 191, 36, 100)  # Желтый
        else:
            return QColor(248, 113, 113, 100) # Красный

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
        """Отрисовывает watchlist в объединенной таблице."""
        # Работаем только с объединенной таблицей
        self._update_combined_from_watchlist(watchlist_data)
    
    def _update_combined_from_watchlist(self, watchlist_data: List[Dict]):
        """Обновляет объединенную таблицу данными из watchlist."""
        for data in watchlist_data:
            try:
                freq = float(data.get('freq', 0))
                span = float(data.get('span', 2.0))
                
                # Создаем диапазон
                freq_start = freq - span/2
                freq_end = freq + span/2
                range_str = f"{freq_start:.1f}-{freq_end:.1f}"
                
                # Находим или создаем строку
                row = self._find_or_create_combined_row(range_str, freq)
                
                if row >= 0:
                    # Обновляем ширину
                    self.combined_table.setItem(row, 2, QTableWidgetItem(f"{span:.1f}"))
                    
                    # Обновляем RSSI для каждого slave
                    for i in range(3):
                        rms_key = f'rms_{i+1}'
                        val = data.get(rms_key)
                        
                        if val is not None:
                            rssi_val = float(val)
                            slave_id = f"slave{i}"
                            col = 3 + i  # Колонки RSSI в объединенной таблице
                            
                            item = QTableWidgetItem(f"{rssi_val:.1f}")
                            item.setTextAlignment(Qt.AlignCenter)
                            item.setBackground(QBrush(self._get_rssi_color(rssi_val)))
                            
                            # Tooltip с дополнительной информацией
                            bins_used = data.get(f'bins_used_{i+1}', 'N/A')
                            timestamp = data.get(f'timestamp_{i+1}', '')
                            item.setToolTip(f"Slave: {slave_id}\nБинов: {bins_used}\nВремя: {timestamp}")
                            
                            self.combined_table.setItem(row, col, item)
                    
                    # Обновляем время
                    updated_time = data.get('updated', time.strftime('%H:%M:%S'))
                    self.combined_table.setItem(row, 9, QTableWidgetItem(updated_time))
                    
                    # Кнопка на карту
                    btn = QPushButton("📍")
                    btn.clicked.connect(lambda _, d=data: self._send_to_map(d))
                    self.combined_table.setCellWidget(row, 10, btn)
                    
                    # Статус - переводим на русский
                    has_measurements = any(data.get(f'rms_{i+1}') for i in range(3))
                    status = "ИЗМЕРЕНИЕ" if has_measurements else "ОЖИДАНИЕ"
                    self.combined_table.setItem(row, 11, QTableWidgetItem(status))
            
            except Exception as e:
                print(f"[SlavesView] Error updating combined row: {e}")
        
        # Обновляем статистику
        self._update_combined_stats()

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
                 if t.get('status') in ['PENDING', 'RUNNING', 'ОЖИДАНИЕ', 'ВЫПОЛНЕНИЕ']]
        self.tasks_table.setRowCount(len(active))
        
        stats = {'pending': 0, 'running': 0, 'completed': 0}
        
        for row, task in enumerate(active):
            # Заполняем колонки
            self.tasks_table.setItem(row, 0, 
                QTableWidgetItem(task.get('id', '')))
            self.tasks_table.setItem(row, 1, 
                QTableWidgetItem(task.get('range', '')))
            
            # Статус - переводим на русский
            status = task.get('status', '')
            
            # Переводим статус на русский
            if status == 'RUNNING':
                status_text = 'ВЫПОЛНЕНИЕ'
                status_item = QTableWidgetItem(status_text)
                status_item.setBackground(QBrush(QColor(74, 222, 128, 100)))
                stats['running'] += 1
            elif status == 'PENDING':
                status_text = 'ОЖИДАНИЕ'
                status_item = QTableWidgetItem(status_text)
                status_item.setBackground(QBrush(QColor(251, 191, 36, 100)))
                stats['pending'] += 1
            elif status == 'COMPLETED':
                status_text = 'ЗАВЕРШЕНО'
                status_item = QTableWidgetItem(status_text)
                status_item.setBackground(QBrush(QColor(200, 200, 200, 100)))
                stats['completed'] += 1
            else:
                status_text = status  # Используем оригинальный статус если не переведен
                status_item = QTableWidgetItem(status_text)
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
        """Добавляет результат трилатерации в объединенную таблицу."""
        try:
            # Находим или создаем строку для данного диапазона
            peak_id = getattr(result, 'peak_id', 'unknown')
            freq = getattr(result, 'freq_mhz', 0.0)
            x = getattr(result, 'x', 0.0)
            y = getattr(result, 'y', 0.0)
            confidence = getattr(result, 'confidence', 0.0)
            
            # Ищем существующую строку по частоте
            range_str = f"{freq-1.0:.1f}-{freq+1.0:.1f}"  # Примерный диапазон
            row = self._find_or_create_combined_row(range_str, freq)
            
            if row >= 0:
                # Обновляем колонки с результатами трилатерации
                self.combined_table.setItem(row, 6, QTableWidgetItem(f"{x:.1f}"))  # X
                self.combined_table.setItem(row, 7, QTableWidgetItem(f"{y:.1f}"))  # Y
                self.combined_table.setItem(row, 8, QTableWidgetItem(f"{confidence*100:.0f}%"))  # Доверие
                self.combined_table.setItem(row, 9, QTableWidgetItem(time.strftime("%H:%M:%S")))  # Время
                
                # Кнопка на карту
                btn = QPushButton("📍")
                btn.clicked.connect(lambda: self.send_to_map.emit({
                    'id': peak_id, 'freq': freq, 'x': x, 'y': y
                }))
                self.combined_table.setCellWidget(row, 10, btn)
                
                # Статус
                self.combined_table.setItem(row, 11, QTableWidgetItem("ОБНАРУЖЕН"))
                
                self._update_combined_stats()
            
        except Exception as e:
            print(f"[SlavesView] Error adding transmitter: {e}")
    
    def _find_or_create_combined_row(self, range_str: str, center_freq: float) -> int:
        """Находит или создает строку в объединенной таблице."""
        # Ищем существующую строку
        for row in range(self.combined_table.rowCount()):
            item = self.combined_table.item(row, 0)
            if item and item.text() == range_str:
                return row
        
        # Создаем новую строку
        row = self.combined_table.rowCount()
        self.combined_table.insertRow(row)
        
        self.combined_table.setItem(row, 0, QTableWidgetItem(range_str))  # Диапазон
        self.combined_table.setItem(row, 1, QTableWidgetItem(f"{center_freq:.1f}"))  # Центр
        self.combined_table.setItem(row, 2, QTableWidgetItem("2.0"))  # Ширина по умолчанию
        
        # Инициализируем RSSI колонки
        for col in range(3, 6):
            self.combined_table.setItem(row, col, QTableWidgetItem("—"))
        
        return row

    def _send_to_map(self, data):
        """Отправляет на карту."""
        self.send_to_map.emit(data)

    def _clear_watchlist(self):
        """Очищает watchlist (в объединенной таблице)."""
        # Очищаем объединенную таблицу
        self._clear_combined_data()

    def _refresh_data(self):
        """Ручное обновление."""
        self._update_data()

    def _clear_data(self):
        """Очищает все данные."""
        if hasattr(self, 'tasks_table') and self.tasks_table:
            self.tasks_table.setRowCount(0)
            
        if hasattr(self, 'combined_table') and self.combined_table:
            self.combined_table.setRowCount(0)
            
        if hasattr(self, 'tasks_log_view') and self.tasks_log_view:
            self.tasks_log_view.clear()
            
        self._update_combined_stats()
    
    # Новые методы для объединенной таблицы
    def _filter_combined_table(self):
        """Фильтрует объединенную таблицу."""
        filter_text = self.range_filter.currentText()
        
        for row in range(self.combined_table.rowCount()):
            item = self.combined_table.item(row, 0)
            if item:
                if filter_text == "Все диапазоны":
                    self.combined_table.setRowHidden(row, False)
                else:
                    self.combined_table.setRowHidden(row, item.text() != filter_text)
    
    def _update_combined_colors(self):
        """Обновляет цвета RSSI в объединенной таблице."""
        threshold = self.threshold_spin.value()
        
        for row in range(self.combined_table.rowCount()):
            for col in range(3, 6):  # RSSI колонки
                item = self.combined_table.item(row, col)
                if item and item.text() != "—":
                    try:
                        rssi = float(item.text())
                        item.setBackground(QBrush(self._get_rssi_color(rssi)))
                    except:
                        pass
    
    def _clear_combined_data(self):
        """Очищает объединенную таблицу."""
        self.combined_table.setRowCount(0)
        self._update_combined_stats()
    
    def _update_combined_stats(self):
        """Обновляет статистику объединенной таблицы."""
        total_count = self.combined_table.rowCount()
        active_count = 0
        all_rssi = []
        
        for row in range(total_count):
            # Проверяем активность (есть ли RSSI данные)
            has_rssi = False
            for col in range(3, 6):
                item = self.combined_table.item(row, col)
                if item and item.text() != "—":
                    has_rssi = True
                    try:
                        all_rssi.append(float(item.text()))
                    except:
                        pass
            
            if has_rssi:
                active_count += 1
        
        self.lbl_combined_count.setText(f"Записей: {total_count}")
        self.lbl_active_ranges.setText(f"Активных: {active_count}")
        
        if all_rssi:
            avg_rssi = np.mean(all_rssi)
            self.lbl_avg_rssi_combined.setText(f"Сред. RSSI: {avg_rssi:.1f} дБм")
        else:
            self.lbl_avg_rssi_combined.setText("Сред. RSSI: — дБм")
    
    # Методы для работы с координатами
    def _initialize_coordinates_table(self):
        """Инициализирует таблицу координат."""
        # Если нет реальных SDR устройств, таблица остается пустой
        self.coordinates_table.setRowCount(0)
        
        # Добавляем информационное сообщение
        if self.coordinates_table.rowCount() == 0:
            # Создаем строку с информацией
            self.coordinates_table.setRowCount(1)
            info_item = QTableWidgetItem("Нет настроенных SDR устройств")
            info_item.setTextAlignment(Qt.AlignCenter)
            info_item.setFlags(Qt.NoItemFlags)  # Неселектируемый
            info_item.setBackground(QBrush(QColor(240, 240, 240, 100)))
            
            # Объединяем все колонки для сообщения
            self.coordinates_table.setItem(0, 0, info_item)
            for col in range(1, 6):
                empty_item = QTableWidgetItem("")
                empty_item.setFlags(Qt.NoItemFlags)
                empty_item.setBackground(QBrush(QColor(240, 240, 240, 100)))
                self.coordinates_table.setItem(0, col, empty_item)
            
            # Объединяем ячейки для информационного сообщения
            self.coordinates_table.setSpan(0, 0, 1, 6)
    
    def _save_coordinates(self):
        """Сохраняет координаты и роли SDR устройств."""
        try:
            devices_config = []
            
            for row in range(self.coordinates_table.rowCount()):
                # Проверяем, что это не информационная строка
                nickname_item = self.coordinates_table.item(row, 0)
                if not nickname_item or nickname_item.flags() == Qt.NoItemFlags:
                    continue
                
                nickname = nickname_item.text()
                
                # Получаем роль из комбобокса
                role_widget = self.coordinates_table.cellWidget(row, 1)
                role = role_widget.currentText() if role_widget else "Измерительное"
                
                # Получаем координаты
                x_item = self.coordinates_table.item(row, 2)
                y_item = self.coordinates_table.item(row, 3)
                z_item = self.coordinates_table.item(row, 4)
                
                if x_item and y_item and z_item:
                    try:
                        x = float(x_item.text())
                        y = float(y_item.text())
                        z = float(z_item.text())
                    except ValueError:
                        x, y, z = 0.0, 0.0, 0.0
                else:
                    x, y, z = 0.0, 0.0, 0.0
                
                device_config = {
                    "nickname": nickname,
                    "role": role,
                    "x": x,
                    "y": y,
                    "z": z,
                    "is_reference": role == "Опорное"
                }
                
                devices_config.append(device_config)
            
            # Эмитируем сигнал для отправки на карту
            self._update_map_with_coordinates(devices_config)
            
            print(f"[SlavesView] Saving configuration for {len(devices_config)} devices:")
            for device in devices_config:
                print(f"  - {device['nickname']}: {device['role']} at ({device['x']}, {device['y']}, {device['z']})")
            
            # TODO: Интеграция с системой конфигурации
            # if hasattr(self, 'orchestrator') and self.orchestrator:
            #     self.orchestrator.update_devices_configuration(devices_config)
            
        except Exception as e:
            print(f"[SlavesView] Error saving coordinates: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_map_with_coordinates(self, devices_config):
        """Обновляет карту с новыми координатами устройств."""
        try:
            slaves_data = []
            
            for device in devices_config:
                slaves_data.append({
                    'id': device['nickname'],
                    'x': device['x'], 
                    'y': device['y'],
                    'z': device['z'],
                    'type': 'sdr_device',
                    'role': device['role'],
                    'is_reference': device['is_reference']
                })
            
            # Эмитируем сигнал для отправки на карту
            self.send_to_map.emit({
                'type': 'update_devices_coordinates',
                'devices': slaves_data
            })
            
            print(f"[SlavesView] Sent {len(slaves_data)} device coordinates to map")
            
        except Exception as e:
            print(f"[SlavesView] Error updating map with coordinates: {e}")
    
    def _reset_coordinates(self):
        """Сбрасывает координаты по умолчанию."""
        self._initialize_coordinates_table()
    
    def _show_slaves_on_map(self):
        """Показывает расположение SDR устройств на карте."""
        try:
            devices_data = []
            
            for row in range(self.coordinates_table.rowCount()):
                # Проверяем, что это не информационная строка
                nickname_item = self.coordinates_table.item(row, 0)
                if not nickname_item or nickname_item.flags() == Qt.NoItemFlags:
                    continue
                
                nickname = nickname_item.text()
                
                # Получаем роль
                role_widget = self.coordinates_table.cellWidget(row, 1)
                role = role_widget.currentText() if role_widget else "Измерительное"
                
                # Получаем координаты
                x_item = self.coordinates_table.item(row, 2)
                y_item = self.coordinates_table.item(row, 3)
                z_item = self.coordinates_table.item(row, 4)
                
                if x_item and y_item and z_item:
                    try:
                        x = float(x_item.text())
                        y = float(y_item.text())
                        z = float(z_item.text())
                    except ValueError:
                        x, y, z = 0.0, 0.0, 0.0
                else:
                    continue
                
                devices_data.append({
                    'id': nickname,
                    'x': x, 'y': y, 'z': z,
                    'type': 'sdr_device',
                    'role': role,
                    'is_reference': role == "Опорное"
                })
            
            # Эмитируем сигнал для отправки на карту
            self.send_to_map.emit({
                'type': 'devices_layout',
                'devices': devices_data
            })
            
            print(f"[SlavesView] Showing {len(devices_data)} devices on map")
            
        except Exception as e:
            print(f"[SlavesView] Error showing devices on map: {e}")
    
    def update_combined_rssi(self, range_str: str, slave_id: str, rssi_rms: float):
        """Обновляет RSSI в объединенной таблице."""
        try:
            # Находим строку с данным диапазоном
            row = -1
            for r in range(self.combined_table.rowCount()):
                item = self.combined_table.item(r, 0)
                if item and item.text() == range_str:
                    row = r
                    break
            
            if row == -1:
                # Создаем новую строку
                center_freq = sum(float(x) for x in range_str.split('-')) / 2
                row = self._find_or_create_combined_row(range_str, center_freq)
            
            # Определяем колонку по slave_id
            col_map = {"slave0": 3, "slave1": 4, "slave2": 5}
            col = col_map.get(slave_id.lower(), -1)
            
            if col > 0 and row >= 0:
                item = QTableWidgetItem(f"{rssi_rms:.1f}")
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(QBrush(self._get_rssi_color(rssi_rms)))
                self.combined_table.setItem(row, col, item)
                
                # Обновляем статистику
                self._update_combined_stats()
                
                # Добавляем в фильтр, если нужно
                if range_str not in [self.range_filter.itemText(i) 
                                   for i in range(self.range_filter.count())]:
                    self.range_filter.addItem(range_str)
        
        except Exception as e:
            print(f"[SlavesView] Error updating combined RSSI: {e}")
            
    
    # Дополнительные методы для удобства
    def manual_refresh(self):
        """Ручное обновление данных (вызывается из главного окна)."""
        self._refresh_data()
    
    def export_current_state(self):
        """Экспорт текущего состояния (заглушка для интеграции)."""
        try:
            import json
            from PyQt5.QtWidgets import QFileDialog, QMessageBox
            
            # Собираем данные для экспорта
            export_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'rssi_data': [],
                'coordinates': {},
                'combined_data': []
            }
            
            # RSSI данные
            for row in range(self.combined_table.rowCount()):
                range_item = self.combined_table.item(row, 0)
                if range_item:
                    row_data = {
                        'range': range_item.text(),
                        'center_freq': self.combined_table.item(row, 1).text() if self.combined_table.item(row, 1) else '',
                        'rssi_slave0': self.combined_table.item(row, 3).text() if self.combined_table.item(row, 3) else '—',
                        'rssi_slave1': self.combined_table.item(row, 4).text() if self.combined_table.item(row, 4) else '—',
                        'rssi_slave2': self.combined_table.item(row, 5).text() if self.combined_table.item(row, 5) else '—',
                        'x': self.combined_table.item(row, 6).text() if self.combined_table.item(row, 6) else '',
                        'y': self.combined_table.item(row, 7).text() if self.combined_table.item(row, 7) else '',
                        'confidence': self.combined_table.item(row, 8).text() if self.combined_table.item(row, 8) else '',
                    }
                    export_data['combined_data'].append(row_data)
            
            # Координаты слейвов
            for row in range(self.coordinates_table.rowCount()):
                slave_id = self.coordinates_table.item(row, 0).text()
                export_data['coordinates'][slave_id] = {
                    'x': float(self.coordinates_table.item(row, 1).text()),
                    'y': float(self.coordinates_table.item(row, 2).text()),
                    'z': float(self.coordinates_table.item(row, 3).text()),
                    'status': self.coordinates_table.item(row, 4).text()
                }
            
            # Диалог сохранения
            filename, _ = QFileDialog.getSaveFileName(
                self, 
                "Экспорт состояния слейвов", 
                f"slaves_state_{time.strftime('%Y%m%d_%H%M%S')}.json",
                "JSON files (*.json)"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                
                QMessageBox.information(self, "Экспорт завершен", 
                                      f"Состояние экспортировано в файл:\n{filename}")
                print(f"[SlavesView] State exported to: {filename}")
                
        except Exception as e:
            print(f"[SlavesView] Export error: {e}")
            if 'QMessageBox' in globals():
                QMessageBox.critical(self, "Ошибка экспорта", f"Не удалось экспортировать данные:\n{e}")
    
    def clear_all_data(self):
        """Очищает все данные (вызывается из главного окна)."""
        self._clear_data()
    
    def update_transmitter_position(self, transmitter_data):
        """Обновляет позицию передатчика (для совместимости)."""
        # Обновляем существующую запись в объединенной таблице
        try:
            freq = transmitter_data.get('freq_mhz', 0.0)
            x = transmitter_data.get('x', 0.0)
            y = transmitter_data.get('y', 0.0)
            confidence = transmitter_data.get('confidence', 0.0)
            
            # Ищем строку по частоте
            for row in range(self.combined_table.rowCount()):
                freq_item = self.combined_table.item(row, 1)
                if freq_item and abs(float(freq_item.text()) - freq) < 0.1:  # Tolerance 100kHz
                    # Обновляем координаты
                    self.combined_table.setItem(row, 6, QTableWidgetItem(f"{x:.1f}"))
                    self.combined_table.setItem(row, 7, QTableWidgetItem(f"{y:.1f}"))
                    self.combined_table.setItem(row, 8, QTableWidgetItem(f"{confidence*100:.0f}%"))
                    self.combined_table.setItem(row, 9, QTableWidgetItem(time.strftime("%H:%M:%S")))
                    
                    # Обновляем статус
                    self.combined_table.setItem(row, 11, QTableWidgetItem("ОТСЛЕЖЕН"))
                    break
                    
        except Exception as e:
            print(f"[SlavesView] Error updating transmitter position: {e}")
    
    def update_available_devices(self, devices_data: list):
        """Обновляет список доступных устройств из диспетчера."""
        try:
            # Обновляем таблицу координат с доступными устройствами
            if hasattr(self, 'coordinates_table'):
                # Сохраняем существующие координаты
                existing_coords = {}
                for row in range(self.coordinates_table.rowCount()):
                    slave_id = self.coordinates_table.item(row, 0).text()
                    x = float(self.coordinates_table.item(row, 1).text())
                    y = float(self.coordinates_table.item(row, 2).text())
                    z = float(self.coordinates_table.item(row, 3).text())
                    existing_coords[slave_id] = (x, y, z)
                
                # Обновляем таблицу с новыми устройствами
                devices_to_show = []
                
                # Всегда добавляем slave0 как опорную точку
                devices_to_show.append({
                    'id': 'slave0', 
                    'coords': existing_coords.get('slave0', (0.0, 0.0, 0.0)),
                    'status': 'REFERENCE'
                })
                
                # Добавляем остальные устройства
                for i, device in enumerate(devices_data[:6], 1):  # Максимум 6 дополнительных устройств
                    slave_id = f"slave{i}"
                    nickname = getattr(device, 'nickname', f'Slave{i}')
                    
                    devices_to_show.append({
                        'id': slave_id,
                        'nickname': nickname, 
                        'coords': existing_coords.get(slave_id, (10.0*i, 0.0, 0.0)),
                        'status': 'AVAILABLE' if getattr(device, 'is_available', True) else 'UNAVAILABLE'
                    })
                
                self._update_coordinates_table_with_devices(devices_to_show)
                print(f"[SlavesView] Updated coordinates table with {len(devices_data)} devices")
                
        except Exception as e:
            print(f"[SlavesView] Error updating available devices: {e}")
    
    def _update_coordinates_table_with_devices(self, devices_list):
        """Обновляет таблицу координат с реальными SDR устройствами."""
        try:
            # Очищаем информацию о span если была
            self.coordinates_table.clearSpans()
            
            # Если нет устройств, показываем сообщение
            if not devices_list:
                self._initialize_coordinates_table()
                return
            
            self.coordinates_table.setRowCount(len(devices_list))
            
            # Определяем опорное устройство (первое в списке)
            reference_device = devices_list[0] if devices_list else None
            
            for row, device in enumerate(devices_list):
                is_reference = (row == 0)  # Первое устройство - опорное
                
                # Никнейм устройства
                nickname = device.get('nickname', f"SDR-{device.get('serial', 'Unknown')[-4:]}")
                nickname_item = QTableWidgetItem(nickname)
                
                if is_reference:
                    nickname_item.setBackground(QBrush(QColor(255, 215, 0, 100)))  # Золотой для опорного
                    nickname_item.setToolTip("Опорное устройство (0,0,0)")
                
                self.coordinates_table.setItem(row, 0, nickname_item)
                
                # Роль устройства - выпадающий список
                role_combo = QComboBox()
                role_combo.addItems([
                    "Опорное", "Измерительное", "Резервное", "Отключено"
                ])
                
                if is_reference:
                    role_combo.setCurrentText("Опорное")
                    role_combo.setEnabled(False)  # Опорное устройство нельзя изменить
                    role_combo.setToolTip("Первое устройство всегда опорное")
                else:
                    role_combo.setCurrentText("Измерительное")
                
                # Сохраняем ссылку на устройство в комбобоксе
                role_combo.setProperty('device_data', device)
                role_combo.currentTextChanged.connect(self._on_role_changed)
                
                self.coordinates_table.setCellWidget(row, 1, role_combo)
                
                # Координаты
                x, y, z = device.get('coords', (0.0, 0.0, 0.0))
                
                # Для опорного устройства координаты всегда (0,0,0)
                if is_reference:
                    x, y, z = 0.0, 0.0, 0.0
                
                x_item = QTableWidgetItem(f"{x:.1f}")
                y_item = QTableWidgetItem(f"{y:.1f}")  
                z_item = QTableWidgetItem(f"{z:.1f}")
                
                # Опорное устройство не редактируется
                if is_reference:
                    for item in [x_item, y_item, z_item]:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        item.setBackground(QBrush(QColor(200, 200, 200, 100)))
                        item.setToolTip("Опорное устройство имеет фиксированные координаты (0,0,0)")
                
                self.coordinates_table.setItem(row, 2, x_item)
                self.coordinates_table.setItem(row, 3, y_item)
                self.coordinates_table.setItem(row, 4, z_item)
                
                # Статус устройства - переводим на русский
                status = device.get('status', 'UNKNOWN')
                if status == 'REFERENCE' or is_reference:
                    status_text = 'ОПОРНОЕ'
                    status_color = QColor(255, 215, 0, 100)  # Золотой
                elif status == 'AVAILABLE' or status == 'ACTIVE':
                    status_text = 'ДОСТУПНО'
                    status_color = QColor(74, 222, 128, 100)  # Зеленый
                elif status == 'UNAVAILABLE':
                    status_text = 'НЕДОСТУПНО'
                    status_color = QColor(248, 113, 113, 100)  # Красный
                else:
                    status_text = 'НЕИЗВЕСТНО'
                    status_color = QColor(200, 200, 200, 100)  # Серый
                
                status_item = QTableWidgetItem(status_text)
                status_item.setBackground(QBrush(status_color))
                self.coordinates_table.setItem(row, 5, status_item)
                
                print(f"[SlavesView] Added device {nickname} as {'reference' if is_reference else 'measurement'}")
                
        except Exception as e:
            print(f"[SlavesView] Error updating coordinates table: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_role_changed(self):
        """Обрабатывает изменение роли устройства."""
        try:
            sender = self.sender()  # QComboBox который изменился
            if sender:
                device_data = sender.property('device_data')
                new_role = sender.currentText()
                
                if device_data:
                    nickname = device_data.get('nickname', 'Unknown')
                    print(f"[SlavesView] Role changed for {nickname}: {new_role}")
                    
                    # Здесь можно добавить логику обработки изменения роли
                    # например, обновить статус устройства или сохранить в конфигурацию
                    
                    # Автоматически сохраняем изменения
                    self._save_coordinates()
                    
        except Exception as e:
            print(f"[SlavesView] Error handling role change: {e}")