#!/usr/bin/env python3
"""
Виджет для отображения и управления watchlist
Показывает активные задачи, их статус и результаты измерений
"""

from __future__ import annotations
from typing import Optional, Dict, List
import time
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QGroupBox, QPushButton, QLabel, QHeaderView, QSplitter, QTextEdit,
    QComboBox, QSpinBox, QFormLayout, QCheckBox
)

from panorama.features.orchestrator.core import Orchestrator, MeasurementTask


class WatchlistView(QWidget):
    """Виджет для отображения watchlist и управления задачами."""
    
    # Сигналы для внешних обработчиков
    task_selected = pyqtSignal(object)  # MeasurementTask
    task_cancelled = pyqtSignal(str)    # task_id
    task_retried = pyqtSignal(str)      # task_id
    
    def __init__(self, orchestrator: Optional[Orchestrator] = None, parent=None):
        super().__init__(parent)
        
        self.orchestrator = orchestrator
        self.tasks: Dict[str, MeasurementTask] = {}
        
        # Таймер обновления
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(1000)  # Обновляем каждую секунду
        
        self._setup_ui()
        self._connect_orchestrator()
        
    def _setup_ui(self):
        """Настраивает пользовательский интерфейс."""
        layout = QVBoxLayout(self)
        
        # Заголовок и статистика
        header_layout = QHBoxLayout()
        
        # Статистика
        self.stats_label = QLabel("Статистика: 0 задач, 0 активных, 0 завершено")
        self.stats_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        header_layout.addWidget(self.stats_label)
        
        header_layout.addStretch()
        
        # Кнопки управления
        self.refresh_btn = QPushButton("Обновить")
        self.refresh_btn.clicked.connect(self._refresh_data)
        header_layout.addWidget(self.refresh_btn)
        
        self.clear_completed_btn = QPushButton("Очистить завершенные")
        self.clear_completed_btn.clicked.connect(self._clear_completed)
        header_layout.addWidget(self.clear_completed_btn)
        
        layout.addLayout(header_layout)
        
        # Основной сплиттер
        splitter = QSplitter(QtCore.Qt.Vertical)
        
        # Таблица задач
        tasks_group = QGroupBox("Задачи Watchlist")
        tasks_layout = QVBoxLayout(tasks_group)
        
        # Фильтры
        filter_layout = QHBoxLayout()
        
        self.status_filter = QComboBox()
        self.status_filter.addItems(["Все", "PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"])
        self.status_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(QLabel("Статус:"))
        filter_layout.addWidget(self.status_filter)
        
        self.freq_filter = QSpinBox()
        self.freq_filter.setRange(0, 10000)
        self.freq_filter.setSuffix(" МГц")
        self.freq_filter.setValue(0)
        self.freq_filter.setSpecialValueText("Все частоты")
        self.freq_filter.valueChanged.connect(self._apply_filters)
        filter_layout.addWidget(QLabel("Частота:"))
        filter_layout.addWidget(self.freq_filter)
        
        filter_layout.addStretch()
        
        tasks_layout.addLayout(filter_layout)
        
        # Таблица задач
        self.tasks_table = QTableWidget()
        self.tasks_table.setColumnCount(8)
        self.tasks_table.setHorizontalHeaderLabels([
            "ID", "Частота (МГц)", "Span (МГц)", "Dwell (мс)", 
            "Статус", "Создана", "Завершена", "Измерения"
        ])
        
        # Настройка таблицы
        header = self.tasks_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # ID
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Частота
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Span
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Dwell
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Статус
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Создана
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Завершена
        header.setSectionResizeMode(7, QHeaderView.Stretch)           # Измерения
        
        self.tasks_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.tasks_table.itemSelectionChanged.connect(self._on_task_selected)
        
        tasks_layout.addWidget(self.tasks_table)
        
        # Кнопки управления задачами
        task_buttons_layout = QHBoxLayout()
        
        self.cancel_task_btn = QPushButton("Отменить задачу")
        self.cancel_task_btn.clicked.connect(self._cancel_selected_task)
        self.cancel_task_btn.setEnabled(False)
        task_buttons_layout.addWidget(self.cancel_task_btn)
        
        self.retry_task_btn = QPushButton("Повторить задачу")
        self.retry_task_btn.clicked.connect(self._retry_selected_task)
        self.retry_task_btn.setEnabled(False)
        task_buttons_layout.addWidget(self.retry_task_btn)
        
        task_buttons_layout.addStretch()
        
        tasks_layout.addLayout(task_buttons_layout)
        
        splitter.addWidget(tasks_group)
        
        # Детали задачи
        details_group = QGroupBox("Детали задачи")
        details_layout = QVBoxLayout(details_group)
        
        self.task_details = QTextEdit()
        self.task_details.setReadOnly(True)
        self.task_details.setMaximumHeight(200)
        details_layout.addWidget(self.task_details)
        
        splitter.addWidget(details_group)
        
        # Настройка сплиттера
        splitter.setSizes([400, 200])
        layout.addWidget(splitter)
        
    def _connect_orchestrator(self):
        """Подключает сигналы оркестратора."""
        if self.orchestrator:
            self.orchestrator.task_created.connect(self._on_task_created)
            self.orchestrator.task_completed.connect(self._on_task_completed)
            self.orchestrator.task_failed.connect(self._on_task_failed)
            self.orchestrator.status_changed.connect(self._on_status_changed)
            
    def set_orchestrator(self, orchestrator: Orchestrator):
        """Устанавливает оркестратор для отслеживания."""
        if self.orchestrator:
            # Отключаем старые сигналы
            try:
                self.orchestrator.task_created.disconnect(self._on_task_created)
                self.orchestrator.task_completed.disconnect(self._on_task_completed)
                self.orchestrator.task_failed.disconnect(self._on_task_failed)
                self.orchestrator.status_changed.disconnect(self._on_status_changed)
            except Exception:
                pass
                
        self.orchestrator = orchestrator
        self._connect_orchestrator()
        
    def _on_task_created(self, task: MeasurementTask):
        """Обработчик создания новой задачи."""
        self.tasks[task.id] = task
        self._update_display()
        
    def _on_task_completed(self, task: MeasurementTask):
        """Обработчик завершения задачи."""
        if task.id in self.tasks:
            self.tasks[task.id] = task
        self._update_display()
        
    def _on_task_failed(self, task: MeasurementTask):
        """Обработчик неудачи задачи."""
        if task.id in self.tasks:
            self.tasks[task.id] = task
        self._update_display()
        
    def _on_status_changed(self, status: dict):
        """Обработчик изменения статуса оркестратора."""
        # Обновляем статистику
        self._update_statistics()
        
    def _update_display(self):
        """Обновляет отображение таблицы."""
        if not self.orchestrator:
            return
            
        # Получаем актуальные данные
        self.tasks = self.orchestrator.tasks.copy()
        
        # Применяем фильтры
        filtered_tasks = self._filter_tasks()
        
        # Обновляем таблицу
        self.tasks_table.setRowCount(len(filtered_tasks))
        
        for row, (task_id, task) in enumerate(filtered_tasks.items()):
            # ID
            self.tasks_table.setItem(row, 0, QTableWidgetItem(task_id[:20]))
            
            # Частота
            freq_mhz = task.peak.f_peak / 1e6
            self.tasks_table.setItem(row, 1, QTableWidgetItem(f"{freq_mhz:.3f}"))
            
            # Span
            span_mhz = task.window.span / 1e6
            self.tasks_table.setItem(row, 2, QTableWidgetItem(f"{span_mhz:.3f}"))
            
            # Dwell
            self.tasks_table.setItem(row, 3, QTableWidgetItem(f"{task.window.dwell_ms}"))
            
            # Статус
            status_item = QTableWidgetItem(task.status)
            self._color_status_item(status_item, task.status)
            self.tasks_table.setItem(row, 4, status_item)
            
            # Время создания
            created_time = time.strftime("%H:%M:%S", time.localtime(task.created_at))
            self.tasks_table.setItem(row, 5, QTableWidgetItem(created_time))
            
            # Время завершения
            if task.completed_at:
                completed_time = time.strftime("%H:%M:%S", time.localtime(task.completed_at))
                self.tasks_table.setItem(row, 6, QTableWidgetItem(completed_time))
            else:
                self.tasks_table.setItem(row, 6, QTableWidgetItem("-"))
                
            # Количество измерений
            measurements_count = len(task.measurements) if task.measurements else 0
            self.tasks_table.setItem(row, 7, QTableWidgetItem(f"{measurements_count}"))
            
        # Обновляем статистику
        self._update_statistics()
        
    def _filter_tasks(self) -> Dict[str, MeasurementTask]:
        """Применяет фильтры к задачам."""
        filtered = {}
        
        status_filter = self.status_filter.currentText()
        freq_filter = self.freq_filter.value()
        
        for task_id, task in self.tasks.items():
            # Фильтр по статусу
            if status_filter != "Все" and task.status != status_filter:
                continue
                
            # Фильтр по частоте
            if freq_filter > 0:
                task_freq_mhz = task.peak.f_peak / 1e6
                if abs(task_freq_mhz - freq_filter) > 1.0:  # ±1 МГц
                    continue
                    
            filtered[task_id] = task
            
        return filtered
        
    def _color_status_item(self, item: QTableWidgetItem, status: str):
        """Раскрашивает ячейку статуса."""
        colors = {
            "PENDING": "#FF9800",    # Оранжевый
            "RUNNING": "#2196F3",    # Синий
            "COMPLETED": "#4CAF50",  # Зеленый
            "FAILED": "#F44336",     # Красный
            "CANCELLED": "#9E9E9E"   # Серый
        }
        
        if status in colors:
            item.setBackground(QtGui.QColor(colors[status]))
            item.setForeground(QtGui.QColor("white"))
            
    def _update_statistics(self):
        """Обновляет статистику."""
        total = len(self.tasks)
        pending = sum(1 for t in self.tasks.values() if t.status == "PENDING")
        running = sum(1 for t in self.tasks.values() if t.status == "RUNNING")
        completed = sum(1 for t in self.tasks.values() if t.status == "COMPLETED")
        failed = sum(1 for t in self.tasks.values() if t.status == "FAILED")
        
        stats_text = f"Статистика: {total} задач, {pending} ожидают, {running} активны, {completed} завершено, {failed} неудачно"
        self.stats_label.setText(stats_text)
        
    def _on_task_selected(self):
        """Обработчик выбора задачи."""
        current_row = self.tasks_table.currentRow()
        if current_row >= 0:
            task_id = self.tasks_table.item(current_row, 0).text()
            
            # Ищем полный ID (если был обрезан)
            full_task_id = None
            for tid in self.tasks.keys():
                if tid.startswith(task_id):
                    full_task_id = tid
                    break
                    
            if full_task_id and full_task_id in self.tasks:
                task = self.tasks[full_task_id]
                self._show_task_details(task)
                
                # Включаем кнопки управления
                self.cancel_task_btn.setEnabled(task.status in ["PENDING", "RUNNING"])
                self.retry_task_btn.setEnabled(task.status in ["FAILED", "CANCELLED"])
                
                # Эмитим сигнал
                self.task_selected.emit(task)
            else:
                self._clear_task_details()
                self.cancel_task_btn.setEnabled(False)
                self.retry_task_btn.setEnabled(False)
        else:
            self._clear_task_details()
            self.cancel_task_btn.setEnabled(False)
            self.retry_task_btn.setEnabled(False)
            
    def _show_task_details(self, task: MeasurementTask):
        """Показывает детали задачи."""
        details = f"""Задача: {task.id}

Частота: {task.peak.f_peak/1e6:.3f} МГц
SNR: {task.peak.snr_db:.1f} дБ
Мощность: {task.peak.power_dbm:.1f} дБм

Окно измерения:
  Центр: {task.window.center/1e6:.3f} МГц
  Span: {task.window.span/1e6:.3f} МГц
  Dwell: {task.window.dwell_ms} мс
  Epoch: {time.strftime('%H:%M:%S', time.localtime(task.window.epoch))}

Статус: {task.status}
Создана: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.created_at))}"""

        if task.completed_at:
            details += f"\nЗавершена: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.completed_at))}"
            
        if task.measurements:
            details += f"\n\nИзмерения ({len(task.measurements)}):"
            for i, meas in enumerate(task.measurements):
                details += f"\n  {i+1}. Slave: {meas.slave_id}, RSSI: {meas.rssi_db:.1f} дБ, SNR: {meas.snr_db:.1f} дБ"
                
        self.task_details.setText(details)
        
    def _clear_task_details(self):
        """Очищает детали задачи."""
        self.task_details.clear()
        
    def _cancel_selected_task(self):
        """Отменяет выбранную задачу."""
        current_row = self.tasks_table.currentRow()
        if current_row >= 0:
            task_id = self.tasks_table.item(current_row, 0).text()
            
            # Ищем полный ID
            full_task_id = None
            for tid in self.tasks.keys():
                if tid.startswith(task_id):
                    full_task_id = tid
                    break
                    
            if full_task_id:
                self.task_cancelled.emit(full_task_id)
                
    def _retry_selected_task(self):
        """Повторяет выбранную задачу."""
        current_row = self.tasks_table.currentRow()
        if current_row >= 0:
            task_id = self.tasks_table.item(current_row, 0).text()
            
            # Ищем полный ID
            full_task_id = None
            for tid in self.tasks.keys():
                if tid.startswith(task_id):
                    full_task_id = tid
                    break
                    
            if full_task_id:
                self.task_retried.emit(full_task_id)
                
    def _refresh_data(self):
        """Обновляет данные."""
        self._update_display()
        
    def _clear_completed(self):
        """Очищает завершенные задачи."""
        if self.orchestrator:
            # Очищаем завершенные задачи из оркестратора
            completed_ids = []
            for task_id, task in self.orchestrator.tasks.items():
                if task.status in ["COMPLETED", "FAILED", "CANCELLED"]:
                    completed_ids.append(task_id)
                    
            for task_id in completed_ids:
                if task_id in self.orchestrator.tasks:
                    del self.orchestrator.tasks[task_id]
                    
            # Очищаем из очереди завершенных
            self.orchestrator.completed.clear()
            
            self._update_display()
            
    def _apply_filters(self):
        """Применяет фильтры."""
        self._update_display()
