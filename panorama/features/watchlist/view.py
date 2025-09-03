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

from .web_table_widget import WebTableWidget


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
        
        # Веб-таблица для RSSI измерений
        self.web_table_widget = None
        
        # Заглушки для обратной совместимости
        self.watchlist_table = None
        self.lbl_watchlist_count = None
        self.rssi_table = None
        self.combined_table = None  # Будет заменена веб-таблицей
        
        # Создаем UI
        self._create_ui()
        
        # Таймер обновления (лайв-данные)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_data)
        self.update_timer.start(300)  # ~3-4 FPS для живых обновлений

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
        
        # Информационная панель удалена по требованиям
        
        # Управляющие элементы (фильтры/пороги/очистка) удалены — система работает автономно
        
        # Веб-таблица вместо QTableWidget для лучшей производительности
        self.web_table_widget = WebTableWidget()
        
        # Подключаем сигналы веб-таблицы
        self.web_table_widget.export_requested.connect(self._on_web_table_export)
        self.web_table_widget.map_navigate_requested.connect(self._on_web_table_map_navigate) 
        self.web_table_widget.clear_data_requested.connect(self._on_web_table_clear_data)
        
        layout.addWidget(self.web_table_widget)
        
        # Создаем заглушку для обратной совместимости
        self.combined_table = self._create_combined_table_proxy()
        
        # Статистика в нативной таблице скрыта — веб-таблица содержит собственные метрики
        
        return widget
    
    def _create_combined_table_proxy(self):
        """Создает прокси-объект для обратной совместимости с combined_table."""
        class CombinedTableProxy:
            def __init__(self, web_widget):
                self.web_widget = web_widget
                self._row_count = 0
            
            def rowCount(self):
                return self._row_count
            
            def setRowCount(self, count):
                self._row_count = count
                if count == 0:
                    self.web_widget.clear_all_data()
            
            def insertRow(self, row):
                self._row_count += 1
                return self._row_count - 1
            
            def setItem(self, row, col, item):
                # Заглушка - данные будут обновляться через веб-интерфейс
                pass
            
            def item(self, row, col):
                # Возвращаем заглушку
                class ItemProxy:
                    def __init__(self, text=""):
                        self._text = text
                    def text(self):
                        return self._text
                    def setText(self, text):
                        self._text = text
                return ItemProxy()
        
        return CombinedTableProxy(self.web_table_widget)
    
    def _on_web_table_export(self, format_name: str):
        """Обрабатывает запрос на экспорт из веб-таблицы."""
        try:
            # Экспортируем данные через веб-таблицу
            data = self.web_table_widget.export_data(format_name)
            
            # Сохраняем в файл
            from PyQt5.QtWidgets import QFileDialog, QMessageBox
            import json
            
            filename, _ = QFileDialog.getSaveFileName(
                self,
                f"Экспорт RSSI данных ({format_name})",
                f"rssi_data_{time.strftime('%Y%m%d_%H%M%S')}.json",
                "JSON files (*.json)"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                QMessageBox.information(
                    self, 
                    "Экспорт завершен",
                    f"Данные экспортированы в файл:\n{filename}"
                )
                
        except Exception as e:
            print(f"[SlavesView] Export error: {e}")
    
    def _on_web_table_map_navigate(self, lat: float, lng: float):
        """Обрабатывает запрос на навигацию по карте из веб-таблицы."""
        try:
            # Эмитируем сигнал для карты
            self.send_to_map.emit({
                'type': 'navigate_to_coordinates',
                'lat': lat,
                'lng': lng
            })
        except Exception as e:
            print(f"[SlavesView] Map navigation error: {e}")
    
    def _on_web_table_clear_data(self):
        """Обрабатывает запрос на очистку данных из веб-таблицы."""
        try:
            self._clear_combined_data()
        except Exception as e:
            print(f"[SlavesView] Clear data error: {e}")


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
        # Таблица задач — только для просмотра
        try:
            from PyQt5.QtWidgets import QAbstractItemView
            self.tasks_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        except Exception:
            pass
        # Отключаем стили для таблицы задач
        self.tasks_table.setStyleSheet("")
        self.tasks_table.verticalHeader().setDefaultSectionSize(22)
        
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
        # Ячейки координатной таблицы редактируются только там, где разрешено явно
        try:
            from PyQt5.QtWidgets import QAbstractItemView
            self.coordinates_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        except Exception:
            pass
        # Цвета устойчивые к темам
        self._color_reference = QColor(255, 215, 0, 180)
        self._color_available = QColor(76, 175, 80, 180)
        self._color_unavailable = QColor(244, 67, 54, 180)
        self._color_locked_bg = QColor(200, 200, 200, 120)
        
        # Настройка столбцов
        header = self.coordinates_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(self.coordinates_table)
        
        # Инициализация координат
        # Не инициализируем автоматом при старте. Ждём данных из диспетчера устройств
        # self._initialize_coordinates_table()
        
        # Кнопки управления координатами
        buttons_layout = QHBoxLayout()
        
        self.btn_sync_coords = QPushButton("🔄 Синхронизация")
        # Сохраняем координаты и сразу отправляем на карту
        self.btn_sync_coords.clicked.connect(lambda: (self._save_coordinates(), self._emit_current_coordinates_to_map()))
        buttons_layout.addWidget(self.btn_sync_coords)
        
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        return widget

    def _create_status_bar(self) -> QWidget:
        """Создает статус бар."""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        layout = QHBoxLayout(widget)
        
        # Нижняя панель статуса скрыта — метрики отображает веб-таблица
        
        return widget


    def update_rssi_value(self, range_str: str, slave_id: str, rssi_rms: float):
        """Обновляет значение RSSI в объединенной таблице."""
        # Теперь работаем только с объединенной таблицей
        self.update_combined_rssi(range_str, slave_id, rssi_rms)

    def _get_rssi_color(self, rssi: float) -> QColor:
        """Цвет для RSSI."""
        # Фиксированный порог для раскраски
        threshold_val = -80
        
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
        
        # Время обновления → отправляем только в веб-таблицу
        try:
            if self.web_table_widget:
                self.web_table_widget.update_performance_stats({'last_update': time.strftime('%H:%M:%S')})
        except Exception:
            pass

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
        """Обновляет веб-таблицу данными из watchlist."""
        if not self.web_table_widget:
            return
            
        try:
            # Преобразуем данные watchlist для веб-таблицы
            rssi_data = {}
            targets_info = {}
            
            for data in watchlist_data:
                try:
                    freq = float(data.get('freq', 0))
                    span = float(data.get('span', 2.0))
                    
                    # Создаем диапазон
                    freq_start = freq - span/2
                    freq_end = freq + span/2
                    range_str = f"{freq_start:.1f}-{freq_end:.1f}"
                    
                    # Собираем RSSI данные для каждого slave
                    range_rssi = {}
                    for i in range(3):
                        rms_key = f'rms_{i+1}'
                        val = data.get(rms_key)
                        
                        if val is not None:
                            rssi_val = float(val)
                            slave_id = f"slave{i}"
                            range_rssi[slave_id] = rssi_val
                    
                    if range_rssi:  # Добавляем только если есть RSSI данные
                        rssi_data[range_str] = range_rssi
                        
                        # Информация о цели
                        targets_info[range_str] = {
                            'center_freq': freq,
                            'span': span,
                            'updated': data.get('updated', time.strftime('%H:%M:%S')),
                            'status': 'ИЗМЕРЕНИЕ' if range_rssi else 'ОЖИДАНИЕ',
                            'bins_used': {f'slave{i}': data.get(f'bins_used_{i+1}', 'N/A') for i in range(3)},
                            'timestamps': {f'slave{i}': data.get(f'timestamp_{i+1}', '') for i in range(3)}
                        }
                
                except Exception as e:
                    print(f"[SlavesView] Error processing watchlist item: {e}")
            
            # Обновляем веб-таблицу
            if rssi_data:
                self.web_table_widget.update_rssi_data(rssi_data)
                self.web_table_widget.update_targets_info(targets_info)
            
            # Обновляем статистику
            self._update_combined_stats()
            
        except Exception as e:
            print(f"[SlavesView] Error updating web table from watchlist: {e}")

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
        """Добавляет результат трилатерации в веб-таблицу."""
        try:
            if not self.web_table_widget:
                return
                
            # Получаем данные результата трилатерации
            peak_id = getattr(result, 'peak_id', 'unknown')
            freq = getattr(result, 'freq_mhz', 0.0)
            x = getattr(result, 'x', 0.0)
            y = getattr(result, 'y', 0.0)
            confidence = getattr(result, 'confidence', 0.0)
            
            # Формируем диапазон
            range_str = f"{freq-1.0:.1f}-{freq+1.0:.1f}"
            
            # Обновляем информацию о цели с результатами трилатерации
            current_data = self.web_table_widget.get_current_data()
            targets_info = current_data.get('targets_info', {})
            
            if range_str not in targets_info:
                targets_info[range_str] = {}
            
            targets_info[range_str].update({
                'center_freq': freq,
                'x': x,
                'y': y,
                'confidence': confidence,
                'peak_id': peak_id,
                'status': 'ОБНАРУЖЕН',
                'trilateration_time': time.strftime("%H:%M:%S"),
                'has_trilateration': True
            })
            
            # Обновляем веб-таблицу
            self.web_table_widget.update_targets_info(targets_info)
            
            # Обновляем статистику
            self._update_combined_stats()
            
        except Exception as e:
            print(f"[SlavesView] Error adding transmitter to web table: {e}")
    
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

    def _on_add_measurement_to_map(self, row: int, payload: dict):
        """Отправляет измерение на карту и блокирует кнопку для этой строки."""
        try:
            self.send_to_map.emit(payload)
            btn = self.combined_table.cellWidget(row, 10)
            if btn:
                btn.setProperty('sent_to_map', True)
                btn.setEnabled(False)
                btn.setText("✅")
                btn.setToolTip("Уже добавлено на карту")
                btn.setStyleSheet(
                    "QPushButton { background-color: #2E7D32; color: #ffffff;"
                    " border: none; padding: 4px 8px; border-radius: 3px; }"
                )
        except Exception:
            pass

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
        """Очищает веб-таблицу."""
        if self.web_table_widget:
            self.web_table_widget.clear_all_data()
        # Также очищаем прокси-таблицу для совместимости
        if hasattr(self.combined_table, 'setRowCount'):
            self.combined_table.setRowCount(0)
        self._update_combined_stats()
    
    def _update_combined_stats(self):
        """Обновляет статистику веб-таблицы."""
        if not self.web_table_widget:
            return
            
        try:
            current_data = self.web_table_widget.get_current_data()
            rssi_data = current_data.get('rssi_data', {})
            
            total_count = len(rssi_data)
            active_count = 0
            all_rssi = []
            
            for range_str, slaves_rssi in rssi_data.items():
                if slaves_rssi:  # Есть RSSI данные
                    active_count += 1
                    for slave_id, rssi_val in slaves_rssi.items():
                        try:
                            all_rssi.append(float(rssi_val))
                        except:
                            pass
            
            # Отправляем статистику в веб-таблицу
            stats = {
                'total_ranges': total_count,
                'active_ranges': active_count,
                'avg_rssi': float(np.mean(all_rssi)) if all_rssi else None,
                'total_measurements': len(all_rssi),
                'last_update': time.strftime('%H:%M:%S')
            }
            
            self.web_table_widget.update_performance_stats(stats)
            
        except Exception as e:
            print(f"[SlavesView] Error updating web table stats: {e}")
    
    # Методы для работы с координатами
    def _initialize_coordinates_table(self):
        """Инициализирует таблицу координат."""
        # Сначала пытаемся загрузить сохраненные устройства
        self.coordinates_table.setRowCount(0)
        
        # Загружаем сохраненные устройства из JSON
        saved_devices = self._load_saved_devices_from_json()
        
        if saved_devices:
            self._populate_coordinates_table_from_saved_data(saved_devices)
            print(f"[SlavesView] Loaded {len(saved_devices)} saved devices from JSON")
            
            # Автоматически отправляем загруженные устройства на карту
            self._send_saved_devices_to_map(saved_devices)
        else:
            # Если нет сохраненных данных, показываем информационное сообщение
            self._show_empty_coordinates_message()
    
    def _send_saved_devices_to_map(self, devices_data):
        """Отправляет сохраненные устройства на карту."""
        try:
            stations_data = []
            
            for device in devices_data:
                x, y, z = device.get('coords', (0.0, 0.0, 0.0))
                stations_data.append({
                    'id': device.get('nickname', 'Unknown'),
                    'x': x,
                    'y': y,
                    'z': z,
                    'is_reference': device.get('is_reference', False),
                    'is_active': True
                })
            
            # Эмитируем сигнал для отправки на карту
            self.send_to_map.emit({
                'type': 'stations_update',
                'stations': stations_data
            })
            
            print(f"[SlavesView] Sent {len(stations_data)} saved stations to map")
            
        except Exception as e:
            print(f"[SlavesView] Error sending saved devices to map: {e}")
    
    def _show_empty_coordinates_message(self):
        """Показывает сообщение о отсутствии SDR устройств."""
        # Пустая таблица без добавления строк
        self.coordinates_table.setRowCount(0)
    
    def _load_saved_devices_from_json(self):
        """Загружает сохраненные устройства из JSON файла."""
        try:
            import json
            from pathlib import Path
            
            config_file = Path.home() / ".panorama" / "device_config.json"
            
            if not config_file.exists():
                return []
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            devices = []
            
            # Загружаем только Slave устройства (Master не участвует в трилатерации)
            slaves = config.get('slaves', [])
            reference_found = False
            
            for i, slave in enumerate(slaves):
                pos = slave.get('pos', [0.0, 0.0, 0.0])
                nickname = slave.get('nickname', f'Slave{i+1}')
                
                # Первый Slave становится опорным
                is_reference = (i == 0) or (pos[0] == 0.0 and pos[1] == 0.0 and pos[2] == 0.0 and not reference_found)
                
                if is_reference:
                    reference_found = True
                    pos = [0.0, 0.0, 0.0]  # Опорное устройство всегда в (0,0,0)
                
                devices.append({
                    'nickname': nickname,
                    'role': 'Опорное' if is_reference else 'Измерительное',
                    'coords': (float(pos[0]), float(pos[1]), float(pos[2])),
                    'status': 'REFERENCE' if is_reference else 'AVAILABLE',
                    'is_reference': is_reference
                })
            
            # Если нет устройств или нет опорного, делаем первое опорным
            if devices and not reference_found:
                devices[0]['role'] = 'Опорное'
                devices[0]['status'] = 'REFERENCE' 
                devices[0]['is_reference'] = True
                devices[0]['coords'] = (0.0, 0.0, 0.0)
            
            return devices
            
        except Exception as e:
            print(f"[SlavesView] Error loading saved devices from JSON: {e}")
            return []
    
    def _populate_coordinates_table_from_saved_data(self, devices_data):
        """Заполняет таблицу координат сохраненными данными."""
        try:
            self.coordinates_table.clearSpans()
            self.coordinates_table.setRowCount(len(devices_data))
            
            for row, device in enumerate(devices_data):
                is_reference = device.get('is_reference', False)
                
                # Никнейм устройства
                nickname = device.get('nickname', f'Device-{row}')
                nickname_item = QTableWidgetItem(nickname)
                # Никнейм централизованно редактируется только в диспетчере устройств
                nickname_item.setFlags(nickname_item.flags() & ~Qt.ItemIsEditable)
                nickname_item.setToolTip("Никнейм редактируется в Диспетчере устройств")
                
                if is_reference:
                    nickname_item.setBackground(QBrush(self._color_reference))
                    nickname_item.setToolTip("Опорное устройство (0,0,0)")
                
                self.coordinates_table.setItem(row, 0, nickname_item)
                
                # Роль устройства
                role_combo = QComboBox()
                role_combo.addItems([
                    "Опорное", "Измерительное", "Резервное", "Отключено"
                ])
                
                saved_role = device.get('role', 'Измерительное')
                role_combo.setCurrentText(saved_role)
                
                if is_reference:
                    role_combo.setEnabled(False)
                    role_combo.setToolTip("Первое устройство всегда опорное")
                
                role_combo.setProperty('device_data', device)
                role_combo.currentTextChanged.connect(self._on_role_changed)
                
                self.coordinates_table.setCellWidget(row, 1, role_combo)
                
                # Координаты
                x, y, z = device.get('coords', (0.0, 0.0, 0.0))
                
                x_item = QTableWidgetItem(f"{x:.1f}")
                y_item = QTableWidgetItem(f"{y:.1f}")
                z_item = QTableWidgetItem(f"{z:.1f}")
                
                # Опорное устройство не редактируется
                if is_reference:
                    for item in [x_item, y_item, z_item]:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        item.setBackground(QBrush(self._color_locked_bg))
                        item.setToolTip("Опорное устройство имеет фиксированные координаты (0,0,0)")
                
                self.coordinates_table.setItem(row, 2, x_item)
                self.coordinates_table.setItem(row, 3, y_item)
                self.coordinates_table.setItem(row, 4, z_item)
                
                # Статус устройства
                status = device.get('status', 'UNKNOWN')
                if status == 'REFERENCE' or is_reference:
                    status_text = 'ОПОРНОЕ'
                    status_color = self._color_reference
                elif status == 'AVAILABLE' or status == 'ACTIVE':
                    status_text = 'ДОСТУПНО'
                    status_color = QColor(74, 222, 128, 100)  # Зеленый
                else:
                    status_text = 'НЕИЗВЕСТНО'
                    status_color = QColor(200, 200, 200, 100)  # Серый
                
                status_item = QTableWidgetItem(status_text)
                status_item.setBackground(QBrush(status_color))
                self.coordinates_table.setItem(row, 5, status_item)
                
                print(f"[SlavesView] Loaded device {nickname} with coords ({x:.1f}, {y:.1f}, {z:.1f})")
                
        except Exception as e:
            print(f"[SlavesView] Error populating coordinates table: {e}")
            import traceback
            traceback.print_exc()
    
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
                
                # Получаем координаты (безопасная конвертация)
                def safe_float_convert(item, default=0.0):
                    if not item or not item.text().strip():
                        return default
                    try:
                        return float(item.text().strip())
                    except (ValueError, AttributeError):
                        return default
                
                # Для опорного устройства координаты всегда (0,0,0)
                if role == "Опорное":
                    x, y, z = 0.0, 0.0, 0.0
                else:
                    x = safe_float_convert(self.coordinates_table.item(row, 2), 0.0)
                    y = safe_float_convert(self.coordinates_table.item(row, 3), 0.0)
                    z = safe_float_convert(self.coordinates_table.item(row, 4), 0.0)
                
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
    
    # Удален отдельный показ на карте — координаты отправляются автоматически
    
    def update_combined_rssi(self, range_str: str, slave_id: str, rssi_rms: float):
        """Обновляет RSSI через веб-таблицу."""
        try:
            if not self.web_table_widget:
                return
            
            # Обновляем данные в веб-таблице
            current_data = self.web_table_widget.get_current_data()
            rssi_data = current_data.get('rssi_data', {})
            
            # Добавляем новое RSSI значение
            if range_str not in rssi_data:
                rssi_data[range_str] = {}
            
            rssi_data[range_str][slave_id] = rssi_rms
            
            # Обновляем веб-таблицу
            self.web_table_widget.update_rssi_data(rssi_data)
            
            # Обновляем статистику
            self._update_combined_stats()
        
        except Exception as e:
            print(f"[SlavesView] Error updating web table RSSI: {e}")

    def _on_combined_item_double_clicked(self, item: QTableWidgetItem):
        """Двойной клик — отправка записи на карту (без тяжёлых кнопок)."""
        try:
            row = item.row()
            # Собираем полезную нагрузку из строки
            range_item = self.combined_table.item(row, 0)
            center_item = self.combined_table.item(row, 1)
            x_item = self.combined_table.item(row, 6)
            y_item = self.combined_table.item(row, 7)
            payload = {
                'type': 'target',
                'range': range_item.text() if range_item else '',
                'freq': float(center_item.text()) if center_item and center_item.text() else 0.0,
                'x': float(x_item.text()) if x_item and x_item.text() else 0.0,
                'y': float(y_item.text()) if y_item and y_item.text() else 0.0,
            }
            self.send_to_map.emit(payload)
        except Exception:
            pass
            
    
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
                    nickname_item = self.coordinates_table.item(row, 0)
                    if not nickname_item or not nickname_item.text().strip():
                        continue
                        
                    slave_id = nickname_item.text()
                    
                    # Безопасная конвертация координат
                    def safe_float_convert(item, default=0.0):
                        if not item or not item.text().strip():
                            return default
                        try:
                            return float(item.text().strip())
                        except (ValueError, AttributeError):
                            return default
                    
                    x = safe_float_convert(self.coordinates_table.item(row, 2), 0.0)
                    y = safe_float_convert(self.coordinates_table.item(row, 3), 0.0) 
                    z = safe_float_convert(self.coordinates_table.item(row, 4), 0.0)
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

    def update_coordinates_from_manager(self, devices_data: list):
        """Получает от диспетчера уже подготовленный список устройств
        для координатной таблицы и синхронизирует её без домыслов.
        Ожидаемый формат каждого элемента:
        {
            'nickname': str,
            'serial': str,
            'driver': str,
            'coords': (x, y, z),
            'status': 'REFERENCE' | 'AVAILABLE' | 'UNAVAILABLE',
            'is_reference': bool
        }
        """
        try:
            if not devices_data:
                self._show_empty_coordinates_message()
                return
            # Прямо готовим список для внутреннего рендера
            prepared = []
            for d in devices_data:
                try:
                    prepared.append({
                        'nickname': d.get('nickname') or f"SDR-{(d.get('serial') or '0000')[-4:]}",
                        'serial': d.get('serial', ''),
                        'coords': d.get('coords', (0.0, 0.0, 0.0)),
                        'status': d.get('status', 'AVAILABLE'),
                        'is_reference': bool(d.get('is_reference', False))
                    })
                except Exception:
                    continue
            self._update_coordinates_table_with_devices(prepared)
            print(f"[SlavesView] Coordinates synced from manager: {len(prepared)} devices")
        except Exception as e:
            print(f"[SlavesView] Error syncing coordinates from manager: {e}")
    
    def _update_coordinates_table_with_devices(self, devices_list):
        """Обновляет таблицу координат с реальными SDR устройствами."""
        try:
            # Очищаем информацию о span если была
            self.coordinates_table.clearSpans()
            
            # Если нет устройств, показываем сообщение
            if not devices_list:
                self._show_empty_coordinates_message()
                return
            
            self.coordinates_table.setRowCount(len(devices_list))
            
            # Ищем опорное устройство или назначаем первое
            reference_found = any(d.get('is_reference', False) for d in devices_list)
            
            for row, device in enumerate(devices_list):
                # Первое устройство становится опорным, если нет другого опорного
                is_reference = device.get('is_reference', False) or (row == 0 and not reference_found)
                
                # Никнейм устройства
                nickname = device.get('nickname', f"SDR-{device.get('serial', 'Unknown')[-4:]}")
                nickname_item = QTableWidgetItem(nickname)
                # Никнейм централизованно редактируется только в диспетчере устройств
                nickname_item.setFlags(nickname_item.flags() & ~Qt.ItemIsEditable)
                nickname_item.setToolTip("Никнейм редактируется в Диспетчере устройств")
                
                if is_reference:
                    nickname_item.setBackground(QBrush(self._color_reference))  # Золотой для опорного
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
                        item.setBackground(QBrush(self._color_locked_bg))
                        item.setToolTip("Опорное устройство имеет фиксированные координаты (0,0,0)")
                
                self.coordinates_table.setItem(row, 2, x_item)
                self.coordinates_table.setItem(row, 3, y_item)
                self.coordinates_table.setItem(row, 4, z_item)
                
                # Статус устройства - переводим на русский
                status = device.get('status', 'UNKNOWN')
                if status == 'REFERENCE' or is_reference:
                    status_text = 'ОПОРНОЕ'
                    status_color = self._color_reference
                elif status == 'AVAILABLE' or status == 'ACTIVE':
                    status_text = 'ДОСТУПНО'
                    status_color = self._color_available
                elif status == 'UNAVAILABLE':
                    status_text = 'НЕДОСТУПНО'
                    status_color = self._color_unavailable
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
                    
                    # Если устройство стало опорным
                    if new_role == "Опорное":
                        self._handle_new_reference_device(sender)
                    
                    # Автоматически сохраняем изменения и отправляем на карту сразу
                    self._save_coordinates()
                    self._emit_current_coordinates_to_map()
                    
        except Exception as e:
            print(f"[SlavesView] Error handling role change: {e}")
    
    def _handle_new_reference_device(self, new_reference_combo):
        """Обрабатывает назначение нового опорного устройства."""
        try:
            # Находим строку нового опорного устройства
            new_reference_row = -1
            for row in range(self.coordinates_table.rowCount()):
                combo = self.coordinates_table.cellWidget(row, 1)
                if combo == new_reference_combo:
                    new_reference_row = row
                    break
            
            if new_reference_row == -1:
                return
            
            # Сбрасываем все устройства на "Измерительное" кроме нового опорного
            for row in range(self.coordinates_table.rowCount()):
                if row == new_reference_row:
                    continue
                    
                combo = self.coordinates_table.cellWidget(row, 1)
                if combo and combo.currentText() == "Опорное":
                    combo.setCurrentText("Измерительное")
                    # Разрешаем менять роль снова
                    combo.setEnabled(True)
                    
                    # Разблокируем координаты предыдущего опорного
                    for col in [2, 3, 4]:  # X, Y, Z колонки
                        item = self.coordinates_table.item(row, col)
                        if item:
                            item.setFlags(item.flags() | Qt.ItemIsEditable)
                            item.setBackground(QBrush(QColor(60, 60, 60)))
                            item.setToolTip("")
                    
                    # Обновляем статус
                    status_item = self.coordinates_table.item(row, 5)
                    if status_item:
                        status_item.setText("ДОСТУПНО")
                        status_item.setBackground(QBrush(self._color_available))
            
            # Настраиваем новое опорное устройство
            self._setup_reference_device(new_reference_row)
            
            print(f"[SlavesView] New reference device set at row {new_reference_row}")
            
        except Exception as e:
            print(f"[SlavesView] Error handling new reference device: {e}")
            import traceback
            traceback.print_exc()
    
    def _setup_reference_device(self, row):
        """Настраивает устройство как опорное."""
        try:
            # Устанавливаем координаты (0,0,0)
            for col, val in [(2, "0.0"), (3, "0.0"), (4, "0.0")]:
                item = self.coordinates_table.item(row, col)
                if item:
                    item.setText(val)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Запрещаем редактирование
                    item.setBackground(QBrush(QColor(200, 200, 200, 100)))
                    item.setToolTip("Опорное устройство имеет фиксированные координаты (0,0,0)")
            
            # Обновляем никнейм с золотым фоном
            nickname_item = self.coordinates_table.item(row, 0)
            if nickname_item:
                nickname_item.setBackground(QBrush(self._color_reference))
                nickname_item.setToolTip("Опорное устройство (0,0,0)")
            
            # Обновляем статус
            status_item = self.coordinates_table.item(row, 5)
            if status_item:
                status_item.setText("ОПОРНОЕ")
                status_item.setBackground(QBrush(self._color_reference))
            
            # Блокируем изменение роли
            combo = self.coordinates_table.cellWidget(row, 1)
            if combo:
                combo.setCurrentText("Опорное")
                combo.setEnabled(False)
                combo.setToolTip("Опорное устройство - основа координатной системы")
            
        except Exception as e:
            print(f"[SlavesView] Error setting up reference device: {e}")

    def _emit_current_coordinates_to_map(self):
        """Собирает текущее состояние таблицы координат и мгновенно отправляет на карту."""
        try:
            devices = []
            for row in range(self.coordinates_table.rowCount()):
                nickname_item = self.coordinates_table.item(row, 0)
                if not nickname_item or not nickname_item.text().strip():
                    continue
                nickname = nickname_item.text()
                role_widget = self.coordinates_table.cellWidget(row, 1)
                role = role_widget.currentText() if role_widget else "Измерительное"
                
                def safe_float_convert(item, default=0.0):
                    if not item or not item.text().strip():
                        return default
                    try:
                        return float(item.text().strip())
                    except (ValueError, AttributeError):
                        return default
                x = safe_float_convert(self.coordinates_table.item(row, 2), 0.0)
                y = safe_float_convert(self.coordinates_table.item(row, 3), 0.0)
                z = safe_float_convert(self.coordinates_table.item(row, 4), 0.0)
                devices.append({
                    'id': nickname,
                    'x': x, 'y': y, 'z': z,
                    'type': 'sdr_device',
                    'role': role,
                    'is_reference': role == "Опорное"
                })
            self.send_to_map.emit({
                'type': 'update_devices_coordinates',
                'devices': devices
            })
        except Exception:
            pass