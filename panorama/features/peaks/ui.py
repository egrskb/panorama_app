from __future__ import annotations
from typing import Optional, List
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import json
from .logic import find_peak_indices, summarize


class PeaksWidget(QtWidgets.QWidget):
    goToFreq = QtCore.pyqtSignal(float)  # freq_hz

    def __init__(self, parent=None):
        super().__init__(parent)
        self._freqs: Optional[np.ndarray] = None
        self._row: Optional[np.ndarray] = None
        self._last_peaks: List[dict] = []  # Сохраняем найденные пики

        # Панель параметров
        top = QtWidgets.QHBoxLayout()
        
        self.th = QtWidgets.QDoubleSpinBox()
        self.th.setRange(-160.0, 30.0)
        self.th.setDecimals(1)
        self.th.setValue(-50.0)  # <-- Изменено с -80 на -50
        self.th.setSuffix(" дБм")
        
        self.min_dist = QtWidgets.QDoubleSpinBox()
        self.min_dist.setRange(0.0, 5000.0)
        self.min_dist.setDecimals(0)
        self.min_dist.setSingleStep(50.0)
        self.min_dist.setValue(100.0)  # Уменьшено для лучшего разрешения
        self.min_dist.setSuffix(" кГц")
        
        self.auto = QtWidgets.QCheckBox("Авто")
        self.auto.setChecked(True)
        
        self.btn = QtWidgets.QPushButton("Найти пики")
        self.btn_export = QtWidgets.QPushButton("Экспорт CSV")
        
        for w, lab in [(self.th, "Порог"), (self.min_dist, "Мин. разнесение")]:
            box = QtWidgets.QVBoxLayout()
            box.addWidget(QtWidgets.QLabel(lab))
            box.addWidget(w)
            top.addLayout(box)
        
        top.addStretch(1)
        top.addWidget(self.auto)
        top.addWidget(self.btn)
        top.addWidget(self.btn_export)

        # Таблица
        self.table = QtWidgets.QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Частота (МГц)", "Уровень (дБм)", "Ширина (кГц)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        
        # Альтернативный цвет строк для лучшей читаемости
        self.table.setAlternatingRowColors(True)

        # Панель поиска частот
        search_panel = QtWidgets.QHBoxLayout()
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Найти частоты (МГц через запятую)")
        self.btn_search = QtWidgets.QPushButton("Найти в таблице")
        search_panel.addWidget(self.search_edit)
        search_panel.addWidget(self.btn_search)

        # Статистика
        self.lbl_stats = QtWidgets.QLabel("Пиков: 0")

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.table)
        lay.addLayout(search_panel)
        lay.addWidget(self.lbl_stats)

        self.btn.clicked.connect(self._run_find)
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_search.clicked.connect(self._search_frequencies)
        self.table.doubleClicked.connect(self._on_double_clicked)
        
        # Автоматическое обновление при изменении параметров
        self.th.valueChanged.connect(self._on_params_changed)
        self.min_dist.valueChanged.connect(self._on_params_changed)

    # --- внешнее API ---
    def update_from_row(self, freqs_hz, row_dbm):
        """Обновление данных от спектра."""
        self._freqs = np.asarray(freqs_hz, dtype=float)
        self._row = np.asarray(row_dbm, dtype=float)
        if self.auto.isChecked():
            self._run_find()

    # --- внутреннее ---
    def _on_params_changed(self):
        """При изменении параметров перезапускаем поиск если авто включено."""
        if self.auto.isChecked() and self._freqs is not None:
            self._run_find()

    def _run_find(self):
        """Поиск пиков с улучшенным алгоритмом."""
        if self._freqs is None or self._row is None:
            return
        
        # Вычисляем количество бинов для минимального расстояния
        if len(self._freqs) > 1:
            bin_width_hz = self._freqs[1] - self._freqs[0]
            bins = max(1, int(round(self.min_dist.value() * 1e3 / bin_width_hz)))
        else:
            bins = 1
        
        # Находим индексы пиков
        idx = find_peak_indices(self._row, min_distance_bins=bins, threshold_dbm=self.th.value())
        
        # Создаем подробную информацию о пиках
        self._last_peaks = []
        for i in idx:
            # Вычисляем ширину пика на уровне -3 дБ
            peak_level = self._row[i]
            threshold_3db = peak_level - 3.0
            
            # Ищем границы пика
            left = i
            while left > 0 and self._row[left] > threshold_3db:
                left -= 1
            
            right = i
            while right < len(self._row) - 1 and self._row[right] > threshold_3db:
                right += 1
            
            # Вычисляем ширину в кГц
            if left != right and len(self._freqs) > 1:
                width_hz = self._freqs[right] - self._freqs[left]
                width_khz = width_hz / 1000.0
            else:
                width_khz = 0.0
            
            self._last_peaks.append({
                'freq_mhz': float(self._freqs[i] / 1e6),
                'level_dbm': float(peak_level),
                'width_khz': float(width_khz),
                'index': int(i)
            })
        
        # Сортируем по уровню (сильнейшие сверху)
        self._last_peaks.sort(key=lambda x: x['level_dbm'], reverse=True)
        
        # Заполняем таблицу
        self._fill_table(self._last_peaks)
        
        # Обновляем статистику
        self.lbl_stats.setText(f"Пиков: {len(self._last_peaks)}")

    def _fill_table(self, peaks: List[dict]):
        """Заполнение таблицы с цветовой индикацией."""
        self.table.setRowCount(len(peaks))
        
        for r, peak in enumerate(peaks):
            # Частота
            item_freq = QtWidgets.QTableWidgetItem(f"{peak['freq_mhz']:.6f}")
            item_freq.setTextAlignment(QtCore.Qt.AlignCenter)
            
            # Уровень с цветовой индикацией
            item_level = QtWidgets.QTableWidgetItem(f"{peak['level_dbm']:.1f}")
            item_level.setTextAlignment(QtCore.Qt.AlignCenter)
            
            # Цветовая индикация по уровню
            level = peak['level_dbm']
            if level >= -30:
                # Очень сильный сигнал - красный
                item_level.setBackground(QtGui.QBrush(QtGui.QColor(255, 200, 200)))
            elif level >= -50:
                # Сильный сигнал - желтый
                item_level.setBackground(QtGui.QBrush(QtGui.QColor(255, 255, 200)))
            elif level >= -70:
                # Средний сигнал - зеленый
                item_level.setBackground(QtGui.QBrush(QtGui.QColor(200, 255, 200)))
            else:
                # Слабый сигнал - голубой
                item_level.setBackground(QtGui.QBrush(QtGui.QColor(200, 230, 255)))
            
            # Ширина
            item_width = QtWidgets.QTableWidgetItem(f"{peak['width_khz']:.1f}")
            item_width.setTextAlignment(QtCore.Qt.AlignCenter)
            
            self.table.setItem(r, 0, item_freq)
            self.table.setItem(r, 1, item_level)
            self.table.setItem(r, 2, item_width)

    def _on_double_clicked(self, mi):
        """Переход к частоте по двойному клику."""
        r = mi.row()
        if 0 <= r < len(self._last_peaks):
            peak = self._last_peaks[r]
            self.goToFreq.emit(peak['freq_mhz'] * 1e6)

    def _search_frequencies(self):
        """Поиск и выделение частот в таблице."""
        text = self.search_edit.text().strip()
        if not text:
            return
        
        # Парсим введенные частоты
        target_freqs = []
        for part in text.replace(',', ' ').replace(';', ' ').split():
            try:
                freq = float(part.strip())
                target_freqs.append(freq)
            except ValueError:
                continue
        
        if not target_freqs:
            return
        
        # Снимаем выделение
        self.table.clearSelection()
        
        # Ищем ближайшие пики для каждой целевой частоты
        for target in target_freqs:
            best_row = -1
            best_diff = float('inf')
            
            for r, peak in enumerate(self._last_peaks):
                diff = abs(peak['freq_mhz'] - target)
                if diff < best_diff:
                    best_diff = diff
                    best_row = r
            
            # Выделяем если разница меньше 1 МГц
            if best_row >= 0 and best_diff < 1.0:
                self.table.selectRow(best_row)
        
        # Прокручиваем к первому выделенному
        selected = self.table.selectionModel().selectedRows()
        if selected:
            self.table.scrollTo(selected[0], QtWidgets.QAbstractItemView.PositionAtCenter)

    def _export_csv(self):
        """Экспорт пиков в CSV."""
        if not self._last_peaks:
            QtWidgets.QMessageBox.information(self, "Экспорт", "Нет пиков для экспорта")
            return
        
        from PyQt5.QtCore import QDateTime
        default_name = f"peaks_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.csv"
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить пики CSV", default_name, "CSV files (*.csv)"
        )
        if not path:
            return
        
        try:
            import csv
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['freq_mhz', 'level_dbm', 'width_khz'])
                for peak in self._last_peaks:
                    writer.writerow([
                        f"{peak['freq_mhz']:.6f}",
                        f"{peak['level_dbm']:.2f}",
                        f"{peak['width_khz']:.1f}"
                    ])
            
            QtWidgets.QMessageBox.information(self, "Экспорт", f"Сохранено {len(self._last_peaks)} пиков")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка экспорта: {e}")

    def save_settings(self, settings):
        """Сохранение настроек."""
        settings.beginGroup("peaks")
        settings.setValue("threshold", self.th.value())
        settings.setValue("min_distance", self.min_dist.value())
        settings.setValue("auto", self.auto.isChecked())
        # Сохраняем последние найденные пики
        settings.setValue("last_peaks", json.dumps(self._last_peaks))
        settings.endGroup()

    def restore_settings(self, settings):
        """Восстановление настроек."""
        settings.beginGroup("peaks")
        self.th.setValue(float(settings.value("threshold", -50.0)))
        self.min_dist.setValue(float(settings.value("min_distance", 100.0)))
        self.auto.setChecked(settings.value("auto", True, type=bool))
        
        # Восстанавливаем последние пики
        try:
            peaks_json = settings.value("last_peaks", "[]")
            self._last_peaks = json.loads(peaks_json)
            if self._last_peaks:
                self._fill_table(self._last_peaks)
                self.lbl_stats.setText(f"Пиков: {len(self._last_peaks)} (сохраненные)")
        except Exception:
            pass
        
        settings.endGroup()