# panorama/features/peaks/ui_improved.py
from __future__ import annotations
from typing import Optional, List, Deque
from collections import deque
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import json


class AdaptivePeaksWidget(QtWidgets.QWidget):
    """Виджет поиска пиков с адаптивным порогом baseline + N."""
    
    goToFreq = QtCore.pyqtSignal(float)  # freq_hz
    peakDetected = QtCore.pyqtSignal(dict)  # peak info
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._freqs: Optional[np.ndarray] = None
        self._row: Optional[np.ndarray] = None
        self._history: Deque[np.ndarray] = deque(maxlen=10)  # История для baseline
        self._baseline: Optional[np.ndarray] = None
        self._last_peaks: List[dict] = []
        
        self._build_ui()
        
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Панель параметров
        params_group = QtWidgets.QGroupBox("Параметры детектора пиков")
        params_layout = QtWidgets.QFormLayout(params_group)
        
        # Режим порога
        self.threshold_mode = QtWidgets.QComboBox()
        self.threshold_mode.addItems(["Адаптивный (baseline + N)", "Фиксированный порог"])
        self.threshold_mode.currentTextChanged.connect(self._on_threshold_mode_changed)
        
        # Адаптивный порог (по умолчанию +20 дБ)
        self.adaptive_offset = QtWidgets.QDoubleSpinBox()
        self.adaptive_offset.setRange(3, 50)
        self.adaptive_offset.setValue(20)  # Изменено на 20 дБ
        self.adaptive_offset.setSuffix(" дБ над шумом")
        self.adaptive_offset.setToolTip("Порог = baseline + это значение")
        
        # Фиксированный порог
        self.fixed_threshold = QtWidgets.QDoubleSpinBox()
        self.fixed_threshold.setRange(-160, 30)
        self.fixed_threshold.setValue(-70)
        self.fixed_threshold.setSuffix(" дБм")
        self.fixed_threshold.setEnabled(False)
        
        # Минимальное расстояние между пиками
        self.min_distance = QtWidgets.QDoubleSpinBox()
        self.min_distance.setRange(0, 5000)
        self.min_distance.setValue(100)
        self.min_distance.setSuffix(" кГц")
        
        # Минимальная ширина пика
        self.min_width = QtWidgets.QSpinBox()
        self.min_width.setRange(1, 100)
        self.min_width.setValue(3)
        self.min_width.setSuffix(" бинов")
        
        # Окно для расчета baseline
        self.baseline_window = QtWidgets.QSpinBox()
        self.baseline_window.setRange(3, 50)
        self.baseline_window.setValue(10)
        self.baseline_window.setSuffix(" свипов")
        self.baseline_window.setToolTip("Количество свипов для расчета шумового порога")
        
        # Метод расчета baseline
        self.baseline_method = QtWidgets.QComboBox()
        self.baseline_method.addItems(["Медиана", "Среднее", "Минимум", "Персентиль"])
        
        # Персентиль для baseline
        self.baseline_percentile = QtWidgets.QSpinBox()
        self.baseline_percentile.setRange(1, 99)
        self.baseline_percentile.setValue(25)
        self.baseline_percentile.setSuffix(" %")
        self.baseline_percentile.setEnabled(False)
        
        # Автопоиск
        self.auto_search = QtWidgets.QCheckBox("Автопоиск")
        self.auto_search.setChecked(True)
        
        params_layout.addRow("Режим порога:", self.threshold_mode)
        params_layout.addRow("Адаптивный порог:", self.adaptive_offset)
        params_layout.addRow("Фиксированный порог:", self.fixed_threshold)
        params_layout.addRow("Мин. расстояние:", self.min_distance)
        params_layout.addRow("Мин. ширина:", self.min_width)
        params_layout.addRow("Окно baseline:", self.baseline_window)
        params_layout.addRow("Метод baseline:", self.baseline_method)
        params_layout.addRow("Персентиль:", self.baseline_percentile)
        params_layout.addRow(self.auto_search)
        
        layout.addWidget(params_group)
        
        # Кнопки управления
        buttons_layout = QtWidgets.QHBoxLayout()
        
        self.btn_find = QtWidgets.QPushButton("🔍 Найти пики")
        self.btn_find.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
        """)
        
        self.btn_clear = QtWidgets.QPushButton("🗑 Очистить")
        self.btn_export = QtWidgets.QPushButton("💾 Экспорт")
        
        buttons_layout.addWidget(self.btn_find)
        buttons_layout.addWidget(self.btn_clear)
        buttons_layout.addWidget(self.btn_export)
        
        layout.addLayout(buttons_layout)
        
        # Таблица пиков с улучшенным отображением
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Частота (МГц)", "Уровень (дБм)", "Над шумом (дБ)", 
            "Ширина (кГц)", "Q-фактор", "Тип"
        ])
        
        # Настройка ширины столбцов для лучшего отображения
        self.table.setColumnWidth(0, 120)  # Частота
        self.table.setColumnWidth(1, 100)  # Уровень
        self.table.setColumnWidth(2, 100)  # Над шумом
        self.table.setColumnWidth(3, 90)   # Ширина
        self.table.setColumnWidth(4, 80)   # Q-фактор
        self.table.setColumnWidth(5, 120)  # Тип
        
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        
        # Улучшенные стили для таблицы
        self.table.setStyleSheet("""
            QTableWidget {
                gridline-color: #555;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 2px;
            }
            QHeaderView::section {
                font-weight: bold;
                padding: 4px;
            }
        """)
        
        layout.addWidget(self.table)
        
        # Статистика
        stats_layout = QtWidgets.QHBoxLayout()
        
        self.lbl_peaks_count = QtWidgets.QLabel("Пиков: 0")
        self.lbl_baseline = QtWidgets.QLabel("Baseline: —")
        self.lbl_threshold = QtWidgets.QLabel("Порог: —")
        
        stats_layout.addWidget(self.lbl_peaks_count)
        stats_layout.addWidget(self.lbl_baseline)
        stats_layout.addWidget(self.lbl_threshold)
        stats_layout.addStretch()
        
        layout.addLayout(stats_layout)
        
        # Подключение сигналов
        self.btn_find.clicked.connect(self._find_peaks)
        self.btn_clear.clicked.connect(self._clear_peaks)
        self.btn_export.clicked.connect(self._export_peaks)
        self.table.doubleClicked.connect(self._on_double_click)
        self.threshold_mode.currentTextChanged.connect(self._on_threshold_mode_changed)
        self.baseline_method.currentTextChanged.connect(self._on_baseline_method_changed)
        
        # Автообновление при изменении параметров
        self.adaptive_offset.valueChanged.connect(self._on_params_changed)
        self.fixed_threshold.valueChanged.connect(self._on_params_changed)
        self.min_distance.valueChanged.connect(self._on_params_changed)
        self.min_width.valueChanged.connect(self._on_params_changed)
        
    def _on_threshold_mode_changed(self, text):
        """Переключение режима порога."""
        is_adaptive = "Адаптивный" in text
        self.adaptive_offset.setEnabled(is_adaptive)
        self.fixed_threshold.setEnabled(not is_adaptive)
        self.baseline_window.setEnabled(is_adaptive)
        self.baseline_method.setEnabled(is_adaptive)
        
        if self.auto_search.isChecked() and self._freqs is not None:
            self._find_peaks()
            
    def _on_baseline_method_changed(self, text):
        """Изменение метода расчета baseline."""
        self.baseline_percentile.setEnabled("Персентиль" in text)
        
        if self.auto_search.isChecked() and self._freqs is not None:
            self._find_peaks()
            
    def _on_params_changed(self):
        """При изменении параметров."""
        if self.auto_search.isChecked() and self._freqs is not None:
            self._find_peaks()
            
    def update_from_row(self, freqs_hz, row_dbm):
        """Обновление данных от спектра."""
        self._freqs = np.asarray(freqs_hz, dtype=float)
        self._row = np.asarray(row_dbm, dtype=float)
        
        # Добавляем в историю для baseline
        self._history.append(self._row.copy())
        
        # Пересчитываем baseline
        self._calculate_baseline()
        
        if self.auto_search.isChecked():
            self._find_peaks()
            
    def _calculate_baseline(self):
        """Вычисляет baseline по истории."""
        if len(self._history) < 3:
            return
            
        # Ограничиваем историю нужным размером
        window_size = min(self.baseline_window.value(), len(self._history))
        history_array = np.array(list(self._history)[-window_size:])
        
        method = self.baseline_method.currentText()
        
        if method == "Медиана":
            self._baseline = np.median(history_array, axis=0)
        elif method == "Среднее":
            self._baseline = np.mean(history_array, axis=0)
        elif method == "Минимум":
            self._baseline = np.min(history_array, axis=0)
        elif method == "Персентиль":
            percentile = self.baseline_percentile.value()
            self._baseline = np.percentile(history_array, percentile, axis=0)
        else:
            self._baseline = np.median(history_array, axis=0)
            
        # Обновляем статистику
        if self._baseline is not None:
            avg_baseline = np.mean(self._baseline)
            self.lbl_baseline.setText(f"Baseline: {avg_baseline:.1f} дБм")
            
    def _get_threshold(self) -> np.ndarray:
        """Получает массив порогов для каждого бина."""
        if self._row is None:
            return None
            
        if "Адаптивный" in self.threshold_mode.currentText():
            if self._baseline is None:
                # Если baseline еще не рассчитан, используем минимум текущей строки
                self._baseline = self._row - 10
                
            threshold = self._baseline + self.adaptive_offset.value()
            avg_threshold = np.mean(threshold)
            self.lbl_threshold.setText(f"Порог: {avg_threshold:.1f} дБм")
            return threshold
        else:
            # Фиксированный порог
            threshold_value = self.fixed_threshold.value()
            self.lbl_threshold.setText(f"Порог: {threshold_value:.1f} дБм")
            return np.full_like(self._row, threshold_value)
            
    def _find_peaks(self):
        """Поиск пиков с адаптивным порогом."""
        if self._freqs is None or self._row is None:
            return
            
        threshold = self._get_threshold()
        if threshold is None:
            return
            
        # Находим точки выше порога
        above_threshold = self._row > threshold
        
        if not np.any(above_threshold):
            self._last_peaks = []
            self._fill_table()
            return
            
        # Находим локальные максимумы
        peaks = []
        
        # Простой поиск локальных максимумов
        for i in range(1, len(self._row) - 1):
            if above_threshold[i] and self._row[i] > self._row[i-1] and self._row[i] >= self._row[i+1]:
                # Проверяем ширину пика
                width = self._calculate_peak_width(i)
                
                if width >= self.min_width.value():
                    peak_info = self._analyze_peak(i, threshold[i])
                    peaks.append(peak_info)
                    
        # Фильтруем по минимальному расстоянию
        if len(self._freqs) > 1:
            bin_width_khz = (self._freqs[1] - self._freqs[0]) / 1000
            min_bins = int(self.min_distance.value() / bin_width_khz)
            peaks = self._filter_close_peaks(peaks, min_bins)
            
        # Сортируем по уровню (сильнейшие сверху)
        peaks.sort(key=lambda x: x['level_dbm'], reverse=True)
        
        self._last_peaks = peaks
        self._fill_table()
        
        # Эмитим сигналы о найденных пиках
        for peak in peaks:
            self.peakDetected.emit(peak)
            
    def _calculate_peak_width(self, peak_idx: int) -> int:
        """Вычисляет ширину пика в бинах."""
        if self._row is None or peak_idx < 0 or peak_idx >= len(self._row):
            return 0
            
        peak_level = self._row[peak_idx]
        threshold_3db = peak_level - 3.0
        
        # Ищем границы на уровне -3dB
        left = peak_idx
        while left > 0 and self._row[left] > threshold_3db:
            left -= 1
            
        right = peak_idx
        while right < len(self._row) - 1 and self._row[right] > threshold_3db:
            right += 1
            
        return right - left + 1
        
    def _analyze_peak(self, peak_idx: int, threshold: float) -> dict:
        """Анализирует характеристики пика."""
        freq_mhz = self._freqs[peak_idx] / 1e6
        level_dbm = self._row[peak_idx]
        
        # Ширина на уровне -3dB
        peak_level = self._row[peak_idx]
        threshold_3db = peak_level - 3.0
        
        left = peak_idx
        while left > 0 and self._row[left] > threshold_3db:
            left -= 1
            
        right = peak_idx
        while right < len(self._row) - 1 and self._row[right] > threshold_3db:
            right += 1
            
        if len(self._freqs) > 1:
            width_hz = self._freqs[right] - self._freqs[left]
            width_khz = width_hz / 1000
        else:
            width_khz = 0
            
        # Q-фактор (добротность)
        q_factor = (freq_mhz * 1000) / width_khz if width_khz > 0 else 0
        
        # Превышение над порогом
        snr = level_dbm - threshold
        
        # Классификация по ширине полосы
        if width_khz < 25:
            signal_type = "Узкополосный"
        elif width_khz < 200:
            signal_type = "Среднеполосный"
        else:
            signal_type = "Широкополосный"
            
        return {
            'index': peak_idx,
            'freq_mhz': float(freq_mhz),
            'level_dbm': float(level_dbm),
            'snr_db': float(snr),
            'width_khz': float(width_khz),
            'q_factor': float(q_factor),
            'type': signal_type
        }
        
    def _filter_close_peaks(self, peaks: List[dict], min_distance_bins: int) -> List[dict]:
        """Фильтрует близко расположенные пики, оставляя сильнейшие."""
        if not peaks or min_distance_bins <= 1:
            return peaks
            
        # Сортируем по уровню (сильнейшие первые)
        sorted_peaks = sorted(peaks, key=lambda x: x['level_dbm'], reverse=True)
        
        filtered = []
        used_indices = set()
        
        for peak in sorted_peaks:
            idx = peak['index']
            
            # Проверяем, не слишком ли близко к уже выбранным
            too_close = False
            for used_idx in used_indices:
                if abs(idx - used_idx) < min_distance_bins:
                    too_close = True
                    break
                    
            if not too_close:
                filtered.append(peak)
                used_indices.add(idx)
                
        return filtered
        
    def _fill_table(self):
        """Заполнение таблицы найденными пиками."""
        self.table.setRowCount(len(self._last_peaks))
        
        for row, peak in enumerate(self._last_peaks):
            # Частота
            freq_item = QtWidgets.QTableWidgetItem(f"{peak['freq_mhz']:.6f}")
            freq_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, 0, freq_item)
            
            # Уровень
            level_item = QtWidgets.QTableWidgetItem(f"{peak['level_dbm']:.1f}")
            level_item.setTextAlignment(QtCore.Qt.AlignCenter)
            
            # Цветовая индикация по уровню
            level = peak['level_dbm']
            if level >= -30:
                color = QtGui.QColor(255, 200, 200)  # Красный - очень сильный
            elif level >= -50:
                color = QtGui.QColor(255, 255, 200)  # Желтый - сильный
            elif level >= -70:
                color = QtGui.QColor(200, 255, 200)  # Зеленый - средний
            else:
                color = QtGui.QColor(200, 230, 255)  # Голубой - слабый
                
            level_item.setBackground(QtGui.QBrush(color))
            self.table.setItem(row, 1, level_item)
            
            # SNR (превышение над порогом)
            snr_item = QtWidgets.QTableWidgetItem(f"+{peak['snr_db']:.1f}")
            snr_item.setTextAlignment(QtCore.Qt.AlignCenter)
            
            # Цвет по SNR
            if peak['snr_db'] > 20:
                snr_item.setForeground(QtGui.QBrush(QtGui.QColor(0, 200, 0)))
            elif peak['snr_db'] > 10:
                snr_item.setForeground(QtGui.QBrush(QtGui.QColor(200, 200, 0)))
            else:
                snr_item.setForeground(QtGui.QBrush(QtGui.QColor(200, 0, 0)))
                
            self.table.setItem(row, 2, snr_item)
            
            # Ширина
            width_item = QtWidgets.QTableWidgetItem(f"{peak['width_khz']:.1f}")
            width_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, 3, width_item)
            
            # Q-фактор
            q_item = QtWidgets.QTableWidgetItem(f"{peak['q_factor']:.0f}")
            q_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, 4, q_item)
            
            # Тип
            type_item = QtWidgets.QTableWidgetItem(peak['type'])
            type_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, 5, type_item)
            
        # Обновляем статистику
        self.lbl_peaks_count.setText(f"Пиков: {len(self._last_peaks)}")
        
    def _on_double_click(self, index):
        """Переход к частоте по двойному клику."""
        row = index.row()
        if 0 <= row < len(self._last_peaks):
            peak = self._last_peaks[row]
            self.goToFreq.emit(peak['freq_mhz'] * 1e6)
            
    def _clear_peaks(self):
        """Очистка найденных пиков."""
        self._last_peaks = []
        self.table.setRowCount(0)
        self.lbl_peaks_count.setText("Пиков: 0")
        
    def _export_peaks(self):
        """Экспорт пиков в CSV."""
        if not self._last_peaks:
            QtWidgets.QMessageBox.information(self, "Экспорт", "Нет пиков для экспорта")
            return
            
        from PyQt5.QtCore import QDateTime
        default_name = f"peaks_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.csv"
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить пики", default_name, "CSV files (*.csv)"
        )
        if not path:
            return
            
        try:
            import csv
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['freq_mhz', 'level_dbm', 'snr_db', 'width_khz', 'q_factor', 'type'])
                
                for peak in self._last_peaks:
                    writer.writerow([
                        f"{peak['freq_mhz']:.6f}",
                        f"{peak['level_dbm']:.2f}",
                        f"{peak['snr_db']:.2f}",
                        f"{peak['width_khz']:.1f}",
                        f"{peak['q_factor']:.0f}",
                        peak['type']
                    ])
                    
            QtWidgets.QMessageBox.information(self, "Экспорт", 
                f"Сохранено {len(self._last_peaks)} пиков в:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка экспорта: {e}")