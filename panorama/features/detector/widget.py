from __future__ import annotations
from typing import List, Tuple, Optional
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np


class DetectorWidget(QtWidgets.QWidget):
    """Виджет детектора активности с ROI и визуализацией."""
    
    rangeSelected = QtCore.pyqtSignal(float, float)  # start_mhz, stop_mhz
    detectionStarted = QtCore.pyqtSignal()
    detectionStopped = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._detecting = False
        self._current_range: Optional[Tuple[float, float]] = None
        self._detections = []
        
        self._build_ui()

    def _build_ui(self):
        v = QtWidgets.QVBoxLayout(self)
        
        # Пресеты диапазонов
        grp_presets = QtWidgets.QGroupBox("ROI-пресеты")
        grid = QtWidgets.QGridLayout(grp_presets)
        
        preset_rows = [
            ("FM (87–108 МГц)", (87.5, 108.0)),
            ("VHF (136–174 МГц)", (136.0, 174.0)),
            ("UHF (400–470 МГц)", (400.0, 470.0)),
            ("Wi-Fi 2.4 ГГц", (2400.0, 2483.5)),
            ("Wi-Fi 5 ГГц", (5170.0, 5895.0)),
            ("5.8 ГГц FPV", (5725.0, 5875.0)),
            ("LTE 700–900", (703.0, 960.0)),
            ("ISM 433", (433.0, 435.0)),
            ("ISM 868", (863.0, 873.0)),
            ("GSM 900", (890.0, 960.0)),
            ("GSM 1800", (1710.0, 1880.0)),
            ("Bluetooth/ZigBee", (2400.0, 2483.5)),
        ]
        
        r = 0
        c = 0
        for title, (start, stop) in preset_rows:
            btn = QtWidgets.QPushButton(title)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, s=start, e=stop: self._select_range(s, e) if checked else self._clear_range())
            grid.addWidget(btn, r, c)
            c += 1
            if c >= 3:  # 3 колонки
                c = 0
                r += 1
        
        v.addWidget(grp_presets)
        
        # Таблица диапазонов
        grp_ranges = QtWidgets.QGroupBox("Диапазоны сканирования")
        vr = QtWidgets.QVBoxLayout(grp_ranges)
        
        self.tbl_ranges = QtWidgets.QTableWidget(0, 3)
        self.tbl_ranges.setHorizontalHeaderLabels(["Начало, МГц", "Конец, МГц", "Активность"])
        self.tbl_ranges.horizontalHeader().setStretchLastSection(True)
        self.tbl_ranges.itemSelectionChanged.connect(self._on_range_selected)
        vr.addWidget(self.tbl_ranges)
        
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Добавить текущий")
        self.btn_del = QtWidgets.QPushButton("Удалить")
        self.btn_merge = QtWidgets.QPushButton("Объединить")
        self.btn_clear = QtWidgets.QPushButton("Очистить все")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_del)
        btn_row.addWidget(self.btn_merge)
        btn_row.addWidget(self.btn_clear)
        vr.addLayout(btn_row)
        
        v.addWidget(grp_ranges)
        
        # Параметры детектора
        grp_params = QtWidgets.QGroupBox("Параметры детектора")
        fp = QtWidgets.QFormLayout(grp_params)
        
        self.th_dbm = QtWidgets.QDoubleSpinBox()
        self.th_dbm.setRange(-160, 30)
        self.th_dbm.setValue(-70)
        self.th_dbm.setSuffix(" дБм")
        
        self.min_width = QtWidgets.QSpinBox()
        self.min_width.setRange(1, 1000)
        self.min_width.setValue(5)
        self.min_width.setSuffix(" бинов")
        
        self.min_duration = QtWidgets.QSpinBox()
        self.min_duration.setRange(1, 60)
        self.min_duration.setValue(3)
        self.min_duration.setSuffix(" сек")
        
        self.chk_burst = QtWidgets.QCheckBox("Детектировать импульсные сигналы")
        self.chk_fhss = QtWidgets.QCheckBox("Детектировать FHSS (прыгающие)")
        
        fp.addRow("Порог:", self.th_dbm)
        fp.addRow("Мин. ширина:", self.min_width)
        fp.addRow("Мин. длительность:", self.min_duration)
        fp.addRow(self.chk_burst)
        fp.addRow(self.chk_fhss)
        
        v.addWidget(grp_params)
        
        # Таблица обнаружений
        grp_detections = QtWidgets.QGroupBox("Обнаруженные сигналы")
        vd = QtWidgets.QVBoxLayout(grp_detections)
        
        self.tbl_detections = QtWidgets.QTableWidget(0, 5)
        self.tbl_detections.setHorizontalHeaderLabels(["Время", "Частота", "Уровень", "Ширина", "Тип"])
        self.tbl_detections.horizontalHeader().setStretchLastSection(True)
        self.tbl_detections.setMaximumHeight(200)
        vd.addWidget(self.tbl_detections)
        
        v.addWidget(grp_detections)
        
        # Кнопки управления
        btns = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Начать детект")
        self.btn_stop = QtWidgets.QPushButton("Остановить")
        self.btn_stop.setEnabled(False)
        self.btn_export = QtWidgets.QPushButton("Экспорт лога")
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.btn_export)
        btns.addStretch(1)
        v.addLayout(btns)
        
        # Статус
        self.lbl_status = QtWidgets.QLabel("Готов к работе")
        self.lbl_status.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        v.addWidget(self.lbl_status)
        
        v.addStretch(1)
        
        # Обработчики
        self.btn_add.clicked.connect(self._add_current_range)
        self.btn_del.clicked.connect(self._delete_selected)
        self.btn_merge.clicked.connect(self._merge_ranges)
        self.btn_clear.clicked.connect(self._clear_ranges)
        self.btn_start.clicked.connect(self._start_detection)
        self.btn_stop.clicked.connect(self._stop_detection)
        self.btn_export.clicked.connect(self._export_log)

    def _select_range(self, start_mhz: float, stop_mhz: float):
        """Выбор диапазона и отправка сигнала для визуализации."""
        self._current_range = (start_mhz, stop_mhz)
        self.rangeSelected.emit(start_mhz, stop_mhz)
        self._add_range(start_mhz, stop_mhz)
        self.lbl_status.setText(f"Выбран диапазон: {start_mhz:.1f} - {stop_mhz:.1f} МГц")

    def _clear_range(self):
        """Очистка выбранного диапазона."""
        self._current_range = None
        self.rangeSelected.emit(0, 0)  # Сигнал для очистки визуализации

    def _add_range(self, start: float, stop: float):
        """Добавление диапазона в таблицу."""
        # Проверяем дубликаты
        for r in range(self.tbl_ranges.rowCount()):
            try:
                s = float(self.tbl_ranges.item(r, 0).text())
                e = float(self.tbl_ranges.item(r, 1).text())
                if abs(s - start) < 0.1 and abs(e - stop) < 0.1:
                    return  # Уже есть
            except Exception:
                continue
        
        row = self.tbl_ranges.rowCount()
        self.tbl_ranges.insertRow(row)
        self.tbl_ranges.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{start:.3f}"))
        self.tbl_ranges.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{stop:.3f}"))
        
        # Индикатор активности
        activity_item = QtWidgets.QTableWidgetItem("—")
        activity_item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tbl_ranges.setItem(row, 2, activity_item)

    def _add_current_range(self):
        """Добавление текущего диапазона из спектра."""
        # Берем из родительского окна текущие настройки
        if self.parent():
            try:
                spectrum = self.parent().spectrum_tab
                start = spectrum.start_mhz.value()
                stop = spectrum.stop_mhz.value()
                self._add_range(start, stop)
            except Exception:
                self._add_range(2400.0, 2483.5)  # По умолчанию

    def _delete_selected(self):
        """Удаление выбранных диапазонов."""
        rows = sorted({i.row() for i in self.tbl_ranges.selectedIndexes()}, reverse=True)
        for r in rows:
            self.tbl_ranges.removeRow(r)

    def _merge_ranges(self):
        """Объединение перекрывающихся диапазонов."""
        ranges = []
        for r in range(self.tbl_ranges.rowCount()):
            try:
                start = float(self.tbl_ranges.item(r, 0).text())
                stop = float(self.tbl_ranges.item(r, 1).text())
                ranges.append((start, stop))
            except Exception:
                continue
        
        if not ranges:
            return
        
        # Сортируем и объединяем
        ranges.sort()
        merged = []
        for start, stop in ranges:
            if not merged or start > merged[-1][1] + 0.1:
                merged.append([start, stop])
            else:
                merged[-1][1] = max(merged[-1][1], stop)
        
        # Обновляем таблицу
        self.tbl_ranges.setRowCount(0)
        for start, stop in merged:
            self._add_range(start, stop)

    def _clear_ranges(self):
        """Очистка всех диапазонов."""
        self.tbl_ranges.setRowCount(0)
        self._clear_range()

    def _on_range_selected(self):
        """При выборе диапазона в таблице."""
        rows = self.tbl_ranges.selectionModel().selectedRows()
        if rows:
            r = rows[0].row()
            try:
                start = float(self.tbl_ranges.item(r, 0).text())
                stop = float(self.tbl_ranges.item(r, 1).text())
                self._current_range = (start, stop)
                self.rangeSelected.emit(start, stop)
            except Exception:
                pass

    def _start_detection(self):
        """Запуск детектора."""
        if self.tbl_ranges.rowCount() == 0:
            QtWidgets.QMessageBox.warning(self, "Детектор", "Добавьте диапазоны для сканирования")
            return
        
        self._detecting = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Детекция активна...")
        self.lbl_status.setStyleSheet("padding: 5px; background-color: #ffcccc;")
        self.detectionStarted.emit()

    def _stop_detection(self):
        """Остановка детектора."""
        self._detecting = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Детекция остановлена")
        self.lbl_status.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        self.detectionStopped.emit()

    def push_data(self, freqs_hz: np.ndarray, row_dbm: np.ndarray):
        """Обработка данных от спектра."""
        if not self._detecting:
            return
        
        # Проверяем активность в каждом диапазоне
        for r in range(self.tbl_ranges.rowCount()):
            try:
                start_mhz = float(self.tbl_ranges.item(r, 0).text())
                stop_mhz = float(self.tbl_ranges.item(r, 1).text())
                
                # Находим индексы в данном диапазоне
                freqs_mhz = freqs_hz / 1e6
                mask = (freqs_mhz >= start_mhz) & (freqs_mhz <= stop_mhz)
                
                if np.any(mask):
                    # Проверяем превышение порога
                    max_power = np.max(row_dbm[mask])
                    
                    activity_item = self.tbl_ranges.item(r, 2)
                    if max_power > self.th_dbm.value():
                        # Обнаружена активность
                        activity_item.setText(f"{max_power:.1f} дБм")
                        activity_item.setBackground(QtGui.QBrush(QtGui.QColor(255, 200, 200)))
                        
                        # Добавляем в таблицу обнаружений
                        self._add_detection(freqs_mhz[mask][np.argmax(row_dbm[mask])], max_power)
                    else:
                        activity_item.setText("—")
                        activity_item.setBackground(QtGui.QBrush())
                        
            except Exception:
                continue

    def _add_detection(self, freq_mhz: float, power_dbm: float):
        """Добавление обнаружения в таблицу."""
        from PyQt5.QtCore import QDateTime
        
        # Ограничиваем количество записей
        if self.tbl_detections.rowCount() >= 100:
            self.tbl_detections.removeRow(0)
        
        row = self.tbl_detections.rowCount()
        self.tbl_detections.insertRow(row)
        
        time_str = QDateTime.currentDateTime().toString("HH:mm:ss")
        self.tbl_detections.setItem(row, 0, QtWidgets.QTableWidgetItem(time_str))
        self.tbl_detections.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{freq_mhz:.3f} МГц"))
        self.tbl_detections.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{power_dbm:.1f} дБм"))
        self.tbl_detections.setItem(row, 3, QtWidgets.QTableWidgetItem("—"))
        self.tbl_detections.setItem(row, 4, QtWidgets.QTableWidgetItem("Непрерывный"))
        
        # Прокручиваем вниз
        self.tbl_detections.scrollToBottom()

    def _export_log(self):
        """Экспорт лога обнаружений."""
        if self.tbl_detections.rowCount() == 0:
            QtWidgets.QMessageBox.information(self, "Экспорт", "Нет обнаружений для экспорта")
            return
        
        from PyQt5.QtCore import QDateTime
        default_name = f"detections_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.csv"
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить лог", default_name, "CSV files (*.csv)"
        )
        if not path:
            return
        
        try:
            import csv
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'freq_mhz', 'power_dbm', 'width', 'type'])
                
                for r in range(self.tbl_detections.rowCount()):
                    row_data = []
                    for c in range(5):
                        item = self.tbl_detections.item(r, c)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
            
            QtWidgets.QMessageBox.information(self, "Экспорт", f"Лог сохранен: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка экспорта: {e}")