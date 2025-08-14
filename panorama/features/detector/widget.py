from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import time
import json


@dataclass
class Detection:
    """Единичное обнаружение сигнала."""
    timestamp: float
    freq_mhz: float
    power_dbm: float
    bandwidth_khz: float
    duration_ms: float
    roi_index: int  # Индекс ROI в котором обнаружен
    confidence: float = 0.0
    
    
@dataclass
class ROIRegion:
    """Регион интереса для мониторинга."""
    id: int
    name: str
    start_mhz: float
    stop_mhz: float
    threshold_dbm: float
    min_width_bins: int
    enabled: bool = True
    detections: List[Detection] = field(default_factory=list)
    last_activity: Optional[float] = None
    

@dataclass
class DetectorState:
    """Состояние детектора."""
    is_active: bool = False
    start_time: Optional[float] = None
    total_detections: int = 0
    regions: List[ROIRegion] = field(default_factory=list)
    detection_history: List[Detection] = field(default_factory=list)
    

class DetectorWidget(QtWidgets.QWidget):
    """Виджет детектора активности с ROI и визуализацией."""
    
    rangeSelected = QtCore.pyqtSignal(float, float)  # start_mhz, stop_mhz
    detectionStarted = QtCore.pyqtSignal()
    detectionStopped = QtCore.pyqtSignal()
    signalDetected = QtCore.pyqtSignal(object)  # Detection
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._state = DetectorState()
        self._roi_id_seq = 0
        self._max_history = 1000  # Максимум записей в истории
        
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
            btn.clicked.connect(lambda checked, s=start, e=stop, n=title: 
                              self._add_roi(s, e, n) if checked else None)
            grid.addWidget(btn, r, c)
            c += 1
            if c >= 3:  # 3 колонки
                c = 0
                r += 1
        
        v.addWidget(grp_presets)
        
        # Таблица диапазонов
        grp_ranges = QtWidgets.QGroupBox("Диапазоны сканирования (ROI)")
        vr = QtWidgets.QVBoxLayout(grp_ranges)
        
        self.tbl_ranges = QtWidgets.QTableWidget(0, 5)
        self.tbl_ranges.setHorizontalHeaderLabels(["Вкл", "Название", "Начало, МГц", "Конец, МГц", "Активность"])
        self.tbl_ranges.horizontalHeader().setStretchLastSection(True)
        self.tbl_ranges.itemSelectionChanged.connect(self._on_range_selected)
        vr.addWidget(self.tbl_ranges)
        
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Добавить текущий")
        self.btn_del = QtWidgets.QPushButton("Удалить")
        self.btn_clear = QtWidgets.QPushButton("Очистить все")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_del)
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
        self.min_width.setRange(1, 100)
        self.min_width.setValue(5)
        self.min_width.setSuffix(" бинов")
        
        self.min_duration = QtWidgets.QSpinBox()
        self.min_duration.setRange(1, 60)
        self.min_duration.setValue(2)
        self.min_duration.setSuffix(" сек")
        
        self.chk_burst = QtWidgets.QCheckBox("Детектировать импульсные сигналы")
        self.chk_fhss = QtWidgets.QCheckBox("Детектировать FHSS (прыгающие)")
        
        fp.addRow("Порог по умолчанию:", self.th_dbm)
        fp.addRow("Мин. ширина:", self.min_width)
        fp.addRow("Мин. длительность:", self.min_duration)
        fp.addRow(self.chk_burst)
        fp.addRow(self.chk_fhss)
        
        v.addWidget(grp_params)
        
        # Таблица обнаружений
        grp_detections = QtWidgets.QGroupBox("Обнаруженные сигналы")
        vd = QtWidgets.QVBoxLayout(grp_detections)
        
        self.tbl_detections = QtWidgets.QTableWidget(0, 6)
        self.tbl_detections.setHorizontalHeaderLabels(["Время", "ROI", "Частота", "Уровень", "Ширина", "Длительность"])
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
        self.btn_clear.clicked.connect(self._clear_ranges)
        self.btn_start.clicked.connect(self._start_detection)
        self.btn_stop.clicked.connect(self._stop_detection)
        self.btn_export.clicked.connect(self._export_log)

    def _add_roi(self, start_mhz: float, stop_mhz: float, name: str = ""):
        """Добавление ROI региона."""
        # Проверяем дубликаты
        for roi in self._state.regions:
            if abs(roi.start_mhz - start_mhz) < 0.1 and abs(roi.stop_mhz - stop_mhz) < 0.1:
                return  # Уже есть
        
        self._roi_id_seq += 1
        roi = ROIRegion(
            id=self._roi_id_seq,
            name=name or f"ROI-{self._roi_id_seq}",
            start_mhz=start_mhz,
            stop_mhz=stop_mhz,
            threshold_dbm=self.th_dbm.value(),
            min_width_bins=self.min_width.value()
        )
        
        self._state.regions.append(roi)
        
        # Добавляем в таблицу
        row = self.tbl_ranges.rowCount()
        self.tbl_ranges.insertRow(row)
        
        # Чекбокс включения
        chk = QtWidgets.QCheckBox()
        chk.setChecked(True)
        chk.toggled.connect(lambda checked, r=roi: setattr(r, 'enabled', checked))
        self.tbl_ranges.setCellWidget(row, 0, chk)
        
        self.tbl_ranges.setItem(row, 1, QtWidgets.QTableWidgetItem(roi.name))
        self.tbl_ranges.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{start_mhz:.3f}"))
        self.tbl_ranges.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{stop_mhz:.3f}"))
        
        # Индикатор активности
        activity_item = QtWidgets.QTableWidgetItem("—")
        activity_item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tbl_ranges.setItem(row, 4, activity_item)
        
        # Визуализация региона
        self.rangeSelected.emit(start_mhz, stop_mhz)

    def _add_current_range(self):
        """Добавление текущего диапазона из спектра."""
        # Берем из родительского окна текущие настройки
        if self.parent():
            try:
                spectrum = self.parent().spectrum_tab
                start = spectrum.start_mhz.value()
                stop = spectrum.stop_mhz.value()
                self._add_roi(start, stop, "Текущий спектр")
            except Exception:
                self._add_roi(2400.0, 2483.5, "По умолчанию")

    def _delete_selected(self):
        """Удаление выбранных диапазонов."""
        rows = sorted({i.row() for i in self.tbl_ranges.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self._state.regions):
                del self._state.regions[r]
            self.tbl_ranges.removeRow(r)

    def _clear_ranges(self):
        """Очистка всех диапазонов."""
        self.tbl_ranges.setRowCount(0)
        self._state.regions.clear()
        self.rangeSelected.emit(0, 0)  # Сигнал для очистки визуализации

    def _on_range_selected(self):
        """При выборе диапазона в таблице."""
        rows = self.tbl_ranges.selectionModel().selectedRows()
        if rows and self._state.regions:
            r = rows[0].row()
            if 0 <= r < len(self._state.regions):
                roi = self._state.regions[r]
                self.rangeSelected.emit(roi.start_mhz, roi.stop_mhz)

    def _start_detection(self):
        """Запуск детектора."""
        if not self._state.regions:
            QtWidgets.QMessageBox.warning(self, "Детектор", "Добавьте диапазоны для сканирования")
            return
        
        self._state.is_active = True
        self._state.start_time = time.time()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Детекция активна...")
        self.lbl_status.setStyleSheet("padding: 5px; background-color: #ffcccc;")
        self.detectionStarted.emit()

    def _stop_detection(self):
        """Остановка детектора."""
        self._state.is_active = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Детекция остановлена")
        self.lbl_status.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        self.detectionStopped.emit()

    def push_data(self, freqs_hz: np.ndarray, row_dbm: np.ndarray):
        """Обработка данных от спектра."""
        if not self._state.is_active or not self._state.regions:
            return
        
        freqs_mhz = freqs_hz / 1e6
        
        # Проверяем каждый активный ROI
        for roi_idx, roi in enumerate(self._state.regions):
            if not roi.enabled:
                continue
            
            # Находим индексы в данном диапазоне
            mask = (freqs_mhz >= roi.start_mhz) & (freqs_mhz <= roi.stop_mhz)
            
            if not np.any(mask):
                continue
            
            # Анализируем сигнал в ROI
            roi_freqs = freqs_mhz[mask]
            roi_power = row_dbm[mask]
            
            # Простой детектор по порогу
            above_threshold = roi_power > roi.threshold_dbm
            
            if np.any(above_threshold):
                # Находим пики
                max_idx = np.argmax(roi_power)
                max_power = roi_power[max_idx]
                max_freq = roi_freqs[max_idx]
                
                # Оцениваем ширину сигнала
                threshold_3db = max_power - 3.0
                above_3db = roi_power > threshold_3db
                bandwidth_bins = np.sum(above_3db)
                
                if bandwidth_bins >= roi.min_width_bins:
                    # Создаем обнаружение
                    detection = Detection(
                        timestamp=time.time(),
                        freq_mhz=max_freq,
                        power_dbm=max_power,
                        bandwidth_khz=bandwidth_bins * (freqs_hz[1] - freqs_hz[0]) / 1000.0 if len(freqs_hz) > 1 else 0,
                        duration_ms=0,  # Будет обновлено при отслеживании
                        roi_index=roi_idx,
                        confidence=min(1.0, (max_power - roi.threshold_dbm) / 30.0)  # Нормализуем
                    )
                    
                    # Добавляем в историю
                    roi.detections.append(detection)
                    roi.last_activity = detection.timestamp
                    self._state.detection_history.append(detection)
                    self._state.total_detections += 1
                    
                    # Ограничиваем размер истории
                    if len(self._state.detection_history) > self._max_history:
                        self._state.detection_history = self._state.detection_history[-self._max_history:]
                    
                    # Обновляем UI
                    self._add_detection_to_table(detection, roi)
                    self._update_roi_activity(roi_idx, max_power)
                    
                    # Эмитим сигнал
                    self.signalDetected.emit(detection)
                else:
                    # Сигнал есть но слишком узкий
                    self._update_roi_activity(roi_idx, max_power, weak=True)
            else:
                # Нет активности
                self._update_roi_activity(roi_idx, None)

    def _add_detection_to_table(self, detection: Detection, roi: ROIRegion):
        """Добавление обнаружения в таблицу."""
        from PyQt5.QtCore import QDateTime
        
        # Ограничиваем количество записей в таблице
        if self.tbl_detections.rowCount() >= 100:
            self.tbl_detections.removeRow(0)
        
        row = self.tbl_detections.rowCount()
        self.tbl_detections.insertRow(row)
        
        time_str = QDateTime.fromSecsSinceEpoch(int(detection.timestamp)).toString("HH:mm:ss")
        self.tbl_detections.setItem(row, 0, QtWidgets.QTableWidgetItem(time_str))
        self.tbl_detections.setItem(row, 1, QtWidgets.QTableWidgetItem(roi.name))
        self.tbl_detections.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{detection.freq_mhz:.3f} МГц"))
        self.tbl_detections.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{detection.power_dbm:.1f} дБм"))
        self.tbl_detections.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{detection.bandwidth_khz:.1f} кГц"))
        self.tbl_detections.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{detection.duration_ms:.0f} мс"))
        
        # Цветовая индикация по уровню
        if detection.power_dbm >= -50:
            color = QtGui.QColor(255, 200, 200)  # Красный - сильный
        elif detection.power_dbm >= -70:
            color = QtGui.QColor(255, 255, 200)  # Желтый - средний
        else:
            color = QtGui.QColor(200, 255, 200)  # Зеленый - слабый
        
        for col in range(6):
            item = self.tbl_detections.item(row, col)
            if item:
                item.setBackground(QtGui.QBrush(color))
        
        # Прокручиваем вниз
        self.tbl_detections.scrollToBottom()
        
        # Обновляем статус
        self.lbl_status.setText(f"Обнаружен сигнал: {detection.freq_mhz:.1f} МГц @ {detection.power_dbm:.1f} дБм")

    def _update_roi_activity(self, roi_idx: int, power_dbm: Optional[float], weak: bool = False):
        """Обновление индикатора активности ROI."""
        if roi_idx >= self.tbl_ranges.rowCount():
            return
        
        activity_item = self.tbl_ranges.item(roi_idx, 4)
        if not activity_item:
            return
        
        if power_dbm is None:
            activity_item.setText("—")
            activity_item.setBackground(QtGui.QBrush())
        elif weak:
            activity_item.setText(f"{power_dbm:.1f} дБм (узкий)")
            activity_item.setBackground(QtGui.QBrush(QtGui.QColor(255, 255, 200)))
        else:
            activity_item.setText(f"{power_dbm:.1f} дБм")
            activity_item.setBackground(QtGui.QBrush(QtGui.QColor(255, 200, 200)))

    def _export_log(self):
        """Экспорт лога обнаружений."""
        if not self._state.detection_history:
            QtWidgets.QMessageBox.information(self, "Экспорт", "Нет обнаружений для экспорта")
            return
        
        from PyQt5.QtCore import QDateTime
        default_name = f"detections_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.csv"
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить лог", default_name, "CSV files (*.csv);;JSON files (*.json)"
        )
        if not path:
            return
        
        try:
            if path.endswith('.json'):
                # Экспорт в JSON
                data = {
                    'metadata': {
                        'export_time': time.time(),
                        'total_detections': self._state.total_detections,
                        'session_start': self._state.start_time,
                        'regions': [
                            {
                                'id': roi.id,
                                'name': roi.name,
                                'start_mhz': roi.start_mhz,
                                'stop_mhz': roi.stop_mhz,
                                'threshold_dbm': roi.threshold_dbm,
                                'detections_count': len(roi.detections)
                            }
                            for roi in self._state.regions
                        ]
                    },
                    'detections': [
                        {
                            'timestamp': d.timestamp,
                            'freq_mhz': d.freq_mhz,
                            'power_dbm': d.power_dbm,
                            'bandwidth_khz': d.bandwidth_khz,
                            'duration_ms': d.duration_ms,
                            'roi_index': d.roi_index,
                            'confidence': d.confidence
                        }
                        for d in self._state.detection_history
                    ]
                }
                
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            else:
                # Экспорт в CSV
                import csv
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'datetime', 'roi_name', 'freq_mhz', 
                                   'power_dbm', 'bandwidth_khz', 'duration_ms', 'confidence'])
                    
                    for d in self._state.detection_history:
                        roi_name = self._state.regions[d.roi_index].name if d.roi_index < len(self._state.regions) else "Unknown"
                        dt_str = QDateTime.fromSecsSinceEpoch(int(d.timestamp)).toString("yyyy-MM-dd HH:mm:ss")
                        writer.writerow([
                            d.timestamp,
                            dt_str,
                            roi_name,
                            f"{d.freq_mhz:.6f}",
                            f"{d.power_dbm:.2f}",
                            f"{d.bandwidth_khz:.1f}",
                            f"{d.duration_ms:.0f}",
                            f"{d.confidence:.3f}"
                        ])
            
            QtWidgets.QMessageBox.information(self, "Экспорт", 
                f"Сохранено {len(self._state.detection_history)} обнаружений:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка экспорта: {e}")

    def get_state(self) -> DetectorState:
        """Получить текущее состояние детектора."""
        return self._state

    def get_active_rois(self) -> List[ROIRegion]:
        """Получить список активных ROI."""
        return [roi for roi in self._state.regions if roi.enabled]

    def get_recent_detections(self, seconds: float = 60.0) -> List[Detection]:
        """Получить обнаружения за последние N секунд."""
        cutoff = time.time() - seconds
        return [d for d in self._state.detection_history if d.timestamp > cutoff]

    def save_settings(self, settings):
        """Сохранение настроек."""
        settings.beginGroup("detector")
        settings.setValue("threshold", self.th_dbm.value())
        settings.setValue("min_width", self.min_width.value())
        settings.setValue("min_duration", self.min_duration.value())
        settings.setValue("detect_burst", self.chk_burst.isChecked())
        settings.setValue("detect_fhss", self.chk_fhss.isChecked())
        
        # Сохраняем ROI регионы
        regions_data = []
        for roi in self._state.regions:
            regions_data.append({
                'name': roi.name,
                'start_mhz': roi.start_mhz,
                'stop_mhz': roi.stop_mhz,
                'threshold_dbm': roi.threshold_dbm,
                'min_width_bins': roi.min_width_bins,
                'enabled': roi.enabled
            })
        settings.setValue("roi_regions", json.dumps(regions_data))
        settings.endGroup()

    def restore_settings(self, settings):
        """Восстановление настроек."""
        settings.beginGroup("detector")
        self.th_dbm.setValue(float(settings.value("threshold", -70.0)))
        self.min_width.setValue(int(settings.value("min_width", 5)))
        self.min_duration.setValue(int(settings.value("min_duration", 2)))
        self.chk_burst.setChecked(settings.value("detect_burst", False, type=bool))
        self.chk_fhss.setChecked(settings.value("detect_fhss", False, type=bool))
        
        # Восстанавливаем ROI регионы
        try:
            regions_json = settings.value("roi_regions", "[]")
            regions_data = json.loads(regions_json)
            for roi_data in regions_data:
                self._add_roi(
                    roi_data['start_mhz'],
                    roi_data['stop_mhz'],
                    roi_data['name']
                )
        except Exception:
            pass
        
        settings.endGroup()