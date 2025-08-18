# panorama/features/peaks/unified_peaks.py
from __future__ import annotations
from typing import Optional, List, Deque, Dict, Tuple
from collections import deque
from dataclasses import dataclass
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QAbstractTableModel, QModelIndex
import numpy as np
import json
import time


@dataclass
class PeakEntry:
    """Структура данных для записи пика."""
    freq_hz: float
    last_dbm: float
    max_dbm: float
    min_dbm: float
    avg_dbm: float
    count: int
    first_seen: float
    last_seen: float
    width_khz: float = 0.0
    q_factor: float = 0.0
    above_noise_db: float = 0.0

    def update(self, dbm: float, now: float, ema_alpha: float = 0.2):
        """Обновляет запись пика."""
        self.last_dbm = dbm
        self.max_dbm = max(self.max_dbm, dbm)
        self.min_dbm = min(self.min_dbm, dbm)
        self.avg_dbm = (1 - ema_alpha) * self.avg_dbm + ema_alpha * dbm
        self.count += 1
        self.last_seen = now


class AutoPeaksEngine(QtCore.QObject):
    """Движок автопиков с накоплением и обновлением."""
    changed = QtCore.pyqtSignal(list)  # список изменённых строк

    def __init__(self, merge_hz: float = 200e3, inactivity_sec: Optional[float] = None, parent=None):
        super().__init__(parent)
        self.merge_hz = merge_hz
        self.inactivity_sec = inactivity_sec
        self._entries: Dict[int, PeakEntry] = {}
        self._order: List[int] = []
        self._bucket_to_row: Dict[int, int] = {}

    def _bucket(self, f_hz: float) -> int:
        """Группирует частоты в бакеты для слияния."""
        return int(round(f_hz / self.merge_hz))

    def entries(self) -> List[PeakEntry]:
        """Возвращает список всех записей."""
        return [self._entries[b] for b in self._order]

    def size(self) -> int:
        """Возвращает количество записей."""
        return len(self._order)

    def ingest(self, peaks: List[Tuple[float, float, float, float, float]], 
               bin_hz: Optional[float] = None, now: Optional[float] = None):
        """Добавляет новые пики. peaks: [(freq_hz, dbm, width_khz, q_factor, above_noise_db)]"""
        if now is None:
            now = time.time()
        if bin_hz is not None:
            self.merge_hz = max(self.merge_hz, bin_hz)

        changed_rows: List[int] = []
        for f_hz, dbm, width_khz, q_factor, above_noise_db in peaks:
            b = self._bucket(f_hz)
            row = self._bucket_to_row.get(b)
            if row is None:
                e = PeakEntry(
                    freq_hz=f_hz, last_dbm=dbm, max_dbm=dbm, min_dbm=dbm,
                    avg_dbm=dbm, count=1, first_seen=now, last_seen=now,
                    width_khz=width_khz, q_factor=q_factor, above_noise_db=above_noise_db
                )
                self._entries[b] = e
                self._order.append(b)
                row = len(self._order) - 1
                self._bucket_to_row[b] = row
                changed_rows.append(row)
            else:
                e = self._entries[self._order[row]]
                e.update(dbm, now)
                # Обновляем дополнительные параметры
                e.width_khz = max(e.width_khz, width_khz)
                e.q_factor = max(e.q_factor, q_factor)
                e.above_noise_db = max(e.above_noise_db, above_noise_db)
                changed_rows.append(row)

        if self.inactivity_sec:
            to_del: List[int] = []
            for i, b in enumerate(self._order):
                if now - self._entries[b].last_seen > self.inactivity_sec:
                    to_del.append(i)
            if to_del:
                for i in reversed(to_del):
                    b = self._order[i]
                    del self._entries[b]
                    del self._order[i]
                self._bucket_to_row = {b: i for i, b in enumerate(self._order)}
                changed_rows = list(range(self.size()))

        if changed_rows:
            self.changed.emit(changed_rows)

    def clear_all(self):
        """Очищает все записи."""
        self._entries.clear()
        self._order.clear()
        self._bucket_to_row.clear()
        self.changed.emit([])


class AutoPeaksTableModel(QAbstractTableModel):
    """Модель данных для таблицы автопиков."""
    COLS = ["Freq, MHz", "Last, dBm", "Max, dBm", "Avg, dBm", "Count", "Width, kHz", "Q-factor", "Above noise, dB", "First seen", "Last seen"]

    def __init__(self, engine: AutoPeaksEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.engine.changed.connect(self._on_changed)
        # Track current row count for efficient updates
        self._row_count = self.engine.size()

    def rowCount(self, parent=QModelIndex()):
        return self._row_count

    def columnCount(self, parent=QModelIndex()):
        return len(self.COLS)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return self.COLS[section]
        return section + 1

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        e = self.engine.entries()[index.row()]
        c = index.column()
        if role == QtCore.Qt.DisplayRole:
            if c == 0: return f"{e.freq_hz/1e6:.4f}"
            if c == 1: return f"{e.last_dbm:.1f}"
            if c == 2: return f"{e.max_dbm:.1f}"
            if c == 3: return f"{e.avg_dbm:.1f}"
            if c == 4: return str(e.count)
            if c == 5: return f"{e.width_khz:.1f}"
            if c == 6: return f"{e.q_factor:.1f}"
            if c == 7: return f"{e.above_noise_db:.1f}"
            if c == 8: return time.strftime("%H:%M:%S", time.localtime(e.first_seen))
            if c == 9: return time.strftime("%H:%M:%S", time.localtime(e.last_seen))
        if role == QtCore.Qt.UserRole:
            return {
                "freq_hz": e.freq_hz, "last_dbm": e.last_dbm, "max_dbm": e.max_dbm,
                "avg_dbm": e.avg_dbm, "count": e.count, "width_khz": e.width_khz,
                "q_factor": e.q_factor, "above_noise_db": e.above_noise_db,
                "first_seen": e.first_seen, "last_seen": e.last_seen
            }
        return None

    def _on_changed(self, rows: List[int]):
        new_size = self.engine.size()
        old_count = self._row_count

        if not rows and new_size == 0:
            self.beginResetModel()
            self._row_count = 0
            self.endResetModel()
            return

        if new_size < old_count:
            self.beginResetModel()
            self._row_count = new_size
            self.endResetModel()
            old_count = new_size
        elif new_size > old_count:
            self.beginInsertRows(QModelIndex(), old_count, new_size - 1)
            self._row_count = new_size
            self.endInsertRows()

        for r in rows:
            if r < old_count:
                tl = self.index(r, 0)
                br = self.index(r, self.columnCount()-1)
                self.dataChanged.emit(tl, br, [QtCore.Qt.DisplayRole, QtCore.Qt.UserRole])


class AutoPeaksFilterProxy(QtCore.QSortFilterProxyModel):
    """Прокси для фильтрации автопиков."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._freq_query: Optional[str] = None
        self._min_count: int = 0
        self._since_sec: Optional[float] = None
        self._min_width: float = 0.0
        self._min_q_factor: float = 0.0
        self.setDynamicSortFilter(True)

    def set_freq_query(self, text: Optional[str]):
        self._freq_query = text.strip() if text else None
        self.invalidateFilter()

    def set_min_count(self, n: int):
        self._min_count = max(0, int(n))
        self.invalidateFilter()

    def set_since_seconds(self, sec: Optional[float]):
        self._since_sec = sec if sec and sec > 0 else None
        self.invalidateFilter()

    def set_min_width(self, width: float):
        self._min_width = max(0.0, float(width))
        self.invalidateFilter()

    def set_min_q_factor(self, q: float):
        self._min_q_factor = max(0.0, float(q))
        self.invalidateFilter()

    def filterAcceptsRow(self, src_row: int, src_parent) -> bool:
        m = self.sourceModel()
        idx = m.index(src_row, 0, src_parent)
        p = m.data(idx, QtCore.Qt.UserRole)
        if not p:
            return True
            
        if int(p["count"]) < self._min_count:
            return False
        if float(p["width_khz"]) < self._min_width:
            return False
        if float(p["q_factor"]) < self._min_q_factor:
            return False
        if self._since_sec is not None and (time.time() - float(p["last_seen"]) > self._since_sec):
            return False
            
        if self._freq_query:
            txt = self._freq_query.replace(" ", "")
            try:
                f = float(p["freq_hz"])
                if "-" in txt:
                    l, r = txt.split("-", 1)
                    f1, f2 = float(l)*1e6, float(r)*1e6
                    if not (min(f1, f2) <= f <= max(f1, f2)):
                        return False
                else:
                    f0 = float(txt)*1e6
                    merge_hz = getattr(m.engine, "merge_hz", 200e3)
                    if abs(f - f0) > (merge_hz/2):
                        return False
            except Exception:
                pass
        return True


class UnifiedPeaksWidget(QtWidgets.QWidget):
    """Объединенный виджет автопиков с настройками и управлением."""
    
    goToFreq = QtCore.pyqtSignal(float)  # freq_hz
    peakDetected = QtCore.pyqtSignal(dict)  # peak info
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Данные спектра
        self._freqs: Optional[np.ndarray] = None
        self._row: Optional[np.ndarray] = None
        self._history: Deque[np.ndarray] = deque(maxlen=10)
        self._baseline: Optional[np.ndarray] = None
        
        # Защита от слишком частых вызовов
        self._last_peaks_update = 0
        self._min_update_interval = 0.1  # Минимум 100мс между обновлениями
        
        # Движок автопиков
        self._auto_engine = AutoPeaksEngine(merge_hz=200e3, inactivity_sec=None, parent=self)
        self._auto_model = AutoPeaksTableModel(self._auto_engine, parent=self)
        self._auto_proxy = AutoPeaksFilterProxy(self)
        self._auto_proxy.setSourceModel(self._auto_model)
        
        self._build_ui()
        self._connect_signals()
        
    def _build_ui(self):
        """Настройка интерфейса."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        
        # Панель параметров автопиков
        params_group = QtWidgets.QGroupBox("Параметры автопиков")
        params_layout = QtWidgets.QFormLayout(params_group)
        
        # Включение/выключение автопиков
        self.auto_peaks_enabled = QtWidgets.QCheckBox("Автопики включены")
        self.auto_peaks_enabled.setChecked(True)
        self.auto_peaks_enabled.setToolTip("Включить/выключить автоматический поиск пиков")
        
        # Статус автопиков
        self.lbl_autopeaks_status = QtWidgets.QLabel("🟢 Автопики активны")
        self.lbl_autopeaks_status.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-weight: bold;
                padding: 5px;
                border: 1px solid #4CAF50;
                border-radius: 3px;
                background-color: rgba(76, 175, 80, 0.1);
            }
        """)
        
        # Режим порога
        self.threshold_mode = QtWidgets.QComboBox()
        self.threshold_mode.addItems(["Адаптивный (baseline + N)", "Фиксированный порог"])
        
        # Адаптивный порог
        self.adaptive_offset = QtWidgets.QDoubleSpinBox()
        self.adaptive_offset.setRange(3, 50)
        self.adaptive_offset.setValue(20)
        self.adaptive_offset.setSuffix(" дБ над шумом")
        
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
        
        # Метод расчета baseline
        self.baseline_method = QtWidgets.QComboBox()
        self.baseline_method.addItems(["Медиана", "Среднее", "Минимум", "Персентиль"])
        
        # Персентиль для baseline
        self.baseline_percentile = QtWidgets.QSpinBox()
        self.baseline_percentile.setRange(1, 99)
        self.baseline_percentile.setValue(25)
        self.baseline_percentile.setSuffix(" %")
        self.baseline_percentile.setEnabled(False)
        
        # Добавляем параметры в форму
        params_layout.addRow("Режим порога:", self.threshold_mode)
        params_layout.addRow("Адаптивный порог:", self.adaptive_offset)
        params_layout.addRow("Фиксированный порог:", self.fixed_threshold)
        params_layout.addRow("Мин. расстояние:", self.min_distance)
        params_layout.addRow("Мин. ширина:", self.min_width)
        params_layout.addRow("Окно baseline:", self.baseline_window)
        params_layout.addRow("Метод baseline:", self.baseline_method)
        params_layout.addRow("Персентиль:", self.baseline_percentile)
        params_layout.addRow(self.auto_peaks_enabled)
        params_layout.addRow("Статус:", self.lbl_autopeaks_status)
        
        layout.addWidget(params_group)
        
        # Панель фильтров
        filt_group = QtWidgets.QGroupBox("Фильтры")
        filt_layout = QtWidgets.QHBoxLayout(filt_group)
        
        # Поиск по частоте
        self._ed_freq = QtWidgets.QLineEdit()
        self._ed_freq.setPlaceholderText("Частота МГц или диапазон: 5658.9  |  5650-5660")
        self._ed_freq.setMinimumWidth(200)
        
        # Минимальное количество срабатываний
        self._sp_min = QtWidgets.QSpinBox()
        self._sp_min.setRange(0, 1_000_000)
        self._sp_min.setPrefix("min hits ≥ ")
        self._sp_min.setValue(0)
        
        # Свежесть данных
        self._sp_sec = QtWidgets.QSpinBox()
        self._sp_sec.setRange(0, 86400)
        self._sp_sec.setPrefix("за последние сек: ")
        self._sp_sec.setValue(0)
        
        # Минимальная ширина для фильтра
        self._sp_min_width = QtWidgets.QDoubleSpinBox()
        self._sp_min_width.setRange(0, 1000)
        self._sp_min_width.setPrefix("мин. ширина ≥ ")
        self._sp_min_width.setSuffix(" кГц")
        self._sp_min_width.setValue(0)
        
        # Минимальный Q-фактор для фильтра
        self._sp_min_q = QtWidgets.QDoubleSpinBox()
        self._sp_min_q.setRange(0, 1000)
        self._sp_min_q.setPrefix("мин. Q ≥ ")
        self._sp_min_q.setValue(0)
        
        filt_layout.addWidget(QtWidgets.QLabel("Поиск:"))
        filt_layout.addWidget(self._ed_freq)
        filt_layout.addWidget(self._sp_min)
        filt_layout.addWidget(self._sp_sec)
        filt_layout.addWidget(self._sp_min_width)
        filt_layout.addWidget(self._sp_min_q)
        
        layout.addWidget(filt_group)
        
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
        
        # Таблица автопиков
        self._tbl_auto = QtWidgets.QTableView()
        self._tbl_auto.setModel(self._auto_proxy)
        self._tbl_auto.setSortingEnabled(True)
        self._tbl_auto.verticalHeader().setVisible(False)
        self._tbl_auto.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._tbl_auto.setAlternatingRowColors(True)
        
        # Настройка колонок
        self._tbl_auto.horizontalHeader().setStretchLastSection(False)
        for i in range(10):
            self._tbl_auto.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
        
        layout.addWidget(self._tbl_auto)
        
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
        
    def _connect_signals(self):
        """Подключение сигналов."""
        # Основные параметры
        self.auto_peaks_enabled.toggled.connect(self._on_autopeaks_toggled)
        self.threshold_mode.currentTextChanged.connect(self._on_threshold_mode_changed)
        self.baseline_method.currentTextChanged.connect(self._on_baseline_method_changed)
        
        # Автообновление при изменении параметров
        self.adaptive_offset.valueChanged.connect(self._on_params_changed)
        self.fixed_threshold.valueChanged.connect(self._on_params_changed)
        self.min_distance.valueChanged.connect(self._on_params_changed)
        self.min_width.valueChanged.connect(self._on_params_changed)
        
        # Фильтры
        self._ed_freq.textChanged.connect(self._auto_proxy.set_freq_query)
        self._sp_min.valueChanged.connect(self._auto_proxy.set_min_count)
        self._sp_sec.valueChanged.connect(lambda v: self._auto_proxy.set_since_seconds(v or None))
        self._sp_min_width.valueChanged.connect(self._auto_proxy.set_min_width)
        self._sp_min_q.valueChanged.connect(self._auto_proxy.set_min_q_factor)
        
        # Кнопки
        self.btn_find.clicked.connect(self._find_peaks)
        self.btn_clear.clicked.connect(self.clear_history)
        self.btn_export.clicked.connect(self._export_peaks)
        
        # Двойной клик по строке
        self._tbl_auto.doubleClicked.connect(self._on_row_double_clicked)
        
        # Обновление статуса при изменении данных
        self._auto_engine.changed.connect(self._update_status)
        
    def _on_autopeaks_toggled(self, enabled: bool):
        """Обработчик переключения автопиков."""
        if enabled:
            self.lbl_autopeaks_status.setText("🟢 Автопики активны")
            self.lbl_autopeaks_status.setStyleSheet("""
                QLabel {
                    color: #4CAF50;
                    font-weight: bold;
                    padding: 5px;
                    border: 1px solid #4CAF50;
                    border-radius: 3px;
                    background-color: rgba(76, 175, 80, 0.1);
                }
            """)
        else:
            self.lbl_autopeaks_status.setText("🔴 Автопики отключены")
            self.lbl_autopeaks_status.setStyleSheet("""
                QLabel {
                    color: #f44336;
                    font-weight: bold;
                    padding: 5px;
                    border: 1px solid #f44336;
                    border-radius: 3px;
                    background-color: rgba(244, 67, 54, 0.1);
                }
            """)
        
        # Если автопики включены и есть данные, сразу ищем пики
        if enabled and self._freqs is not None and self._row is not None:
            self._find_peaks()
            
    def _on_threshold_mode_changed(self, text):
        """Переключение режима порога."""
        is_adaptive = "Адаптивный" in text
        self.adaptive_offset.setEnabled(is_adaptive)
        self.fixed_threshold.setEnabled(not is_adaptive)
        self.baseline_window.setEnabled(is_adaptive)
        self.baseline_method.setEnabled(is_adaptive)
        
        if self.auto_peaks_enabled.isChecked() and self._freqs is not None:
            self._find_peaks()
            
    def _on_baseline_method_changed(self, text):
        """Изменение метода расчета baseline."""
        self.baseline_percentile.setEnabled("Персентиль" in text)
        
        if self.auto_peaks_enabled.isChecked() and self._freqs is not None:
            self._find_peaks()
            
    def _on_params_changed(self):
        """При изменении параметров."""
        if self.auto_peaks_enabled.isChecked() and self._freqs is not None:
            self._find_peaks()
            
    def _on_row_double_clicked(self, index):
        """Обработка двойного клика по строке - переход к частоте."""
        proxy_index = self._auto_proxy.index(index.row(), 0)
        source_index = self._auto_proxy.mapToSource(proxy_index)
        if source_index.isValid():
            entry = self._auto_engine.entries()[source_index.row()]
            self.goToFreq.emit(entry.freq_hz)
            
    def _update_status(self, changed_rows):
        """Обновление статусной строки."""
        total = self._auto_engine.size()
        if total == 0:
            self.lbl_peaks_count.setText("Пиков: 0")
        else:
            self.lbl_peaks_count.setText(f"Пиков: {total}")
            
    def update_from_row(self, freqs_hz, row_dbm):
        """Обновление из данных спектра."""
        freqs_hz = np.asarray(freqs_hz, dtype=float)
        row_dbm = np.asarray(row_dbm, dtype=float)
        
        # Проверяем, изменились ли размеры данных
        if self._freqs is not None and self._freqs.size != freqs_hz.size:
            self.clear_history()
            
        self._freqs = freqs_hz
        self._row = row_dbm
        
        if self._freqs.size != self._row.size:
            return
        
        # Добавляем в историю для baseline
        if self._row.size > 0:
            self._history.append(self._row.copy())
            self._calculate_baseline()
            
            # Автопики работают всегда, если включены, но с защитой от слишком частых вызовов
            if self.auto_peaks_enabled.isChecked():
                current_time = time.time()
                if current_time - self._last_peaks_update >= self._min_update_interval:
                    self._last_peaks_update = current_time
                    self._find_peaks()
                
    def _calculate_baseline(self):
        """Вычисляет baseline по истории."""
        if len(self._history) < 3:
            return
            
        if self._freqs is None:
            return
            
        expected_size = self._freqs.size
        valid_history = []
        
        for hist_row in self._history:
            if hist_row.size == expected_size:
                valid_history.append(hist_row)
        
        if len(valid_history) < 3:
            return
            
        try:
            window_size = min(self.baseline_window.value(), len(valid_history))
            history_array = np.array(valid_history[-window_size:])
            
            # Проверяем, что массив не пустой и содержит валидные данные
            if history_array.size == 0 or np.any(np.isnan(history_array)) or np.any(np.isinf(history_array)):
                return
                
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
                
            # Проверяем результат на валидность
            if self._baseline is not None and not np.any(np.isnan(self._baseline)) and not np.any(np.isinf(self._baseline)):
                avg_baseline = np.mean(self._baseline)
                self.lbl_baseline.setText(f"Baseline: {avg_baseline:.1f} дБм")
            else:
                self._baseline = None
                
        except Exception as e:
            print(f"Ошибка при вычислении baseline: {e}")
            self._baseline = None
            
    def _get_threshold(self) -> np.ndarray:
        """Получает массив порогов для каждого бина."""
        if self._row is None:
            return None
            
        try:
            if "Адаптивный" in self.threshold_mode.currentText():
                if self._baseline is None:
                    self._baseline = self._row - 10
                elif self._baseline.size != self._row.size:
                    self._baseline = self._row - 10
                    
                # Проверяем валидность baseline
                if self._baseline is not None and not np.any(np.isnan(self._baseline)) and not np.any(np.isinf(self._baseline)):
                    threshold = self._baseline + self.adaptive_offset.value()
                    # Проверяем валидность результата
                    if not np.any(np.isnan(threshold)) and not np.any(np.isinf(threshold)):
                        avg_threshold = np.mean(threshold)
                        self.lbl_threshold.setText(f"Порог: {avg_threshold:.1f} дБм")
                        return threshold
                
                # Если что-то пошло не так, используем фиксированный порог
                threshold_value = self.fixed_threshold.value()
                self.lbl_threshold.setText(f"Порог: {threshold_value:.1f} дБм (fallback)")
                return np.full_like(self._row, threshold_value)
            else:
                threshold_value = self.fixed_threshold.value()
                self.lbl_threshold.setText(f"Порог: {threshold_value:.1f} дБм")
                return np.full_like(self._row, threshold_value)
        except Exception as e:
            print(f"Ошибка при вычислении порога: {e}")
            # Fallback на фиксированный порог
            threshold_value = self.fixed_threshold.value()
            self.lbl_threshold.setText(f"Порог: {threshold_value:.1f} дБм (error)")
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
            return
            
        # Находим локальные максимумы
        peaks = []
        
        # Защита от деления на ноль
        freq_step = self._freqs[1] - self._freqs[0]
        if freq_step <= 0:
            return
            
        min_distance_bins = max(1, int(self.min_distance.value() * 1000 / freq_step))
        
        # Ограничиваем количество пиков для предотвращения зависания
        max_peaks = 100
        
        for i in range(1, len(self._row) - 1):
            if len(peaks) >= max_peaks:
                break
                
            if above_threshold[i] and self._row[i] > self._row[i-1] and self._row[i] >= self._row[i+1]:
                # Проверяем ширину пика
                width = self._calculate_peak_width(i)
                
                if width >= self.min_width.value():
                    # Вычисляем Q-фактор с защитой от деления на ноль
                    if width > 0 and freq_step > 0:
                        q_factor = self._freqs[i] / (width * freq_step)
                    else:
                        q_factor = 0
                    
                    # Вычисляем уровень над шумом
                    above_noise = self._row[i] - threshold[i]
                    
                    peaks.append((
                        self._freqs[i],           # freq_hz
                        self._row[i],             # dbm
                        width * freq_step / 1000,  # width_khz
                        q_factor,                 # q_factor
                        above_noise               # above_noise_db
                    ))
                    
        if peaks:
            # Фильтруем по минимальному расстоянию
            filtered_peaks = self._filter_peaks_by_distance(peaks, min_distance_bins)
            if filtered_peaks:
                self._auto_engine.ingest(filtered_peaks, bin_hz=freq_step)
            
    def _calculate_peak_width(self, peak_idx: int) -> int:
        """Вычисляет ширину пика в бинах."""
        if self._row is None:
            return 0
            
        threshold = self._get_threshold()
        if threshold is None:
            return 1
            
        # Ищем левую границу с защитой от бесконечного цикла
        left = peak_idx
        left_limit = max(0, peak_idx - 100)  # Ограничиваем поиск
        while left > left_limit and self._row[left] > threshold[left]:
            left -= 1
            
        # Ищем правую границу с защитой от бесконечного цикла
        right = peak_idx
        right_limit = min(len(self._row) - 1, peak_idx + 100)  # Ограничиваем поиск
        while right < right_limit and self._row[right] > threshold[right]:
            right += 1
            
        return right - left + 1
        
    def _filter_peaks_by_distance(self, peaks: List[Tuple], min_distance_bins: int) -> List[Tuple]:
        """Фильтрует пики по минимальному расстоянию."""
        if not peaks:
            return []
            
        if self._freqs is None or len(self._freqs) < 2:
            return peaks
            
        # Сортируем по амплитуде (убывание)
        sorted_peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
        filtered = []
        taken = set()
        
        freq_step = self._freqs[1] - self._freqs[0]
        if freq_step <= 0:
            return peaks
            
        # Ограничиваем количество обрабатываемых пиков
        max_peaks_to_process = 50
        
        for i, peak in enumerate(sorted_peaks):
            if i >= max_peaks_to_process:
                break
                
            freq_hz = peak[0]
            freq_bin = int(round((freq_hz - self._freqs[0]) / freq_step))
            
            # Проверяем, не слишком ли близко к уже взятым пикам
            too_close = False
            for taken_bin in taken:
                if abs(freq_bin - taken_bin) < min_distance_bins:
                    too_close = True
                    break
                    
            if not too_close:
                filtered.append(peak)
                taken.add(freq_bin)
                
        return filtered
        
    def clear_history(self):
        """Очищает всю историю автопиков."""
        self._history.clear()
        self._baseline = None
        self._auto_engine.clear_all()
        self.lbl_peaks_count.setText("Пиков: 0")
        self.lbl_baseline.setText("Baseline: —")
        self.lbl_threshold.setText("Порог: —")
        
    def _export_peaks(self):
        """Экспорт пиков в CSV."""
        if self._auto_engine.size() == 0:
            QtWidgets.QMessageBox.information(self, "Экспорт", "Нет данных для экспорта")
            return
            
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Экспорт пиков", "", "CSV файлы (*.csv)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    # Заголовки
                    f.write("Частота (МГц),Последний (дБм),Максимум (дБм),Среднее (дБм),Количество,Ширина (кГц),Q-фактор,Над шумом (дБ),Первое появление,Последнее появление\n")
                    
                    # Данные
                    for entry in self._auto_engine.entries():
                        f.write(f"{entry.freq_hz/1e6:.6f},{entry.last_dbm:.2f},{entry.max_dbm:.2f},{entry.avg_dbm:.2f},"
                               f"{entry.count},{entry.width_khz:.2f},{entry.q_factor:.2f},{entry.above_noise_db:.2f},"
                               f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry.first_seen))},"
                               f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry.last_seen))}\n")
                               
                QtWidgets.QMessageBox.information(self, "Экспорт", f"Данные экспортированы в {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка при экспорте: {e}")
                
    def get_engine(self):
        """Возвращает движок автопиков для внешнего доступа."""
        return self._auto_engine
