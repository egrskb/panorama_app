# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Dict, Any

from PyQt5 import QtCore, QtGui, QtWidgets
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # pandas не обязателен; используем лёгкий in-memory storage


class PandasTableModel(QtCore.QAbstractTableModel):
    """Простой QAbstractTableModel поверх pandas.DataFrame для отображения логов RMS."""

    def __init__(self, df: Optional[object] = None, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._default_columns = [
            'time', 'peak_id', 'slave', 'range_mhz', 'center_mhz', 'halfspan_mhz', 'rms_dbm', 'noise_dbm', 'snr_db', 'confidence'
        ]
        # Индекс для upsert по ключу (peak_id, slave)
        self._row_index_by_key = {}
        self._rows_by_peak = {}
        if pd is not None:
            # pandas backend
            if df is None:
                self._df = pd.DataFrame(columns=list(self._default_columns))  # type: ignore
            else:
                try:
                    self._df = pd.DataFrame(df)  # type: ignore
                except Exception:
                    self._df = pd.DataFrame(columns=list(self._default_columns))  # type: ignore
            self._rows = None
            self._columns = list(self._df.columns)
            self._rebuild_index()
        else:
            # lightweight backend (list of dicts)
            self._df = None
            self._columns = list(self._default_columns)
            self._rows = []  # type: list
            self._rebuild_index()

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        if parent.isValid():
            return 0
        if pd is not None:
            return len(self._df)  # type: ignore
        return len(self._rows)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        if parent.isValid():
            return 0
        if pd is not None:
            return len(self._df.columns)  # type: ignore
        return len(self._columns)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):
        if not index.isValid() or role not in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole, QtCore.Qt.ToolTipRole):
            return None
        if pd is not None:
            value = self._df.iat[index.row(), index.column()]  # type: ignore
        else:
            try:
                col = self._columns[index.column()]
                value = self._rows[index.row()].get(col, '')
            except Exception:
                value = ''
        if isinstance(value, float):
            # компактный формат для чисел
            return f"{value:.1f}"
        return str(value)

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            if pd is not None:
                return str(self._df.columns[section])  # type: ignore
            else:
                return str(self._columns[section])
        else:
            return str(section + 1)

    def append_row(self, row: Dict[str, Any]):
        key = (str(row.get('peak_id', '')), str(row.get('slave', '')))
        if pd is not None:
            self.beginInsertRows(QtCore.QModelIndex(), len(self._df), len(self._df))  # type: ignore
            self._df.loc[len(self._df)] = row  # type: ignore
            self.endInsertRows()
            # sync columns in case of new keys
            self._columns = list(self._df.columns)  # type: ignore
            # update index
            self._row_index_by_key[key] = len(self._df) - 1  # type: ignore
            self._rows_by_peak.setdefault(key[0], set()).add(self._row_index_by_key[key])
        else:
            # normalize row keys
            normalized = {c: row.get(c, '') for c in self._columns}
            self.beginInsertRows(QtCore.QModelIndex(), len(self._rows), len(self._rows))
            self._rows.append(normalized)
            self.endInsertRows()
            self._row_index_by_key[key] = len(self._rows) - 1
            self._rows_by_peak.setdefault(key[0], set()).add(self._row_index_by_key[key])

    def set_dataframe(self, df: object):
        self.beginResetModel()
        if pd is not None:
            try:
                self._df = pd.DataFrame(df)  # type: ignore
            except Exception:
                self._df = pd.DataFrame(columns=list(self._default_columns))  # type: ignore
            self._rows = None
            self._columns = list(self._df.columns)  # type: ignore
            self._rebuild_index()
        else:
            # attempt to coerce to list of dicts
            try:
                if isinstance(df, list):
                    self._rows = [
                        {c: r.get(c, '') for c in self._columns} if isinstance(r, dict) else {c: '' for c in self._columns}
                        for r in df
                    ]
                else:
                    self._rows = []
            except Exception:
                self._rows = []
            self._df = None
            self._rebuild_index()
        self.endResetModel()

    def dataframe(self):  # -> Union[pd.DataFrame, List[Dict]]
        return self._df if pd is not None else self._rows

    def columns(self):
        return list(self._columns) if pd is None else list(self._df.columns)  # type: ignore

    # ---------- upsert helpers ----------
    def _rebuild_index(self):
        self._row_index_by_key = {}
        self._rows_by_peak = {}
        try:
            if pd is not None:
                for idx in range(len(self._df)):  # type: ignore
                    row = self._df.iloc[idx]  # type: ignore
                    key = (str(row.get('peak_id', '')), str(row.get('slave', '')))
                    self._row_index_by_key[key] = idx
                    self._rows_by_peak.setdefault(key[0], set()).add(idx)
            else:
                for idx, row in enumerate(self._rows):
                    key = (str(row.get('peak_id', '')), str(row.get('slave', '')))
                    self._row_index_by_key[key] = idx
                    self._rows_by_peak.setdefault(key[0], set()).add(idx)
        except Exception:
            self._row_index_by_key = {}
            self._rows_by_peak = {}

    def upsert_row(self, row: Dict[str, Any]) -> int:
        key = (str(row.get('peak_id', '')), str(row.get('slave', '')))
        if key in self._row_index_by_key:
            idx = self._row_index_by_key[key]
            self.update_row(idx, row)
            return idx
        else:
            self.append_row(row)
            return self._row_index_by_key.get(key, -1)

    def update_row(self, idx: int, row: Dict[str, Any]):
        if idx < 0:
            return
        try:
            if pd is not None:
                # Update values in DataFrame
                for col, val in row.items():
                    if col not in self._df.columns:  # type: ignore
                        # add new column with empty values
                        self._df[col] = ''  # type: ignore
                        self._columns = list(self._df.columns)  # type: ignore
                    self._df.at[idx, col] = val  # type: ignore
                # Notify view about row update
                left = self.index(idx, 0)
                right = self.index(idx, self.columnCount() - 1)
                self.dataChanged.emit(left, right, [QtCore.Qt.DisplayRole])
            else:
                # list backend
                for col in self._columns:
                    if col in row:
                        self._rows[idx][col] = row[col]
                left = self.index(idx, 0)
                right = self.index(idx, self.columnCount() - 1)
                self.dataChanged.emit(left, right, [QtCore.Qt.DisplayRole])
        except Exception:
            pass

    def update_confidence_for_peak(self, peak_id: str, confidence: float):
        if not peak_id:
            return
        indices = list(self._rows_by_peak.get(str(peak_id), []))
        if not indices:
            return
        sorting = self.parent().table.isSortingEnabled() if hasattr(self, 'parent') and hasattr(self.parent(), 'table') else False
        if sorting and hasattr(self, 'parent'):
            try:
                self.parent().table.setSortingEnabled(False)
            except Exception:
                pass
        try:
            for idx in indices:
                try:
                    if pd is not None:
                        if 'confidence' not in self._df.columns:  # type: ignore
                            self._df['confidence'] = ''  # type: ignore
                            self._columns = list(self._df.columns)  # type: ignore
                        self._df.at[idx, 'confidence'] = float(confidence)  # type: ignore
                    else:
                        if 'confidence' in self._columns:
                            self._rows[idx]['confidence'] = float(confidence)
                    left = self.index(idx, 0)
                    right = self.index(idx, self.columnCount() - 1)
                    self.dataChanged.emit(left, right, [QtCore.Qt.DisplayRole])
                except Exception:
                    continue
        finally:
            if sorting and hasattr(self, 'parent'):
                try:
                    self.parent().table.setSortingEnabled(True)
                except Exception:
                    pass


class TestTableWidget(QtWidgets.QWidget):
    """Простая таблица (QTableView) для отображения live-логов RMS по слейвам.

    Ожидаемые столбцы:
      - time, peak_id, slave, range_mhz, center_mhz, halfspan_mhz, rms_dbm, noise_dbm, snr_db, confidence
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        # Фильтр
        filter_layout = QtWidgets.QHBoxLayout()
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Фильтр (подстрока по всем столбцам)")
        filter_layout.addWidget(QtWidgets.QLabel("Фильтр:"))
        filter_layout.addWidget(self.filter_edit, 1)

        self.table = QtWidgets.QTableView(self)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        self.model = PandasTableModel()
        # Оборачиваем в прокси для фильтрации/сортировки
        self.proxy = QtCore.QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.proxy.setFilterKeyColumn(-1)  # по всем столбцам
        self.table.setModel(self.proxy)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)

        layout.addLayout(filter_layout)
        layout.addWidget(self.table)

        # Сигналы
        self.filter_edit.textChanged.connect(self._on_filter_changed)

    @QtCore.pyqtSlot(dict)
    def add_log_entry(self, entry: Dict[str, Any]):
        """Добавляет строку в таблицу. Расчёт range_mhz, если не передан."""
        row = dict(entry)
        try:
            if 'range_mhz' not in row:
                cf = float(row.get('center_mhz', 0.0))
                hs = float(row.get('halfspan_mhz', 0.0))
                row['range_mhz'] = f"{cf - hs:.1f}-{cf + hs:.1f}"
        except Exception:
            pass
        # Заполним отсутствующие колонки пустыми значениями
        for col in self.model.columns():
            if col not in row:
                row[col] = ''
        # Временно отключим сортировку, чтобы избежать «скачков» при апдейтах
        sorting = self.table.isSortingEnabled()
        if sorting:
            self.table.setSortingEnabled(False)
        try:
            self.model.upsert_row(row)
        finally:
            if sorting:
                self.table.setSortingEnabled(True)

    def _on_filter_changed(self, text: str):
        try:
            self.proxy.setFilterFixedString(text)
        except Exception:
            pass

    def set_dataframe(self, df: pd.DataFrame):
        self.model.set_dataframe(df)

    @QtCore.pyqtSlot(str, float)
    def update_confidence_for_peak(self, peak_id: str, confidence: float):
        self.model.update_confidence_for_peak(peak_id, confidence)


