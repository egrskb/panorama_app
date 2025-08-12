from __future__ import annotations
from typing import Optional, Callable, List
from PyQt5 import QtWidgets

class DeviceDialog(QtWidgets.QDialog):
    """
    Диалог выбора HackRF (для libhackrf).
    Показывает список серийников и поле для ручного ввода суффикса.
    """
    def __init__(self, serials: List[str], current: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор HackRF (libhackrf)")
        self.resize(420, 320)

        self._provider: Optional[Callable[[], List[str]]] = None

        # список устройств
        self.list = QtWidgets.QListWidget()
        self.list.addItems(serials or [])

        # поле ручного ввода суффикса серийника
        self.edit = QtWidgets.QLineEdit(current or "")
        self.edit.setPlaceholderText("Суффикс серийника (можно пусто)")

        # кнопки
        self.btn_refresh = QtWidgets.QPushButton("Обновить")
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        # раскладка
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("Доступные устройства:"))
        lay.addWidget(self.list, stretch=1)
        lay.addWidget(QtWidgets.QLabel("Или укажи суффикс вручную:"))
        lay.addWidget(self.edit)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(self.btn_refresh)
        bottom.addWidget(btns)
        lay.addLayout(bottom)

        # выделим текущий при наличии
        if current:
            items = self.list.findItems(current, QtCore.Qt.MatchFlag.MatchEndsWith | QtCore.Qt.MatchFlag.MatchCaseSensitive)  # type: ignore
            if items:
                self.list.setCurrentItem(items[0])

    def set_provider(self, provider: Callable[[], List[str]]):
        """provider() -> List[str] для кнопки «Обновить»."""
        self._provider = provider

    def selected_serial_suffix(self) -> str:
        """Возвращает выбранный/введённый суффикс серийника (может быть пустым)."""
        txt = (self.edit.text() or "").strip()
        if txt:
            return txt
        it = self.list.currentItem()
        return (it.text().strip() if it else "")

    def _on_refresh_clicked(self):
        if not self._provider:
            return
        try:
            serials = self._provider() or []
        except Exception:
            serials = []
        self.list.clear()
        self.list.addItems(serials)

# экспортируем символ явно (на всякий случай)
__all__ = ["DeviceDialog"]
