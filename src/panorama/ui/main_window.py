from PyQt5 import QtCore, QtWidgets

from ..core import parsing, colormaps, settings_store
from ..detection import peaks
from .peaks_dialog import PeaksDialog
from .trilateration_window import TrilaterationWindow
from .widgets.time_axis import TimeAxis


class MainWindow(QtWidgets.QMainWindow):
    """Simplified placeholder main window.

    The original project contains a very feature rich window.  In this
    refactoring exercise we provide only a minimal stub that preserves
    the public API used by other modules and tests.  The full featured
    implementation can be added later without changing the package
    layout.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("ПАНОРАМА — версия 0.1 бета")
        label = QtWidgets.QLabel(
            "PANORAMA placeholder UI",
            alignment=QtCore.Qt.AlignCenter,
        )
        self.setCentralWidget(label)

    # placeholder methods to keep API surface
    def on_start(self):  # pragma: no cover - GUI
        pass

    def on_stop(self):  # pragma: no cover - GUI
        pass
