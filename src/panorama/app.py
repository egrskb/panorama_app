import logging
from pathlib import Path
from PyQt5 import QtWidgets

from .ui.main_window import MainWindow
from .core import config

LOGGER = logging.getLogger("panorama")


def main():
    logging.basicConfig(level=logging.INFO)
    cfg = config.merged_config()
    app = QtWidgets.QApplication([])
    win = MainWindow(cfg)
    win.show()
    return app.exec_()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
