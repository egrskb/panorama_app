from PyQt5 import QtCore


class SDRWorker(QtCore.QThread):
    """Basic interface for SDR workers."""

    spectrumReady = QtCore.pyqtSignal(object, object)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def stop(self):  # pragma: no cover - interface
        raise NotImplementedError
