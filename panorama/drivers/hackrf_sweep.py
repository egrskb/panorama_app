from __future__ import annotations
import shutil
from typing import Optional
from PyQt5 import QtCore
from .base import SourceBackend, SweepConfig
from panorama.shared.parsing import parse_sweep_line, SweepLine


class HackRFSweepSource(SourceBackend):
    def __init__(self, executable: str = "hackrf_sweep", parent=None):
        super().__init__(parent)
        import shutil as _sh
        self._exe = _sh.which(executable) or executable
        self._p = None
        self._buf = bytearray()


    def is_running(self) -> bool:
        return self._p is not None and self._p.state() != QtCore.QProcess.NotRunning

    def start(self, config: SweepConfig):
        if self.is_running():
            self.status.emit("Уже запущено")
            return

        import shutil
        if not shutil.which(self._exe):
            self.error.emit(f"Не найден исполняемый файл: {self._exe}")
            return

        args = config.to_args()
        self.status.emit(f"Запуск: {self._exe} {' '.join(args)}") 

        self._p = QtCore.QProcess(self)
        # Ensure we get text lines; we'll decode manually to be safe
        self._p.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self._p.readyReadStandardOutput.connect(self._on_ready)
        self._p.finished.connect(self._on_finished)

        args = config.to_args()
        self.status.emit(f"Запуск: {self._exe} {' '.join(args)}")
        self._p.start(self._exe, args)

        if not self._p.waitForStarted(3000):
            self.error.emit("Не удалось запустить hackrf_sweep")
            self._cleanup(emit_finished=False)
            return

        self.started.emit()

    def stop(self):
        if not self.is_running():
            return
        # try graceful
        self._p.terminate()
        if not self._p.waitForFinished(1500):
            self._p.kill()
            self._p.waitForFinished(1500)
        self._cleanup(emit_finished=True)

    # -------- internals --------
    def _on_ready(self):
        if not self._p:
            return
        self._buf.extend(self._p.readAllStandardOutput().data())
        # Split by lines; keep tail (incomplete)
        *lines, tail = self._buf.split(b"\n")
        self._buf = bytearray(tail)
        for raw in lines:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            # hackrf_sweep sometimes prints header lines; skip non-CSV
            if "," not in line:
                # forward as status to help with debugging
                self.status.emit(line)
                continue
            try:
                sw: SweepLine = parse_sweep_line(line)
                self.sweepLine.emit(sw)
            except Exception as e:
                # don't spam: only short prefix
                self.status.emit(f"parse skip: {str(e)[:120]}")

    def _on_finished(self, code: int, _status):
        self._cleanup(emit_finished=False)
        self.finished.emit(int(code))

    def _cleanup(self, emit_finished: bool):
        if self._p:
            try:
                self._p.readyReadStandardOutput.disconnect(self._on_ready)
            except Exception:
                pass
            try:
                self._p.finished.disconnect(self._on_finished)
            except Exception:
                pass
            self._p = None
        self._buf.clear()
        if emit_finished:
            self.finished.emit(0)
