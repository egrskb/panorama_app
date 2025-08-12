import sys, os, stat, getpass, pathlib, logging
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QSettings

from panorama.features.spectrum import SpectrumView
from panorama.features.peaks import PeaksWidget
from panorama.features.devices import DeviceDialog

from panorama.drivers.hackrf_sweep import HackRFSweepSource
try:
    from panorama.drivers.hackrf_lib import HackRFLibSource  # CFFI
    _LIB_AVAILABLE = True
except Exception:
    HackRFLibSource = None  # type: ignore
    _LIB_AVAILABLE = False

from panorama.shared import write_row_csv, setup_logging, merged_defaults


APP_TITLE = "ПАНОРАМА 0.1 бета"


def _fix_runtime_dir():
    """Чиним XDG_RUNTIME_DIR с правами 0700 (PyQt предупреждение)."""
    path = os.environ.get("XDG_RUNTIME_DIR")
    ok = False
    if path and os.path.isdir(path):
        try:
            ok = (stat.S_IMODE(os.stat(path).st_mode) == 0o700)
        except Exception:
            ok = False
    if not ok:
        new = f"/tmp/xdg-runtime-{getpass.getuser()}"
        pathlib.Path(new).mkdir(parents=True, exist_ok=True)
        os.chmod(new, 0o700)
        os.environ["XDG_RUNTIME_DIR"] = new


class LogDock(QtWidgets.QDockWidget):
    """Приклеенный (неотстыковываемый) док с логом."""
    def __init__(self, parent=None):
        super().__init__("Лог", parent)
        self.setObjectName("LogDock")
        self.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)  # нельзя открепить/закрыть/переместить
        self.view = QtWidgets.QPlainTextEdit()
        self.view.setReadOnly(True)
        self.setWidget(self.view)

    def append(self, text: str):
        self.view.appendPlainText(text)


class DetectorPlaceholder(QtWidgets.QWidget):
    """Вкладка 'Детекция' — заглушка под будущую логику."""
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel(
            "Детекция (заглушка)\n\n"
            "Здесь будет ядро детектора, таблица событий и оверлеи.\n"
            "API предусмотрим позже: detector.push(freqs_hz, row_dbm)."
        ))
        v.addStretch(1)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, logger: logging.Logger, settings: QSettings):
        super().__init__()
        self.log = logger
        self.settings = settings

        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 760)

        # --- центральные вкладки ---
        self.tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # Спектр
        self.spectrum_tab = SpectrumView()

        # Источники
        self._sweep_source = HackRFSweepSource()
        if _LIB_AVAILABLE:
            try:
                self._lib_source = HackRFLibSource()
                self._lib_available = True
            except Exception:
                self._lib_source = None
                self._lib_available = False
        else:
            self._lib_source = None
            self._lib_available = False

        self._source = self._sweep_source
        self.spectrum_tab.set_source(self._source)
        self._wire_source(self._source)

        # Пики / Детекция (заглушка)
        self.peaks_tab = PeaksWidget()
        self.detector_tab = DetectorPlaceholder()

        # Провязка: спектр → пики
        self.spectrum_tab.newRowReady.connect(self.peaks_tab.update_from_row)
        self.peaks_tab.goToFreq.connect(self.spectrum_tab.set_cursor_freq)

        # Вкладки
        self.tabs.addTab(self.spectrum_tab, "Спектр")
        self.tabs.addTab(self.peaks_tab, "Пики")
        self.tabs.addTab(self.detector_tab, "Детекция")

        # Неотстыковываемый «Лог»
        self.logdock = LogDock(self)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.logdock)
        self.logdock.setVisible(True)

        # Меню/горячие клавиши/статус
        self._build_menu()
        self._build_shortcuts()
        self.statusBar().showMessage("Готово")

        # Состояние окна + настройки спектра
        self._restore_window_state()
        self.spectrum_tab.restore_settings(self.settings, merged_defaults())

    # ---------------- внутреннее ----------------
    def _wire_source(self, src):
        # отписаться от старых
        for s in (self._sweep_source, getattr(self, "_lib_source", None)):
            if not s:
                continue
            try:
                s.status.disconnect()
                s.error.disconnect()
                s.started.disconnect()
                s.finished.disconnect()
            except Exception:
                pass

        if not src:
            return
        src.status.connect(self._append_log)
        src.error.connect(lambda m: self._append_log("ERROR: " + m))
        src.started.connect(lambda: self._append_log("Источник: запущен"))
        src.finished.connect(lambda code: self._append_log(f"Источник: завершён (код {code})"))

    def _build_menu(self):
        menubar = self.menuBar()

        # Файл
        m_file = menubar.addMenu("Файл")
        act_export = QtWidgets.QAction("Экспорт текущего свипа…", self)
        act_export.triggered.connect(self._export_current)
        m_file.addAction(act_export)
        m_file.addSeparator()
        act_exit = QtWidgets.QAction("Выход", self)
        act_exit.triggered.connect(self.close)
        m_file.addAction(act_exit)

        # Инструменты
        m_tools = menubar.addMenu("Инструменты")
        # Источник
        src_menu = m_tools.addMenu("Источник")
        self.src_group = QtWidgets.QActionGroup(self)
        self.src_group.setExclusive(True)
        self.act_src_sweep = QtWidgets.QAction("hackrf_sweep", self, checkable=True, checked=True)
        src_menu.addAction(self.act_src_sweep)
        if self._lib_available:
            self.act_src_lib = QtWidgets.QAction("libhackrf (CFFI)", self, checkable=True, checked=False)
            src_menu.addAction(self.act_src_lib)
            self.src_group.addAction(self.act_src_lib)
            self.act_src_lib.toggled.connect(lambda on: on and self._switch_source("lib"))
        self.src_group.addAction(self.act_src_sweep)
        self.act_src_sweep.toggled.connect(lambda on: on and self._switch_source("sweep"))

        # Устройство (для lib)
        m_dev = menubar.addMenu("Устройство")
        act_pick = QtWidgets.QAction("Выбрать HackRF…", self)
        act_pick.triggered.connect(self._choose_libhackrf)
        act_pick.setEnabled(self._lib_available)
        m_dev.addAction(act_pick)

        act_cal_load = QtWidgets.QAction("Загрузить калибровку CSV…", self)
        act_cal_load.triggered.connect(self._load_calibration_csv)
        act_cal_load.setEnabled(self._lib_available)
        m_dev.addAction(act_cal_load)

        self.act_cal_enable = QtWidgets.QAction("Применять калибровку", self, checkable=True, checked=False)
        self.act_cal_enable.toggled.connect(self._toggle_calibration)
        self.act_cal_enable.setEnabled(self._lib_available)
        m_dev.addAction(self.act_cal_enable)

        # Справка
        m_help = menubar.addMenu("Справка")
        act_about = QtWidgets.QAction("О программе", self)
        act_about.triggered.connect(self._show_about)
        m_help.addAction(act_about)

    def _build_shortcuts(self):
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self._toggle_start_stop)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+E"), self, activated=self._export_current)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, activated=self.spectrum_tab._on_reset_view)

    def _show_about(self):
        QtWidgets.QMessageBox.information(
            self,
            "О программе",
            "ПАНОРАМА 0.1 бета\n"
            "HackRF Sweep Analyzer (модульная версия)\n\n"
            "Источник: hackrf_sweep / libhackrf (CFFI)\n"
            "Спектр + водопад, маркеры, экспорт CSV/PNG."
        )
    # ---------------- действия ----------------
    def _toggle_start_stop(self):
        try:
            if self._source and self._source.is_running():
                self._source.stop()
            else:
                self.spectrum_tab._on_start_clicked()
        except Exception as e:
            self._append_log(f"ERROR start/stop: {e}")

    def _switch_source(self, name: str):
        try:
            if self._source and self._source.is_running():
                self._source.stop()
        except Exception:
            pass

        if name == "lib" and self._lib_available and self._lib_source:
            self._source = self._lib_source
            self._append_log("Переключён источник: libhackrf (CFFI)")
        else:
            self._source = self._sweep_source
            self._append_log("Переключён источник: hackrf_sweep")

        self.spectrum_tab.set_source(self._source)
        self._wire_source(self._source)

    def _append_log(self, text: str):
        self.log.info(text)
        self.logdock.append(text)

    def _export_current(self):
        freqs, row = self.spectrum_tab.get_current_row()
        if freqs is None or row is None:
            QtWidgets.QMessageBox.information(self, "Экспорт", "Пока нет полной строки для экспорта.")
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Куда сохранить файлы?")
        if not out_dir:
            return
        from PyQt5.QtCore import QDateTime
        ts = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        base = os.path.join(out_dir, f"sweep_{ts}")
        csv_path = f"{base}.csv"
        try:
            write_row_csv(csv_path, freqs, row)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Экспорт CSV", f"Ошибка: {e}")
            return
        spec_png, water_png = self.spectrum_tab.save_plots(base) if hasattr(self.spectrum_tab, "save_plots") else ("", "")
        QtWidgets.QMessageBox.information(
            self, "Экспорт завершён",
            f"CSV: {csv_path}\nPNG спектра: {spec_png}\nPNG водопада: {water_png}"
        )

    # -------- libhackrf: устройство/калибровка --------
    def _choose_libhackrf(self):
        if not self._lib_available or not self._lib_source:
            QtWidgets.QMessageBox.information(self, "libhackrf", "libhackrf (CFFI) недоступен.")
            return
        serials = self._lib_source.list_serials()
        current = self.settings.value("source/libhackrf_serial", "", type=str) or ""
        dlg = DeviceDialog(serials, current, self)
        dlg.set_provider(self._lib_source.list_serials)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            sel = dlg.selected_serial_suffix().strip()
            try:
                self._lib_source.set_serial_suffix(sel or None)
            except Exception:
                pass
            self.settings.setValue("source/libhackrf_serial", sel)
            self._append_log(f"libhackrf: выбран серийник *{sel or '(любой)'}")
            if self._source is self._lib_source and self._source.is_running():
                try:
                    self._source.stop()
                except Exception:
                    pass

    def _load_calibration_csv(self):
        if not self._lib_available or not self._lib_source:
            QtWidgets.QMessageBox.information(self, "Калибровка", "libhackrf (CFFI) недоступен.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Файл калибровки CSV", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        ok = self._lib_source.load_calibration(path)
        self.act_cal_enable.setChecked(ok)
        if ok:
            self._append_log(f"Калибровка загружена: {path}")
            QtWidgets.QMessageBox.information(self, "Калибровка", "Успешно загружено и применено.")
        else:
            err = self._lib_source.last_error()
            QtWidgets.QMessageBox.warning(self, "Калибровка", f"Не удалось загрузить.\n{err}")

    def _toggle_calibration(self, on: bool):
        if self._lib_available and self._lib_source:
            self._lib_source.set_calibration_enabled(bool(on))
            self._append_log(f"Калибровка: {'вкл' if on else 'выкл'}")

    # -------- окно/настройки --------
    def _restore_window_state(self):
        self.restoreGeometry(self.settings.value("main/geometry", type=QtCore.QByteArray) or QtCore.QByteArray())
        self.restoreState(self.settings.value("main/windowState", type=QtCore.QByteArray) or QtCore.QByteArray())

    def _save_window_state(self):
        self.settings.setValue("main/geometry", self.saveGeometry())
        self.settings.setValue("main/windowState", self.saveState())

    def closeEvent(self, e):
        try:
            if self._source and self._source.is_running():
                self._source.stop()
        except Exception:
            pass
        # сохранить настройки виджета спектра
        try:
            self.spectrum_tab.save_settings(self.settings)
        except Exception:
            pass
        self._save_window_state()
        super().closeEvent(e)


def main():
    _fix_runtime_dir()
    QtCore.QCoreApplication.setOrganizationName("panorama")
    QtCore.QCoreApplication.setApplicationName("panorama")
    settings = QSettings(QSettings.IniFormat, QSettings.UserScope, "panorama", "panorama")

    logger = setup_logging("panorama")
    logger.info(f"Logging initialized → {os.path.expanduser('~')}/.local/state/panorama/logs/panorama.log")

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(logger, settings)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
