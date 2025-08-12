import sys, os, stat, getpass, pathlib, logging
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QSettings

from panorama.features.spectrum import SpectrumView
from panorama.features.map3d import MapView
from panorama.features.peaks import PeaksWidget
from panorama.features.devices import DeviceDialog

from panorama.drivers.hackrf_sweep import HackRFSweepSource
from panorama.drivers.hackrf_lib import HackRFLibSource

from panorama.shared import write_row_csv, setup_logging, merged_defaults

APP_TITLE = "ПАНОРАМА 0.1 бета"


def _fix_runtime_dir():
    """Qt/DBus может зависать, если XDG_RUNTIME_DIR имеет права != 0700."""
    path = os.environ.get("XDG_RUNTIME_DIR")
    ok = False
    if path and os.path.isdir(path):
        try:
            mode = stat.S_IMODE(os.stat(path).st_mode)
            ok = (mode == 0o700)
        except Exception:
            ok = False
    if not ok:
        new = f"/tmp/xdg-runtime-{getpass.getuser()}"
        pathlib.Path(new).mkdir(parents=True, exist_ok=True)
        os.chmod(new, 0o700)
        os.environ["XDG_RUNTIME_DIR"] = new


class LogDock(QtWidgets.QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Лог", parent)
        self.setObjectName("LogDock")
        self.view = QtWidgets.QPlainTextEdit()
        self.view.setReadOnly(True)
        self.setWidget(self.view)

    def append(self, text: str):
        self.view.appendPlainText(text)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, logger: logging.Logger, settings: QSettings):
        super().__init__()
        self.log = logger
        self.settings = settings
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 750)

        # --- центральные вкладки ---
        self.tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # Спектр
        self.spectrum_tab = SpectrumView()

        # Источники
        self._sweep_source = HackRFSweepSource()
        try:
            self._lib_source = HackRFLibSource()
            self._lib_available = True
        except Exception:
            self._lib_source = None
            self._lib_available = False

        self._source = self._sweep_source
        self.spectrum_tab.set_source(self._source)

        # Восстановить сохранённый серийник для libhackrf
        saved_ser = self.settings.value("source/libhackrf_serial", "", type=str) or ""
        if self._lib_available and self._lib_source and saved_ser:
            try:
                self._lib_source.set_serial_suffix(saved_ser)
                self._append_log(f"libhackrf: выбран серийник *{saved_ser}")
            except Exception:
                pass

        # Карта и Пики
        self.map_tab = MapView()
        self.peaks_tab = PeaksWidget()
        self.spectrum_tab.newRowReady.connect(self.peaks_tab.update_from_row)
        self.peaks_tab.goToFreq.connect(self.spectrum_tab.set_cursor_freq)

        self.tabs.addTab(self.spectrum_tab, "Спектр")
        self.tabs.addTab(self.map_tab, "Карта")
        self.tabs.addTab(self.peaks_tab, "Пики")

        # Док-панель «Лог»
        self.logdock = LogDock(self)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.logdock)
        self.logdock.setVisible(True)

        # Подключаем источники к логу
        self._wire_source(self._source)

        self._build_menu()
        self._build_shortcuts()
        self.statusBar().showMessage("Готово")

        # Восстановление состояния + настроек спектра
        self._restore_window_state()
        self.spectrum_tab.restore_settings(self.settings, merged_defaults())

    # ---------- служебное ----------
    def _wire_source(self, src):
        # отцепим предыдущие коннекты (если были)
        for s in (self._sweep_source, getattr(self, "_lib_source", None)):
            if s is None:
                continue
            try:
                s.status.disconnect()
            except Exception:
                pass
            try:
                s.error.disconnect()
            except Exception:
                pass
            try:
                s.started.disconnect()
            except Exception:
                pass
            try:
                s.finished.disconnect()
            except Exception:
                pass

        if src is None:
            return
        src.status.connect(self._append_log)
        src.error.connect(lambda m: self._append_log("ERROR: " + m))
        src.started.connect(lambda: self._append_log("Источник: запущен"))
        src.finished.connect(lambda code: self._append_log(f"Источник: завершён (код {code})"))

    # ---------- меню / хоткеи ----------
    def _build_menu(self):
        menubar = self.menuBar()

        # Файл
        file_menu = menubar.addMenu("Файл")
        self.act_export = QtWidgets.QAction("Экспорт текущего свипа…", self)
        self.act_export.triggered.connect(self._export_current)
        file_menu.addAction(self.act_export)
        file_menu.addSeparator()
        act_exit = QtWidgets.QAction("Выход", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Инструменты
        tools_menu = menubar.addMenu("Инструменты")
        self.act_log = QtWidgets.QAction("Показать лог", self, checkable=True, checked=True)
        self.act_log.toggled.connect(self.logdock.setVisible)
        tools_menu.addAction(self.act_log)

        # Источник
        src_menu = tools_menu.addMenu("Источник")
        self.src_group = QtWidgets.QActionGroup(self)
        self.src_group.setExclusive(True)

        self.act_src_sweep = QtWidgets.QAction("hackrf_sweep", self, checkable=True)
        self.act_src_lib   = QtWidgets.QAction("libhackrf (CFFI)", self, checkable=True)
        self.act_src_lib.setEnabled(self._lib_available)
        self.act_src_sweep.setChecked(True)

        self.src_group.addAction(self.act_src_sweep)
        self.src_group.addAction(self.act_src_lib)
        src_menu.addAction(self.act_src_sweep)
        src_menu.addAction(self.act_src_lib)

        self.act_src_sweep.toggled.connect(lambda on: on and self._switch_source("sweep"))
        self.act_src_lib.toggled.connect(lambda on: on and self._switch_source("lib"))

        # Устройство
        dev_menu = menubar.addMenu("Устройство")
        act_pick = QtWidgets.QAction("Выбрать HackRF…", self)
        act_pick.triggered.connect(self._choose_libhackrf)
        dev_menu.addAction(act_pick)

        act_cal_load = QtWidgets.QAction("Загрузить калибровку CSV…", self)
        act_cal_load.triggered.connect(self._load_calibration_csv)
        dev_menu.addAction(act_cal_load)

        self.act_cal_enable = QtWidgets.QAction("Применять калибровку", self, checkable=True, checked=False)
        self.act_cal_enable.toggled.connect(self._toggle_calibration)
        dev_menu.addAction(self.act_cal_enable)

        # Справка
        help_menu = menubar.addMenu("Справка")
        act_about = QtWidgets.QAction("О программе", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _build_shortcuts(self):
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self._toggle_start_stop)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+E"), self, activated=self._export_current)

    # ---------- действия ----------
    def _toggle_start_stop(self):
        if self._source and self._source.is_running():
            self._source.stop()
        else:
            # эмулируем нажатие кнопки «Старт» у вкладки спектра
            self.spectrum_tab._on_start_clicked()

    def _switch_source(self, name: str):
        # остановить текущий
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
        spec_png, water_png = self.spectrum_tab.save_plots(base)
        QtWidgets.QMessageBox.information(
            self, "Экспорт завершён",
            f"CSV: {csv_path}\nPNG спектра: {spec_png}\nPNG водопада: {water_png}"
        )

    def _choose_libhackrf(self):
        if not self._lib_available or not self._lib_source:
            QtWidgets.QMessageBox.information(self, "libhackrf", "libhackrf (CFFI) недоступен.")
            return
        try:
            serials = self._lib_source.list_serials()
        except Exception:
            serials = []
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
            if self.act_src_lib.isChecked():
                try:
                    if self._source and self._source.is_running():
                        self._source.stop()
                except Exception:
                    pass
                self.spectrum_tab.set_source(self._lib_source)

    def _load_calibration_csv(self):
        if not self._lib_available or not self._lib_source:
            QtWidgets.QMessageBox.information(self, "Калибровка", "libhackrf (CFFI) недоступен.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Файл калибровки CSV", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        ok = False
        try:
            ok = self._lib_source.load_calibration(path)
        except Exception:
            ok = False
        self.act_cal_enable.setChecked(ok)
        if ok:
            self._append_log(f"Калибровка загружена: {path}")
            QtWidgets.QMessageBox.information(self, "Калибровка", "Успешно загружено и применено.")
        else:
            err = ""
            try:
                err = self._lib_source.last_error()
            except Exception:
                pass
            QtWidgets.QMessageBox.warning(self, "Калибровка", f"Не удалось загрузить.\n{err}")

    def _toggle_calibration(self, on: bool):
        if self._lib_available and self._lib_source:
            try:
                self._lib_source.set_calibration_enabled(bool(on))
            except Exception:
                pass
            self._append_log(f"Калибровка: {'вкл' if on else 'выкл'}")

    def _show_about(self):
        QtWidgets.QMessageBox.information(
            self,
            "О программе",
            f"{APP_TITLE}\nHackRF Sweep Analyzer\n\n"
            "Структура: feature-first (features/, drivers/, shared/).",
        )

    # ---------- сохранение/восстановление ----------
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
        self.spectrum_tab.save_settings(self.settings)
        self._save_window_state()
        super().closeEvent(e)


def main():
    _fix_runtime_dir()
    # QSettings ini: ~/.config/panorama/panorama.ini
    QtCore.QCoreApplication.setOrganizationName("panorama")
    QtCore.QCoreApplication.setApplicationName("panorama")
    settings = QSettings(QSettings.IniFormat, QSettings.UserScope, "panorama", "panorama")

    logger = setup_logging("panorama")
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(logger, settings)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
