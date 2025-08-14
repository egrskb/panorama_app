import sys, os, stat, getpass, pathlib, logging, time
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QSettings

from panorama.features.spectrum import SpectrumView
from panorama.features.peaks import PeaksWidget
from panorama.features.devices import DeviceDialog
from panorama.features.map3d import MapView

from panorama.drivers.hackrf_sweep import HackRFSweepSource
from panorama.shared.calibration import load_calibration_csv, get_calibration_lut

try:
    from panorama.drivers.hackrf_lib import HackRFLibSource
    _LIB_AVAILABLE = True
except Exception:
    HackRFLibSource = None  # type: ignore
    _LIB_AVAILABLE = False

from panorama.shared import write_row_csv, setup_logging, merged_defaults


APP_TITLE = "ПАНОРАМА 0.1 бета"


def _fix_runtime_dir():
    """Чиним XDG_RUNTIME_DIR с правами 0700."""
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


class DetectorWidget(QtWidgets.QWidget):
    """Вкладка детектора активности с ROI."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        v = QtWidgets.QVBoxLayout(self)
        
        # Пресеты диапазонов
        grp_presets = QtWidgets.QGroupBox("Пресеты")
        grid = QtWidgets.QGridLayout(grp_presets)
        
        preset_rows = [
            ("FM (87–108 МГц)", [(87.5, 108.0)]),
            ("VHF (136–174 МГц)", [(136.0, 174.0)]),
            ("UHF (400–470 МГц)", [(400.0, 470.0)]),
            ("Wi-Fi 2.4 ГГц", [(2400.0, 2483.5)]),
            ("Wi-Fi 5 ГГц", [(5170.0, 5895.0)]),
            ("5.8 ГГц FPV", [(5725.0, 5875.0)]),
            ("LTE 700–900", [(703.0, 960.0)]),
            ("ISM 868", [(863.0, 873.0)]),
        ]
        
        r = 0; c = 0
        for title, ranges in preset_rows:
            btn = QtWidgets.QPushButton(title)
            btn.clicked.connect(lambda _, rr=ranges: self._add_ranges(rr))
            grid.addWidget(btn, r, c)
            c += 1
            if c >= 2:
                c = 0; r += 1
        
        v.addWidget(grp_presets)
        
        # Таблица диапазонов
        grp_ranges = QtWidgets.QGroupBox("Диапазоны сканирования")
        vr = QtWidgets.QVBoxLayout(grp_ranges)
        
        self.tbl_ranges = QtWidgets.QTableWidget(0, 2)
        self.tbl_ranges.setHorizontalHeaderLabels(["Начало, МГц", "Конец, МГц"])
        self.tbl_ranges.horizontalHeader().setStretchLastSection(True)
        vr.addWidget(self.tbl_ranges)
        
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Добавить +")
        self.btn_del = QtWidgets.QPushButton("Удалить −")
        self.btn_merge = QtWidgets.QPushButton("Объединить")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_del)
        btn_row.addWidget(self.btn_merge)
        vr.addLayout(btn_row)
        
        v.addWidget(grp_ranges)
        
        # Параметры детектора
        grp_params = QtWidgets.QGroupBox("Параметры детектора")
        fp = QtWidgets.QFormLayout(grp_params)
        
        self.th_dbm = QtWidgets.QDoubleSpinBox()
        self.th_dbm.setRange(-160, 30)
        self.th_dbm.setValue(-80)
        self.th_dbm.setSuffix(" дБм")
        
        self.min_width = QtWidgets.QSpinBox()
        self.min_width.setRange(1, 1000)
        self.min_width.setValue(5)
        self.min_width.setSuffix(" бинов")
        
        fp.addRow("Порог:", self.th_dbm)
        fp.addRow("Мин. ширина:", self.min_width)
        
        v.addWidget(grp_params)
        
        # Кнопки управления
        btns = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Начать детект")
        self.btn_stop = QtWidgets.QPushButton("Остановить")
        self.btn_stop.setEnabled(False)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addStretch(1)
        v.addLayout(btns)
        
        v.addStretch(1)
        
        # Обработчики
        self.btn_add.clicked.connect(self._add_current)
        self.btn_del.clicked.connect(self._delete_selected)
        self.btn_merge.clicked.connect(self._merge_ranges)
        self.btn_start.clicked.connect(self._start_detection)
        self.btn_stop.clicked.connect(self._stop_detection)
        
        self._detecting = False
    
    def _add_ranges(self, ranges):
        for start, stop in ranges:
            r = self.tbl_ranges.rowCount()
            self.tbl_ranges.insertRow(r)
            self.tbl_ranges.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{start:.3f}"))
            self.tbl_ranges.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{stop:.3f}"))
    
    def _add_current(self):
        # Заглушка - берем диапазон из спектра
        self._add_ranges([(2400.0, 2483.5)])
    
    def _delete_selected(self):
        rows = sorted({i.row() for i in self.tbl_ranges.selectedIndexes()}, reverse=True)
        for r in rows:
            self.tbl_ranges.removeRow(r)
    
    def _merge_ranges(self):
        # Простое объединение перекрывающихся диапазонов
        ranges = []
        for r in range(self.tbl_ranges.rowCount()):
            try:
                start = float(self.tbl_ranges.item(r, 0).text())
                stop = float(self.tbl_ranges.item(r, 1).text())
                ranges.append((start, stop))
            except Exception:
                continue
        
        if not ranges:
            return
        
        ranges.sort()
        merged = []
        for start, stop in ranges:
            if not merged or start > merged[-1][1] + 0.1:
                merged.append([start, stop])
            else:
                merged[-1][1] = max(merged[-1][1], stop)
        
        self.tbl_ranges.setRowCount(0)
        for start, stop in merged:
            r = self.tbl_ranges.rowCount()
            self.tbl_ranges.insertRow(r)
            self.tbl_ranges.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{start:.3f}"))
            self.tbl_ranges.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{stop:.3f}"))
    
    def _start_detection(self):
        self._detecting = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        # Здесь будет логика детектора
        QtWidgets.QMessageBox.information(self, "Детектор", "Детекция запущена (заглушка)")
    
    def _stop_detection(self):
        self._detecting = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
    
    def push_data(self, freqs_hz, row_dbm):
        """API для подачи данных в детектор."""
        if not self._detecting:
            return
        # Здесь будет анализ данных
        pass


class TrilaterationWindow(QtWidgets.QMainWindow):
    """Окно трилатерации с 3 SDR."""
    
    def __init__(self, master_serial: str, slave1_serial: str, slave2_serial: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Трилатерация")
        
        self.master_serial = master_serial
        self.slave1_serial = slave1_serial
        self.slave2_serial = slave2_serial
        
        # Центральный виджет - карта
        self.map = MapView()
        self.setCentralWidget(self.map)
        
        # Док с логом
        self.log_dock = QtWidgets.QDockWidget("Лог трилатерации", self)
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log_dock.setWidget(self.log)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.log_dock)
        
        # Тулбар
        toolbar = self.addToolBar("Управление")
        
        act_start = toolbar.addAction("Старт")
        act_stop = toolbar.addAction("Стоп")
        act_clear = toolbar.addAction("Очистить")
        
        act_start.triggered.connect(self._start)
        act_stop.triggered.connect(self._stop)
        act_clear.triggered.connect(self._clear)
        
        self._running = False
        
        self.resize(900, 700)
        self._log(f"Master: {master_serial}")
        self._log(f"Slave1: {slave1_serial}")
        self._log(f"Slave2: {slave2_serial}")
    
    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")
    
    def _start(self):
        if self._running:
            return
        self._running = True
        self._log("Трилатерация запущена")
        # Здесь будет запуск 3 источников
    
    def _stop(self):
        self._running = False
        self._log("Трилатерация остановлена")
    
    def _clear(self):
        self.log.clear()


class MainWindow(QtWidgets.QMainWindow):
    """Главное окно приложения ПАНОРАМА."""
    
    def __init__(self, logger: logging.Logger, settings: QSettings):
        super().__init__()
        self.log = logger
        self.settings = settings
        self._calibration_profiles = {}  # Загруженные профили калибровки

        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 900)

        # --- центральные вкладки ---
        self.tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # Спектр
        self.spectrum_tab = SpectrumView()

        # Источники
        self._sweep_source = HackRFSweepSource()
        self._lib_source = None
        
        if _LIB_AVAILABLE:
            try:
                self._lib_source = HackRFLibSource()
                self._lib_available = True
            except Exception as e:
                self._lib_available = False
                self.log.warning(f"libhackrf недоступна: {e}")
        else:
            self._lib_available = False

        self._source = self._sweep_source
        self.spectrum_tab.set_source(self._source)
        self._wire_source(self._source)

        # Остальные вкладки
        self.map_tab = MapView()
        self.peaks_tab = PeaksWidget()
        self.detector_tab = DetectorWidget()

        # Провязка: спектр → пики и детектор
        self.spectrum_tab.newRowReady.connect(self.peaks_tab.update_from_row)
        self.spectrum_tab.newRowReady.connect(self.detector_tab.push_data)
        self.peaks_tab.goToFreq.connect(self.spectrum_tab.set_cursor_freq)

        # Добавляем вкладки
        self.tabs.addTab(self.spectrum_tab, "Спектр")
        self.tabs.addTab(self.peaks_tab, "Пики")
        self.tabs.addTab(self.detector_tab, "Детектор")
        self.tabs.addTab(self.map_tab, "Карта")

        # Меню и статусбар
        self._build_menu()
        self._build_shortcuts()
        self.statusBar().showMessage("Готово")

        # Восстановление состояния
        self._restore_window_state()
        if hasattr(self.spectrum_tab, "restore_settings"):
            self.spectrum_tab.restore_settings(self.settings, merged_defaults())
        
        # Автозагрузка калибровки если есть
        self._try_load_default_calibration()

    # ---------------- внутреннее ----------------
    def _wire_source(self, src):
        """Подключает обработчики к источнику."""
        # Отписываемся от старых
        for s in [self._sweep_source, self._lib_source]:
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
        
        src.status.connect(lambda m: self.statusBar().showMessage(m, 3000))
        src.error.connect(lambda m: self._show_error("Источник", m))
        src.started.connect(lambda: self._on_source_started())
        src.finished.connect(lambda code: self._on_source_finished(code))

    def _on_source_started(self):
        """Вызывается при старте источника."""
        # Применяем калибровку для hackrf_sweep
        if isinstance(self._source, HackRFSweepSource) and self._calibration_profiles:
            lna = self.spectrum_tab.lna_db.value()
            vga = self.spectrum_tab.vga_db.value()
            amp = 1 if self.spectrum_tab.amp_on.isChecked() else 0
            
            lut = get_calibration_lut(self._calibration_profiles, lna, vga, amp)
            if lut is not None:
                self._source.set_calibration_lut(lut)
                self.statusBar().showMessage(f"Калибровка применена (LNA:{lna} VGA:{vga} AMP:{amp})", 5000)

    def _on_source_finished(self, code: int):
        """Вызывается при остановке источника."""
        if code != 0:
            self.statusBar().showMessage(f"Источник завершен с кодом {code}", 5000)

    def _build_menu(self):
        menubar = self.menuBar()

        # === Файл ===
        m_file = menubar.addMenu("&Файл")
        
        act_export_csv = QtWidgets.QAction("Экспорт свипа CSV…", self)
        act_export_csv.triggered.connect(self._export_current_csv)
        
        act_export_png = QtWidgets.QAction("Экспорт водопада PNG…", self)
        act_export_png.triggered.connect(self._export_waterfall_png)
        
        m_file.addAction(act_export_csv)
        m_file.addAction(act_export_png)
        m_file.addSeparator()
        
        act_exit = QtWidgets.QAction("Выход", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        m_file.addAction(act_exit)

        # === Источник ===
        m_source = menubar.addMenu("&Источник")
        
        self.src_group = QtWidgets.QActionGroup(self)
        self.src_group.setExclusive(True)
        
        self.act_src_sweep = QtWidgets.QAction("hackrf_sweep", self, checkable=True, checked=True)
        m_source.addAction(self.act_src_sweep)
        self.src_group.addAction(self.act_src_sweep)
        self.act_src_sweep.toggled.connect(lambda on: on and self._switch_source("sweep"))
        
        if self._lib_available:
            self.act_src_lib = QtWidgets.QAction("libhackrf (CFFI)", self, checkable=True, checked=False)
            m_source.addAction(self.act_src_lib)
            self.src_group.addAction(self.act_src_lib)
            self.act_src_lib.toggled.connect(lambda on: on and self._switch_source("lib"))
        
        m_source.addSeparator()
        
        act_devices = QtWidgets.QAction("Выбрать устройство…", self)
        act_devices.triggered.connect(self._choose_device)
        m_source.addAction(act_devices)

        # === Калибровка ===
        m_cal = menubar.addMenu("&Калибровка")
        
        act_cal_load = QtWidgets.QAction("Загрузить CSV…", self)
        act_cal_load.triggered.connect(self._load_calibration_csv)
        m_cal.addAction(act_cal_load)
        
        self.act_cal_enable = QtWidgets.QAction("Применять калибровку", self, checkable=True, checked=False)
        self.act_cal_enable.toggled.connect(self._toggle_calibration)
        m_cal.addAction(self.act_cal_enable)
        
        m_cal.addSeparator()
        
        act_cal_clear = QtWidgets.QAction("Очистить калибровку", self)
        act_cal_clear.triggered.connect(self._clear_calibration)
        m_cal.addAction(act_cal_clear)

        # === Справка ===
        m_help = menubar.addMenu("&Справка")
        
        act_hotkeys = QtWidgets.QAction("Горячие клавиши", self)
        act_hotkeys.triggered.connect(self._show_hotkeys)
        m_help.addAction(act_hotkeys)
        
        act_about = QtWidgets.QAction("О программе", self)
        act_about.triggered.connect(self._show_about)
        m_help.addAction(act_about)

    def _build_shortcuts(self):
        """Горячие клавиши."""
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self._toggle_start_stop)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+E"), self, activated=self._export_current_csv)
        QtWidgets.QShortcut(QtGui.QKeySequence("R"), self, activated=self.spectrum_tab._on_reset_view)
        QtWidgets.QShortcut(QtGui.QKeySequence("+"), self, activated=lambda: self._zoom_x(0.8))
        QtWidgets.QShortcut(QtGui.QKeySequence("-"), self, activated=lambda: self._zoom_x(1.25))

    def _zoom_x(self, factor: float):
        """Зум по X для спектра."""
        vb = self.spectrum_tab.plot.getViewBox()
        x0, x1 = vb.viewRange()[0]
        cx = (x0 + x1) / 2
        w = (x1 - x0) * factor
        vb.setXRange(cx - w/2, cx + w/2, padding=0)

    def _show_hotkeys(self):
        QtWidgets.QMessageBox.information(
            self, "Горячие клавиши",
            "Space - Старт/Стоп\n"
            "+ - Приблизить\n"
            "- - Отдалить\n"
            "R - Сброс вида\n"
            "Ctrl+E - Экспорт CSV\n"
            "Ctrl+Q - Выход\n\n"
            "Двойной клик на спектре/водопаде - добавить маркер"
        )

    def _show_about(self):
        QtWidgets.QMessageBox.information(
            self, "О программе",
            f"{APP_TITLE}\n"
            "HackRF Sweep Analyzer\n\n"
            "Модульная версия с поддержкой:\n"
            "• hackrf_sweep и libhackrf (CFFI)\n"
            "• Калибровка CSV (SDR Console format)\n"
            "• Спектр, водопад, пики, маркеры\n"
            "• Детектор активности с ROI\n"
            "• Трилатерация (3 SDR)\n"
            "• Экспорт CSV/PNG"
        )

    def _show_error(self, title: str, msg: str):
        QtWidgets.QMessageBox.critical(self, title, msg)

    # ---------------- действия ----------------
    def _toggle_start_stop(self):
        """Переключает старт/стоп по пробелу."""
        if self._source and self._source.is_running():
            self._source.stop()
        else:
            self.spectrum_tab.btn_start.click()

    def _switch_source(self, name: str):
        """Переключает источник данных."""
        # Останавливаем текущий
        if self._source and self._source.is_running():
            self._source.stop()

        if name == "lib" and self._lib_available and self._lib_source:
            self._source = self._lib_source
            self.statusBar().showMessage("Источник: libhackrf (CFFI)", 3000)
        else:
            self._source = self._sweep_source
            self.statusBar().showMessage("Источник: hackrf_sweep", 3000)

        self.spectrum_tab.set_source(self._source)
        self._wire_source(self._source)

    def _choose_device(self):
        """Выбор устройства HackRF."""
        serials = []
        
        if self._lib_available and self._lib_source:
            serials = self._lib_source.list_serials()
        
        if not serials:
            serials = ["(auto)"]
        
        current = self.settings.value("device/serial", "", type=str) or ""
        
        dlg = DeviceDialog(serials, current, self)
        if self._lib_source:
            dlg.set_provider(self._lib_source.list_serials)
        
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            sel = dlg.selected_serial_suffix().strip()
            
            # Сохраняем выбор
            self.settings.setValue("device/serial", sel)
            
            # Применяем к источникам
            if self._lib_source:
                self._lib_source.set_serial_suffix(sel or None)
            
            # Для hackrf_sweep нужно будет передать через SweepConfig.serial
            
            self.statusBar().showMessage(f"Выбран: {sel or '(auto)'}", 3000)

    # ---------------- калибровка ----------------
    def _try_load_default_calibration(self):
        """Пытается загрузить hackrf_cal.csv из текущей директории."""
        default_path = "hackrf_cal.csv"
        if os.path.isfile(default_path):
            self._load_calibration_file(default_path)

    def _load_calibration_csv(self):
        """Загрузка калибровки через диалог."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Файл калибровки CSV", "",
            "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        
        self._load_calibration_file(path)

    def _load_calibration_file(self, path: str):
        """Загружает файл калибровки."""
        try:
            self._calibration_profiles = load_calibration_csv(path)
            
            if not self._calibration_profiles:
                self._show_error("Калибровка", "Файл пуст или неверный формат")
                return
            
            # Для libhackrf загружаем прямо в библиотеку
            if self._lib_available and self._lib_source:
                ok = self._lib_source.load_calibration(path)
                if ok:
                    self.act_cal_enable.setChecked(True)
                    self.statusBar().showMessage(f"Калибровка загружена: {os.path.basename(path)}", 5000)
                else:
                    self._show_error("Калибровка", "Ошибка загрузки в libhackrf")
            else:
                # Для hackrf_sweep просто сохраняем профили
                self.act_cal_enable.setChecked(True)
                self.statusBar().showMessage(f"Калибровка загружена: {os.path.basename(path)}", 5000)
            
            self.log.info(f"Загружено профилей калибровки: {len(self._calibration_profiles)}")
            
        except Exception as e:
            self._show_error("Калибровка", f"Ошибка загрузки: {e}")

    def _toggle_calibration(self, on: bool):
        """Включает/выключает применение калибровки."""
        if self._lib_available and self._lib_source:
            self._lib_source.set_calibration_enabled(on)
        
        self.statusBar().showMessage(f"Калибровка: {'включена' if on else 'выключена'}", 3000)

    def _clear_calibration(self):
        """Очищает загруженную калибровку."""
        self._calibration_profiles = {}
        self.act_cal_enable.setChecked(False)
        self.statusBar().showMessage("Калибровка очищена", 3000)

    # ---------------- трилатерация ----------------
    def _open_trilateration(self):
        """Открывает окно трилатерации."""
        if not self._lib_available:
            self._show_error("Трилатерация", "Требуется libhackrf (CFFI)")
            return
        
        # Диалог выбора 3 устройств
        serials = self._lib_source.list_serials() if self._lib_source else []
        if len(serials) < 3:
            self._show_error("Трилатерация", f"Требуется минимум 3 устройства (найдено: {len(serials)})")
            return
        
        # Простой диалог выбора
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Выбор устройств для трилатерации")
        dlg.resize(400, 200)
        
        layout = QtWidgets.QFormLayout(dlg)
        
        master_combo = QtWidgets.QComboBox()
        slave1_combo = QtWidgets.QComboBox()
        slave2_combo = QtWidgets.QComboBox()
        
        for combo in [master_combo, slave1_combo, slave2_combo]:
            combo.addItems(serials)
        
        if len(serials) >= 3:
            master_combo.setCurrentIndex(0)
            slave1_combo.setCurrentIndex(1)
            slave2_combo.setCurrentIndex(2)
        
        layout.addRow("Master:", master_combo)
        layout.addRow("Slave 1:", slave1_combo)
        layout.addRow("Slave 2:", slave2_combo)
        
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        master = master_combo.currentText()
        slave1 = slave1_combo.currentText()
        slave2 = slave2_combo.currentText()
        
        # Проверка уникальности
        if len({master, slave1, slave2}) < 3:
            self._show_error("Трилатерация", "Выберите 3 разных устройства")
            return
        
        # Открываем окно
        self.trilat_window = TrilaterationWindow(master, slave1, slave2, self)
        self.trilat_window.show()

    # ---------------- экспорт ----------------
    def _export_current_csv(self):
        """Экспорт текущего свипа в CSV."""
        freqs, row = self.spectrum_tab.get_current_row()
        if freqs is None or row is None:
            QtWidgets.QMessageBox.information(self, "Экспорт", "Нет данных для экспорта")
            return
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить CSV", 
            f"sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV files (*.csv)"
        )
        if not path:
            return
        
        try:
            write_row_csv(path, freqs, row)
            self.statusBar().showMessage(f"Экспортировано: {os.path.basename(path)}", 5000)
        except Exception as e:
            self._show_error("Экспорт", f"Ошибка: {e}")

    def _export_waterfall_png(self):
        """Экспорт водопада в PNG."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить PNG",
            f"waterfall_{time.strftime('%Y%m%d_%H%M%S')}.png",
            "PNG files (*.png)"
        )
        if not path:
            return
        
        try:
            self.spectrum_tab.export_waterfall_png(path)
            self.statusBar().showMessage(f"Экспортировано: {os.path.basename(path)}", 5000)
        except Exception as e:
            self._show_error("Экспорт", f"Ошибка: {e}")

    # -------- окно/настройки --------
    def _restore_window_state(self):
        self.restoreGeometry(self.settings.value("main/geometry", type=QtCore.QByteArray) or QtCore.QByteArray())
        self.restoreState(self.settings.value("main/windowState", type=QtCore.QByteArray) or QtCore.QByteArray())

    def _save_window_state(self):
        self.settings.setValue("main/geometry", self.saveGeometry())
        self.settings.setValue("main/windowState", self.saveState())

    def closeEvent(self, e):
        # Останавливаем источник
        try:
            if self._source and self._source.is_running():
                self._source.stop()
        except Exception:
            pass
        
        # Сохраняем настройки
        try:
            if hasattr(self.spectrum_tab, "save_settings"):
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
    logger.info(f"ПАНОРАМА запущена")

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")  # Современный стиль
    
    win = MainWindow(logger, settings)
    win.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()