# panorama/main.py
import sys, os, stat, getpass, pathlib, logging, time
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QSettings

from panorama.features.spectrum import SpectrumView
from panorama.features.peaks.ui_improved import AdaptivePeaksWidget  # Новый модуль пиков
from panorama.features.detector.widget import DetectorWidget  # Улучшенный детектор
from panorama.features.devices.manager import DeviceManager, DeviceConfigDialog  # Новый менеджер
from panorama.features.map3d import MapView
from panorama.features.trilateration.engine import TrilaterationEngine, SignalMeasurement  # Новый движок

from panorama.drivers.hackrf_sweep import HackRFSweepSource
from panorama.shared.calibration import load_calibration_csv, get_calibration_lut

try:
    from panorama.drivers.hackrf_lib import HackRFLibSource
    _LIB_AVAILABLE = True
except Exception:
    HackRFLibSource = None  # type: ignore
    _LIB_AVAILABLE = False

from panorama.shared import write_row_csv, setup_logging, merged_defaults


APP_TITLE = "ПАНОРАМА 0.2 Pro"


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


class MainWindow(QtWidgets.QMainWindow):
    """Главное окно приложения ПАНОРАМА."""
    
    def __init__(self, logger: logging.Logger, settings: QSettings):
        super().__init__()
        self.log = logger
        self.settings = settings
        self._calibration_profiles = {}
        
        # Менеджер устройств
        self.device_manager = DeviceManager()
        
        # Движок трилатерации
        self.trilateration_engine = TrilaterationEngine()

        self.setWindowTitle(APP_TITLE)
        self.resize(1600, 950)
        
        # Применяем темную тему
        self._apply_dark_theme()

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
        self.peaks_tab = AdaptivePeaksWidget()  # Новый виджет пиков
        self.detector_tab = DetectorWidget()  # Улучшенный детектор

        # Провязка сигналов
        self._connect_signals()

        # Добавляем вкладки
        self.tabs.addTab(self.spectrum_tab, "📊 Спектр")
        self.tabs.addTab(self.peaks_tab, "📍 Пики")
        self.tabs.addTab(self.detector_tab, "🎯 Детектор")
        self.tabs.addTab(self.map_tab, "🗺️ Карта")

        # Меню и статусбар
        self._build_menu()
        self._build_shortcuts()
        
        # Статусбар с индикаторами
        self._build_statusbar()

        # Восстановление состояния
        self._restore_window_state()
        if hasattr(self.spectrum_tab, "restore_settings"):
            self.spectrum_tab.restore_settings(self.settings, merged_defaults())
        
        # Автозагрузка калибровки если есть
        self._try_load_default_calibration()

    def _apply_dark_theme(self):
        """Применяет темную тему к приложению."""
        dark_stylesheet = """
        QMainWindow {
            background-color: #2b2b2b;
        }
        QTabWidget::pane {
            background-color: #2b2b2b;
            border: 1px solid #444;
        }
        QTabBar::tab {
            background-color: #3c3c3c;
            color: #ccc;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #4a4a4a;
            color: white;
        }
        QGroupBox {
            color: #ccc;
            border: 1px solid #555;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #3c3c3c;
            color: #ccc;
            border: 1px solid #555;
            padding: 6px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
        }
        QPushButton:pressed {
            background-color: #555;
        }
        QTableWidget {
            background-color: #2b2b2b;
            color: #ccc;
            gridline-color: #444;
            selection-background-color: #4a4a4a;
        }
        QHeaderView::section {
            background-color: #3c3c3c;
            color: #ccc;
            padding: 4px;
            border: 1px solid #444;
        }
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
            background-color: #3c3c3c;
            color: #ccc;
            border: 1px solid #555;
            padding: 4px;
            border-radius: 2px;
        }
        QLabel {
            color: #ccc;
        }
        QCheckBox {
            color: #ccc;
        }
        QMenuBar {
            background-color: #2b2b2b;
            color: #ccc;
        }
        QMenuBar::item:selected {
            background-color: #4a4a4a;
        }
        QMenu {
            background-color: #3c3c3c;
            color: #ccc;
            border: 1px solid #555;
        }
        QMenu::item:selected {
            background-color: #4a4a4a;
        }
        QStatusBar {
            background-color: #2b2b2b;
            color: #ccc;
            border-top: 1px solid #444;
        }
        """
        self.setStyleSheet(dark_stylesheet)

    def _connect_signals(self):
        """Подключает все сигналы между компонентами."""
        # Спектр → Пики и Детектор
        self.spectrum_tab.newRowReady.connect(self.peaks_tab.update_from_row)
        self.spectrum_tab.newRowReady.connect(self.detector_tab.push_data)
        self.spectrum_tab.newRowReady.connect(self._process_for_trilateration)
        
        # Пики → Спектр
        self.peaks_tab.goToFreq.connect(self.spectrum_tab.set_cursor_freq)
        
        # Детектор → Карта
        self.detector_tab.sendToMap.connect(self._send_detection_to_map)
        self.detector_tab.rangeSelected.connect(self.spectrum_tab.add_roi_region)
        
        # Карта → Трилатерация
        self.map_tab.trilaterationStarted.connect(self._start_trilateration)
        self.map_tab.trilaterationStopped.connect(self._stop_trilateration)

    def _build_statusbar(self):
        """Создает статусбар с индикаторами."""
        self.statusBar().showMessage("Готово")
        
        # Индикатор источника
        self.lbl_source = QtWidgets.QLabel("Источник: hackrf_sweep")
        self.lbl_source.setStyleSheet("padding: 0 10px;")
        self.statusBar().addPermanentWidget(self.lbl_source)
        
        # Индикатор устройств
        self.lbl_devices = QtWidgets.QLabel("SDR: 0")
        self.lbl_devices.setStyleSheet("padding: 0 10px;")
        self.statusBar().addPermanentWidget(self.lbl_devices)
        
        # Индикатор калибровки
        self.lbl_calibration = QtWidgets.QLabel("CAL: ✗")
        self.lbl_calibration.setStyleSheet("padding: 0 10px; color: #ff6666;")
        self.statusBar().addPermanentWidget(self.lbl_calibration)
        
        # Индикатор трилатерации
        self.lbl_trilateration = QtWidgets.QLabel("TRI: ⚪")
        self.lbl_trilateration.setStyleSheet("padding: 0 10px;")
        self.statusBar().addPermanentWidget(self.lbl_trilateration)

    def _wire_source(self, src):
        """Подключает обработчики к источнику."""
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
        
        act_export_csv = QtWidgets.QAction("📄 Экспорт свипа CSV…", self)
        act_export_csv.triggered.connect(self._export_current_csv)
        
        act_export_png = QtWidgets.QAction("🖼️ Экспорт водопада PNG…", self)
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
        
        act_devices = QtWidgets.QAction("⚙️ Настройка SDR устройств…", self)
        act_devices.triggered.connect(self._configure_devices)
        m_source.addAction(act_devices)

        # === Калибровка ===
        m_cal = menubar.addMenu("&Калибровка")
        
        act_cal_load = QtWidgets.QAction("📁 Загрузить CSV…", self)
        act_cal_load.triggered.connect(self._load_calibration_csv)
        m_cal.addAction(act_cal_load)
        
        self.act_cal_enable = QtWidgets.QAction("✓ Применять калибровку", self, checkable=True, checked=False)
        self.act_cal_enable.toggled.connect(self._toggle_calibration)
        m_cal.addAction(self.act_cal_enable)
        
        m_cal.addSeparator()
        
        act_cal_clear = QtWidgets.QAction("🗑️ Очистить калибровку", self)
        act_cal_clear.triggered.connect(self._clear_calibration)
        m_cal.addAction(act_cal_clear)

        # === Инструменты ===
        m_tools = menubar.addMenu("&Инструменты")
        
        act_trilateration = QtWidgets.QAction("📡 Трилатерация (3 SDR)", self)
        act_trilateration.triggered.connect(self._open_trilateration_settings)
        m_tools.addAction(act_trilateration)
        
        act_signal_db = QtWidgets.QAction("📊 База сигналов", self)
        act_signal_db.triggered.connect(self._open_signal_database)
        m_tools.addAction(act_signal_db)

        # === Справка ===
        m_help = menubar.addMenu("&Справка")
        
        act_hotkeys = QtWidgets.QAction("⌨️ Горячие клавиши", self)
        act_hotkeys.triggered.connect(self._show_hotkeys)
        m_help.addAction(act_hotkeys)
        
        act_about = QtWidgets.QAction("ℹ️ О программе", self)
        act_about.triggered.connect(self._show_about)
        m_help.addAction(act_about)

    def _build_shortcuts(self):
        """Горячие клавиши."""
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self._toggle_start_stop)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+E"), self, activated=self._export_current_csv)
        QtWidgets.QShortcut(QtGui.QKeySequence("R"), self, activated=self.spectrum_tab._on_reset_view)
        QtWidgets.QShortcut(QtGui.QKeySequence("+"), self, activated=lambda: self._zoom_x(0.8))
        QtWidgets.QShortcut(QtGui.QKeySequence("-"), self, activated=lambda: self._zoom_x(1.25))
        QtWidgets.QShortcut(QtGui.QKeySequence("D"), self, activated=lambda: self.tabs.setCurrentWidget(self.detector_tab))
        QtWidgets.QShortcut(QtGui.QKeySequence("S"), self, activated=lambda: self.tabs.setCurrentWidget(self.spectrum_tab))
        QtWidgets.QShortcut(QtGui.QKeySequence("P"), self, activated=lambda: self.tabs.setCurrentWidget(self.peaks_tab))
        QtWidgets.QShortcut(QtGui.QKeySequence("M"), self, activated=lambda: self.tabs.setCurrentWidget(self.map_tab))

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
            "**Управление:**\n"
            "Space - Старт/Стоп\n"
            "+ - Приблизить\n"
            "- - Отдалить\n"
            "R - Сброс вида\n\n"
            "**Навигация:**\n"
            "S - Вкладка Спектр\n"
            "P - Вкладка Пики\n"
            "D - Вкладка Детектор\n"
            "M - Вкладка Карта\n\n"
            "**Экспорт:**\n"
            "Ctrl+E - Экспорт CSV\n"
            "Ctrl+Q - Выход\n\n"
            "**Мышь:**\n"
            "Двойной клик на спектре/водопаде - добавить маркер\n"
            "Колесо мыши - зум по X"
        )

    def _show_about(self):
        QtWidgets.QMessageBox.information(
            self, "О программе",
            f"**{APP_TITLE}**\n"
            "Advanced HackRF Sweep Analyzer\n\n"
            "**Возможности:**\n"
            "• Адаптивный детектор с baseline + N порогом\n"
            "• Трилатерация целей с 3 SDR\n"
            "• Классификация сигналов по диапазонам\n"
            "• Менеджер устройств с никнеймами\n"
            "• Фильтр Калмана для траекторий\n"
            "• Калибровка CSV (SDR Console format)\n"
            "• Экспорт CSV/PNG/JSON\n\n"
            "**Версия:** 0.2 Pro\n"
            "**Лицензия:** MIT"
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
        if self._source and self._source.is_running():
            self._source.stop()

        if name == "lib" and self._lib_available and self._lib_source:
            self._source = self._lib_source
            self.lbl_source.setText("Источник: libhackrf")
            self.statusBar().showMessage("Источник: libhackrf (CFFI)", 3000)
        else:
            self._source = self._sweep_source
            self.lbl_source.setText("Источник: hackrf_sweep")
            self.statusBar().showMessage("Источник: hackrf_sweep", 3000)

        self.spectrum_tab.set_source(self._source)
        self._wire_source(self._source)

    def _configure_devices(self):
        """Открывает диалог настройки SDR устройств."""
        # Получаем список доступных устройств
        serials = []
        if self._lib_available and self._lib_source:
            serials = self._lib_source.list_serials()
        
        if not serials:
            # Заглушка для демонстрации
            serials = ["HACKRF_001", "HACKRF_002", "HACKRF_003", "HACKRF_004"]
        
        dlg = DeviceConfigDialog(self.device_manager, serials, self)
        dlg.devicesConfigured.connect(self._on_devices_configured)
        
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._update_device_status()

    def _on_devices_configured(self):
        """Вызывается после настройки устройств."""
        self._update_device_status()
        
        # Обновляем движок трилатерации
        master, slave1, slave2 = self.device_manager.get_trilateration_devices()
        
        if master and slave1 and slave2:
            self.trilateration_engine.set_device_positions(
                (master.position_x, master.position_y, master.position_z),
                (slave1.position_x, slave1.position_y, slave1.position_z),
                (slave2.position_x, slave2.position_y, slave2.position_z),
                master.serial, slave1.serial, slave2.serial
            )
            
            # Обновляем карту
            self.map_tab._on_sdr_moved()

    def _update_device_status(self):
        """Обновляет статус устройств в статусбаре."""
        online_count = sum(1 for d in self.device_manager.devices.values() if d.is_online)
        self.lbl_devices.setText(f"SDR: {online_count}")
        
        # Проверяем готовность к трилатерации
        master, slave1, slave2 = self.device_manager.get_trilateration_devices()
        if master and slave1 and slave2:
            self.lbl_trilateration.setText("TRI: ✓")
            self.lbl_trilateration.setStyleSheet("padding: 0 10px; color: #66ff66;")
        else:
            self.lbl_trilateration.setText("TRI: ✗")
            self.lbl_trilateration.setStyleSheet("padding: 0 10px; color: #ff6666;")

    # ---------------- трилатерация ----------------
    def _open_trilateration_settings(self):
        """Открывает настройки трилатерации."""
        master, slave1, slave2 = self.device_manager.get_trilateration_devices()
        
        if not (master and slave1 and slave2):
            QtWidgets.QMessageBox.warning(
                self, "Трилатерация",
                "Необходимо настроить 3 SDR устройства!\n"
                "Источник → Настройка SDR устройств"
            )
            return
        
        # Показываем информацию о готовности
        QtWidgets.QMessageBox.information(
            self, "Трилатерация готова",
            f"**Устройства настроены:**\n\n"
            f"Master: {master.nickname} ({master.serial})\n"
            f"Позиция: ({master.position_x:.1f}, {master.position_y:.1f}, {master.position_z:.1f})\n\n"
            f"Slave 1: {slave1.nickname} ({slave1.serial})\n"
            f"Позиция: ({slave1.position_x:.1f}, {slave1.position_y:.1f}, {slave1.position_z:.1f})\n\n"
            f"Slave 2: {slave2.nickname} ({slave2.serial})\n"
            f"Позиция: ({slave2.position_x:.1f}, {slave2.position_y:.1f}, {slave2.position_z:.1f})\n\n"
            f"Переключитесь на вкладку **Карта** для запуска!"
        )
        
        # Переключаемся на карту
        self.tabs.setCurrentWidget(self.map_tab)

    def _start_trilateration(self):
        """Запускает трилатерацию."""
        self.trilateration_engine.start()
        self.lbl_trilateration.setText("TRI: 🔴")
        self.lbl_trilateration.setStyleSheet("padding: 0 10px; color: #ff6666;")
        self.statusBar().showMessage("Трилатерация запущена", 3000)

    def _stop_trilateration(self):
        """Останавливает трилатерацию."""
        self.trilateration_engine.stop()
        self.lbl_trilateration.setText("TRI: ✓")
        self.lbl_trilateration.setStyleSheet("padding: 0 10px; color: #66ff66;")
        self.statusBar().showMessage("Трилатерация остановлена", 3000)

    def _process_for_trilateration(self, freqs_hz, power_dbm):
        """Обрабатывает данные для трилатерации."""
        if not self.trilateration_engine.is_running:
            return
        
        # Определяем какое это устройство
        # В реальной системе это будет определяться по источнику
        device_serial = self.device_manager.master or "UNKNOWN"
        
        # Находим пики для трилатерации
        threshold = np.median(power_dbm) + 10  # Простой порог
        peaks_mask = power_dbm > threshold
        
        if np.any(peaks_mask):
            peak_idx = np.argmax(power_dbm)
            
            from panorama.features.trilateration.engine import SignalMeasurement
            measurement = SignalMeasurement(
                timestamp=time.time(),
                device_serial=device_serial,
                freq_mhz=freqs_hz[peak_idx] / 1e6,
                power_dbm=power_dbm[peak_idx],
                bandwidth_khz=200,  # Заглушка
                noise_floor_dbm=np.median(power_dbm)
            )
            
            self.trilateration_engine.add_measurement(measurement)

    def _send_detection_to_map(self, detection):
        """Отправляет обнаружение на карту."""
        # Здесь должна быть логика добавления цели на карту
        # Используем трилатерацию для определения позиции
        positions = self.trilateration_engine.get_current_positions()
        
        if detection.freq_mhz in positions:
            pos = positions[detection.freq_mhz]
            # Добавляем на карту с вычисленными координатами
            self.statusBar().showMessage(
                f"Цель на карте: {detection.freq_mhz:.1f} МГц @ ({pos.x:.1f}, {pos.y:.1f})",
                5000
            )

    def _open_signal_database(self):
        """Открывает базу данных сигналов."""
        QtWidgets.QMessageBox.information(
            self, "База сигналов",
            "**Классификация сигналов:**\n\n"
            "• 87.5-108 МГц - FM Radio\n"
            "• 118-137 МГц - Aviation\n"
            "• 144-148 МГц - Amateur 2m\n"
            "• 156-163 МГц - Marine VHF\n"
            "• 430-440 МГц - Amateur 70cm\n"
            "• 446 МГц - PMR446\n"
            "• 433 МГц - ISM 433\n"
            "• 868 МГц - ISM 868\n"
            "• 900-960 МГц - GSM\n"
            "• 2.4 ГГц - WiFi/Bluetooth\n"
            "• 5.8 ГГц - FPV Video\n"
            "• 1.5-1.6 ГГц - GPS/GNSS\n"
        )

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
                    self.lbl_calibration.setText("CAL: ✓")
                    self.lbl_calibration.setStyleSheet("padding: 0 10px; color: #66ff66;")
                    self.statusBar().showMessage(f"Калибровка загружена: {os.path.basename(path)}", 5000)
                else:
                    self._show_error("Калибровка", "Ошибка загрузки в libhackrf")
            else:
                # Для hackrf_sweep просто сохраняем профили
                self.act_cal_enable.setChecked(True)
                self.lbl_calibration.setText("CAL: ✓")
                self.lbl_calibration.setStyleSheet("padding: 0 10px; color: #66ff66;")
                self.statusBar().showMessage(f"Калибровка загружена: {os.path.basename(path)}", 5000)
            
            self.log.info(f"Загружено профилей калибровки: {len(self._calibration_profiles)}")
            
        except Exception as e:
            self._show_error("Калибровка", f"Ошибка загрузки: {e}")

    def _toggle_calibration(self, on: bool):
        """Включает/выключает применение калибровки."""
        if self._lib_available and self._lib_source:
            self._lib_source.set_calibration_enabled(on)
        
        if on and self._calibration_profiles:
            self.lbl_calibration.setText("CAL: ✓")
            self.lbl_calibration.setStyleSheet("padding: 0 10px; color: #66ff66;")
        else:
            self.lbl_calibration.setText("CAL: ✗")
            self.lbl_calibration.setStyleSheet("padding: 0 10px; color: #ff6666;")
        
        self.statusBar().showMessage(f"Калибровка: {'включена' if on else 'выключена'}", 3000)

    def _clear_calibration(self):
        """Очищает загруженную калибровку."""
        self._calibration_profiles = {}
        self.act_cal_enable.setChecked(False)
        self.lbl_calibration.setText("CAL: ✗")
        self.lbl_calibration.setStyleSheet("padding: 0 10px; color: #ff6666;")
        self.statusBar().showMessage("Калибровка очищена", 3000)

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
        
        # Останавливаем трилатерацию
        try:
            if self.trilateration_engine.is_running:
                self.trilateration_engine.stop()
        except Exception:
            pass
        
        # Сохраняем настройки
        try:
            if hasattr(self.spectrum_tab, "save_settings"):
                self.spectrum_tab.save_settings(self.settings)
        except Exception:
            pass
        
        # Сохраняем конфигурацию устройств
        self.device_manager.save_config()
        
        self._save_window_state()
        super().closeEvent(e)


def main():
    _fix_runtime_dir()
    
    QtCore.QCoreApplication.setOrganizationName("panorama")
    QtCore.QCoreApplication.setApplicationName("panorama")
    settings = QSettings(QSettings.IniFormat, QSettings.UserScope, "panorama", "panorama")

    logger = setup_logging("panorama")
    logger.info(f"ПАНОРАМА Pro запущена")

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Устанавливаем иконку приложения
    app.setWindowIcon(QtGui.QIcon.fromTheme("radio"))
    
    win = MainWindow(logger, settings)
    win.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()