# panorama/main.py
import sys, os, stat, getpass, pathlib, logging, time
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QSettings
import numpy as np

from panorama.features.spectrum import SpectrumView
from panorama.features.peaks.ui_improved import AdaptivePeaksWidget
from panorama.features.detector.widget import DetectorWidget
from panorama.features.devices.manager import DeviceManager, DeviceConfigDialog
from panorama.features.map3d import MapView
from panorama.features.trilateration.engine import TrilaterationEngine, SignalMeasurement

from panorama.drivers.hackrf_sweep import HackRFSweepSource
from panorama.shared.calibration import load_calibration_csv, get_calibration_lut

try:
    from panorama.drivers.hackrf_lib import HackRFLibSource
    _LIB_AVAILABLE = True
except Exception:
    HackRFLibSource = None
    _LIB_AVAILABLE = False

from panorama.shared import write_row_csv, merged_defaults


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


class QuietLogger:
    """Менеджер для фильтрации лишних сообщений."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self._last_messages = {}
        self._suppressed_prefixes = [
            "First few values:",
            "hq_get_master_spectrum: returned",
            "Spectrum update #",
            "Master: Update #",
            "Active signals:",
            "Signal timeout:",
            "Processing peaks",
        ]
        
    def log(self, message, level="info", dedupe_key=None):
        """Логирует с дедупликацией и фильтрацией."""
        # Фильтруем лишние сообщения
        if not self.verbose:
            for prefix in self._suppressed_prefixes:
                if message.startswith(prefix):
                    return
                    
        # Дедупликация повторяющихся сообщений
        if dedupe_key:
            if dedupe_key in self._last_messages:
                if self._last_messages[dedupe_key] == message:
                    return
            self._last_messages[dedupe_key] = message
            
        # Выводим только важные сообщения
        if level == "error":
            print(f"[ERROR] {message}")
        elif level == "warning":
            print(f"[WARN] {message}")
        elif self.verbose or level == "important":
            print(f"[INFO] {message}")


class MainWindow(QtWidgets.QMainWindow):
    """Главное окно приложения ПАНОРАМА с multi-SDR поддержкой."""
    
    def __init__(self, logger: logging.Logger, settings: QSettings):
        super().__init__()
        self.log = logger
        self.settings = settings
        self._calibration_profiles = {}
        
        # Флаги состояния multi-SDR
        self._multi_sdr_active = False
        self._trilateration_active = False
        self._detector_active = False
        
        # Загружаем шрифт с эмодзи поддержкой
        self._load_emoji_font()
        
        # Менеджер устройств
        self.device_manager = DeviceManager()
        
        # Движок трилатерации
        self.trilateration_engine = TrilaterationEngine()

        self.setWindowTitle(APP_TITLE)
        self.resize(1600, 950)
        
        # Применяем улучшенную темную тему
        self._apply_dark_theme()

        # --- центральные вкладки ---
        self.tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # Спектр
        self.spectrum_tab = SpectrumView()

        # Источники
        self._sweep_source = HackRFSweepSource()
        self._lib_source = None
        self._current_source_type = "sweep"
        
        if _LIB_AVAILABLE:
            try:
                self._lib_source = HackRFLibSource()
                self._lib_available = True
                self.log.log("libhackrf_multi успешно загружена", level="important")
            except Exception as e:
                self._lib_available = False
                self.log.log(f"libhackrf_multi недоступна: {e}", level="warning")
        else:
            self._lib_available = False
            self.log.log("libhackrf_multi не скомпилирована", level="warning")

        self._source = self._sweep_source
        self.spectrum_tab.set_source(self._source)
        self._wire_source(self._source)

        # Остальные вкладки
        self.map_tab = MapView()
        self.peaks_tab = AdaptivePeaksWidget()
        self.detector_tab = DetectorWidget()

        # Добавляем вкладки
        self.tabs.addTab(self.spectrum_tab, "📊 Спектр")
        self.tabs.addTab(self.peaks_tab, "📍 Пики")
        self.tabs.addTab(self.detector_tab, "🎯 Детектор")
        self.tabs.addTab(self.map_tab, "🗺️ Карта")
        
        # Провязка сигналов
        self._connect_signals()
        
        # Отключаем вкладки, требующие libhackrf, если используется hackrf_sweep
        self._update_tabs_availability()
        
        # Инициализируем подключение к пикам
        self._on_tab_changed(0)
        
        # Связываем параметры детектора с multi-SDR
        self._connect_detector_to_multisdr()

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
        
        # Таймер для обновления статуса multi-SDR
        self._status_timer = QtCore.QTimer()
        self._status_timer.timeout.connect(self._update_multi_sdr_status)
        self._status_timer.setInterval(1000)  # 1 Hz
        
    def _load_emoji_font(self):
        """Загружает шрифт с поддержкой эмодзи."""
        from PyQt5.QtGui import QFontDatabase
        
        # Пробуем загрузить системные шрифты с эмодзи
        emoji_fonts = [
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
            "/System/Library/Fonts/Apple Color Emoji.ttc",
            "C:/Windows/Fonts/seguiemj.ttf"
        ]
        
        for font_path in emoji_fonts:
            if os.path.exists(font_path):
                QFontDatabase.addApplicationFont(font_path)
                break

    def _apply_dark_theme(self):
        """Применяет улучшенную темную тему."""
        dark_stylesheet = """
        /* Основные элементы */
        QMainWindow {
            background-color: #2b2b2b;
            color: #e0e0e0;
        }
        
        QWidget {
            background-color: #2b2b2b;
            color: #e0e0e0;
        }
        
        /* Вкладки */
        QTabWidget::pane {
            background-color: #353535;
            border: 1px solid #555;
        }
        
        QTabBar::tab {
            background-color: #404040;
            color: #e0e0e0;
            padding: 8px 16px;
            margin-right: 2px;
            border: 1px solid #555;
            border-bottom: none;
        }
        
        QTabBar::tab:selected {
            background-color: #4a90e2;
            color: white;
        }
        
        QTabBar::tab:hover {
            background-color: #505050;
        }
        
        QTabBar::tab:disabled {
            background-color: #2b2b2b;
            color: #666;
        }
        
        /* Группы */
        QGroupBox {
            color: #e0e0e0;
            border: 2px solid #555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            background-color: #2b2b2b;
        }
        
        /* Кнопки */
        QPushButton {
            background-color: #404040;
            color: #e0e0e0;
            border: 1px solid #555;
            padding: 6px 12px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #4a90e2;
            border: 1px solid #5aa0f2;
        }
        
        QPushButton:pressed {
            background-color: #3a80d2;
        }
        
        QPushButton:disabled {
            background-color: #2b2b2b;
            color: #666;
            border: 1px solid #444;
        }
        
        /* Таблицы */
        QTableWidget {
            background-color: #353535;
            color: #e0e0e0;
            gridline-color: #555;
            selection-background-color: #4a90e2;
            border: 1px solid #555;
        }
        
        QTableWidget::item {
            padding: 4px;
            border: none;
        }
        
        QTableWidget::item:selected {
            background-color: #4a90e2;
            color: white;
        }
        
        QHeaderView::section {
            background-color: #404040;
            color: #e0e0e0;
            padding: 6px;
            border: 1px solid #555;
            font-weight: bold;
        }
        
        /* Остальные стили */
        QLineEdit, QTextEdit, QPlainTextEdit {
            background-color: #404040;
            color: #e0e0e0;
            border: 1px solid #555;
            padding: 4px;
            border-radius: 3px;
            selection-background-color: #4a90e2;
        }
        
        QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: #404040;
            color: #e0e0e0;
            border: 1px solid #555;
            padding: 4px;
            border-radius: 3px;
        }
        
        QLabel {
            color: #e0e0e0;
            background-color: transparent;
        }
        
        QCheckBox {
            color: #e0e0e0;
            spacing: 5px;
        }
        
        QMenuBar {
            background-color: #353535;
            color: #e0e0e0;
            border-bottom: 1px solid #555;
        }
        
        QMenu {
            background-color: #404040;
            color: #e0e0e0;
            border: 1px solid #555;
        }
        
        QStatusBar {
            background-color: #353535;
            color: #e0e0e0;
            border-top: 1px solid #555;
        }
        """
        self.setStyleSheet(dark_stylesheet)

    def _connect_signals(self):
        """Подключает все сигналы между компонентами."""
        # Спектр → Пики (только при активации вкладки пиков)
        # НЕ подключаем автоматически к детектору!
        
        # Подключение к пикам только когда вкладка активна
        self.tabs.currentChanged.connect(self._on_tab_changed)
        
        # Детектор работает только при явном запуске
        self.detector_tab.detectionStarted.connect(self._on_detector_started_manual)
        self.detector_tab.detectionStopped.connect(self._on_detector_stopped_manual)
        
        # Спектр → Очистка истории при изменении конфигурации
        self.spectrum_tab.configChanged.connect(self.peaks_tab.clear_history)
        
        # Пики → Спектр (навигация)
        self.peaks_tab.goToFreq.connect(self.spectrum_tab.set_cursor_freq)
        
        # Детектор → Карта и multi-SDR
        self.detector_tab.sendToMap.connect(self._send_detection_to_map)
        self.detector_tab.rangeSelected.connect(self.spectrum_tab.add_roi_region)
        self.detector_tab.signalDetected.connect(self._on_signal_detected)
        
        # Карта → Трилатерация
        self.map_tab.trilaterationStarted.connect(self._start_trilateration)
        self.map_tab.trilaterationStopped.connect(self._stop_trilateration)
        
        # Обработка данных для трилатерации
        if self._lib_available:
            self.spectrum_tab.newRowReady.connect(self._process_for_trilateration)
            
    def _on_signal_detected(self, detection):
        """Обработчик обнаружения сигнала детектором для multi-SDR."""
        if not self._multi_sdr_active:
            return
            
        # Логируем обнаружение
        self.log.log(f"Signal detected: {detection.freq_mhz:.3f} MHz, "
                     f"{detection.power_dbm:.1f} dBm, "
                     f"BW: {detection.bandwidth_khz:.1f} kHz", level="important")
        
        # Для multi-SDR сигналы автоматически передаются через watchlist в libhackrf
        self.statusBar().showMessage(
            f"📡 Цель {detection.freq_mhz:.1f} МГц → Slave SDR", 
            3000
        )
        
    def _on_tab_changed(self, index):
        """При смене вкладки."""
        # Пики работают только на своей вкладке
        if self.tabs.widget(index) == self.peaks_tab:
            self.spectrum_tab.newRowReady.connect(self.peaks_tab.update_from_row)
        else:
            try:
                self.spectrum_tab.newRowReady.disconnect(self.peaks_tab.update_from_row)
            except:
                pass
                
    def _on_detector_started_manual(self):
        """Детектор запущен вручную пользователем."""
        # Только теперь подключаем поток данных к детектору
        self.spectrum_tab.newRowReady.connect(self.detector_tab.push_data)
        
        # Если multi-SDR активен, передаем параметры в библиотеку
        if self._multi_sdr_active and self._lib_source:
            threshold_offset = self.detector_tab.threshold_offset.value()
            min_width = self.detector_tab.min_width.value()
            min_sweeps = self.detector_tab.min_sweeps.value()
            timeout = self.detector_tab.signal_timeout.value()
            
            # Передаем параметры в C библиотеку через FFI
            if hasattr(self._lib_source, '_lib'):
                try:
                    # Добавить эту функцию в cdef библиотеки
                    self._lib_source._ffi.cdef("""
                        void hq_set_detector_params(float threshold_offset_db, 
                                                   int min_width_bins,
                                                   int min_sweeps, 
                                                   float timeout_sec);
                    """)
                    self._lib_source._lib.hq_set_detector_params(
                        threshold_offset, min_width, min_sweeps, timeout
                    )
                except:
                    pass
        
        self._detector_active = True
        self.statusBar().showMessage("🎯 Детектор запущен", 5000)
        
    def _on_detector_stopped_manual(self):
        """Детектор остановлен пользователем."""
        # Отключаем поток данных от детектора
        try:
            self.spectrum_tab.newRowReady.disconnect(self.detector_tab.push_data)
        except:
            pass
            
        self._detector_active = False
        self.statusBar().showMessage("⚪ Детектор остановлен", 5000)
        
    def _connect_detector_to_multisdr(self):
        """Связывает параметры детектора с multi-SDR библиотекой."""
        def update_detector_params(threshold, min_width, min_sweeps, timeout):
            if self._multi_sdr_active and self._lib_source:
                self._lib_source.set_detector_params(
                    threshold, min_width, min_sweeps, timeout
                )
        
        # Подключаем сигнал изменения параметров
        self.detector_tab.parametersChanged.connect(update_detector_params)

    def _build_statusbar(self):
        """Создает статусбар с индикаторами."""
        self.statusBar().showMessage("Готово")
        
        # Индикатор источника
        self.lbl_source = QtWidgets.QLabel("Источник: hackrf_sweep")
        self.lbl_source.setStyleSheet("padding: 0 10px; font-weight: bold;")
        self.statusBar().addPermanentWidget(self.lbl_source)
        
        # Индикатор устройств
        self.lbl_devices = QtWidgets.QLabel("SDR: 0")
        self.lbl_devices.setStyleSheet("padding: 0 10px;")
        self.statusBar().addPermanentWidget(self.lbl_devices)
        
        # Индикатор multi-SDR
        self.lbl_multi_sdr = QtWidgets.QLabel("MULTI: OFF")
        self.lbl_multi_sdr.setStyleSheet("padding: 0 10px;")
        self.statusBar().addPermanentWidget(self.lbl_multi_sdr)
        
        # Индикатор калибровки
        self.lbl_calibration = QtWidgets.QLabel("CAL: НЕТ")
        self.lbl_calibration.setStyleSheet("padding: 0 10px; color: #ff6666;")
        self.statusBar().addPermanentWidget(self.lbl_calibration)
        
        # Индикатор трилатерации
        self.lbl_trilateration = QtWidgets.QLabel("TRI: ВЫКЛ")
        self.lbl_trilateration.setStyleSheet("padding: 0 10px;")
        self.statusBar().addPermanentWidget(self.lbl_trilateration)

    def _update_tabs_availability(self):
        """Обновляет доступность вкладок в зависимости от источника."""
        if self._current_source_type == "sweep":
            # При hackrf_sweep отключаем карту и предупреждаем в детекторе
            self.tabs.setTabEnabled(3, False)  # Карта
            self.tabs.setTabToolTip(3, "Требуется libhackrf для трилатерации")
            
            # Добавляем предупреждение в детектор
            if hasattr(self.detector_tab, 'btn_send_to_map'):
                self.detector_tab.btn_send_to_map.setEnabled(False)
                self.detector_tab.btn_send_to_map.setToolTip("Требуется libhackrf для работы с картой")
        else:
            # При libhackrf все доступно
            self.tabs.setTabEnabled(3, True)
            self.tabs.setTabToolTip(3, "")
            
            if hasattr(self.detector_tab, 'btn_send_to_map'):
                self.detector_tab.btn_send_to_map.setEnabled(True)
                self.detector_tab.btn_send_to_map.setToolTip("")

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
        
        # Отключаем multi-SDR режим при остановке
        if self._multi_sdr_active:
            self._stop_multi_sdr()

    def _update_multi_sdr_status(self):
        """Обновляет статус multi-SDR системы без спама."""
        if not self._lib_available or not self._lib_source:
            return
        
        try:
            status = self._lib_source.get_status()
            if status:
                # Обновляем UI без вывода в консоль
                parts = []
                if status['master_running']:
                    parts.append("M:OK")
                if status['slave1_running']:
                    parts.append("S1:OK")
                if status['slave2_running']:
                    parts.append("S2:OK")
                if status['watch_items'] > 0:
                    parts.append(f"T:{status['watch_items']}")
                
                status_text = f"MULTI: {' '.join(parts)}" if parts else "MULTI: READY"
                
                # Обновляем только если изменилось
                if self.lbl_multi_sdr.text() != status_text:
                    self.lbl_multi_sdr.setText(status_text)
                    if parts:
                        self.lbl_multi_sdr.setStyleSheet("padding: 0 10px; color: #66ff66;")
                    else:
                        self.lbl_multi_sdr.setStyleSheet("padding: 0 10px; color: #ffff66;")
        except:
            pass  # Тихо игнорируем ошибки

    def _build_menu(self):
        menubar = self.menuBar()

        # === Файл ===
        m_file = menubar.addMenu("&Файл")
        
        act_export_csv = QtWidgets.QAction("Экспорт свипа CSV...", self)
        act_export_csv.triggered.connect(self._export_current_csv)
        
        act_export_png = QtWidgets.QAction("Экспорт водопада PNG...", self)
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
            self.act_src_lib = QtWidgets.QAction("libhackrf_multi (CFFI)", self, checkable=True, checked=False)
            m_source.addAction(self.act_src_lib)
            self.src_group.addAction(self.act_src_lib)
            self.act_src_lib.toggled.connect(lambda on: on and self._switch_source("lib"))
        
        m_source.addSeparator()
        
        act_devices = QtWidgets.QAction("Настройка SDR устройств...", self)
        act_devices.triggered.connect(self._configure_devices)
        m_source.addAction(act_devices)
        
        m_source.addSeparator()
        
        # Multi-SDR режим
        self.act_multi_sdr = QtWidgets.QAction("Multi-SDR режим (3 устройства)", self, checkable=True)
        self.act_multi_sdr.triggered.connect(self._toggle_multi_sdr)
        m_source.addAction(self.act_multi_sdr)

        # === Калибровка ===
        m_cal = menubar.addMenu("&Калибровка")
        
        act_cal_load = QtWidgets.QAction("Загрузить CSV...", self)
        act_cal_load.triggered.connect(self._load_calibration_csv)
        m_cal.addAction(act_cal_load)
        
        self.act_cal_enable = QtWidgets.QAction("Применять калибровку", self, checkable=True, checked=False)
        self.act_cal_enable.toggled.connect(self._toggle_calibration)
        m_cal.addAction(self.act_cal_enable)
        
        m_cal.addSeparator()
        
        act_cal_clear = QtWidgets.QAction("Очистить калибровку", self)
        act_cal_clear.triggered.connect(self._clear_calibration)
        m_cal.addAction(act_cal_clear)

        # === Инструменты ===
        m_tools = menubar.addMenu("&Инструменты")
        
        act_trilateration = QtWidgets.QAction("Трилатерация (3 SDR)", self)
        act_trilateration.triggered.connect(self._open_trilateration_settings)
        m_tools.addAction(act_trilateration)
        
        act_signal_db = QtWidgets.QAction("База сигналов", self)
        act_signal_db.triggered.connect(self._open_signal_database)
        m_tools.addAction(act_signal_db)

        # === Справка ===
        m_help = menubar.addMenu("&Справка")
        
        act_hotkeys = QtWidgets.QAction("Горячие клавиши", self)
        act_hotkeys.triggered.connect(self._show_hotkeys)
        m_help.addAction(act_hotkeys)
        
        act_about = QtWidgets.QAction("О программе", self)
        act_about.triggered.connect(self._show_about)
        m_help.addAction(act_about)

    def _toggle_multi_sdr(self):
        """Переключает multi-SDR режим."""
        if not self._lib_available or self._current_source_type != "lib":
            QtWidgets.QMessageBox.warning(
                self, "Multi-SDR",
                "Multi-SDR режим требует источник libhackrf_multi!\n\n"
                "Переключитесь на libhackrf_multi в меню Источник."
            )
            self.act_multi_sdr.setChecked(False)
            return
        
        if self.act_multi_sdr.isChecked():
            self._start_multi_sdr()
        else:
            self._stop_multi_sdr()

    def _start_multi_sdr(self):
        """Запускает multi-SDR режим."""
        # Проверяем настройку устройств
        master, slave1, slave2 = self.device_manager.get_trilateration_devices()
        if not (master and slave1 and slave2):
            QtWidgets.QMessageBox.warning(
                self, "Multi-SDR",
                "Настройте 3 SDR устройства для multi-SDR режима!\n"
                "Источник → Настройка SDR устройств"
            )
            self.act_multi_sdr.setChecked(False)
            return
        
        # Устанавливаем количество устройств
        if self._lib_source:
            self._lib_source.set_num_devices(3)
            
            # Если есть worker, передаем параметры детектора
            if hasattr(self._lib_source, '_multi_worker') and self._lib_source._multi_worker:
                if hasattr(self.detector_tab, 'threshold_offset'):
                    self._lib_source._multi_worker.set_detector_params(
                        self.detector_tab.threshold_offset.value(),
                        self.detector_tab.min_width.value()
                    )
        
        self._multi_sdr_active = True
        self._status_timer.start()
        
        self.lbl_multi_sdr.setText("MULTI: INIT")
        self.lbl_multi_sdr.setStyleSheet("padding: 0 10px; color: #ffff66;")
        
        # Тихий вывод - только важное
        self.log.log("Multi-SDR режим активирован", level="important")

    def _stop_multi_sdr(self):
        """Останавливает multi-SDR режим."""
        self._multi_sdr_active = False
        self._status_timer.stop()
        
        # Возвращаем на 1 устройство
        if self._lib_source:
            self._lib_source.set_num_devices(1)
        
        self.lbl_multi_sdr.setText("MULTI: OFF")
        self.lbl_multi_sdr.setStyleSheet("padding: 0 10px;")
        
        self.statusBar().showMessage("Multi-SDR режим деактивирован", 5000)
        self.log.log("Multi-SDR mode disabled", level="important")

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
            "<b>Управление:</b><br>"
            "Space - Старт/Стоп<br>"
            "+ - Приблизить<br>"
            "- - Отдалить<br>"
            "R - Сброс вида<br><br>"
            "<b>Навигация:</b><br>"
            "S - Вкладка Спектр<br>"
            "P - Вкладка Пики<br>"
            "D - Вкладка Детектор<br>"
            "M - Вкладка Карта<br><br>"
            "<b>Экспорт:</b><br>"
            "Ctrl+E - Экспорт CSV<br>"
            "Ctrl+Q - Выход<br><br>"
            "<b>Мышь:</b><br>"
            "Двойной клик на спектре/водопаде - добавить маркер<br>"
            "Колесо мыши - зум по X"
        )

    def _show_about(self):
        QtWidgets.QMessageBox.information(
            self, "О программе",
            f"<b>{APP_TITLE}</b><br>"
            "Advanced HackRF Multi-SDR Analyzer<br><br>"
            "<b>Возможности:</b><br>"
            "• Multi-SDR режим: Master sweep + Slave tracking<br>"
            "• Адаптивный детектор с baseline + N порогом<br>"
            "• Трилатерация целей с 3 SDR<br>"
            "• Группировка широкополосных сигналов<br>"
            "• Классификация сигналов по диапазонам<br>"
            "• Менеджер устройств с никнеймами<br>"
            "• Фильтр Калмана для траекторий<br>"
            "• Калибровка CSV (SDR Console format)<br>"
            "• Экспорт CSV/PNG/JSON<br><br>"
            "<b>Версия:</b> 0.2 Pro<br>"
            "<b>Лицензия:</b> MIT"
        )

    def _show_error(self, title: str, msg: str):
        QtWidgets.QMessageBox.critical(self, title, msg)

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

        # Отключаем multi-SDR при смене источника
        if self._multi_sdr_active:
            self._stop_multi_sdr()
            self.act_multi_sdr.setChecked(False)

        if name == "lib" and self._lib_available and self._lib_source:
            self._source = self._lib_source
            self._current_source_type = "lib"
            self.lbl_source.setText("Источник: libhackrf_multi")
            self.statusBar().showMessage("Источник: libhackrf_multi (CFFI)", 3000)
            
            # Включаем опцию multi-SDR
            self.act_multi_sdr.setEnabled(True)
        else:
            self._source = self._sweep_source
            self._current_source_type = "sweep"
            self.lbl_source.setText("Источник: hackrf_sweep")
            self.statusBar().showMessage("Источник: hackrf_sweep", 3000)
            
            # Отключаем опцию multi-SDR
            self.act_multi_sdr.setEnabled(False)

        self.spectrum_tab.set_source(self._source)
        self._wire_source(self._source)
        self._update_tabs_availability()

    def _configure_devices(self):
        """Открывает диалог настройки SDR устройств."""
        # Получаем список реальных устройств
        serials = []
        if self._lib_available and self._lib_source:
            try:
                serials = self._lib_source.list_serials()
            except Exception as e:
                self.log.log(f"Не удалось получить список устройств: {e}", level="warning")
        
        if not serials:
            QtWidgets.QMessageBox.warning(
                self, "SDR устройства",
                "Не найдено подключенных HackRF устройств.\n\n"
                "Подключите устройства и попробуйте снова."
            )
            return
        
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
            
            # ВАЖНО: Синхронизируем с картой
            self.map_tab.update_devices(master, slave1, slave2)

    def _update_device_status(self):
        """Обновляет статус устройств в статусбаре."""
        online_count = sum(1 for d in self.device_manager.devices.values() if d.is_online)
        self.lbl_devices.setText(f"SDR: {online_count}")
        
        # Проверяем готовность к трилатерации
        master, slave1, slave2 = self.device_manager.get_trilateration_devices()
        if master and slave1 and slave2:
            self.lbl_trilateration.setText("TRI: ГОТОВ")
            self.lbl_trilateration.setStyleSheet("padding: 0 10px; color: #66ff66;")
        else:
            self.lbl_trilateration.setText("TRI: НЕ ГОТОВ")
            self.lbl_trilateration.setStyleSheet("padding: 0 10px; color: #ff6666;")

    def _open_trilateration_settings(self):
        """Открывает настройки трилатерации."""
        if self._current_source_type == "sweep":
            QtWidgets.QMessageBox.warning(
                self, "Трилатерация",
                "Трилатерация требует libhackrf_multi!\n\n"
                "Переключитесь на источник libhackrf_multi в меню Источник."
            )
            return
            
        master, slave1, slave2 = self.device_manager.get_trilateration_devices()
        
        if not (master and slave1 and slave2):
            QtWidgets.QMessageBox.warning(
                self, "Трилатерация",
                "Необходимо настроить 3 SDR устройства!\n"
                "Источник → Настройка SDR устройств"
            )
            return
        
        # Показываем информацию о готовности
        info_msg = f"<b>Устройства настроены:</b><br><br>"
        info_msg += f"Master: {master.nickname} ({master.serial})<br>"
        info_msg += f"Позиция: ({master.position_x:.1f}, {master.position_y:.1f}, {master.position_z:.1f})<br><br>"
        info_msg += f"Slave 1: {slave1.nickname} ({slave1.serial})<br>"
        info_msg += f"Позиция: ({slave1.position_x:.1f}, {slave1.position_y:.1f}, {slave1.position_z:.1f})<br><br>"
        info_msg += f"Slave 2: {slave2.nickname} ({slave2.serial})<br>"
        info_msg += f"Позиция: ({slave2.position_x:.1f}, {slave2.position_y:.1f}, {slave2.position_z:.1f})<br><br>"
        
        if self._multi_sdr_active:
            info_msg += "<b>Multi-SDR режим активен!</b><br>"
            info_msg += "Переключитесь на вкладку <b>Карта</b> для запуска трилатерации!"
        else:
            info_msg += "Активируйте <b>Multi-SDR режим</b> в меню Источник,<br>"
            info_msg += "затем переключитесь на вкладку <b>Карта</b>!"
        
        QtWidgets.QMessageBox.information(self, "Трилатерация", info_msg)
        
        # Переключаемся на карту если multi-SDR активен
        if self._multi_sdr_active:
            self.tabs.setCurrentWidget(self.map_tab)

    def _start_trilateration(self):
        """Запускает трилатерацию."""
        if not self._multi_sdr_active:
            QtWidgets.QMessageBox.warning(
                self, "Трилатерация",
                "Активируйте Multi-SDR режим перед запуском трилатерации!\n"
                "Источник → Multi-SDR режим"
            )
            return
        
        self.trilateration_engine.start()
        self._trilateration_active = True
        self.lbl_trilateration.setText("TRI: АКТИВНА")
        self.lbl_trilateration.setStyleSheet("padding: 0 10px; color: #ff6666; font-weight: bold;")
        self.statusBar().showMessage("Трилатерация запущена", 3000)

    def _stop_trilateration(self):
        """Останавливает трилатерацию."""
        self.trilateration_engine.stop()
        self._trilateration_active = False
        self.lbl_trilateration.setText("TRI: ГОТОВ")
        self.lbl_trilateration.setStyleSheet("padding: 0 10px; color: #66ff66;")
        self.statusBar().showMessage("Трилатерация остановлена", 3000)

    def _process_for_trilateration(self, freqs_hz, power_dbm):
        """Обрабатывает данные для трилатерации."""
        if not self._trilateration_active or not self.trilateration_engine.is_running:
            return
        
        # Определяем какое это устройство
        device_serial = self.device_manager.master.serial if self.device_manager.master else "UNKNOWN"
        
        # Находим пики для трилатерации
        threshold = np.median(power_dbm) + 10
        peaks_mask = power_dbm > threshold
        
        if np.any(peaks_mask):
            peak_idx = np.argmax(power_dbm)
            
            measurement = SignalMeasurement(
                timestamp=time.time(),
                device_serial=device_serial,
                freq_mhz=freqs_hz[peak_idx] / 1e6,
                power_dbm=power_dbm[peak_idx],
                bandwidth_khz=200,
                noise_floor_dbm=np.median(power_dbm)
            )
            
            self.trilateration_engine.add_measurement(measurement)

    def _send_detection_to_map(self, detection):
        """Отправляет обнаружение на карту (вызывается только пользователем через кнопку)."""
        if self._current_source_type == "sweep":
            QtWidgets.QMessageBox.warning(
                self, "Карта",
                "Работа с картой требует libhackrf_multi!\n\n"
                "Переключитесь на источник libhackrf_multi."
            )
            return
        
        # Добавляем цель на карту
        target = self.map_tab.add_target_from_detector(detection)
        
        if target:
            self.statusBar().showMessage(
                f"Цель добавлена на карту: {detection.freq_mhz:.1f} МГц",
                5000
            )
            
            # Переключаемся на вкладку карты
            self.tabs.setCurrentWidget(self.map_tab)

    def _open_signal_database(self):
        """Открывает базу данных сигналов."""
        QtWidgets.QMessageBox.information(
            self, "База сигналов",
            "<b>Классификация сигналов:</b><br><br>"
            "• 87.5-108 МГц - FM Radio<br>"
            "• 118-137 МГц - Aviation<br>"
            "• 144-148 МГц - Amateur 2m<br>"
            "• 156-163 МГц - Marine VHF<br>"
            "• 430-440 МГц - Amateur 70cm<br>"
            "• 446 МГц - PMR446<br>"
            "• 433 МГц - ISM 433<br>"
            "• 868 МГц - ISM 868<br>"
            "• 900-960 МГц - GSM<br>"
            "• 1090 МГц - ADS-B<br>"
            "• 1575 МГц - GPS L1<br>"
            "• 2.4 ГГц - WiFi/Bluetooth<br>"
            "• 5.8 ГГц - FPV Video/WiFi 5G<br><br>"
            "<b>Типы по ширине полосы:</b><br>"
            "• < 25 кГц - Narrowband (PMR, голос)<br>"
            "• 25-200 кГц - Voice/Data<br>"
            "• 200 кГц-2 МГц - Wideband<br>"
            "• 2-10 МГц - Video/WiFi<br>"
            "• > 10 МГц - Ultra-Wide"
        )

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
                    self.lbl_calibration.setText("CAL: ДА")
                    self.lbl_calibration.setStyleSheet("padding: 0 10px; color: #66ff66;")
                    self.statusBar().showMessage(f"Калибровка загружена: {os.path.basename(path)}", 5000)
                else:
                    self._show_error("Калибровка", "Ошибка загрузки в libhackrf")
            else:
                # Для hackrf_sweep просто сохраняем профили
                self.act_cal_enable.setChecked(True)
                self.lbl_calibration.setText("CAL: ДА")
                self.lbl_calibration.setStyleSheet("padding: 0 10px; color: #66ff66;")
                self.statusBar().showMessage(f"Калибровка загружена: {os.path.basename(path)}", 5000)
            
            self.log.log(f"Загружено профилей калибровки: {len(self._calibration_profiles)}", level="important")
            
        except Exception as e:
            self._show_error("Калибровка", f"Ошибка загрузки: {e}")

    def _toggle_calibration(self, on: bool):
        """Включает/выключает применение калибровки."""
        if self._lib_available and self._lib_source:
            self._lib_source.set_calibration_enabled(on)
        
        if on and self._calibration_profiles:
            self.lbl_calibration.setText("CAL: ДА")
            self.lbl_calibration.setStyleSheet("padding: 0 10px; color: #66ff66;")
        else:
            self.lbl_calibration.setText("CAL: НЕТ")
            self.lbl_calibration.setStyleSheet("padding: 0 10px; color: #ff6666;")
        
        self.statusBar().showMessage(f"Калибровка: {'включена' if on else 'выключена'}", 3000)

    def _clear_calibration(self):
        """Очищает загруженную калибровку."""
        self._calibration_profiles = {}
        self.act_cal_enable.setChecked(False)
        self.lbl_calibration.setText("CAL: НЕТ")
        self.lbl_calibration.setStyleSheet("padding: 0 10px; color: #ff6666;")
        self.statusBar().showMessage("Калибровка очищена", 3000)

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
        
        # Останавливаем multi-SDR
        try:
            if self._multi_sdr_active:
                self._stop_multi_sdr()
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


# Полная логика работы multi-SDR с детектором
class MultiSDRController:
    """Контроллер для управления multi-SDR режимом."""
    
    def __init__(self, lib_source, detector_widget):
        self.lib_source = lib_source
        self.detector = detector_widget
        self.active = False
        self.targets = {}  # freq -> target_info
        
    def start(self):
        """Запуск multi-SDR с правильной последовательностью."""
        if not self.lib_source or self.active:
            return False
            
        # 1. Устанавливаем параметры детектора
        self._sync_detector_params()
        
        # 2. Запускаем master для sweep
        # Master автоматически начнет заполнять watchlist
        
        # 3. Slaves автоматически начнут отслеживать цели из watchlist
        
        self.active = True
        return True
        
    def _sync_detector_params(self):
        """Синхронизирует параметры детектора с C библиотекой."""
        if not self.detector:
            return
            
        # Получаем текущие параметры из UI
        if self.detector.threshold_mode.currentText().startswith("Авто"):
            threshold = self.detector.threshold_offset.value()
        else:
            # Для фиксированного порога вычисляем offset от baseline
            threshold = self.detector.fixed_threshold.value() + 110
            
        min_width = self.detector.min_width.value()
        min_sweeps = self.detector.min_sweeps.value()
        timeout = self.detector.signal_timeout.value()
        
        # Передаем в библиотеку
        if self.lib_source:
            self.lib_source.set_detector_params(
                threshold, min_width, min_sweeps, timeout
            )
            
    def process_watchlist(self, watchlist):
        """Обрабатывает watchlist от C библиотеки."""
        # Обновляем список целей
        new_targets = {}
        
        for item in watchlist:
            key = f"{item.f_center_hz:.1f}"
            
            if key in self.targets:
                # Обновляем существующую цель
                target = self.targets[key]
                target['rssi'] = item.rssi_ema
                target['hits'] = item.hit_count
                target['last_seen'] = time.time()
            else:
                # Новая цель
                target = {
                    'freq_hz': item.f_center_hz,
                    'bandwidth_hz': item.bw_hz,
                    'rssi': item.rssi_ema,
                    'hits': item.hit_count,
                    'first_seen': time.time(),
                    'last_seen': time.time()
                }
                
            new_targets[key] = target
            
        # Удаляем устаревшие цели
        timeout = 5.0  # секунд
        current_time = time.time()
        
        for key in list(self.targets.keys()):
            if key not in new_targets:
                if current_time - self.targets[key]['last_seen'] > timeout:
                    del self.targets[key]
                    
        self.targets = new_targets
        
        return self.targets


def main():
    _fix_runtime_dir()
    
    QtCore.QCoreApplication.setOrganizationName("panorama")
    QtCore.QCoreApplication.setApplicationName("panorama")
    settings = QSettings(QSettings.IniFormat, QSettings.UserScope, "panorama", "panorama")

    # Используем тихий логгер
    logger = QuietLogger(verbose=False)  # verbose=True для отладки
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Подавляем Qt warnings если не в режиме отладки
    if not logger.verbose:
        QtCore.qInstallMessageHandler(lambda *args: None)
    
    win = MainWindow(logger, settings)
    win.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()