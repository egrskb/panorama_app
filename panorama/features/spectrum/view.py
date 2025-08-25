from __future__ import annotations
from typing import Optional, Deque, Dict, Tuple, Any
from collections import deque
import time, json, os

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui

from panorama.drivers.base import SweepConfig, SourceBackend
from panorama.features.spectrum.model import SpectrumModel
from panorama.shared.palettes import get_colormap
from panorama.features.settings.storage import load_detector_settings

MAX_DISPLAY_COLS = 4096   # максимум колонок на экране (≈4k)
WATER_ROWS_TARGET = 200   # число строк водопада по вертикали


class SpectrumView(QtWidgets.QWidget):
    """Главный виджет спектра с водопадом, маркерами и полной функциональностью."""

    newRowReady    = QtCore.pyqtSignal(object, object)  # (freqs_hz, row_dbm)
    rangeSelected  = QtCore.pyqtSignal(float, float)
    configChanged  = QtCore.pyqtSignal()

    def __init__(self, parent=None, orchestrator=None):
        super().__init__(parent)

        # ---- данные / состояние ----
        self._model = SpectrumModel(rows=WATER_ROWS_TARGET)
        self._source: Optional[SourceBackend] = None
        self._current_cfg: Optional[SweepConfig] = None
        self._orchestrator = orchestrator

        # стабильный «видимый» буфер водопада (даунсемпленный по колонкам)
        self._water_view: Optional[np.ndarray] = None
        self._wf_ds_factor: int = 1
        self._wf_cols_ds: int = 0
        self._wf_x_ds: Optional[np.ndarray] = None
        self._wf_max_cols: int = MAX_DISPLAY_COLS

        # статистика свипов
        self._sweep_count = 0
        self._last_full_ts: Optional[float] = None
        self._dt_ema: Optional[float] = None
        self._running = False

        # --- верхняя панель параметров ---
        self.start_mhz = QtWidgets.QDoubleSpinBox()
        self._cfg_dsb(self.start_mhz, 24, 7000, 1, 50.0, " МГц")  # ИЗМЕНЕНО: начало 50 МГц
        self.stop_mhz = QtWidgets.QDoubleSpinBox()
        self._cfg_dsb(self.stop_mhz, 25, 7000, 1, 6000.0, " МГц")  # ИЗМЕНЕНО: конец 6000 МГц
        self.bin_khz = QtWidgets.QDoubleSpinBox()
        self._cfg_dsb(self.bin_khz, 1, 5000, 0, 800, " кГц")  # ОПТИМИЗАЦИЯ: увеличено с 200 до 800 для лучшей производительности
        self.lna_db = QtWidgets.QSpinBox()
        self.lna_db.setRange(0, 40); self.lna_db.setSingleStep(8); self.lna_db.setValue(24)
        self.vga_db = QtWidgets.QSpinBox()
        self.vga_db.setRange(0, 62); self.vga_db.setSingleStep(2); self.vga_db.setValue(20)
        self.amp_on = QtWidgets.QCheckBox("AMP")
        
        # Параметры для оркестратора
        self.spanSpin = QtWidgets.QDoubleSpinBox()
        self.spanSpin.setRange(1, 1000)
        self.spanSpin.setValue(5)
        self.spanSpin.setSuffix(" МГц")
        
        self.dwellSpin = QtWidgets.QSpinBox()
        self.dwellSpin.setRange(100, 10000)
        self.dwellSpin.setValue(1000)
        self.dwellSpin.setSuffix(" мс")
        
        self.btn_start = QtWidgets.QPushButton("Старт")
        self.btn_stop  = QtWidgets.QPushButton("Стоп"); self.btn_stop.setEnabled(False)
        self.btn_reset = QtWidgets.QPushButton("Сброс вида")

        top = QtWidgets.QHBoxLayout()
        def col(lbl, w):
            v = QtWidgets.QVBoxLayout()
            v.addWidget(QtWidgets.QLabel(lbl)); v.addWidget(w); return v
        for w, lab in [
            (self.start_mhz, "F нач (МГц)"),
            (self.stop_mhz,  "F конец (МГц)"),
            (self.bin_khz,   "Bin (кГц)"),
            (self.lna_db,    "LNA"),
            (self.vga_db,    "VGA"),
        ]:
            top.addLayout(col(lab, w))
        top.addWidget(self.amp_on)
        
        # Добавляем параметры оркестратора
        top.addLayout(col("Span (МГц)", self.spanSpin))
        top.addLayout(col("Dwell (мс)", self.dwellSpin))
        
        top.addStretch(1)
        top.addWidget(self.btn_start); top.addWidget(self.btn_stop); top.addWidget(self.btn_reset)

        # --------- левый блок: графики ----------
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Частота (МГц)")
        self.plot.setLabel("left",   "Мощность (дБм)")
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        vb = self.plot.getViewBox()
        vb.enableAutoRange(x=False, y=False)
        vb.setMouseEnabled(x=True, y=False)

        # линии спектра
        self.curve_now = pg.PlotCurveItem([], [], pen=pg.mkPen('#FFFFFF', width=1))
        self.curve_avg = pg.PlotCurveItem([], [], pen=pg.mkPen('#00FF00', width=1))
        self.curve_min = pg.PlotCurveItem([], [], pen=pg.mkPen((120, 120, 255), width=1))
        self.curve_max = pg.PlotCurveItem([], [], pen=pg.mkPen('#FFC800', width=1))
        self.plot.addItem(self.curve_now)
        self.plot.addItem(self.curve_avg)
        self.plot.addItem(self.curve_min)
        self.plot.addItem(self.curve_max)

        # курсор
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((80, 80, 80, 120)))
        self._hline = pg.InfiniteLine(angle=0,  movable=False, pen=pg.mkPen((80, 80, 80, 120)))
        self.plot.addItem(self._vline, ignoreBounds=True)
        self.plot.addItem(self._hline, ignoreBounds=True)
        self._cursor_text = pg.TextItem(color=pg.mkColor(255, 255, 255), anchor=(0, 1))
        self.plot.addItem(self._cursor_text)

        # водопад
        self.water_plot = pg.PlotItem()
        self.water_plot.setLabel("bottom", "Частота (МГц)")
        self.water_plot.setLabel("left",   "Время →")
        self.water_plot.invertY(True)               # свежие строки внизу
        self.water_plot.setMouseEnabled(x=True, y=False)

        self.water_img = pg.ImageItem(axisOrder="row-major")
        self.water_img.setAutoDownsample(False)     # сами управляем downsample по колонкам
        self.water_plot.addItem(self.water_img)
        self.water_plot.setXLink(self.plot)

        # палитра и уровни
        self._lut_name = "turbo"
        self._lut = get_colormap(self._lut_name, 256)
        self._wf_levels = (-110.0, -20.0)

        # контейнер для графиков
        graphs = QtWidgets.QVBoxLayout()
        graphs.addLayout(top)
        graphs.addWidget(self.plot, stretch=2)
        glw = pg.GraphicsLayoutWidget()
        glw.addItem(self.water_plot)
        graphs.addWidget(glw, stretch=3)
        graphs_w = QtWidgets.QWidget(); graphs_w.setLayout(graphs)

        # --------- правая панель ----------
        self.panel = self._build_right_panel()

        # общий сплиттер
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        split.addWidget(graphs_w); split.addWidget(self.panel)
        split.setStretchFactor(0, 4); split.setStretchFactor(1, 1)

        # главная раскладка
        root = QtWidgets.QVBoxLayout(self); root.addWidget(split)

        # обработчики
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_reset.clicked.connect(self._on_reset_view)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(lambda ev: self._on_add_marker(ev, from_water=False))
        self.water_plot.scene().sigMouseClicked.connect(lambda ev: self._on_add_marker(ev, from_water=True))

        # таймер «коалессации» обновлений водопада (теперь почти не нужен)
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setInterval(80)
        self._update_timer.timeout.connect(self._refresh_water)
        self._pending_water_update = False

        # накопители линий
        self._avg_queue: Deque[np.ndarray] = deque(maxlen=8)
        self._minhold: Optional[np.ndarray] = None
        self._maxhold: Optional[np.ndarray] = None
        self._ema_last: Optional[np.ndarray] = None

        # маркеры
        self._marker_seq = 0
        self._markers: Dict[int, Dict[str, Any]] = {}
        self._marker_colors = ["#FF5252", "#40C4FF", "#FFD740", "#69F0AE", "#B388FF",
                               "#FFAB40", "#18FFFF", "#FF6E40", "#64FFDA", "#EEFF41"]

        # ROI регионы
        self._roi_regions = []

        # стартовый вид
        self._on_reset_view()
        self._apply_visibility()
        
        # Загружаем настройки сглаживания из файла
        self._load_smoothing_settings()

    # ----- правая панель -----
    def _build_right_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)

        # Линии
        grp_lines = QtWidgets.QGroupBox("Линии")
        gl = QtWidgets.QFormLayout(grp_lines)
        self.chk_now = QtWidgets.QCheckBox("Текущая"); self.chk_now.setChecked(True)
        self.chk_avg = QtWidgets.QCheckBox("Средняя"); self.chk_avg.setChecked(True)
        self.chk_min = QtWidgets.QCheckBox("Мин");     self.chk_min.setChecked(False)
        self.chk_max = QtWidgets.QCheckBox("Макс");    self.chk_max.setChecked(True)
        self.avg_win = QtWidgets.QSpinBox(); self.avg_win.setRange(1, 200); self.avg_win.setValue(8); self.avg_win.setSuffix(" свипов")
        gl.addRow(self.chk_now); gl.addRow(self.chk_avg)
        h = QtWidgets.QHBoxLayout(); h.addWidget(QtWidgets.QLabel("Окно:")); h.addWidget(self.avg_win); gl.addRow(h)
        gl.addRow(self.chk_min); gl.addRow(self.chk_max)
        for w in (self.chk_now, self.chk_avg, self.chk_min, self.chk_max):
            w.toggled.connect(self._apply_visibility)

        # Сглаживание
        grp_smooth = QtWidgets.QGroupBox("Сглаживание")
        gs = QtWidgets.QFormLayout(grp_smooth)
        self.chk_smooth = QtWidgets.QCheckBox("По частоте"); self.chk_smooth.setChecked(False)
        self.smooth_win = QtWidgets.QSpinBox(); self.smooth_win.setRange(3, 301); self.smooth_win.setSingleStep(2); self.smooth_win.setValue(5)
        self.smooth_win.valueChanged.connect(self._ensure_odd_window)
        self.chk_ema = QtWidgets.QCheckBox("EMA по времени"); self.chk_ema.setChecked(False)
        self.alpha = QtWidgets.QDoubleSpinBox(); self.alpha.setRange(0.01, 1.00); self.alpha.setSingleStep(0.05); self.alpha.setValue(0.30)
        gs.addRow(self.chk_smooth); gs.addRow("Окно:", self.smooth_win)
        gs.addRow(self.chk_ema);    gs.addRow("α:", self.alpha)

        # Водопад
        grp_wf = QtWidgets.QGroupBox("Водопад")
        gw = QtWidgets.QFormLayout(grp_wf)
        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(["qsa", "turbo", "viridis", "inferno", "plasma", "magma", "gray"])
        self.cmb_cmap.setCurrentText(self._lut_name)
        self.sp_wf_min = QtWidgets.QDoubleSpinBox(); self.sp_wf_min.setRange(-200, 50); self.sp_wf_min.setValue(self._wf_levels[0]); self.sp_wf_min.setSuffix(" дБм")
        self.sp_wf_max = QtWidgets.QDoubleSpinBox(); self.sp_wf_max.setRange(-200, 50); self.sp_wf_max.setValue(self._wf_levels[1]); self.sp_wf_max.setSuffix(" дБм")
        self.btn_auto_levels = QtWidgets.QPushButton("Авто уровни")
        self.chk_wf_invert = QtWidgets.QCheckBox("Инвертировать (свежие снизу)")
        self.chk_wf_invert.setChecked(True)
        gw.addRow("Палитра:", self.cmb_cmap)
        gw.addRow("Мин:", self.sp_wf_min)
        gw.addRow("Макс:", self.sp_wf_max)
        gw.addRow(self.chk_wf_invert)
        gw.addRow(self.btn_auto_levels)
        self.cmb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        self.sp_wf_min.valueChanged.connect(self._on_wf_levels)
        self.sp_wf_max.valueChanged.connect(self._on_wf_levels)
        self.btn_auto_levels.clicked.connect(self._auto_levels)
        self.chk_wf_invert.toggled.connect(self._on_wf_invert)

        # Маркеры
        grp_mrk = QtWidgets.QGroupBox("Маркеры")
        gm = QtWidgets.QVBoxLayout(grp_mrk)
        self.list_markers = QtWidgets.QListWidget(); self.list_markers.setMaximumHeight(150)
        gm.addWidget(self.list_markers)
        btn_clear = QtWidgets.QPushButton("Очистить все"); gm.addWidget(btn_clear)
        btn_clear.clicked.connect(self._clear_markers)
        self.list_markers.itemDoubleClicked.connect(self._jump_to_marker)

        # Статус
        self.lbl_sweep = QtWidgets.QLabel("Свипов: 0")

        # собрать
        lay.addWidget(grp_lines)
        lay.addWidget(grp_smooth)
        lay.addWidget(grp_wf)
        lay.addWidget(grp_mrk)
        lay.addStretch(1)
        lay.addWidget(self.lbl_sweep)
        return panel

    # ---------- даунсемплинг для линий ----------
    def _downsample_row(self, y: np.ndarray, max_cols: int = MAX_DISPLAY_COLS) -> np.ndarray:
        n = int(y.size)
        if n <= max_cols:
            return y
        factor = int(np.ceil(n / max_cols))
        pad = (-n) % factor
        y2 = np.pad(y, (0, pad), mode='edge') if pad else y
        y2 = y2.reshape(-1, factor)
        return y2.max(axis=1).astype(np.float32, copy=False)  # максимум — чтобы пики были видны

    def _downsample_x(self, x: np.ndarray, max_cols: int = MAX_DISPLAY_COLS) -> np.ndarray:
        n = int(x.size)
        if n <= max_cols:
            return x
        factor = int(np.ceil(n / max_cols))
        pad = (-n) % factor
        x2 = np.pad(x, (0, pad), mode='edge') if pad else x
        x2 = x2.reshape(-1, factor)
        return x2.mean(axis=1).astype(np.float64, copy=False)

    # ---------- сервис ----------
    def _ensure_odd_window(self, value):
        if value % 2 == 0:
            self.smooth_win.setValue(value + 1)

    def _on_wf_invert(self, checked):
        self.water_plot.invertY(checked)

    def _on_cmap_changed(self, name: str):
        self._lut_name = name
        self._lut = get_colormap(name, 256)
        # не трогаем форму буфера/rect — только LUT
        self.water_img.setLookupTable(self._lut)

    def _on_wf_levels(self):
        self._wf_levels = (float(self.sp_wf_min.value()), float(self.sp_wf_max.value()))
        self.water_img.setLevels(self._wf_levels)

    @staticmethod
    def _cfg_dsb(sp: QtWidgets.QDoubleSpinBox, a, b, dec, val, suffix=""):
        sp.setRange(a, b); sp.setDecimals(dec); sp.setValue(val)
        if suffix: sp.setSuffix(suffix)

    def _apply_visibility(self):
        self.curve_now.setVisible(self.chk_now.isChecked())
        self.curve_avg.setVisible(self.chk_avg.isChecked())
        self.curve_min.setVisible(self.chk_min.isChecked())
        self.curve_max.setVisible(self.chk_max.isChecked())

    def _auto_levels(self):
        """
        Автонастройка уровней палитры для водопада.
        Берём данные из отображаемого буфера (_water_view, если есть),
        иначе — из полного self._model.water. Добавляем лёгкое сглаживание,
        чтобы уровни не дёргались от кадра к кадру.
        """
        # 1) источник данных
        if getattr(self, "_water_view", None) is not None:
            Z = self._water_view
        else:
            Z = self._model.water

        if Z is None:
            return

        # отбрасываем «пустые» значения
        data = Z[np.isfinite(Z) & (Z > -200.0)]
        if data.size == 0:
            return

        # 2) робастные уровни по перцентилям
        vmin_new = float(np.percentile(data, 5))
        vmax_new = float(np.percentile(data, 99))

        # гарантируем минимальный разлёт
        if vmax_new - vmin_new < 5.0:
            mid = 0.5 * (vmin_new + vmax_new)
            vmin_new = mid - 2.5
            vmax_new = mid + 2.5

        # 3) лёгкое сглаживание (чтоб не мигало при шуме)
        vmin_old, vmax_old = self._wf_levels
        alpha = 0.3
        vmin_s = alpha * vmin_new + (1.0 - alpha) * vmin_old
        vmax_s = alpha * vmax_new + (1.0 - alpha) * vmax_old

        # 4) применяем: обновляем спинбоксы (они вызовут _on_wf_levels сами)
        # чтобы избежать лишних сигналов, если разница мала — не трогаем
        if abs(vmin_s - vmin_old) > 0.2:
            self.sp_wf_min.setValue(vmin_s)
        if abs(vmax_s - vmax_old) > 0.2:
            self.sp_wf_max.setValue(vmax_s)

    # ---------- Source wiring ----------
    def set_source(self, src: SourceBackend):
        if self._source is not None:
            for sig, slot in [
                (self._source.fullSweepReady, self._on_full_sweep),
                (self._source.status,        self._on_status),
                (self._source.error,         self._on_error),
                (self._source.started,       self._on_started),
                (self._source.finished,      self._on_finished),
            ]:
                try: sig.disconnect(slot)
                except Exception: pass

        self._source = src
        self._source.fullSweepReady.connect(self._on_full_sweep)
        self._source.status.connect(self._on_status)
        self._source.error.connect(self._on_error)
        self._source.started.connect(self._on_started)
        self._source.finished.connect(self._on_finished)

    def _check_sdr_master_configured(self) -> bool:
        """Проверяет, что SDR master настроен в sdr_settings.json."""
        try:
            from panorama.features.settings.storage import load_sdr_settings
            sdr_settings = load_sdr_settings()
            
            if not sdr_settings or 'master' not in sdr_settings:
                return False
            
            master_config = sdr_settings['master']
            if not master_config or 'serial' not in master_config:
                return False
            
            master_serial = master_config['serial']
            if not master_serial or len(master_serial) < 16:
                return False
            
            return True
            
        except Exception as e:
            print(f"[SpectrumView] Error checking SDR master: {e}")
            return False

    # ---------- сигналы источника ----------
    def _on_status(self, msg: object) -> None:
        try:
            text = str(msg)
        except Exception:
            text = repr(msg)
        if hasattr(self, "lbl_sweep") and self.lbl_sweep is not None:
            base = self.lbl_sweep.text().split(" • ")[0]
            self.lbl_sweep.setText(f"{base} • {text}")
        print(f"[Spectrum] status: {text}")

    def _on_error(self, err: object) -> None:
        try:
            text = str(err)
        except Exception:
            text = repr(err)
        print(f"[Spectrum] ERROR: {text}")
        QtWidgets.QMessageBox.critical(self, "Ошибка источника", text)
        self._set_controls_enabled(True)
        self._running = False

    def _on_started(self) -> None:
        self._set_controls_enabled(False)
        self._running = True
        print("[Spectrum] started")

    def _on_finished(self, code: int) -> None:
        self._set_controls_enabled(True)
        self._running = False
        print(f"[Spectrum] finished, code={code}")

    # ---------- кнопки ----------
    def _on_start_clicked(self):
        if not self._source:
            QtWidgets.QMessageBox.information(self, "Источник", "Источник данных не подключён")
            return

        # Проверяем, что SDR master настроен
        if not self._check_sdr_master_configured():
            QtWidgets.QMessageBox.critical(self, "SDR Master не настроен", 
                "Для запуска спектра необходимо настроить SDR Master устройство.\n\n"
                "Перейдите в Настройки → Диспетчер устройств и выберите HackRF устройство как Master.")
            return

        f0, f1, bw = self._snap_bounds(
            self.start_mhz.value(),
            self.stop_mhz.value(),
            self.bin_khz.value() * 1e3
        )
        self.start_mhz.setValue(f0 / 1e6)
        self.stop_mhz.setValue(f1 / 1e6)

        # Получаем серийный номер master из настроек
        master_serial = None
        try:
            from panorama.features.settings.storage import load_sdr_settings
            sdr_settings = load_sdr_settings()
            if sdr_settings and 'master' in sdr_settings:
                master_config = sdr_settings['master']
                if master_config and 'serial' in master_config:
                    master_serial = master_config['serial']
        except Exception:
            pass
        
        cfg = SweepConfig(
            freq_start_hz=int(f0),
            freq_end_hz=int(f1),
            bin_hz=int(bw),
            lna_db=int(self.lna_db.value()),
            vga_db=int(self.vga_db.value()),
            amp_on=bool(self.amp_on.isChecked()),
            serial=master_serial,  # Передаем серийный номер
        )
        self._current_cfg = cfg

        # Сетка модели
        self._model.set_grid(cfg.freq_start_hz, cfg.freq_end_hz, cfg.bin_hz)

        # Сброс статистик
        self._minhold = None; self._maxhold = None
        self._avg_queue.clear(); self._ema_last = None

        # Жёсткие пределы и центрирование графика (фикс выхода за границы)
        freqs = self._model.freqs_hz
        if freqs is not None and freqs.size:
            x_mhz = freqs.astype(np.float64) / 1e6
            x0, x1 = float(x_mhz[0]), float(x_mhz[-1])
            self.plot.setLimits(xMin=x0, xMax=x1)
            self.water_plot.setLimits(xMin=x0, xMax=x1)

            self.plot.setXRange(x0, x1, padding=0.0)
            self.plot.setYRange(-110.0, -20.0, padding=0)
            self.water_plot.setXRange(x0, x1, padding=0.0)
            self.water_plot.setYRange(0, self._model.rows, padding=0)

            # инициализация картинки (пока без данных — с пустыми уровнями/LUT)
            self.water_img.setLookupTable(self._lut)
            self.water_img.setLevels(self._wf_levels)

        # Пробрасываем параметры в оркестратор (есть публичный API)
        if self._orchestrator:
            span_mhz = float(self.spanSpin.value())
            dwell_ms = int(self.dwellSpin.value())
            self._orchestrator.set_global_parameters(span_hz=span_mhz * 1e6, dwell_ms=dwell_ms)

        self._source.start(cfg)
        self.configChanged.emit()
        self._set_controls_enabled(False)
        self._running = True

    def _on_stop_clicked(self):
        if self._source and self._source.is_running():
            self._source.stop()
        self._set_controls_enabled(True)
        self._running = False

    def _on_reset_view(self):
        self.plot.setYRange(-110.0, -20.0, padding=0)
        self._water_view = None   # → следующая полная строка заново проинициализирует водопад

    # ---------- обработка ПОЛНОГО свипа (анти-мерцание) ----------
    def _on_full_sweep(self, freqs_hz: np.ndarray, power_dbm: np.ndarray):
        """
        Пришла ПОЛНАЯ строка спектра без пропусков.
        Рисуем без мерцания: поддерживаем стабильный даунсемплированный буфер self._water_view.
        """
        print(f"[SpectrumView] Received full sweep: freqs={freqs_hz.size}, power={power_dbm.size}, "
              f"freq_range=[{freqs_hz[0]/1e6:.1f}, {freqs_hz[-1]/1e6:.1f}] MHz, "
              f"power_range=[{power_dbm.min():.1f}, {power_dbm.max():.1f}] dBm")
        
        # Дополнительная отладка
        print(f"[SpectrumView] Data types: freqs={type(freqs_hz)}, power={type(power_dbm)}")
        print(f"[SpectrumView] Data shapes: freqs={freqs_hz.shape}, power={power_dbm.shape}")
        print(f"[SpectrumView] Power range: min={power_dbm.min():.2f}, max={power_dbm.max():.2f}")
        print(f"[SpectrumView] Has NaN: freqs={np.any(np.isnan(freqs_hz))}, power={np.any(np.isnan(power_dbm))}")
        print(f"[SpectrumView] Has Inf: freqs={np.any(np.isinf(freqs_hz))}, power={np.any(np.isinf(power_dbm))}")
        
        if freqs_hz is None or power_dbm is None:
            return
        if freqs_hz.size == 0 or power_dbm.size == 0 or freqs_hz.size != power_dbm.size:
            return

        # ВАЖНО: инициализируем не только при смене размера, но и если _water_view ещё нет.
        first_or_need_init = (
            self._model.freqs_hz is None
            or self._model.freqs_hz.size != freqs_hz.size
            or self._water_view is None
        )

        if first_or_need_init:
            print(f"[SpectrumView] Initializing display grid: freqs={freqs_hz.size}")
            # Полная сетка (храним для экспорта/курсора и т.д.)
            self._model.freqs_hz = freqs_hz.astype(np.float64, copy=True)
            # Используем правильный метод для обновления last_row
            self._model.update_full_sweep(freqs_hz.astype(np.float32, copy=True), power_dbm.astype(np.float32, copy=True))
            n = int(freqs_hz.size)
            self._model.water = np.full((self._model.rows, n), -120.0, dtype=np.float32)

            # Фикс осей
            x_mhz = self._model.freqs_hz / 1e6
            x0, x1 = float(x_mhz[0]), float(x_mhz[-1])
            self.plot.setLimits(xMin=x0, xMax=x1)
            self.plot.setXRange(x0, x1, padding=0.0)
            self.plot.setYRange(-110.0, -20.0, padding=0)
            self.water_plot.setLimits(xMin=x0, xMax=x1)
            self.water_plot.setXRange(x0, x1, padding=0.0)
            self.water_plot.setYRange(0, self._model.rows, padding=0)

            # --- даунсемплирование колонок водопада под дисплей ---
            x_mhz = self._model.freqs_hz.astype(np.float64) / 1e6
            self._wf_max_cols = 4096
            self._wf_ds_factor = int(np.ceil(n / self._wf_max_cols)) or 1
            self._wf_cols_ds = int(np.ceil(n / self._wf_ds_factor))

            # X-ось для отображения (усредняем группы)
            self._wf_x_ds = self._downsample_cols_fixed(x_mhz, self._wf_ds_factor, agg="mean").astype(np.float64, copy=False)
            if self._wf_x_ds.size != self._wf_cols_ds:
                # из-за округления приведём к нужной длине
                if self._wf_x_ds.size > self._wf_cols_ds:
                    self._wf_x_ds = self._wf_x_ds[:self._wf_cols_ds]
                else:
                    self._wf_x_ds = np.pad(self._wf_x_ds, (0, self._wf_cols_ds - self._wf_x_ds.size), mode='edge')

            # Сам видимый буфер: форма не меняется кадр-за-кадром
            self._water_view = np.full((self._model.rows, self._wf_cols_ds), -120.0, dtype=np.float32)

            # Один раз назначаем LUT/levels/rect и показываем буфер
            self.water_img.setLookupTable(self._lut)
            self.water_img.setLevels(self._wf_levels)
            self.water_img.setImage(self._water_view, autoLevels=False)
            self._update_water_rect(self._wf_x_ds, self._water_view)

            # Сброс статистик линий
            self._avg_queue.clear()
            self._minhold = None
            self._maxhold = None
            self._ema_last = None

        # Храним полную строку в модели (для экспорта/аналитики)
        self._model.push_waterfall_row(power_dbm.astype(np.float32, copy=False))

        # FPS/счётчик
        t = time.time()
        if self._last_full_ts is not None:
            dt = t - self._last_full_ts
            self._dt_ema = dt if self._dt_ema is None else (0.3 * dt + 0.7 * self._dt_ema)
        self._last_full_ts = t
        self._sweep_count += 1
        fps_text = f" • {1.0/self._dt_ema:.1f} св/с" if self._dt_ema else ""
        self.lbl_sweep.setText(f"Свипов: {self._sweep_count}{fps_text}")

        # Линии спектра
        print(f"[SpectrumView] Calling _refresh_spectrum...")
        self._refresh_spectrum()
        print(f"[SpectrumView] _refresh_spectrum completed")

        # ---- Обновляем СТАБИЛЬНЫЙ буфер водопада ----
        if self._water_view is not None:
            last_row = self._model.last_row
            row_ds = self._downsample_cols_fixed(last_row, self._wf_ds_factor, agg="max").astype(np.float32, copy=False)
            if row_ds.size != self._wf_cols_ds:
                if row_ds.size > self._wf_cols_ds:
                    row_ds = row_ds[:self._wf_cols_ds]
                else:
                    row_ds = np.pad(row_ds, (0, self._wf_cols_ds - row_ds.size), mode='edge')
            
            # Сдвигаем вниз и пишем свежую строку в самый низ
            self._water_view = np.roll(self._water_view, -1, axis=0)
            self._water_view[-1, :] = row_ds
            self.water_img.setImage(self._water_view, autoLevels=False)
        else:
            print(f"[SpectrumView] Waterfall view not initialized yet")
            
        # Когерентная перерисовка (таймер)
        self._pending_water_update = True
        if not self._update_timer.isActive():
            self._update_timer.start()

        # Немного автоматики по уровням после первых кадров
        if self._sweep_count == 5:
            self._auto_levels()

        # Сигнал для внешних обработчиков (пики и т.п.)
        self.newRowReady.emit(freqs_hz, power_dbm)

    def reload_detector_settings(self):
        """Перезагружает настройки детектора из файла."""
        self._model.reload_detector_settings()
        print("[SpectrumView] Настройки детектора перезагружены")

    def _load_smoothing_settings(self):
        """Загружает настройки сглаживания из файла детектора."""
        try:
            detector_settings = load_detector_settings()
            
            # Применяем настройки сглаживания
            if detector_settings.get("smoothing_enabled", False):
                self.chk_smooth.setChecked(True)
                print("[SpectrumView] Сглаживание по частоте включено")
            
            if detector_settings.get("ema_enabled", False):
                self.chk_ema.setChecked(True)
                print("[SpectrumView] EMA по времени включено")
            
            # Устанавливаем значения параметров
            smoothing_window = detector_settings.get("smoothing_window", 7)
            self.smooth_win.setValue(smoothing_window)
            
            ema_alpha = detector_settings.get("ema_alpha", 0.3)
            self.alpha.setValue(ema_alpha)
            
            print(f"[SpectrumView] Загружены настройки сглаживания: окно={smoothing_window}, α={ema_alpha}")
            
        except Exception as e:
            print(f"[SpectrumView] Ошибка загрузки настроек сглаживания: {e}")

    # ---------- сглаживания ----------
    def _smooth_freq(self, y: np.ndarray) -> np.ndarray:
        if not self.chk_smooth.isChecked() or y.size < 3:
            return y
        w = int(self.smooth_win.value())
        if w < 3: w = 3
        if w % 2 == 0: w += 1
        kernel = np.ones(w, dtype=np.float32) / float(w)
        smoothed = np.convolve(y, kernel, mode='same')
        half_w = w // 2
        # аккуратные края — усреднение неполными окнами
        for i in range(half_w):
            smoothed[i]        = np.mean(y[0:i+half_w+1])
            smoothed[-(i+1)]   = np.mean(y[-(i+half_w+1):])
        return smoothed.astype(np.float32)

    def _smooth_time_ema(self, y: np.ndarray) -> np.ndarray:
        if not self.chk_ema.isChecked():
            return y
        a = float(self.alpha.value())
        if self._ema_last is None or self._ema_last.shape != y.shape:
            self._ema_last = y.copy(); return y
        self._ema_last = (a * y) + ((1.0 - a) * self._ema_last)
        return self._ema_last.copy()

    # ---------- вспомогательное ----------
    def _downsample_cols_fixed(self, arr: np.ndarray, factor: int, agg: str = "max") -> np.ndarray:
        """
        Даунсемплирует 1D-массив по колонкам фиксированным фактором (agg: 'max' или 'mean').
        Используется и для X-оси (mean), и для строк водопада (max).
        """
        if factor <= 1 or arr.size == 0:
            return arr.astype(arr.dtype, copy=False)
        pad = (-arr.size) % factor
        arr2 = np.pad(arr, (0, pad), mode='edge') if pad else arr
        blocks = arr2.reshape(-1, factor)
        out = blocks.mean(axis=1) if agg == "mean" else blocks.max(axis=1)
        return out.astype(arr.dtype, copy=False)

    def _set_controls_enabled(self, enabled: bool):
        self.start_mhz.setEnabled(enabled)
        self.stop_mhz.setEnabled(enabled)
        self.bin_khz.setEnabled(enabled)
        self.lna_db.setEnabled(enabled)
        self.vga_db.setEnabled(enabled)
        self.amp_on.setEnabled(enabled)
        self.btn_start.setEnabled(enabled)
        self.btn_stop.setEnabled(not enabled)

    # ---------- публичный API навигации ----------
    def set_cursor_freq(self, f: float, center: bool = True) -> None:
        """
        Установить курсор на частоту f (Гц или МГц) и опционально центрировать вид.
        """
        try:
            f = float(f)
        except Exception:
            return
        f_hz = f * 1e6 if f < 1e7 else f

        if self._model.freqs_hz is None or self._model.freqs_hz.size == 0:
            return

        x_mhz = self._model.freqs_hz.astype(np.float64) / 1e6
        x_val = f_hz / 1e6

        self._vline.setPos(x_val)

        yrow = self._model.last_row
        if yrow is not None and yrow.size == x_mhz.size:
            i = int(np.clip(np.searchsorted(x_mhz, x_val), 1, len(x_mhz) - 1))
            x0, x1 = x_mhz[i - 1], x_mhz[i]
            y0, y1 = float(yrow[i - 1]), float(yrow[i])
            y_at = y0 if x1 == x0 else ((x_val - x0) / (x1 - x0)) * (y1 - y0) + y0
            self._hline.setPos(y_at)

        if center:
            vb = self.plot.getViewBox()
            xr = vb.viewRange()[0] if vb is not None else (x_mhz[0], x_mhz[-1])
            width = xr[1] - xr[0]
            if width <= 0:
                width = max((x_mhz[-1] - x_mhz[0]) * 0.25, 1.0)
            x0 = max(x_val - width / 2.0, float(x_mhz[0]))
            x1 = min(x_val + width / 2.0, float(x_mhz[-1]))
            self.plot.setXRange(x0, x1, padding=0.0)

    # ---------- ROI ----------
    def add_roi_region(self,
                       f_start_hz: float,
                       f_stop_hz: float,
                       label: str | None = None,
                       color: QtGui.QColor | str | None = None,
                       zoom: bool = True) -> int:
        if self._model.freqs_hz is None or self._model.freqs_hz.size == 0:
            return -1

        try:
            f0 = float(f_start_hz)
            f1 = float(f_stop_hz)
        except Exception:
            return -1
        if f1 < f0:
            f0, f1 = f1, f0

        x_mhz = self._model.freqs_hz.astype(np.float64) / 1e6
        x0_lim, x1_lim = float(x_mhz[0]), float(x_mhz[-1])

        x0 = max(x0_lim, min(x1_lim, f0 / 1e6))
        x1 = max(x0_lim, min(x1_lim, f1 / 1e6))
        if x1 <= x0:
            return -1

        if color is None:
            color = QtGui.QColor(64, 196, 255, 60)
        pen   = pg.mkPen(QtGui.QColor(64, 196, 255, 200), width=1)
        brush = pg.mkBrush(color)

        region = pg.LinearRegionItem(values=(x0, x1), orientation=pg.LinearRegionItem.Vertical, movable=True)
        region.setBrush(brush)
        region.setPen(pen)
        region.setZValue(5)
        self.plot.addItem(region)

        txt = label or f"{x0:.3f}–{x1:.3f} MHz"
        text_item = pg.TextItem(txt, color=pen.color(), anchor=(0, 1))
        y_top = self.plot.getViewBox().viewRange()[1][1]
        text_item.setPos(x0, y_top)
        text_item.setZValue(6)
        self.plot.addItem(text_item)

        def _on_change_finished():
            a, b = sorted(region.getRegion())
            text_item.setText(f"{a:.3f}–{b:.3f} MHz")
            y_top2 = self.plot.getViewBox().viewRange()[1][1]
            text_item.setPos(a, y_top2)

        region.sigRegionChangeFinished.connect(_on_change_finished)

        rid = self._marker_seq
        self._marker_seq += 1
        self._roi_regions.append({
            "id": rid,
            "region": region,
            "label": text_item,
        })

        if zoom:
            self.plot.setXRange(x0, x1, padding=0.0)

        return rid

    def clear_roi_regions(self) -> None:
        for it in self._roi_regions:
            try:
                self.plot.removeItem(it["region"])
                self.plot.removeItem(it["label"])
            except Exception:
                pass
        self._roi_regions.clear()

    def remove_roi_region(self, rid: int) -> bool:
        for i, it in enumerate(self._roi_regions):
            if it.get("id") == rid:
                try:
                    self.plot.removeItem(it["region"])
                    self.plot.removeItem(it["label"])
                except Exception:
                    pass
                self._roi_regions.pop(i)
                return True
        return False

    # ---------- вспомогательные UI ----------
    def _snap_bounds(self, f0_mhz: float, f1_mhz: float, bin_hz: float) -> Tuple[float, float, float]:
        bin_mhz = bin_hz / 1e6
        seg_mhz = 5.0
        f0 = np.floor(f0_mhz + 1e-9); f1 = np.floor(f1_mhz + 1e-9)
        width = max(seg_mhz, f1 - f0)
        width = np.floor(width / seg_mhz) * seg_mhz
        if width < seg_mhz: width = seg_mhz
        width = np.round(width / bin_mhz) * bin_mhz
        return float(f0) * 1e6, float(f0 + width) * 1e6, bin_hz

    def _update_water_rect(self, x_mhz: np.ndarray, z: np.ndarray):
        self.water_img.setRect(QtCore.QRectF(
            float(x_mhz[0]), 0.0,
            float(x_mhz[-1] - x_mhz[0]),
            float(z.shape[0])
        ))

    def _refresh_water(self):
        """
        Таймерная перерисовка. Важно: использовать видимый буфер `_water_view`,
        а не полный self._model.water, чтобы не было рассинхрона по размерам.
        """
        if not self._pending_water_update:
            self._update_timer.stop()
            return
        self._pending_water_update = False

        if getattr(self, "_water_view", None) is not None:
            # Перерисовываем только стабильный даунсемплированный буфер
            self.water_img.setImage(self._water_view, autoLevels=False)
            # rect/уровни уже выставлены и не трогаются — никаких мерцаний
        else:
            # Фолбек (если вдруг ещё не инициализировали _water_view)
            z = self._model.water
            if z is None:
                return
            z = np.ascontiguousarray(z)
            self.water_img.setLookupTable(self._lut)
            self.water_img.setLevels(self._wf_levels)
            self.water_img.setImage(z, autoLevels=False)
            freqs = self._model.freqs_hz
            if freqs is not None and freqs.size:
                self._update_water_rect(freqs.astype(np.float64)/1e6, z)
    # ---------- курсор ----------
    def _on_mouse_moved(self, pos):
        vb = self.plot.getViewBox()
        if vb is None:
            return
        p = vb.mapSceneToView(pos)
        fx = float(p.x()); fy = float(p.y())
        self._vline.setPos(fx); self._hline.setPos(fy)
        self._update_cursor_label()

    def _update_cursor_label(self):
        freqs = self._model.freqs_hz; row = self._model.last_row
        fx = float(self._vline.value()); fy = float(self._hline.value())
        if freqs is None or row is None or row.size == 0:
            self._cursor_text.setText(f"{fx:.3f} МГц, {fy:.1f} дБм")
            self._cursor_text.setPos(fx, self.plot.getViewBox().viewRange()[1][1] - 5)
            return
        x_mhz = freqs.astype(np.float64) / 1e6
        i = int(np.clip(np.searchsorted(x_mhz, fx), 1, len(x_mhz)-1))
        x0, x1 = x_mhz[i-1], x_mhz[i]; y0, y1 = row[i-1], row[i]
        y_at = float(y0) if x1 == x0 else float((fx - x0) / (x1 - x0) * (y1 - y0) + y0)
        self._cursor_text.setText(f"{fx:.3f} МГц, {y_at:.1f} дБм")
        self._cursor_text.setPos(fx, self.plot.getViewBox().viewRange()[1][1] - 5)

    # ---------- маркеры ----------
    def _on_add_marker(self, ev, from_water: bool):
        if not ev.double() or ev.button() != QtCore.Qt.LeftButton:
            return
        vb = (self.water_plot if from_water else self.plot).getViewBox()
        p = vb.mapSceneToView(ev.scenePos())
        fx = float(p.x())

        name, ok = QtWidgets.QInputDialog.getText(
            self, "Новый маркер",
            f"Частота: {fx:.6f} МГц\nНазвание:",
            QtWidgets.QLineEdit.Normal,
            f"M{self._marker_seq + 1}"
        )
        if not ok or not name:
            return
        self._add_marker(fx, name)

    def _add_marker(self, f_mhz: float, name: str, color: Optional[str] = None):
        self._marker_seq += 1
        mid = self._marker_seq
        if color is None:
            color = self._marker_colors[mid % len(self._marker_colors)]

        line_spec = pg.InfiniteLine(
            pos=f_mhz, angle=90, movable=True,
            pen=pg.mkPen(color, width=2, style=QtCore.Qt.DashLine)
        )
        line_water = pg.InfiniteLine(
            pos=f_mhz, angle=90, movable=True,
            pen=pg.mkPen(color, width=1.5, style=QtCore.Qt.DashLine)
        )
        label = pg.TextItem(name, anchor=(0.5, 1), color=color)
        label.setPos(f_mhz, self.plot.getViewBox().viewRange()[1][1])

        self.plot.addItem(line_spec)
        self.plot.addItem(label)
        self.water_plot.addItem(line_water)

        line_spec.sigPositionChanged.connect(lambda: self._sync_marker(mid, line_spec.value(), "spec"))
        line_water.sigPositionChanged.connect(lambda: self._sync_marker(mid, line_water.value(), "water"))

        self._markers[mid] = {
            "freq": f_mhz,
            "name": name,
            "color": color,
            "line_spec": line_spec,
            "line_water": line_water,
            "label": label
        }

        item = QtWidgets.QListWidgetItem(f"{name}: {f_mhz:.6f} МГц")
        item.setData(QtCore.Qt.UserRole, mid)
        item.setForeground(QtGui.QBrush(QtGui.QColor(color)))
        self.list_markers.addItem(item)

    def _sync_marker(self, mid: int, new_freq: float, source: str):
        if mid not in self._markers:
            return
        m = self._markers[mid]
        m["freq"] = new_freq
        if source == "spec":
            m["line_water"].setValue(new_freq)
        else:
            m["line_spec"].setValue(new_freq)
        m["label"].setPos(new_freq, self.plot.getViewBox().viewRange()[1][1])
        for i in range(self.list_markers.count()):
            item = self.list_markers.item(i)
            if item.data(QtCore.Qt.UserRole) == mid:
                item.setText(f"{m['name']}: {new_freq:.6f} МГц")
                break

    def _jump_to_marker(self, item: QtWidgets.QListWidgetItem):
        mid = item.data(QtCore.Qt.UserRole)
        if mid not in self._markers:
            return
        freq = self._markers[mid]["freq"]
        vb = self.plot.getViewBox()
        x0, x1 = vb.viewRange()[0]
        center = (x0 + x1) / 2
        offset = freq - center
        self.plot.setXRange(x0 + offset, x1 + offset, padding=0)

    def _clear_markers(self):
        for m in self._markers.values():
            try:
                self.plot.removeItem(m["line_spec"])
                self.plot.removeItem(m["label"])
                self.water_plot.removeItem(m["line_water"])
            except Exception:
                pass
        self._markers.clear()
        self.list_markers.clear()

    # ---------- настройки ----------
    def restore_settings(self, settings, defaults: dict):
        d = (defaults or {}).get("spectrum", {})
        # Дефолтные значения 50-6000 МГц
        start_default_mhz = float(d.get("start_mhz", 50.0))
        stop_default_mhz  = float(d.get("stop_mhz", 6000.0))
        bin_default_khz   = float(d.get("bin_khz", 800.0))  # ОПТИМИЗАЦИЯ: увеличено с 200.0 до 800.0 для лучшей производительности

        settings.beginGroup("spectrum")
        try:
            self.start_mhz.setValue(float(settings.value("start_mhz", start_default_mhz)))
            self.stop_mhz.setValue(float(settings.value("stop_mhz",  stop_default_mhz)))
            self.bin_khz.setValue(float(settings.value("bin_khz",   bin_default_khz)))
            self.lna_db.setValue(int(settings.value("lna_db", d.get("lna_db", 24))))
            self.vga_db.setValue(int(settings.value("vga_db", d.get("vga_db", 20))))
            self.amp_on.setChecked(str(settings.value("amp_on", d.get("amp_on", False))).lower() in ("1","true","yes"))

            self.chk_now.setChecked(settings.value("chk_now", True,  type=bool))
            self.chk_avg.setChecked(settings.value("chk_avg", True,  type=bool))
            self.chk_min.setChecked(settings.value("chk_min", False, type=bool))
            self.chk_max.setChecked(settings.value("chk_max", True,  type=bool))
            self.avg_win.setValue(int(settings.value("avg_win", 8)))

            self.chk_smooth.setChecked(settings.value("chk_smooth", False, type=bool))
            self.smooth_win.setValue(int(settings.value("smooth_win", 5)))
            self.chk_ema.setChecked(settings.value("chk_ema", False, type=bool))
            self.alpha.setValue(float(settings.value("alpha", 0.3)))

            self._lut_name = settings.value("cmap", "turbo")
            self.cmb_cmap.setCurrentText(self._lut_name)
            self._lut = get_colormap(self._lut_name, 256)

            self.sp_wf_min.setValue(float(settings.value("wf_min", -110.0)))
            self.sp_wf_max.setValue(float(settings.value("wf_max", -20.0)))
            self._wf_levels = (self.sp_wf_min.value(), self.sp_wf_max.value())

            # Маркеры
            markers_json = settings.value("markers", "[]")
            try:
                markers = json.loads(markers_json)
                for m in markers:
                    self._add_marker(
                        float(m.get("freq", 0)),
                        str(m.get("name", "M")),
                        str(m.get("color", "#FFFFFF"))
                    )
            except Exception:
                pass
        finally:
            settings.endGroup()

        self._on_reset_view()
        self._apply_visibility()

    def save_settings(self, settings):
        settings.beginGroup("spectrum")
        try:
            settings.setValue("start_mhz", float(self.start_mhz.value()))
            settings.setValue("stop_mhz",  float(self.stop_mhz.value()))
            settings.setValue("bin_khz",   float(self.bin_khz.value()))
            settings.setValue("lna_db",    int(self.lna_db.value()))
            settings.setValue("vga_db",    int(self.vga_db.value()))
            settings.setValue("amp_on",    bool(self.amp_on.isChecked()))

            settings.setValue("chk_now", self.chk_now.isChecked())
            settings.setValue("chk_avg", self.chk_avg.isChecked())
            settings.setValue("chk_min", self.chk_min.isChecked())
            settings.setValue("chk_max", self.chk_max.isChecked())
            settings.setValue("avg_win", self.avg_win.value())

            settings.setValue("chk_smooth", self.chk_smooth.isChecked())
            settings.setValue("smooth_win", self.smooth_win.value())
            settings.setValue("chk_ema", self.chk_ema.isChecked())
            settings.setValue("alpha", self.alpha.value())

            settings.setValue("cmap", self._lut_name)
            settings.setValue("wf_min", self.sp_wf_min.value())
            settings.setValue("wf_max", self.sp_wf_max.value())

            markers = []
            for m in self._markers.values():
                markers.append({
                    "freq":  m["freq"],
                    "name":  m["name"],
                    "color": m["color"]
                })
            settings.setValue("markers", json.dumps(markers))
        finally:
            settings.endGroup()

    # ---------- экспорт ----------
    def export_waterfall_png(self, path: str):
        """Простой экспорт текущего вида водопада в PNG (best-effort)."""
        from PyQt5.QtWidgets import QGraphicsScene
        from PyQt5.QtGui import QPixmap, QPainter
        from PyQt5.QtCore import QRectF

        scene = QGraphicsScene()
        scene.addItem(self.water_plot)
        rect = self.water_plot.boundingRect()
        pixmap = QPixmap(int(rect.width()), int(rect.height()))
        pixmap.fill(QtCore.Qt.white)
        painter = QPainter(pixmap)
        scene.render(painter, QRectF(pixmap.rect()), rect)
        painter.end()
        pixmap.save(path)

    def export_current_csv(self, path: str):
        freqs = self._model.freqs_hz
        row   = self._model.last_row
        if freqs is None or row is None:
            raise ValueError("Нет данных для экспорта")
        with open(path, "w", encoding="utf-8") as f:
            f.write("freq_hz,freq_mhz,dbm\n")
            for fz, y in zip(freqs, row):
                f.write(f"{float(fz):.3f},{float(fz)/1e6:.6f},{float(y):.2f}\n")

    # ---------- обновление линий спектра ----------
    def _refresh_spectrum(self):
        """
        Обновляет линии спектра с даунсемплингом по X и опциональным сглаживанием.
        """
        freqs = self._model.freqs_hz
        row   = self._model.last_row
        
        # Отладка
        print(f"[SpectrumView] _refresh_spectrum: freqs={freqs is not None}, row={row is not None}, "
              f"freqs_size={freqs.size if freqs is not None else 'None'}")
        
        if freqs is None or row is None or not freqs.size:
            print(f"[SpectrumView] _refresh_spectrum: early return - no data")
            return

        x_mhz = freqs.astype(np.float64) / 1e6
        y     = row.astype(np.float32, copy=False)

        y_smoothed = self._smooth_freq(y)
        y_now      = self._smooth_time_ema(y_smoothed)

        x_ds = self._downsample_x(x_mhz)
        y_ds = self._downsample_row(y_now)
        self.curve_now.setData(x_ds, y_ds)

        # Средняя по N последних строк
        self._avg_queue.append(y.copy())
        while len(self._avg_queue) > self.avg_win.value():
            self._avg_queue.popleft()
        if self._avg_queue and all(arr.size == y.size for arr in self._avg_queue):
            y_avg  = np.mean(np.stack(self._avg_queue, axis=0), axis=0).astype(np.float32, copy=False)
            y_avgS = self._smooth_freq(y_avg)
            self.curve_avg.setData(x_ds, self._downsample_row(y_avgS))

        # Min/Max hold
        if self._maxhold is None or self._maxhold.shape != y.shape:
            self._maxhold = y.copy()
        else:
            self._maxhold = np.maximum(self._maxhold, y)

        if self._minhold is None or self._minhold.shape != y.shape:
            self._minhold = y.copy()
        else:
            self._minhold = np.minimum(self._minhold, y)

        self.curve_min.setData(x_ds, self._downsample_row(self._minhold))
        self.curve_max.setData(x_ds, self._downsample_row(self._maxhold))

        self._update_cursor_label()
