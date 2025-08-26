# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

import numpy as np

# --- Qt (опционально) ---
try:
    from PyQt5.QtCore import QObject, pyqtSignal  # type: ignore
    _HAS_QT = True
except Exception:  # pragma: no cover
    _HAS_QT = False
    class _Dummy(object):  # type: ignore
        pass
    def pyqtSignal(*_a, **_k):  # type: ignore
        return None
    QObject = _Dummy  # type: ignore

# --- Настройки детектора ---
from panorama.features.settings.storage import load_detector_settings


# ---------- Типы данных ----------

@dataclass(frozen=True)
class DetectedPeak:
    """Информация о пике, для мастера/оркестратора/слейвов."""
    # Основные поля (совместимость с новым кодом)
    freq_hz: float           # Частота пика (по максимуму)
    snr_db: float            # Оценка SNR (peak - noise_floor)
    power_dbm: float         # Абсолютная мощность в точке пика
    band_hz: float           # Полуширина окна +/- (из настроек)
    idx: int                 # Индекс бина на текущей сетке
    
    # Дополнительные поля для совместимости со старым кодом
    id: str = ""             # Идентификатор пика
    f_peak: float = 0.0     # Частота пика (дублирует freq_hz)
    bin_hz: float = 0.0     # Ширина бина
    t0: float = 0.0         # Время первого обнаружения
    last_seen: float = 0.0  # Время последнего обнаружения
    span_user: float = 0.0  # Пользовательский span
    status: str = "ACTIVE"  # Статус пика


# ---------- Общая реализация детекции (ядро) ----------

def _detect_peaks_core(
    freqs_hz: np.ndarray,
    power_dbm: np.ndarray,
    snr_threshold_db: float = 10.0,
    min_peak_bins: int = 3,
    min_peak_distance_bins: int = 5,
    peak_band_hz: float = 5e6,
) -> List[DetectedPeak]:
    """Чистая детекция пиков без зависимостей от Qt/модели."""
    if freqs_hz is None or power_dbm is None:
        return []
    f = np.asarray(freqs_hz, dtype=np.float32)
    p = np.asarray(power_dbm, dtype=np.float32)
    if f.ndim != 1 or p.ndim != 1 or f.size == 0 or f.size != p.size:
        return []

    floor = np.median(p)
    snr = p - floor

    # локальные максимумы (не смотрим крайние точки)
    cand = np.where((p[1:-1] > p[:-2]) & (p[1:-1] > p[2:]))[0] + 1
    if cand.size == 0:
        return []

    # порог по SNR
    cand = cand[snr[cand] >= snr_threshold_db]
    if cand.size == 0:
        return []

    # грубая оценка ширины на -3 дБ
    good_idx: List[int] = []
    for i in cand:
        left = i
        right = i
        half = p[i] - 3.0
        while left > 0 and p[left] > half:
            left -= 1
        while right < p.size - 1 and p[right] > half:
            right += 1
        width_bins = right - left + 1
        if width_bins >= min_peak_bins:
            good_idx.append(i)
    if not good_idx:
        return []

    # подавление близких пиков — оставляем более мощный
    idxs = np.array(good_idx, dtype=int)
    if idxs.size > 1:
        idxs = idxs[np.argsort(p[idxs])[::-1]]  # по убыванию мощности
        selected: List[int] = []
        taken = np.zeros_like(p, dtype=bool)
        for i in idxs:
            if taken[i]:
                continue
            lo = max(0, i - min_peak_distance_bins)
            hi = min(p.size, i + min_peak_distance_bins + 1)
            taken[lo:hi] = True
            selected.append(int(i))
        idxs = np.array(sorted(selected), dtype=int)

    band = float(peak_band_hz)
    peaks: List[DetectedPeak] = []
    for i in idxs:
        peaks.append(
            DetectedPeak(
                freq_hz=float(f[i]),
                snr_db=float(snr[i]),
                power_dbm=float(p[i]),
                band_hz=band,
                idx=int(i),
            )
        )
    return peaks


# ---------- Класс PeakDetector (ожидается другими модулями) ----------

class PeakDetector:
    """
    Независимый детектор пиков.
    Совместим с импортом: from panorama.features.spectrum.model import PeakDetector
    """
    def __init__(
        self,
        snr_threshold_db: float = 10.0,
        min_peak_bins: int = 3,
        min_peak_distance_bins: int = 5,
        peak_band_hz: float = 5e6,
    ) -> None:
        self.snr_threshold_db = float(snr_threshold_db)
        self.min_peak_bins = int(min_peak_bins)
        self.min_peak_distance_bins = int(min_peak_distance_bins)
        self.peak_band_hz = float(peak_band_hz)

    def detect(self, freqs_hz: np.ndarray, power_dbm: np.ndarray) -> List[DetectedPeak]:
        """Вернуть список пиков по текущим настройкам детектора."""
        return _detect_peaks_core(
            freqs_hz=freqs_hz,
            power_dbm=power_dbm,
            snr_threshold_db=self.snr_threshold_db,
            min_peak_bins=self.min_peak_bins,
            min_peak_distance_bins=self.min_peak_distance_bins,
            peak_band_hz=self.peak_band_hz,
        )

    def detect_peaks(self, freqs_hz: np.ndarray, power_dbm: np.ndarray) -> List[DetectedPeak]:
        """Совместимый метод для обратной совместимости."""
        return self.detect(freqs_hz, power_dbm)


# ---------- Модель спектра (для UI + хранение водопада) ----------

class SpectrumModel(QObject if _HAS_QT else object):
    """
    Модель данных спектра + водопад.

    Совместимость с UI:
    - __init__(rows=...) — параметр rows поддерживается.
    - set_grid(freq_start_hz, freq_end_hz, bin_hz) — задаёт сетку частот.
    - last_row — последняя строка водопада (используется курсором).
    - update_full_sweep(freqs_hz, power_dbm), push_waterfall_row(row), reset().
    - detect_peaks() — использует те же параметры, что и PeakDetector.
    """

    spectrum_updated = pyqtSignal() if _HAS_QT else None
    waterfall_updated = pyqtSignal() if _HAS_QT else None
    status_changed = pyqtSignal(str) if _HAS_QT else None

    def __init__(self, rows: int = 200) -> None:
        if _HAS_QT:
            super().__init__()  # type: ignore
        self._rows_limit: int = int(rows)
        self.freqs_hz: np.ndarray = np.array([], dtype=np.float32)
        self.power_dbm: np.ndarray = np.array([], dtype=np.float32)

        self.waterfall: deque[np.ndarray] = deque(maxlen=self._rows_limit)
        self._last_row: Optional[np.ndarray] = None

        # Загружаем параметры детектора из настроек
        detector_settings = load_detector_settings()
        self.snr_threshold_db: float = float(detector_settings.get("snr_threshold_db", 10.0))
        self.min_peak_bins: int = int(detector_settings.get("min_peak_bins", 3))
        self.min_peak_distance_bins: int = int(detector_settings.get("min_peak_distance_bins", 5))
        self.peak_band_hz: float = float(detector_settings.get("peak_band_hz", 5e6))

    # ----- Совместимость с view.py -----

    def set_grid(self, freq_start_hz: float, freq_end_hz: float, bin_hz: float) -> None:
        """
        Создаёт сетку частот под указанный диапазон/бин и очищает модель.
        Вызывается из SpectrumView в момент старта скана.
        """
        try:
            f0 = float(freq_start_hz)
            f1 = float(freq_end_hz)
            bw = float(bin_hz)
            if bw <= 0 or f1 <= f0:
                raise ValueError
        except Exception:
            self._emit_status("Некорректные параметры сетки")
            return

        n_bins = int(np.floor((f1 - f0) / bw)) + 1
        self.freqs_hz = (f0 + np.arange(n_bins, dtype=np.float32) * bw).astype(np.float32)

        # Сбрасываем текущие данные
        self.power_dbm = np.full_like(self.freqs_hz, -120.0, dtype=np.float32)
        self.waterfall.clear()
        self._last_row = None

        self._emit_spectrum()
        self._emit_waterfall()
        self._emit_status(f"Инициализирована сетка: {f0/1e6:.1f}-{f1/1e6:.1f} МГц, bin={bw/1e3:.0f} кГц")

    @property
    def last_row(self) -> Optional[np.ndarray]:
        """Последняя строка водопада (используется подсказкой курсора в UI)."""
        return self._last_row

    # ----- Общие методы -----

    @property
    def rows(self) -> int:
        return self._rows_limit

    def set_rows(self, rows: int) -> None:
        self._rows_limit = int(rows)
        old = list(self.waterfall)
        self.waterfall = deque(old, maxlen=self._rows_limit)
        self._emit_waterfall()

    def reset(self) -> None:
        self.freqs_hz = np.array([], dtype=np.float32)
        self.power_dbm = np.array([], dtype=np.float32)
        self.waterfall.clear()
        self._last_row = None
        self._emit_spectrum()
        self._emit_waterfall()
        self._emit_status("Сброс модели")

    def configure_detector(
        self,
        snr_threshold_db: Optional[float] = None,
        min_peak_bins: Optional[int] = None,
        min_peak_distance_bins: Optional[int] = None,
        peak_band_hz: Optional[float] = None,
    ) -> None:
        if snr_threshold_db is not None:
            self.snr_threshold_db = float(snr_threshold_db)
        if min_peak_bins is not None:
            self.min_peak_bins = int(min_peak_bins)
        if min_peak_distance_bins is not None:
            self.min_peak_distance_bins = int(min_peak_distance_bins)
        if peak_band_hz is not None:
            self.peak_band_hz = float(peak_band_hz)

    def reload_detector_settings(self) -> None:
        """Перезагружает настройки детектора из файла."""
        detector_settings = load_detector_settings()
        self.snr_threshold_db = float(detector_settings.get("snr_threshold_db", 10.0))
        self.min_peak_bins = int(detector_settings.get("min_peak_bins", 3))
        self.min_peak_distance_bins = int(detector_settings.get("min_peak_distance_bins", 5))
        self.peak_band_hz = float(detector_settings.get("peak_band_hz", 5e6))
        self._emit_status("Настройки детектора перезагружены")

    def update_full_sweep(self, freqs_hz: np.ndarray, power_dbm: np.ndarray) -> None:
        if freqs_hz is None or power_dbm is None:
            return
        freqs_hz = np.asarray(freqs_hz, dtype=np.float32)
        power_dbm = np.asarray(power_dbm, dtype=np.float32)
        if freqs_hz.ndim != 1 or power_dbm.ndim != 1 or len(freqs_hz) != len(power_dbm):
            self._emit_status("Некорректные размеры данных панорамы")
            return
            
        if self.freqs_hz is None:
            # инициализируем глобальную сетку частот один раз
            self.freqs_hz = freqs_hz
            self.power_dbm = np.zeros_like(power_dbm)
            self.waterfall = []

        # обновляем только мощность
        self.power_dbm = power_dbm
        # добавляем строку в водопад
        self.waterfall.append(power_dbm.copy())
        if len(self.waterfall) > self._rows_limit:
            self.waterfall.pop(0)
            
        # поддержим last_row для курсора — берём текущий спектр
        self._last_row = power_dbm.copy()
        
        self._emit_spectrum()
        self._emit_waterfall()

    def push_waterfall_row(self, row_dbm: np.ndarray) -> None:
        if self.freqs_hz.size == 0:
            return
        row = np.asarray(row_dbm, dtype=np.float32)
        if row.ndim != 1 or row.size != self.freqs_hz.size:
            return
        self._last_row = row
        self.waterfall.append(row)
        self._emit_waterfall()

    def detect_peaks(self) -> List[DetectedPeak]:
        return _detect_peaks_core(
            freqs_hz=self.freqs_hz,
            power_dbm=self.power_dbm,
            snr_threshold_db=self.snr_threshold_db,
            min_peak_bins=self.min_peak_bins,
            min_peak_distance_bins=self.min_peak_distance_bins,
            peak_band_hz=self.peak_band_hz,
        )

    # --- сигналы/статусы ---
    def _emit_spectrum(self) -> None:
        if _HAS_QT and self.spectrum_updated is not None:
            try:
                self.spectrum_updated.emit()  # type: ignore
            except Exception:
                pass

    def _emit_waterfall(self) -> None:
        if _HAS_QT and self.waterfall_updated is not None:
            try:
                self.waterfall_updated.emit()  # type: ignore
            except Exception:
                pass

    def _emit_status(self, text: str) -> None:
        if _HAS_QT and self.status_changed is not None:
            try:
                self.status_changed.emit(text)  # type: ignore
            except Exception:
                pass


# ---------- Утилита для слейвов (watchlist) ----------

def peak_window_for_slaves(
    peak: DetectedPeak,
    clamp: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    Диапазон [f0, f1] = [freq - band_hz, freq + band_hz], с опц. ограничением clamp=(fmin,fmax).
    """
    f0 = peak.freq_hz - peak.band_hz
    f1 = peak.freq_hz + peak.band_hz
    if clamp is not None:
        fmin, fmax = clamp
        f0 = max(f0, fmin)
        f1 = min(f1, fmax)
    return (f0, f1)
