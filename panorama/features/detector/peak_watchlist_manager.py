# panorama/features/detector/peak_watchlist_manager.py
"""
Менеджер детектора пиков для видеосигналов дронов и управления watchlist.
Обнаруживает широкополосные сигналы (типично 5-20 МГц для видео).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from panorama.features.settings.storage import (load_detector_settings,
                                                save_detector_settings)


@dataclass
class VideoSignalPeak:
    """Обнаруженный пик видеосигнала."""
    center_freq_hz: float  # Центральная частота пика (F_max - частота максимума)
    peak_power_dbm: float  # Пиковая мощность
    bandwidth_hz: float    # Ширина сигнала на уровне -3dB
    snr_db: float         # Отношение сигнал/шум
    timestamp: float      # Время обнаружения
    consecutive_detections: int = 1  # Количество последовательных обнаружений
    last_seen: float = 0.0
    id: str = ""
    # Новые метрики кластера
    centroid_freq_hz: float = 0.0  # Центроид (взвешенный по мощности)
    bandwidth_3db_hz: float = 0.0  # Ширина по уровню -3dB от пика
    cluster_start_hz: float = 0.0  # Начало кластера
    cluster_end_hz: float = 0.0    # Конец кластера

    def __post_init__(self):
        if not self.id:
            self.id = f"peak_{int(self.center_freq_hz/1e6)}MHz_{int(time.time()*1000)}"
        if self.last_seen == 0.0:
            self.last_seen = self.timestamp


@dataclass
class WatchlistEntry:
    """Запись в watchlist для slave SDR."""
    peak_id: str
    center_freq_hz: float
    span_hz: float  # Полная ширина окна для измерения RSSI
    freq_start_hz: float
    freq_stop_hz: float
    created_at: float
    last_update: float
    rssi_measurements: Dict[str, float] = field(default_factory=dict)  # slave_id -> RSSI_RMS
    is_active: bool = True
    # Полные границы кластера видеосигнала (для отображения и логики пропуска диапазонов на мастере)
    cluster_start_hz: float = 0.0
    cluster_end_hz: float = 0.0

    def __post_init__(self):
        # Вычисляем границы окна
        half_span = self.span_hz / 2.0
        self.freq_start_hz = self.center_freq_hz - half_span
        self.freq_stop_hz = self.center_freq_hz + half_span


class PeakWatchlistManager(QObject):
    """
    Менеджер детектора пиков и watchlist для Master->Slave координации.
    """

    # Сигналы
    peak_detected = pyqtSignal(object)  # VideoSignalPeak
    watchlist_updated = pyqtSignal(list)  # List[WatchlistEntry]
    watchlist_task_ready = pyqtSignal(dict)  # Задача для slaves

    def __init__(self, parent=None):
        super().__init__(parent)

        # Загружаем настройки детектора
        detector_settings = load_detector_settings()

        # Параметры детектора для видеосигналов (настроены для стабильной работы)
        self.video_bandwidth_min_mhz = 2.0   # Минимум 2 МГц для видеосигналов
        self.video_bandwidth_max_mhz = 30.0  # Разрешаем широкие сигналы
        self.threshold_mode = "adaptive"      # "adaptive" или "fixed"
        self.baseline_offset_db = 15.0       # Порог над baseline для устойчивости
        self.threshold_dbm = -65.0           # Фиксированный порог (повышен)
        self.min_snr_db = 8.0                # Минимальный SNR для фильтрации шума
        self.min_peak_width_bins = 10        # Увеличена минимальная ширина в бинах
        self.min_peak_distance_bins = 20     # Увеличено минимальное расстояние между пиками
        self.merge_if_gap_hz = detector_settings.get("cluster", {}).get("merge_if_gap_hz", 3e6)  # По умолчанию чуть шире

        # Параметры watchlist
        self.watchlist_span_hz = 10e6        # Окно ±5 МГц вокруг пика по умолчанию
        self.max_watchlist_size = 10         # Максимум целей в watchlist
        self.peak_timeout_sec = 120.0        # Увеличенный таймаут, чтобы записи не исчезали быстро
        self.min_confirmation_sweeps = 3     # Требуем 3 подтверждения для добавления в watchlist
        self.center_mode: str = 'fmax'       # 'fmax' | 'centroid'
        # Гистерезис для устойчивого выделения полос видеосигнала (менее чувствительный)
        self.hysteresis_high_db = 10.0  # seed-порог SNR для начала региона
        self.hysteresis_low_db = 5.0    # порог SNR для расширения региона
        self.bridge_gap_bins_default = 4  # допускаемый короткий разрыв в бинах
        # Границы по возврату к baseline
        self.boundary_end_delta_db = 1.2   # насколько выше baseline, чтобы считать конец
        self.edge_min_run_bins = 3         # минимум подряд для фиксации конца
        # Анти‑выбросные фильтры
        self.prefilter_median_bins = 5     # скользящая медиана по частоте перед SNR
        self.min_region_occupancy = 0.40   # доля бинов в [a,b], где snr>=low, чтобы считать регион валидным
        self.min_region_area = 3.0         # интеграл SNR (дБ*бин) минимальный для региона

        # Состояние
        self.detected_peaks: Dict[str, VideoSignalPeak] = {}
        self.watchlist: Dict[str, WatchlistEntry] = {}
        self.baseline_history = deque(maxlen=50)  # История для адаптивного порога
        self.last_spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._previous_peaks: Set[float] = set()  # Для предотвращения дублирования
        self._ever_added_centers: Set[float] = set()  # Центры, уже добавленные в watchlist за сессию
        self.watchlist_entry_retention_sec = 600.0     # Храним записи watchlist не менее 10 минут

        # Таймер очистки
        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.timeout.connect(self._cleanup_old_entries)
        self.cleanup_timer.start(5000)  # Каждые 5 секунд

        # Анти-дубликат: минимальная пауза между соседними добавлениями (сек)
        self._add_hysteresis_sec = 4.0

    def process_spectrum(self, freqs_hz: np.ndarray, power_dbm: np.ndarray,
                         user_span_hz: Optional[float] = None) -> List[VideoSignalPeak]:
        # Python‑детектор удалён: используем C‑детектор через координатор.
        return []

    # --- NEW: ingestion from C-side detector ---
    def ingest_detected_bands(self, bands: List[tuple], user_span_hz: Optional[float] = None) -> None:
        """Принимает список диапазонов от C-детектора и обновляет watchlist без дублей.
        bands: List of tuples (start_hz, stop_hz, center_hz, peak_dbm)
        Логика сопоставления идентична python-детектору:
        - если новый диапазон перекрывается с существующим окном/кластером или центр рядом — обновляем запись
        - иначе, добавляем новую запись только если _should_add_entry(...) возвращает True
        """
        if not bands:
            return
        now = time.time()
        span_default = float(user_span_hz or self.watchlist_span_hz)
        updated = False

        # Упорядочим по убыванию предполагаемой мощности (если есть)
        try:
            bands_sorted = sorted(bands, key=lambda b: float(b[3]) if len(b) > 3 else 0.0, reverse=True)
        except Exception:
            bands_sorted = list(bands)

        # 1) Сначала слияние пересекающихся/вложенных диапазонов от мастера: «нет вложенных диапазонов»
        #    Отсортируем по началу и объединим все пересечения/малые разрывы (<= merge_if_gap_hz).
        try:
            bands_sorted = sorted(bands_sorted, key=lambda b: float(b[0]))
        except Exception:
            pass
        merged_master: List[Tuple[float, float, float, float]] = []
        for b in bands_sorted:
            try:
                f0, f1, fc = float(b[0]), float(b[1]), float(b[2])
                pk = float(b[3]) if len(b) > 3 else 0.0
            except Exception:
                continue
            if f1 <= f0:
                continue
            if not merged_master:
                merged_master.append((f0, f1, fc, pk))
                continue
            mf0, mf1, mfc, mpk = merged_master[-1]
            # Если пересекаются или разрыв мал (<= merge_if_gap_hz) — объединяем
            if f0 <= mf1 + float(self.merge_if_gap_hz):
                nf0 = min(mf0, f0)
                nf1 = max(mf1, f1)
                # Центр и пик — берём по максимуму pk
                if pk >= mpk:
                    nfc, npk = fc, pk
                else:
                    nfc, npk = mfc, mpk
                merged_master[-1] = (nf0, nf1, nfc, npk)
            else:
                merged_master.append((f0, f1, fc, pk))

        matched_ids: Set[str] = set()
        for band in merged_master:
            try:
                f0, f1, fc = float(band[0]), float(band[1]), float(band[2])
            except Exception:
                continue
            if not np.isfinite(f0) or not np.isfinite(f1) or not np.isfinite(fc):
                continue
            if f1 <= f0:
                continue

            # Пытаемся найти существующую запись, соответствующую этому диапазону
            match_id = self._find_existing_entry_for_band(center_hz=fc,
                                                          span_hz=span_default,
                                                          cluster_start_hz=f0,
                                                          cluster_end_hz=f1)
            if match_id:
                # Обновляем существующую запись (центр/границы/время)
                entry = self.watchlist.get(match_id)
                if entry:
                    # Центр мастера = середина полного диапазона, НЕ Fmax
                    double_center = 0.5 * (float(f0) + float(f1))
                    entry.center_freq_hz = float(double_center)
                    entry.cluster_start_hz = float(f0)
                    entry.cluster_end_hz = float(f1)
                    entry.freq_start_hz = entry.center_freq_hz - entry.span_hz / 2.0
                    entry.freq_stop_hz = entry.center_freq_hz + entry.span_hz / 2.0
                    entry.last_update = now
                    matched_ids.add(match_id)
                    updated = True
                continue

            # Если не нашли совпадение — проверим, что добавление не создаст дубль
            if not self._should_add_entry(center_hz=fc, span_hz=span_default,
                                          cluster_start_hz=f0, cluster_end_hz=f1):
                continue

            # Учитываем лимит размера списка — удаляем самый старый при необходимости
            if len(self.watchlist) >= self.max_watchlist_size:
                try:
                    oldest_id = min(self.watchlist.keys(), key=lambda k: self.watchlist[k].created_at)
                    del self.watchlist[oldest_id]
                except Exception:
                    pass

            peak_id = f"peak_{int(((f0+f1)/2)/1e6)}MHz_{int(now*1000)}"
            entry = WatchlistEntry(
                peak_id=peak_id,
                # Центр мастера = середина полного диапазона, НЕ Fmax
                center_freq_hz=float(0.5*(f0+f1)),
                span_hz=span_default,
                freq_start_hz=float(0.5*(f0+f1)) - span_default/2.0,
                freq_stop_hz=float(0.5*(f0+f1)) + span_default/2.0,
                created_at=now,
                last_update=now,
                cluster_start_hz=float(f0),
                cluster_end_hz=float(f1),
            )
            self.watchlist[peak_id] = entry
            matched_ids.add(peak_id)
            updated = True

        # Удаляем записи, которые не пришли в текущем кадре (синхронизация с мастером)
        if matched_ids:
            to_drop = [pid for pid in self.watchlist.keys() if pid not in matched_ids]
            for pid in to_drop:
                try:
                    del self.watchlist[pid]
                except Exception:
                    pass
            if to_drop:
                updated = True

        if updated:
            self.watchlist_updated.emit(list(self.watchlist.values()))

    def _find_existing_entry_for_band(self, center_hz: float, span_hz: float,
                                       cluster_start_hz: float, cluster_end_hz: float) -> Optional[str]:
        """Находит существующую запись watchlist, соответствующую диапазону из C-детектора.
        Критерии совпадения (любой):
          - центр попадает в окно существующей записи
          - пересечение окон ≥ 60% меньшего окна
          - центр существующей записи попадает внутрь нового кластера
          - пересечение [cluster] с окном записи ≥ 30%
          - |Δf центров| < 0.6×span существующей записи
        Возвращает peak_id или None.
        """
        if not self.watchlist:
            return None
        new_start = center_hz - span_hz / 2.0
        new_end = center_hz + span_hz / 2.0
        for pid, entry in self.watchlist.items():
            exist_start = entry.freq_start_hz
            exist_end = entry.freq_stop_hz
            # 1) центр внутри
            if exist_start <= center_hz <= exist_end:
                return pid
            # 2) Перекрытие окон ≥ 60%
            inter = max(0.0, min(new_end, exist_end) - max(new_start, exist_start))
            ref_span = min(span_hz, entry.span_hz)
            if ref_span > 0 and (inter / ref_span) >= 0.60:
                return pid
            # 3) Центр записи в новом кластере или перекрытие ≥ 30%
            cluster_span = max(1.0, cluster_end_hz - cluster_start_hz)
            if cluster_start_hz <= entry.center_freq_hz <= cluster_end_hz:
                return pid
            cluster_inter = max(0.0, min(cluster_end_hz, exist_end) - max(cluster_start_hz, exist_start))
            if (cluster_inter / min(cluster_span, (exist_end - exist_start))) >= 0.30:
                return pid
            # 4) Близость центров
            if abs(center_hz - entry.center_freq_hz) < 0.6 * entry.span_hz:
                return pid
        return None

    def _compute_baseline(self, power_dbm: np.ndarray) -> float:
        """
        Вычисляет baseline (шумовой пол) для адаптивного порога.
        Использует нижний квартиль (25-й перцентиль) для устойчивости к сильным пикам.
        """
        # Текущее значение baseline по 25-му перцентилю
        baseline_value = float(np.percentile(power_dbm, 25))
        # Добавляем в историю
        self.baseline_history.append(baseline_value)

        # Используем медиану по истории для сглаживания
        if len(self.baseline_history) > 0:
            return float(np.median(list(self.baseline_history)))
        else:
            return baseline_value

    def _find_video_peaks(self, freqs_hz: np.ndarray, power_dbm: np.ndarray,
                         threshold: float, baseline: float) -> List[VideoSignalPeak]:
        """
        Ищет широкополосные пики характерные для видеосигналов с улучшенной кластеризацией.
        Подход:
        1) Вычисляем SNR по отношению к baseline и сглаживаем по частоте
        2) Гистерезис-маска: seed по high-порогу, расширение по low-порогу, мостим короткие дыры
        3) Для региона считаем метрики и уточняем края по возврату к baseline (±delta)
        """
        peaks: List[VideoSignalPeak] = []

        # Оценка ширины бина
        try:
            bin_hz = float(np.median(np.diff(freqs_hz)))
            if not np.isfinite(bin_hz) or bin_hz <= 0:
                bin_hz = float(freqs_hz[1] - freqs_hz[0])
        except Exception:
            bin_hz = max(1e3, float(freqs_hz[1] - freqs_hz[0])) if freqs_hz.size > 1 else 100e3

        # Сглаженный SNR: предварительно медианный фильтр, затем усреднение
        snr = (power_dbm - baseline).astype(np.float32, copy=False)
        try:
            if self.prefilter_median_bins and self.prefilter_median_bins >= 3:
                w = int(self.prefilter_median_bins | 1)
                pad = w // 2
                padded = np.pad(snr, (pad, pad), mode='edge')
                strd = np.lib.stride_tricks.as_strided(
                    padded,
                    shape=(snr.size, w),
                    strides=(padded.strides[0], padded.strides[0])
                )
                snr = np.median(strd, axis=1).astype(np.float32, copy=False)
        except Exception:
            pass
        win_bins = int(max(5, min(401, round(200e3 / max(1.0, bin_hz)))))
        if (win_bins % 2) == 0:
            win_bins += 1
        try:
            kernel = np.ones(win_bins, dtype=np.float32) / float(win_bins)
            snr_s = np.convolve(snr, kernel, mode='same')
        except Exception:
            snr_s = snr

        hi = float(self.hysteresis_high_db)
        lo = float(self.hysteresis_low_db)
        lo = min(lo, hi)
        mask_low = snr_s >= lo
        mask_high = snr_s >= hi

        # Мостим короткие дыры
        bridge = int(max(0, self.bridge_gap_bins_default))
        if bridge > 0 and mask_low.any():
            holes = []
            start = None
            for i, v in enumerate(mask_low):
                if not v and start is None:
                    start = i
                elif v and start is not None:
                    holes.append((start, i - 1))
                    start = None
            if start is not None:
                holes.append((start, len(mask_low) - 1))
            for a, b in holes:
                if (b - a + 1) <= bridge:
                    mask_low[a:b+1] = True

        # Регионы по низкому порогу, но требуем seed по высокому
        regions_low = self._find_connected_regions(mask_low)
        merged_regions: List[Tuple[int, int]] = []
        for a, b in regions_low:
            if not np.any(mask_high[a:b+1]):
                continue
            # Фильтр занятости: какой процент бинов внутри региона >= low
            occ = float(np.mean(mask_low[a:b+1])) if (b > a) else 0.0
            if occ < float(self.min_region_occupancy):
                continue
            # Фильтр площади SNR (в дБ*бин), чтобы отсечь одиночные узкие иглы
            area = float(np.sum(np.maximum(0.0, snr_s[a:b+1])))
            if area < float(self.min_region_area):
                continue
            merged_regions.append((a, b))
        if not merged_regions:
            return peaks

        for start_idx, end_idx in merged_regions:
            region_freqs = freqs_hz[start_idx:end_idx+1]
            region_power = power_dbm[start_idx:end_idx+1]

            if len(region_freqs) < self.min_peak_width_bins:
                continue

            # 3. Рассчитываем метрики кластера
            peak_freq, peak_power, centroid_freq, bandwidth_3db, cluster_start, cluster_end = \
                self._calculate_cluster_metrics(freqs_hz, power_dbm, start_idx, end_idx, baseline)

            # Полная ширина кластера
            cluster_bandwidth_hz = cluster_end - cluster_start
            cluster_bandwidth_mhz = cluster_bandwidth_hz / 1e6

            # Проверяем, подходит ли под видеосигнал
            # Используем минимальную ширину, но допускаем узкие кластеры, если SNR сильно выше порога
            min_width_mhz = float(self.video_bandwidth_min_mhz)
            if cluster_bandwidth_mhz < min_width_mhz:
                # допустим узкую полосу, если пик значительно (>= 10 дБ) выше baseline
                pass  # проверим после вычисления SNR
            if cluster_bandwidth_mhz > self.video_bandwidth_max_mhz:
                continue  # Слишком широкий

            # Вычисляем SNR
            snr = peak_power - baseline
            if cluster_bandwidth_mhz < min_width_mhz and snr >= max(self.min_snr_db + 4.0, 10.0):
                pass  # разрешаем узкий кластер при очень большом SNR
            elif snr < self.min_snr_db:
                continue
                
            # Дополнительная проверка на стабильность: пик должен быть значительно выше соседей
            region_median = np.median(region_power)
            if peak_power - region_median < 4.0:  # Чуть мягче: 4 дБ над медианой региона
                continue

            # 4. Создаем объект пика с новыми метриками
            center_choice = peak_freq if self.center_mode == 'fmax' else centroid_freq
            peak = VideoSignalPeak(
                center_freq_hz=center_choice,                # Выбранный центр окна
                peak_power_dbm=peak_power,                   # Мощность в максимуме
                bandwidth_hz=cluster_bandwidth_hz,           # Полная ширина кластера (для совместимости)
                snr_db=snr,
                timestamp=time.time(),
                # Новые метрики кластера
                centroid_freq_hz=centroid_freq,              # Центроид (взвешенный по мощности)
                bandwidth_3db_hz=bandwidth_3db,              # Ширина по уровню -3 дБ
                cluster_start_hz=cluster_start,              # Начало кластера
                cluster_end_hz=cluster_end                   # Конец кластера
            )
            peaks.append(peak)

        return peaks

    def _find_connected_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Находит связные области в булевой маске.
        """
        regions = []
        in_region = False
        start = 0

        for i in range(len(mask)):
            if mask[i] and not in_region:
                start = i
                in_region = True
            elif not mask[i] and in_region:
                regions.append((start, i-1))
                in_region = False

        if in_region:
            regions.append((start, len(mask)-1))

        return regions

    def _merge_nearby_regions(self, regions: List[Tuple[int, int]], freqs_hz: np.ndarray, merge_gap_hz: Optional[float] = None) -> List[Tuple[int, int]]:
        """
        Объединяет близко расположенные регионы если разрыв меньше merge_if_gap_hz.
        """
        if len(regions) <= 1:
            return regions

        merged = []
        current_start, current_end = regions[0]

        allowed_gap = float(self.merge_if_gap_hz if merge_gap_hz is None else merge_gap_hz)
        for start, end in regions[1:]:
            # Проверяем разрыв между регионами
            gap_hz = freqs_hz[start] - freqs_hz[current_end]
            if gap_hz <= allowed_gap:
                # Объединяем регионы
                current_end = end
            else:
                # Добавляем текущий регион и начинаем новый
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        # Добавляем последний регион
        merged.append((current_start, current_end))
        return merged

    def _should_add_entry(self, center_hz: float, span_hz: float,
                          cluster_start_hz: float = None, cluster_end_hz: float = None) -> bool:
        """Проверяет, что новая запись не дублирует существующую.
        Логика:
        - если есть активная запись и центр нового окна попадает в диапазон существующей — не добавляем
        - если есть пересечение окон ≥ 60% меньшего окна — не добавляем
        - если границы кластера известны и пересекаются с существующей записью
          (центр существующей попадает внутрь кластера ИЛИ пересечение ≥ 30%) — не добавляем
        - запрет по близости центров: |Δf| < 0.6 × span существующей
        - гистерезис по времени
        """
        if not self.watchlist:
            return True
        now = time.time()
        new_start = center_hz - span_hz / 2.0
        new_end = center_hz + span_hz / 2.0
        for entry in self.watchlist.values():
            exist_start = entry.freq_start_hz
            exist_end = entry.freq_stop_hz
            # 1) Попадание центра
            if exist_start <= center_hz <= exist_end:
                return False
            # 2) Перекрытие окон
            inter = max(0.0, min(new_end, exist_end) - max(new_start, exist_start))
            if inter > 0:
                ref_span = min(span_hz, entry.span_hz)
                if ref_span > 0 and (inter / ref_span) >= 0.60:
                    return False
            # 3) Кластерная проверка (если доступны границы кластера)
            if cluster_start_hz is not None and cluster_end_hz is not None and cluster_end_hz > cluster_start_hz:
                cluster_inter = max(0.0, min(cluster_end_hz, exist_end) - max(cluster_start_hz, exist_start))
                cluster_span = max(1.0, cluster_end_hz - cluster_start_hz)
                if (exist_start <= (cluster_start_hz + cluster_end_hz) / 2.0 <= exist_end) or (cluster_inter / min(cluster_span, (exist_end - exist_start))) >= 0.30:
                    return False
            # 4) Близость центров (усилена до 0.6×span)
            if abs(center_hz - entry.center_freq_hz) < 0.6 * entry.span_hz:
                return False
            # 5) Хистерезис (время)
            if (now - entry.created_at) < self._add_hysteresis_sec and abs(center_hz - entry.center_freq_hz) < entry.span_hz:
                return False
        return True

    def _calculate_cluster_metrics(self, freqs_hz: np.ndarray, power_dbm: np.ndarray,
                                 start_idx: int, end_idx: int, baseline: float) -> Tuple[float, float, float, float, float, float]:
        """
        Рассчитывает метрики кластера:
        - peak_freq: частота максимума (F_max)
        - peak_power: мощность в максимуме
        - centroid_freq: центроид (взвешенный по мощности)
        - bandwidth_3db: ширина по уровню -3 дБ от пика
        - cluster_start: начальная частота кластера
        - cluster_end: конечная частота кластера
        """
        try:
            # Защитные проверки
            if start_idx < 0 or end_idx >= len(freqs_hz) or start_idx > end_idx:
                # Fallback значения
                center_freq = float(freqs_hz[min(max(start_idx, 0), len(freqs_hz)-1)])
                return center_freq, -80.0, center_freq, 1e6, center_freq, center_freq

            region_freqs = freqs_hz[start_idx:end_idx+1]
            region_power = power_dbm[start_idx:end_idx+1]

            if len(region_freqs) == 0:
                center_freq = float(freqs_hz[start_idx] if start_idx < len(freqs_hz) else freqs_hz[0])
                return center_freq, -80.0, center_freq, 1e6, center_freq, center_freq

            # 1. Частота и мощность максимума (F_max)
            peak_idx = np.argmax(region_power)
            peak_freq = float(region_freqs[peak_idx])
            peak_power = float(region_power[peak_idx])

            # 2. Центроид (взвешенный по мощности)
            # Переводим мощности в линейные единицы для взвешивания
            power_linear = np.power(10.0, region_power / 10.0)
            power_sum = np.sum(power_linear)
            if power_sum > 0:
                centroid_freq = float(np.sum(region_freqs * power_linear) / power_sum)
            else:
                centroid_freq = peak_freq

            # 3. Границы кластера и ширина по уровню -3 дБ
            threshold_3db = peak_power - 3.0
            above_3db = region_power >= threshold_3db
            if np.any(above_3db):
                indices_above_3db = np.where(above_3db)[0]
                start_3db_idx = int(indices_above_3db[0])
                end_3db_idx = int(indices_above_3db[-1])
                bandwidth_3db = float(region_freqs[end_3db_idx] - region_freqs[start_3db_idx])
            else:
                start_3db_idx = 0
                end_3db_idx = len(region_freqs) - 1
                bandwidth_3db = float(region_freqs[-1] - region_freqs[0])

            # Границы: когда сглаженная мощность возвращается к baseline (±delta) стабильно несколько бинов
            # Сглаживание небольшим окном по частоте, чтобы убрать мелкие колебания
            try:
                k = np.ones(5, dtype=np.float32) / 5.0
                smooth_power = np.convolve(region_power, k, mode='same')
            except Exception:
                smooth_power = region_power

            end_level = float(baseline + float(self.boundary_end_delta_db))
            min_run = int(max(1, self.edge_min_run_bins))

            # Левая граница
            left = int(peak_idx)
            run = 0
            for i in range(int(peak_idx), -1, -1):
                if smooth_power[i] <= end_level:
                    run += 1
                    if run >= min_run:
                        left = i
                        break
                else:
                    run = 0

            # Правая граница
            right = int(peak_idx)
            run = 0
            for i in range(int(peak_idx), len(smooth_power)):
                if smooth_power[i] <= end_level:
                    run += 1
                    if run >= min_run:
                        right = i
                        break
                else:
                    run = 0

            # Защита индексов
            left = max(0, min(left, len(region_freqs) - 1))
            right = max(left, min(right, len(region_freqs) - 1))

            cluster_start = float(region_freqs[left])
            cluster_end = float(region_freqs[right])

            return peak_freq, peak_power, centroid_freq, bandwidth_3db, cluster_start, cluster_end

        except Exception as e:
            # В случае ошибки возвращаем безопасные значения
            try:
                center_freq = float(freqs_hz[min(max(start_idx, 0), len(freqs_hz)-1)])
                print(f"[ERROR] _calculate_cluster_metrics failed: {e}, using fallback")
                return center_freq, -80.0, center_freq, 1e6, center_freq, center_freq
            except:
                return 5640e6, -80.0, 5640e6, 1e6, 5640e6, 5640e6

    def _add_to_watchlist(self, peak: VideoSignalPeak, span_hz: float):
        """
        Добавляет пик в watchlist для мониторинга slave устройствами.
        Использует F_max или centroid в зависимости от center_mode.
        """
        # Проверяем лимит watchlist
        if len(self.watchlist) >= self.max_watchlist_size:
            # Удаляем самый старый
            oldest_id = min(self.watchlist.keys(),
                          key=lambda k: self.watchlist[k].created_at)
            del self.watchlist[oldest_id]

        # Выбираем центр для RSSI окна в зависимости от режима
        if self.center_mode == 'centroid' and peak.centroid_freq_hz > 0.0:
            rssi_center_hz = peak.centroid_freq_hz
            print(f"[Watchlist] Using centroid center: {rssi_center_hz/1e6:.3f} MHz")
        else:
            rssi_center_hz = peak.center_freq_hz  # F_max
            print(f"[Watchlist] Using F_max center: {rssi_center_hz/1e6:.3f} MHz")

        # Создаем запись watchlist
        entry = WatchlistEntry(
            peak_id=peak.id,
            center_freq_hz=rssi_center_hz,  # Используем выбранный центр
            span_hz=span_hz,
            freq_start_hz=rssi_center_hz - span_hz/2,
            freq_stop_hz=rssi_center_hz + span_hz/2,
            created_at=time.time(),
            last_update=time.time(),
            cluster_start_hz=float(peak.cluster_start_hz or (rssi_center_hz - span_hz/2)),
            cluster_end_hz=float(peak.cluster_end_hz or (rssi_center_hz + span_hz/2))
        )

        self.watchlist[peak.id] = entry

        # Эмитим обновление watchlist
        self.watchlist_updated.emit(list(self.watchlist.values()))

        # Эмитим задачу для slaves
        task = {
            'peak_id': peak.id,
            'center_freq_hz': entry.center_freq_hz,
            'span_hz': entry.span_hz,
            'freq_start_hz': entry.freq_start_hz,
            'freq_stop_hz': entry.freq_stop_hz,
            'timestamp': entry.created_at
        }
        self.watchlist_task_ready.emit(task)

        # Безопасные логи с проверкой наличия новых метрик
        if peak.centroid_freq_hz > 0.0 and peak.bandwidth_3db_hz > 0.0:
            print(f"[Watchlist] Added: Peak={peak.center_freq_hz/1e6:.1f} MHz, "
                  f"Centroid={peak.centroid_freq_hz/1e6:.1f} MHz, "
                  f"BW-3dB={peak.bandwidth_3db_hz/1e6:.1f} MHz, "
                  f"Cluster=[{peak.cluster_start_hz/1e6:.1f}-{peak.cluster_end_hz/1e6:.1f}] MHz, "
                  f"span={span_hz/1e6:.1f} MHz")
        else:
            # Fallback для старых пиков без новых метрик
            print(f"[Watchlist] Added: {peak.center_freq_hz/1e6:.1f} MHz, "
                  f"span={span_hz/1e6:.1f} MHz, "
                  f"window=[{entry.freq_start_hz/1e6:.1f}-{entry.freq_stop_hz/1e6:.1f}] MHz")

    def update_rssi_measurement(self, peak_id: str, slave_id: str, rssi_dbm: float):
        """
        Обновляет измерение RSSI от slave для записи в watchlist.
        """
        if peak_id in self.watchlist:
            self.watchlist[peak_id].rssi_measurements[slave_id] = rssi_dbm
            self.watchlist[peak_id].last_update = time.time()
            # Консольный лог для дебага
            try:
                print(f"[Watchlist] RSSI update: {peak_id} {slave_id} = {rssi_dbm:.1f} dBm")
            except Exception:
                pass
            # Автоудаление, если сигнал сравнялся с шумом на всех 3+ SDR
            try:
                baseline_med = float(np.median(list(self.baseline_history))) if len(self.baseline_history) > 0 else -90.0
                threshold = baseline_med + max(3.0, self.min_snr_db)  # немного выше порога SNR
                entry = self.watchlist[peak_id]
                vals = list(entry.rssi_measurements.values())
                strong = [v for v in vals if v is not None and v > threshold]
                
                # Новая логика: удаляем только если есть данные минимум от 3 слейвов И все показывают шум
                if len(vals) >= 3 and not strong and (time.time() - entry.last_update) > 30.0:
                    try:
                        del self.watchlist[peak_id]
                        self.watchlist_updated.emit(list(self.watchlist.values()))
                        print(f"[Watchlist] Removed inactive peak {peak_id} - all {len(vals)} slaves show noise level (< {threshold:.1f} dBm)")
                    except Exception:
                        pass
            except Exception:
                pass

    def get_rssi_for_trilateration(self, peak_id: str) -> Optional[Dict[str, float]]:
        """
        Получает измерения RSSI для трилатерации.

        Returns:
            Словарь {slave_id: rssi_dbm} или None если недостаточно данных
        """
        if peak_id not in self.watchlist:
            return None

        measurements = self.watchlist[peak_id].rssi_measurements

        # Нужно минимум 3 измерения для трилатерации
        if len(measurements) < 3:
            return None

        return measurements.copy()

    def _cleanup_old_entries(self):
        """
        Очищает устаревшие записи из пиков и watchlist.
        """
        current_time = time.time()

        # Очищаем старые пики
        peaks_to_remove = []
        for peak_id, peak in self.detected_peaks.items():
            if current_time - peak.last_seen > self.peak_timeout_sec:
                peaks_to_remove.append(peak_id)
                self._previous_peaks.discard(peak.center_freq_hz)

        for peak_id in peaks_to_remove:
            peak = self.detected_peaks.get(peak_id)
            del self.detected_peaks[peak_id]
            # Не удаляем запись из watchlist, если она моложе retention порога
            if peak_id in self.watchlist:
                age = current_time - self.watchlist[peak_id].created_at
                if age > self.watchlist_entry_retention_sec:
                    del self.watchlist[peak_id]

        # Обновляем watchlist
        if peaks_to_remove:
            self.watchlist_updated.emit(list(self.watchlist.values()))

    def clear_watchlist(self):
        """Полная очистка watchlist."""
        self.watchlist.clear()
        self.detected_peaks.clear()
        self._previous_peaks.clear()
        self.watchlist_updated.emit([])

    def set_detection_parameters(self, threshold_offset_db: float = None,
                                min_snr_db: float = None,
                                min_peak_width_bins: int = None,
                                watchlist_span_hz: float = None,
                                merge_if_gap_hz: float = None):
        """Обновляет параметры детектора."""
        if threshold_offset_db is not None:
            self.baseline_offset_db = threshold_offset_db
        if min_snr_db is not None:
            self.min_snr_db = min_snr_db
        if min_peak_width_bins is not None:
            self.min_peak_width_bins = min_peak_width_bins
        if watchlist_span_hz is not None:
            self.watchlist_span_hz = watchlist_span_hz
        if merge_if_gap_hz is not None:
            self.merge_if_gap_hz = merge_if_gap_hz
            # Сохраняем в настройки
            detector_settings = load_detector_settings()
            if "cluster" not in detector_settings:
                detector_settings["cluster"] = {}
            detector_settings["cluster"]["merge_if_gap_hz"] = merge_if_gap_hz
            save_detector_settings(detector_settings)
