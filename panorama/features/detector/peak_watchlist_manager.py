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
        self.baseline_offset_db = 15.0       # Увеличен порог над baseline для меньших ложных срабатываний
        self.threshold_dbm = -65.0           # Фиксированный порог (повышен)
        self.min_snr_db = 8.0                # Увеличен минимальный SNR для фильтрации шума
        self.min_peak_width_bins = 10        # Увеличена минимальная ширина в бинах
        self.min_peak_distance_bins = 20     # Увеличено минимальное расстояние между пиками
        self.merge_if_gap_hz = detector_settings.get("cluster", {}).get("merge_if_gap_hz", 2e6)  # Объединять кластеры при разрыве

        # Параметры watchlist
        self.watchlist_span_hz = 10e6        # Окно ±5 МГц вокруг пика по умолчанию
        self.max_watchlist_size = 10         # Максимум целей в watchlist
        self.peak_timeout_sec = 120.0        # Увеличенный таймаут, чтобы записи не исчезали быстро
        self.min_confirmation_sweeps = 3     # Требуем 3 подтверждения для добавления в watchlist
        self.center_mode: str = 'fmax'       # 'fmax' | 'centroid'

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
        """
        Обрабатывает спектр и ищет пики видеосигналов.

        Args:
            freqs_hz: Частоты в Гц
            power_dbm: Мощности в дБм
            user_span_hz: Пользовательская ширина окна для watchlist

        Returns:
            Список обнаруженных пиков
        """
        if freqs_hz is None or power_dbm is None or freqs_hz.size < 10:
            return []

        # Сохраняем для анализа
        self.last_spectrum = (freqs_hz.copy(), power_dbm.copy())

        # Отладка
        try:
            print(f"[DEBUG] Processing spectrum: {freqs_hz.size} points, power range: {power_dbm.min():.1f} to {power_dbm.max():.1f} dBm")
        except Exception:
            pass

        # Вычисляем baseline
        baseline = self._compute_baseline(power_dbm)
        try:
            print(f"[DEBUG] Baseline: {baseline:.1f} dBm")
        except Exception:
            pass

        # Определяем порог
        if self.threshold_mode == "adaptive":
            threshold = baseline + self.baseline_offset_db
        else:
            threshold = self.threshold_dbm
        try:
            print(f"[DEBUG] Threshold: {threshold:.1f} dBm (mode={self.threshold_mode})")
        except Exception:
            pass

        # Находим пики
        peaks = self._find_video_peaks(freqs_hz, power_dbm, threshold, baseline)
        try:
            print(f"[DEBUG] Found {len(peaks)} peaks")
        except Exception:
            pass

        # Обновляем состояние и watchlist
        new_peaks = []
        current_time = time.time()

        for peak in peaks:
            # Проверяем, не дубликат ли это (в пределах 1 МГц)
            is_duplicate = False
            for existing_freq in self._previous_peaks:
                if abs(peak.center_freq_hz - existing_freq) < 1e6:
                    is_duplicate = True
                    break

            if is_duplicate:
                # Обновляем существующий пик и при достижении подтверждения добавляем в watchlist
                for peak_id, existing_peak in self.detected_peaks.items():
                    if abs(existing_peak.center_freq_hz - peak.center_freq_hz) < 1e6:
                        existing_peak.last_seen = current_time
                        existing_peak.consecutive_detections += 1
                        existing_peak.peak_power_dbm = max(existing_peak.peak_power_dbm,
                                                          peak.peak_power_dbm)
                        # Обновляем новые метрики кластера
                        if peak.centroid_freq_hz > 0.0:
                            existing_peak.centroid_freq_hz = peak.centroid_freq_hz
                        if peak.bandwidth_3db_hz > 0.0:
                            existing_peak.bandwidth_3db_hz = peak.bandwidth_3db_hz
                        if peak.cluster_start_hz > 0.0:
                            existing_peak.cluster_start_hz = peak.cluster_start_hz
                        if peak.cluster_end_hz > 0.0:
                            existing_peak.cluster_end_hz = peak.cluster_end_hz
                        try:
                            print(f"[DEBUG] Peak {peak_id} detections: {existing_peak.consecutive_detections}")
                        except Exception:
                            pass
                        # Если достигнуто требуемое число подтверждений — добавляем в watchlist
                        if (existing_peak.consecutive_detections >= max(1, self.min_confirmation_sweeps)
                                and peak_id not in self.watchlist):
                            proposed_span = float(user_span_hz or self.watchlist_span_hz)
                            if self._should_add_entry(existing_peak.center_freq_hz, proposed_span):
                                if not any(abs(existing_peak.center_freq_hz - c) < 1e6 for c in self._ever_added_centers):
                                    self._add_to_watchlist(existing_peak, proposed_span)
                                    self._ever_added_centers.add(existing_peak.center_freq_hz)
                        break
            else:
                # Новый пик
                self.detected_peaks[peak.id] = peak
                new_peaks.append(peak)
                self._previous_peaks.add(peak.center_freq_hz)

                # Добавляем в watchlist если подтвержден и ранее не добавляли такой центр
                if peak.consecutive_detections >= max(1, self.min_confirmation_sweeps):
                    proposed_span = float(user_span_hz or self.watchlist_span_hz)
                    if self._should_add_entry(peak.center_freq_hz, proposed_span):
                        if not any(abs(peak.center_freq_hz - c) < 1e6 for c in self._ever_added_centers):
                            self._add_to_watchlist(peak, proposed_span)
                            self._ever_added_centers.add(peak.center_freq_hz)

        # Эмитим сигналы для новых пиков
        for peak in new_peaks:
            self.peak_detected.emit(peak)

        return peaks

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
        Кластеризация: contiguous бины над порогом + объединение при разрыве < merge_if_gap_hz
        Метрики: ширина по −3 дБ, центроид (взвешенный по мощности), пик (F_max)
        """
        peaks = []

        # Маска точек выше порога
        above_threshold = power_dbm > threshold
        if not np.any(above_threshold):
            return peaks

        # 1. Находим связные области
        regions = self._find_connected_regions(above_threshold)

        # 2. Объединяем близко расположенные регионы (динамический порог = 0.25×span)
        current_span_hz = float(self.watchlist_span_hz)
        merged_regions = self._merge_nearby_regions(regions, freqs_hz, merge_gap_hz=max(0.0, 0.25 * current_span_hz))

        for start_idx, end_idx in merged_regions:
            region_freqs = freqs_hz[start_idx:end_idx+1]
            region_power = power_dbm[start_idx:end_idx+1]

            if len(region_freqs) < self.min_peak_width_bins:
                continue

            # 3. Рассчитываем метрики кластера
            peak_freq, peak_power, centroid_freq, bandwidth_3db, cluster_start, cluster_end = \
                self._calculate_cluster_metrics(freqs_hz, power_dbm, start_idx, end_idx)

            # Полная ширина кластера
            cluster_bandwidth_hz = cluster_end - cluster_start
            cluster_bandwidth_mhz = cluster_bandwidth_hz / 1e6

            # Проверяем, подходит ли под видеосигнал
            # Используем единую минимальную ширину для всех диапазонов
            min_width_mhz = self.video_bandwidth_min_mhz

            if cluster_bandwidth_mhz < min_width_mhz:
                continue  # Слишком узкий
            if cluster_bandwidth_mhz > self.video_bandwidth_max_mhz:
                continue  # Слишком широкий

            # Вычисляем SNR
            snr = peak_power - baseline
            if snr < self.min_snr_db:
                continue
                
            # Дополнительная проверка на стабильность: пик должен быть значительно выше соседей
            region_median = np.median(region_power)
            if peak_power - region_median < 5.0:  # Пик должен быть минимум на 5 дБ выше медианы региона
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

    def _should_add_entry(self, center_hz: float, span_hz: float) -> bool:
        """Проверяет, что новая запись не вложена/не дубликат существующей.
        Условия запрета:
        - перекрытие окон ≥ 60% меньшего окна
        - |Δf| < 0.5 × span существующей записи
        - запись рядом моложе hysteresis
        """
        if not self.watchlist:
            return True
        now = time.time()
        new_start = center_hz - span_hz / 2.0
        new_end = center_hz + span_hz / 2.0
        for entry in self.watchlist.values():
            exist_start = entry.freq_start_hz
            exist_end = entry.freq_stop_hz
            # Перекрытие
            inter = max(0.0, min(new_end, exist_end) - max(new_start, exist_start))
            if inter > 0:
                ref_span = min(span_hz, entry.span_hz)
                if ref_span > 0 and (inter / ref_span) >= 0.60:
                    return False
            # Близость центров
            if abs(center_hz - entry.center_freq_hz) < 0.5 * entry.span_hz:
                return False
            # Хистерезис (время)
            if (now - entry.created_at) < self._add_hysteresis_sec and abs(center_hz - entry.center_freq_hz) < entry.span_hz:
                return False
        return True

    def _calculate_cluster_metrics(self, freqs_hz: np.ndarray, power_dbm: np.ndarray,
                                 start_idx: int, end_idx: int) -> Tuple[float, float, float, float, float, float]:
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

            # 3. Ширина по уровню -3 дБ от пика
            threshold_3db = peak_power - 3.0
            above_3db = region_power >= threshold_3db

            if np.any(above_3db):
                indices_above_3db = np.where(above_3db)[0]
                start_3db_idx = indices_above_3db[0]
                end_3db_idx = indices_above_3db[-1]
                bandwidth_3db = float(region_freqs[end_3db_idx] - region_freqs[start_3db_idx])
            else:
                # Если нет точек выше -3 дБ, используем весь регион
                bandwidth_3db = float(region_freqs[-1] - region_freqs[0])

            # 4. Границы кластера
            cluster_start = float(region_freqs[0])
            cluster_end = float(region_freqs[-1])

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
            last_update=time.time()
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
