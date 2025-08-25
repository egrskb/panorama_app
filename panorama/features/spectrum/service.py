# panorama/features/spectrum/service.py
from __future__ import annotations
from typing import Optional, Any, Tuple
import numpy as np
import time
from panorama.features.settings.storage import get_coverage_threshold, get_interpolation_enabled


class SweepAssembler:
    """
    Собирает одну полную строку спектра из sweep-сегментов (четверти окна).
    """

    def __init__(self, coverage_threshold: float = None, wrap_guard_hz: float = 100e6):
        if coverage_threshold is None:
            coverage_threshold = get_coverage_threshold()
        self.threshold = float(coverage_threshold)
        self.wrap_guard_hz = float(wrap_guard_hz)
        self.interpolation_enabled = get_interpolation_enabled()

        # сетка
        self.f0: float = 0.0
        self.f1: float = 0.0
        self.bin_hz: float = 0.0
        self.nbins: int = 0

        # буферы
        self.row: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None

        # служебное
        self._last_low: Optional[float] = None
        self._last_high: Optional[float] = None
        self._segment_count: int = 0
        self._start_time: float = 0.0

        self._nan_fill_value = -120.0
        self._debug_log_interval = 200

    def configure(self, f0_hz: int, f1_hz: int, bin_hz: int, lut: Any = None) -> None:
        f0 = float(f0_hz); f1 = float(f1_hz); b = float(bin_hz)
        if f1 <= f0 or b <= 0.0:
            raise ValueError("Bad grid: f1<=f0 or bin<=0")

        nb = int(round((f1 - f0) / b))
        if nb <= 0:
            raise ValueError("nbins <= 0")

        self.f0, self.f1, self.bin_hz, self.nbins = f0, f1, b, nb
        self.row  = np.full(nb, np.nan, dtype=np.float32)
        self.mask = np.zeros(nb, dtype=bool)

        self._last_low = None
        self._last_high = None
        self._segment_count = 0
        self._start_time = time.time()

        print(f"[SweepAssembler] Configured: {f0/1e6:.1f}-{f1/1e6:.1f} MHz, {nb} bins, bin_width={b/1e3:.1f} kHz")

    def reset_pass(self) -> None:
        if self.row is not None:
            self.row.fill(np.nan)
        if self.mask is not None:
            self.mask[:] = False
        self._last_low = None
        self._last_high = None
        self._segment_count = 0
        self._start_time = time.time()

    def freq_grid(self) -> np.ndarray:
        if self.nbins <= 0 or self.bin_hz <= 0:
            return np.array([])
        return self.f0 + (np.arange(self.nbins, dtype=np.float64) + 0.5) * self.bin_hz

    def feed(self, sw: Any) -> Tuple[Optional[np.ndarray], float]:
        if self.row is None or self.mask is None:
            return None, 0.0

        freqs = _get(sw, "freqs_hz", None)
        data  = _get(sw, "data_dbm", None)
        if data is None:
            return None, float(self.mask.mean())

        low = float(_get(sw, "hz_low", _infer_low(freqs)))
        if freqs is not None and len(freqs) > 0:
            high = float(np.max(freqs))
        else:
            high = low + len(data) * self.bin_hz

        self._segment_count += 1

        # wrap-детектор
        if self._last_high is not None:
            if (low < self._last_high - self.wrap_guard_hz) and (low < self.f0 + 50e6):
                cov_before = float(self.mask.mean())
                if cov_before >= self.threshold:
                    if self.interpolation_enabled:
                        out = self._interpolate_row()
                    else:
                        out = self._finalize_row_no_interpolation()
                    self.reset_pass()
                    print(f"[SweepAssembler] Full pass ready: coverage={cov_before:.3f}, reason=wrap, time={time.time()-self._start_time:.1f}s")
                    return out.copy(), cov_before

        self._last_low  = low
        self._last_high = high

        y = np.asarray(data, dtype=np.float32).ravel()
        if y.size == 0:
            return None, float(self.mask.mean())

        if freqs is not None:
            f = np.asarray(freqs, dtype=np.float64).ravel()
            idx = np.floor((f - self.f0) / self.bin_hz).astype(np.int64)
            valid = (idx >= 0) & (idx < self.nbins)
            if np.any(valid):
                self.row[idx[valid]] = y[valid]
                self.mask[idx[valid]] = True
        else:
            i0 = int(np.floor((low - self.f0) / self.bin_hz))
            if i0 < 0:
                y = y[-i0:]
                i0 = 0
            if i0 < self.nbins and y.size > 0:
                span = min(y.size, self.nbins - i0)
                if span > 0:
                    self.row[i0:i0+span] = y[:span]
                    self.mask[i0:i0+span] = True

        cov = float(self.mask.mean())

        if self._segment_count % self._debug_log_interval == 0:
            nset = int(self.mask.sum())
            print(f"[SweepAssembler] Segment #{self._segment_count}: coverage={cov:.3f} ({nset}/{self.nbins} bins), time={time.time()-self._start_time:.1f}s, freq_range={self._last_low/1e6:.1f}-{self._last_high/1e6:.1f} MHz")

        if cov >= self.threshold:
            if self.interpolation_enabled:
                out = self._interpolate_row()
            else:
                out = self._finalize_row_no_interpolation()
            print(f"[SweepAssembler] Full pass ready: coverage={cov:.3f}, time={time.time()-self._start_time:.1f}s")
            self.reset_pass()
            return out.copy(), cov

        return None, cov

    def _interpolate_row(self) -> np.ndarray:
        """
        Интерполирует пропуски в строке спектра для устранения ступенчатости.
        Использует улучшенную интерполяцию с учетом соседних значений.
        """
        if self.row is None:
            return np.array([], dtype=np.float32)
        
        # Копируем строку для интерполяции
        interpolated = self.row.copy()
        
        # Находим индексы известных и неизвестных значений
        known_mask = ~np.isnan(self.row)
        unknown_mask = np.isnan(self.row)
        
        if not np.any(unknown_mask):
            # Нет пропусков - возвращаем как есть
            return interpolated.astype(np.float32)
        
        if not np.any(known_mask):
            # Нет известных значений - заполняем значением по умолчанию
            return np.full_like(interpolated, self._nan_fill_value, dtype=np.float32)
        
        # Находим границы известных участков
        known_indices = np.where(known_mask)[0]
        
        # Интерполируем пропуски с улучшенным алгоритмом
        for i in range(len(known_indices) - 1):
            start_idx = known_indices[i]
            end_idx = known_indices[i + 1]
            
            if end_idx - start_idx > 1:  # Есть пропуски между известными точками
                start_val = self.row[start_idx]
                end_val = self.row[end_idx]
                
                # Используем кубическую интерполяцию для более гладких переходов
                for j in range(start_idx + 1, end_idx):
                    # Нормализованная позиция от 0 до 1
                    t = (j - start_idx) / (end_idx - start_idx)
                    
                    # Кубическая интерполяция с плавными производными на краях
                    t2 = t * t
                    t3 = t2 * t
                    
                    # Коэффициенты для плавного перехода
                    a = 2 * t3 - 3 * t2 + 1
                    b = -2 * t3 + 3 * t2
                    
                    interpolated[j] = a * start_val + b * end_val
                    
                    # Добавляем небольшое сглаживание для устранения шума
                    if j > start_idx + 1 and j < end_idx - 1:
                        # Усредняем с соседними интерполированными значениями
                        prev_val = interpolated[j-1]
                        if not np.isnan(prev_val):
                            interpolated[j] = 0.7 * interpolated[j] + 0.3 * prev_val
        
        # Обрабатываем края с экстраполяцией
        if known_indices[0] > 0:
            # Экстраполируем начало с учетом тренда
            if len(known_indices) > 1:
                trend = self.row[known_indices[1]] - self.row[known_indices[0]]
                for j in range(known_indices[0]):
                    interpolated[j] = self.row[known_indices[0]] + trend * (j - known_indices[0])
            else:
                interpolated[:known_indices[0]] = self.row[known_indices[0]]
        
        if known_indices[-1] < len(interpolated) - 1:
            # Экстраполируем конец с учетом тренда
            if len(known_indices) > 1:
                trend = self.row[known_indices[-1]] - self.row[known_indices[-2]]
                for j in range(known_indices[-1] + 1, len(interpolated)):
                    interpolated[j] = self.row[known_indices[-1]] + trend * (j - known_indices[-1])
            else:
                interpolated[known_indices[-1] + 1:] = self.row[known_indices[-1]]
        
        # Заменяем оставшиеся NaN на значение по умолчанию
        interpolated = np.where(np.isnan(interpolated), self._nan_fill_value, interpolated)
        
        # Финальное сглаживание для устранения возможных артефактов интерполяции
        if len(interpolated) > 5:
            # Применяем легкое сглаживание по 3 точкам
            smoothed = interpolated.copy()
            for i in range(1, len(interpolated) - 1):
                smoothed[i] = 0.5 * interpolated[i] + 0.25 * (interpolated[i-1] + interpolated[i+1])
            interpolated = smoothed
        
        return interpolated.astype(np.float32)

    def _finalize_row_no_interpolation(self) -> np.ndarray:
        """
        Финализирует строку спектра без интерполяции.
        Просто заменяет NaN на значение по умолчанию.
        """
        if self.row is None:
            return np.array([], dtype=np.float32)
        
        # Копируем строку и заменяем NaN на значение по умолчанию
        finalized = self.row.copy()
        finalized = np.where(np.isnan(finalized), self._nan_fill_value, finalized)
        
        return finalized.astype(np.float32)


def _get(d: Any, key: str, default=None):
    if d is None: return default
    if isinstance(d, dict): return d.get(key, default)
    return getattr(d, key, default)

def _infer_low(freqs: Optional[np.ndarray]) -> float:
    if freqs is None: return float("nan")
    f = np.asarray(freqs, dtype=np.float64)
    return float(f.min()) if f.size else float("nan")
