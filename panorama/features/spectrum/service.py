# panorama/features/spectrum/service.py
from __future__ import annotations

from typing import Optional, Any, Tuple
import numpy as np


class SweepAssembler:
    """
    Собирает ОДНУ полную строку спектра из приходящих сегментов sweep.
    ОПТИМИЗИРОВАНО: улучшена производительность для большого количества точек.
    """

    def __init__(self, coverage_threshold: float = 0.45, wrap_guard_hz: float = 100e6):
        # Снижаем порог покрытия для широкого диапазона
        self.threshold = float(coverage_threshold)
        # Увеличиваем wrap guard для работы с диапазоном 50-6000 МГц
        self.wrap_guard_hz = float(wrap_guard_hz)

        # конфигурация сетки
        self.f0: float = 0.0
        self.f1: float = 0.0
        self.bin_hz: float = 0.0
        self.nbins: int = 0

        # накопители текущего прохода
        self.row: Optional[np.ndarray] = None  # float32, len=nbins
        self.mask: Optional[np.ndarray] = None  # bool, len=nbins

        # служебное для wrap-детектора
        self._last_low: Optional[float] = None
        self._last_high: Optional[float] = None
        self._segment_count: int = 0
        self._start_time: float = 0
        
        # ОПТИМИЗАЦИЯ: увеличиваем интервал логирования для лучшей производительности
        self._debug_log_interval = 200  # было 50, теперь 200 - логируем реже
        
        # ОПТИМИЗАЦИЯ: кэш для частых вычислений
        self._nan_fill_value = -120.0
        
        # ОПТИМИЗАЦИЯ: автоматический даунсемплинг для больших массивов
        self._max_display_bins = 10000  # Максимальное количество точек для отображения
        self._downsample_factor = 1  # Фактор даунсемплинга

    # ---------------- конфигурация/сброс ----------------

    def configure(self, f0_hz: int, f1_hz: int, bin_hz: int, lut: Any = None) -> None:
        f0 = float(f0_hz)
        f1 = float(f1_hz)
        b  = float(bin_hz)
        if f1 <= f0 or b <= 0.0:
            raise ValueError("Bad grid: f1<=f0 or bin<=0")

        # Используем round для согласования с мастером (Fs/N)
        nb = int(round((f1 - f0) / b))
        if nb <= 0:
            raise ValueError("nbins <= 0")

        self.f0, self.f1, self.bin_hz, self.nbins = f0, f1, b, nb
        self.row  = np.full(nb, np.nan, dtype=np.float32)
        self.mask = np.zeros(nb, dtype=bool)
        self._last_low = None
        self._last_high = None
        self._segment_count = 0
        
        # ОПТИМИЗАЦИЯ: автоматический даунсемплинг для больших массивов
        if nb > self._max_display_bins:
            self._downsample_factor = max(1, nb // self._max_display_bins)
            print(f"[SweepAssembler] Auto-downsample: {nb} bins -> {nb // self._downsample_factor} bins (factor={self._downsample_factor})")
        else:
            self._downsample_factor = 1
        
        import time
        self._start_time = time.time()
        
        # ОПТИМИЗАЦИЯ: выводим информацию только один раз при конфигурации
        print(f"[SweepAssembler] Configured: {f0/1e6:.1f}-{f1/1e6:.1f} MHz, "
              f"{nb} bins, bin_width={b/1e3:.1f} kHz")
        
        # ОПТИМИЗАЦИЯ: предупреждение о большом количестве точек
        if nb > 10000:
            print(f"[SweepAssembler] WARNING: Large number of bins ({nb}) may impact UI performance")
            print(f"[SweepAssembler] Consider increasing bin_width to reduce bins")

    def reset_pass(self) -> None:
        if self.row is not None:
            self.row.fill(np.nan)
        if self.mask is not None:
            self.mask[:] = False
        self._last_low = None
        self._last_high = None
        self._segment_count = 0
        
        import time
        self._start_time = time.time()

    # ---------------- основная укладка ----------------

    def feed(self, sw: Any) -> Tuple[Optional[np.ndarray], float]:
        """Кладёт сегмент. При достаточном покрытии возвращает (row, coverage)."""
        if self.row is None or self.mask is None:
            return None, 0.0

        # извлекаем поля сегмента
        freqs = _get(sw, "freqs_hz", None)
        data  = _get(sw, "data_dbm", None)
        if data is None:
            return None, float(self.mask.mean())

        # нижняя частота сегмента
        low = float(_get(sw, "hz_low", _infer_low(freqs)))
        
        # Вычисляем верхнюю частоту
        if freqs is not None and len(freqs) > 0:
            high = float(np.max(freqs))
        else:
            # Примерная оценка на основе размера данных и bin_hz
            high = low + len(data) * self.bin_hz
        
        self._segment_count += 1
        
        # Детекция wrap - более интеллектуальная логика
        # Wrap происходит, когда мы возвращаемся к началу диапазона после прохода до конца
        if self._last_high is not None:
            # Если текущий low намного меньше последнего high И близок к началу диапазона
            if (low < self._last_high - self.wrap_guard_hz and 
                low < self.f0 + 50e6):  # В пределах 50 МГц от начала
                
                # Проверяем покрытие перед сбросом
                cov_before = float(self.mask.mean())
                if cov_before >= self.threshold:
                    # У нас достаточное покрытие, возвращаем строку
                    out = np.where(np.isnan(self.row), self._nan_fill_value, self.row).astype(np.float32, copy=False)
                    self.reset_pass()
                    print(f"[SweepAssembler] Full pass ready: coverage={cov_before:.3f}, "
                          f"reason=threshold, time={time.time()-self._start_time:.1f}s, "
                          f"freq_range={self.f0/1e6:.1f}-{self.f1/1e6:.1f} MHz")
                    return out.copy(), cov_before
                elif high > self.f1 - 50e6:  # Достигли конца диапазона
                    # Проверяем достаточно ли у нас покрытия для возврата
                    if cov_before >= 0.4:  # Более мягкий порог для широкого диапазона
                        out = np.where(np.isnan(self.row), self._nan_fill_value, self.row).astype(np.float32, copy=False)
                        print(f"[SweepAssembler] Full pass ready: coverage={cov_before:.3f}, "
                              f"reason=range, time={time.time()-self._start_time:.1f}s, "
                              f"freq_range={self.f0/1e6:.1f}-{high/1e6:.1f} MHz")
                        self.reset_pass()
                        return out.copy(), cov_before
                    else:
                        # Недостаточное покрытие, но обнаружен wrap - сбрасываем
                        print(f"[SweepAssembler] Wrap detected: {high/1e6:.1f} MHz < {low/1e6:.1f} MHz")
                        print(f"[SweepAssembler] Incomplete pass, coverage={cov_before:.3f}, resetting")
                        self.reset_pass()

        self._last_low = low
        self._last_high = high

        y = np.asarray(data, dtype=np.float32).ravel()
        if y.size == 0:
            return None, float(self.mask.mean())

        # Укладка данных
        if freqs is not None:
            # укладка по индексам глобальной сетки (через freqs)
            f = np.asarray(freqs, dtype=np.float64).ravel()
            # округление к ближайшему бину глобальной сетки:
            idx = np.round((f - self.f0) / self.bin_hz).astype(np.int64)

            # фильтруем валидные индексы
            valid = (idx >= 0) & (idx < self.nbins)
            if np.any(valid):
                self.row[idx[valid]] = y[valid]
                self.mask[idx[valid]] = True
            elif low == low:  # low is finite - fallback на непрерывную укладку
                i0 = int(round((low - self.f0) / self.bin_hz))
                if i0 < 0:
                    y = y[-i0:]
                    i0 = 0
                if i0 < self.nbins and y.size > 0:
                    span = min(y.size, self.nbins - i0)
                    if span > 0:
                        self.row[i0:i0 + span] = y[:span]
                        self.mask[i0:i0 + span] = True
        else:
            # быстрый непрерывный сегмент
            i0 = int(round((low - self.f0) / self.bin_hz))
            if i0 < 0:
                y = y[-i0:]
                i0 = 0
            if i0 < self.nbins and y.size > 0:
                span = min(y.size, self.nbins - i0)
                if span > 0:
                    self.row[i0:i0 + span] = y[:span]
                    self.mask[i0:i0 + span] = True

        # покрытие
        cov = float(self.mask.mean())
        
        # ОПТИМИЗАЦИЯ: логируем прогресс реже для лучшей производительности
        if self._segment_count % self._debug_log_interval == 0:
            import time
            filled_bins = np.sum(self.mask)
            if filled_bins > 0:
                freq_min = self.f0 + np.min(np.where(self.mask)[0]) * self.bin_hz
                freq_max = self.f0 + np.max(np.where(self.mask)[0]) * self.bin_hz
                print(f"[SweepAssembler] Segment #{self._segment_count}: "
                      f"coverage={cov:.3f} ({filled_bins}/{self.nbins} bins), "
                      f"time={time.time()-self._start_time:.1f}s, "
                      f"freq_range={freq_min/1e6:.1f}-{freq_max/1e6:.1f} MHz")

        # ОПТИМИЗАЦИЯ: интерполяция только при необходимости и с ограничениями
        if cov > 0.3 and cov < self.threshold:
            holes = np.where(~self.mask)[0]
            # Ограничиваем количество заполняемых дырок для производительности
            max_holes_to_fill = min(holes.size, int(self.nbins * 0.05))  # Максимум 5% дырок
            if holes.size > 0 and holes.size < self.nbins * 0.1:  # Заполняем только если дырок < 10%
                # ОПТИМИЗАЦИЯ: более эффективная интерполяция
                holes_to_fill = holes[:max_holes_to_fill]
                for k in holes_to_fill:
                    # ищем ближайший установленный бин слева/справа
                    left = k - 1
                    right = k + 1
                    if left >= 0 and self.mask[left] and right < self.nbins and self.mask[right]:
                        # Интерполяция между соседями
                        self.row[k] = (self.row[left] + self.row[right]) / 2
                        self.mask[k] = True
                cov = float(self.mask.mean())

        # Возвращаем результат при достаточном покрытии
        if cov >= self.threshold:
            out = np.where(np.isnan(self.row), self._nan_fill_value, self.row).astype(np.float32, copy=False)
            
            # ОПТИМИЗАЦИЯ: автоматический даунсемплинг для больших массивов
            if self._downsample_factor > 1:
                # Даунсемплинг с усреднением
                new_size = len(out) // self._downsample_factor
                out_downsampled = np.zeros(new_size, dtype=np.float32)
                for i in range(new_size):
                    start_idx = i * self._downsample_factor
                    end_idx = start_idx + self._downsample_factor
                    # Усредняем по группам бинов
                    valid_values = out[start_idx:end_idx]
                    valid_mask = ~np.isnan(valid_values)
                    if np.any(valid_mask):
                        out_downsampled[i] = np.mean(valid_values[valid_mask])
                    else:
                        out_downsampled[i] = self._nan_fill_value
                out = out_downsampled
                print(f"[SweepAssembler] Downsampled output: {len(out_downsampled)} bins")
            
            import time
            print(f"[SweepAssembler] Full pass ready: coverage={cov:.3f}, "
                  f"reason=coverage, time={time.time()-self._start_time:.1f}s")
            self.reset_pass()
            return out.copy(), cov

        return None, cov


# ---------------- утилиты ----------------

def _get(d: Any, key: str, default=None):
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _infer_low(freqs: Optional[np.ndarray]) -> float:
    if freqs is None:
        return float("nan")
    f = np.asarray(freqs, dtype=np.float64)
    return float(f.min()) if f.size else float("nan")