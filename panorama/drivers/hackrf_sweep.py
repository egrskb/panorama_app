from __future__ import annotations
import shutil, threading
from typing import Optional, Tuple
from PyQt5 import QtCore
import numpy as np

from .base import SourceBackend, SweepConfig
from panorama.shared.parsing import parse_sweep_line, SweepLine


class HackRFSweepSource(SourceBackend):
    """
    Источник данных через hackrf_sweep с поддержкой калибровки и сборки полных проходов.
    """
    
    def __init__(self, executable: str = "hackrf_sweep", parent=None):
        super().__init__(parent)
        
        self._exe = shutil.which(executable) or executable
        self._p = None
        self._buf = bytearray()
        self._assembler = _SweepAssembler()
        self._calibration_lut = None  # Для калибровки

    def set_calibration_lut(self, lut: Optional[Tuple[np.ndarray, np.ndarray]]):
        """
        Устанавливает LUT калибровки: (freq_mhz, offset_db).
        LUT должен быть отфильтрован для текущих LNA/VGA/AMP.
        """
        self._calibration_lut = lut

    def is_running(self) -> bool:
        return self._p is not None and self._p.state() != QtCore.QProcess.NotRunning

    def start(self, config: SweepConfig):
        if self.is_running():
            self.status.emit("Уже запущено")
            return

        if not shutil.which(self._exe):
            self.error.emit(f"Не найден исполняемый файл: {self._exe}")
            return

        # Конфигурируем сборщик
        self._assembler.configure(config, self._calibration_lut)

        # Запускаем процесс
        self._p = QtCore.QProcess(self)
        self._p.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self._p.readyReadStandardOutput.connect(self._on_ready)
        self._p.finished.connect(self._on_finished)

        args = config.to_args()
        self.status.emit(f"Запуск: {self._exe} {' '.join(args)}")
        self._p.start(self._exe, args)

        if not self._p.waitForStarted(3000):
            self.error.emit("Не удалось запустить hackrf_sweep")
            self._cleanup(emit_finished=False)
            return

        self.started.emit()

    def stop(self):
        if not self.is_running():
            return
        
        self._p.terminate()
        if not self._p.waitForFinished(1500):
            self._p.kill()
            self._p.waitForFinished(1500)
        self._cleanup(emit_finished=True)

    # -------- internals --------
    def _on_ready(self):
        if not self._p:
            return
        
        self._buf.extend(self._p.readAllStandardOutput().data())
        
        # Разбиваем по строкам
        *lines, tail = self._buf.split(b"\n")
        self._buf = bytearray(tail)
        
        for raw in lines:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            
            # Пропускаем служебные сообщения
            if "," not in line or line.startswith("hackrf_"):
                self.status.emit(line)
                continue
            
            try:
                sw: SweepLine = parse_sweep_line(line)
                
                # Эмитим сырой сегмент
                self.sweepLine.emit(sw)
                
                # Собираем полный проход
                result = self._assembler.add_segment(sw)
                if result is not None:
                    full_freqs, full_power = result
                    self.fullSweepReady.emit(full_freqs, full_power)
                    
            except Exception as e:
                self.status.emit(f"Parse error: {str(e)[:120]}")

    def _on_finished(self, code: int, _status):
        self._cleanup(emit_finished=False)
        self.finished.emit(int(code))

    def _cleanup(self, emit_finished: bool):
        if self._p:
            try:
                self._p.readyReadStandardOutput.disconnect(self._on_ready)
            except Exception:
                pass
            try:
                self._p.finished.disconnect(self._on_finished)
            except Exception:
                pass
            self._p = None
        
        self._buf.clear()
        self._assembler.reset()
        
        if emit_finished:
            self.finished.emit(0)


class _SweepAssembler:
    """Собирает полный проход из сегментов hackrf_sweep с применением калибровки."""
    
    def __init__(self):
        self.f0_hz = 0
        self.f1_hz = 0
        self.bin_hz = 0
        self.grid = None
        self.n_bins = 0
        self.sum = None
        self.cnt = None
        self.seen = None
        self.prev_low = None
        self.lut = None
        
    def configure(self, cfg: SweepConfig, lut: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.f0_hz = cfg.freq_start_hz
        self.f1_hz = cfg.freq_end_hz
        self.bin_hz = cfg.bin_hz
        self.lut = lut
        
        # Создаем сетку частот
        self.grid = np.arange(
            self.f0_hz + self.bin_hz * 0.5,
            self.f1_hz + self.bin_hz * 0.5,
            self.bin_hz,
            dtype=np.float64
        )
        self.n_bins = len(self.grid)
        self.reset()
    
    def reset(self):
        if self.n_bins == 0:
            return
        self.sum = np.zeros(self.n_bins, np.float64)
        self.cnt = np.zeros(self.n_bins, np.int32)
        self.seen = np.zeros(self.n_bins, bool)
        self.prev_low = None
    
    def add_segment(self, sw: SweepLine) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Добавляет сегмент из SweepLine.
        Возвращает (freqs_hz, power_dbm) когда проход завершен.
        """
        if self.grid is None or self.n_bins == 0:
            return None
        
        # Вычисляем частоты для бинов
        n = sw.n_bins
        freqs = sw.f_low_hz + (np.arange(n, dtype=np.float64) + 0.5) * sw.bin_hz
        power = sw.power_dbm.astype(np.float32)
        
        # Применяем калибровку если есть
        if self.lut is not None:
            power = self._apply_calibration(freqs, power)
        
        # Детекция обмотки
        if self.prev_low is not None and sw.f_low_hz < self.prev_low - 10e6:
            # Новый проход начался
            result = self._finalize()
            self.reset()
            self.prev_low = sw.f_low_hz
            self._add_to_grid(freqs, power)
            return result
        
        self.prev_low = sw.f_low_hz
        self._add_to_grid(freqs, power)
        
        # Проверяем покрытие
        coverage = float(self.seen.sum()) / float(self.n_bins) if self.n_bins else 0
        if coverage >= 0.95:
            result = self._finalize()
            self.reset()
            return result
        
        return None
    
    def _apply_calibration(self, freqs_hz: np.ndarray, power_dbm: np.ndarray) -> np.ndarray:
        """Применяет LUT калибровки."""
        if self.lut is None:
            return power_dbm
        
        f_mhz_lut, offset_db_lut = self.lut
        f_mhz = freqs_hz / 1e6
        
        # Интерполируем оффсеты для наших частот
        offsets = np.interp(f_mhz, f_mhz_lut, offset_db_lut).astype(np.float32)
        
        return power_dbm + offsets
    
    def _add_to_grid(self, f_hz: np.ndarray, p_dbm: np.ndarray):
        """Раскладывает сегмент в сетку."""
        idx = np.rint((f_hz - self.grid[0]) / self.bin_hz).astype(np.int32)
        mask = (idx >= 0) & (idx < self.n_bins)
        if not np.any(mask):
            return
        
        idx = idx[mask]
        p = p_dbm[mask].astype(np.float64)
        
        np.add.at(self.sum, idx, p)
        np.add.at(self.cnt, idx, 1)
        self.seen[idx] = True
    
    def _finalize(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Финализирует текущий проход."""
        if self.n_bins == 0:
            return None
        
        coverage = float(self.seen.sum()) / float(self.n_bins)
        if coverage < 0.5:
            return None
        
        p = np.full(self.n_bins, np.nan, np.float32)
        valid = self.cnt > 0
        p[valid] = (self.sum[valid] / self.cnt[valid]).astype(np.float32)
        
        # Интерполируем пропуски
        if np.isnan(p).any():
            vmask = ~np.isnan(p)
            if vmask.any():
                p = np.interp(np.arange(self.n_bins), np.flatnonzero(vmask), p[vmask]).astype(np.float32)
            p[np.isnan(p)] = -120.0
        
        return self.grid.copy(), p