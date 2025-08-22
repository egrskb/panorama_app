"""
Master sweep controller for real-time spectrum analysis and peak detection.
Uses HackRFQSABackend (from example.c) for real HackRF sweep operations.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from pathlib import Path

# Импортируем новый backend
try:
    from panorama.drivers.hrf_backend import HackRFQSABackend
    C_LIB_AVAILABLE = True
except ImportError:
    HackRFQSABackend = None
    C_LIB_AVAILABLE = False
    print("WARNING: HackRFQSABackend not available. Check libhackrf_qsa.so")


@dataclass
class SweepTile:
    """Структура для передачи данных sweep плитки."""
    f_start: float  # Начальная частота (Гц)
    bin_hz: float   # Ширина бина (Гц)
    count: int      # Количество бинов
    power: List[float]  # Массив мощностей (дБм)
    t0: float      # Временная метка


@dataclass
class DetectedPeak:
    """Обнаруженный пик сигнала."""
    id: str
    f_peak: float      # Центральная частота пика (Гц)
    snr_db: float      # SNR в дБ
    bin_hz: float      # Ширина бина
    t0: float          # Время обнаружения
    last_seen: float   # Последнее время наблюдения
    span_user: float   # Пользовательская ширина полосы (Гц)
    status: str        # Статус: ACTIVE, QUIET, EXPIRED


class PeakDetector:
    """Детектор пиков с подавлением боковых лепестков."""
    
    def __init__(self, min_snr_db: float = 10.0, min_peak_distance_bins: int = 2):
        self.min_snr_db = min_snr_db
        self.min_peak_distance_bins = min_peak_distance_bins
        self.last_spectrum = None
        
    def detect_peaks_in_spectrum(self, freqs_hz: np.ndarray, powers_dbm: np.ndarray) -> List[DetectedPeak]:
        """Обнаруживает пики в спектре."""
        peaks = []
        
        if len(powers_dbm) < 3:
            return peaks
        
        # Сохраняем для последующего анализа
        self.last_spectrum = (freqs_hz, powers_dbm)
        
        # Находим локальные максимумы
        for i in range(1, len(powers_dbm) - 1):
            if powers_dbm[i] > powers_dbm[i-1] and powers_dbm[i] > powers_dbm[i+1]:
                # Оцениваем шум
                noise_level = self._estimate_noise_level(powers_dbm, i)
                snr_db = powers_dbm[i] - noise_level
                
                if snr_db >= self.min_snr_db:
                    # Проверяем расстояние до других пиков
                    too_close = False
                    for peak in peaks:
                        freq_diff = abs(freqs_hz[i] - peak.f_peak)
                        if freq_diff < (freqs_hz[1] - freqs_hz[0]) * self.min_peak_distance_bins:
                            too_close = True
                            break
                    
                    if not too_close:
                        peaks.append(DetectedPeak(
                            id=f"peak_{freqs_hz[i]:.0f}",
                            f_peak=freqs_hz[i],
                            snr_db=snr_db,
                            bin_hz=freqs_hz[1] - freqs_hz[0] if len(freqs_hz) > 1 else 200e3,
                            t0=time.time(),
                            last_seen=time.time(),
                            span_user=2e6,
                            status="ACTIVE"
                        ))
        
        return peaks
    
    def _estimate_noise_level(self, powers: np.ndarray, peak_idx: int) -> float:
        """Оценивает уровень шума вокруг пика."""
        # Исключаем область пика
        start_idx = max(0, peak_idx - 5)
        end_idx = min(len(powers), peak_idx + 6)
        
        noise_powers = []
        for i in range(len(powers)):
            if i < start_idx or i >= end_idx:
                noise_powers.append(powers[i])
        
        if noise_powers:
            return np.median(noise_powers)
        return np.mean(powers)


class MasterSweepController(QObject):
    """Контроллер Master sweep для управления HackRF и детекции пиков."""
    
    # Сигналы
    sweep_tile_received = pyqtSignal(object)  # SweepTile
    peak_detected = pyqtSignal(object)        # DetectedPeak
    sweep_error = pyqtSignal(str)             # Ошибка sweep
    full_sweep_ready = pyqtSignal(object, object)  # (freqs_hz, power_dbm)
    
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.log = logger
        
        # Проверяем доступность библиотеки
        if not C_LIB_AVAILABLE:
            self.log.error("HackRFQSABackend not available")
            self.sweep_source = None
        else:
            self.sweep_source = None  # Будет создан при необходимости
            self.log.info("HackRFQSABackend available")
        
        self.peak_detector = PeakDetector()
        self.is_running = False
        self._sdr_initialized = False
        
        # Параметры sweep
        self.start_hz = 24e6      # 24 МГц
        self.stop_hz = 6e9        # 6 ГГц
        self.bin_hz = 200e3       # 200 кГц
        self.dwell_ms = 100       # 100 мс
        
        # Таймер для обновления детекции пиков
        self.peak_detection_timer = QTimer()
        self.peak_detection_timer.timeout.connect(self._update_peak_detection)
        self.peak_detection_timer.start(1000)  # Каждую секунду
        
        # Буфер для watchlist
        self.watchlist: Dict[str, DetectedPeak] = {}
        
        # Для хранения последнего спектра
        self.last_spectrum = None
    
    def initialize_sdr(self):
        """Инициализирует SDR устройство."""
        if self._sdr_initialized:
            self.log.info("SDR already initialized")
            return True
            
        if not C_LIB_AVAILABLE:
            self.log.error("HackRFQSABackend not available")
            return False
            
        try:
            # Создаем backend
            self.sweep_source = HackRFQSABackend()
            
            # Подключаем сигналы
            self.sweep_source.fullSweepReady.connect(self._on_full_sweep)
            self.sweep_source.error.connect(self._on_error)
            self.sweep_source.started.connect(self._on_started)
            self.sweep_source.finished.connect(self._on_finished)
            
            self._sdr_initialized = True
            self.log.info("HackRFQSABackend initialized successfully")
            return True
            
        except Exception as e:
            self.log.error(f"Failed to initialize HackRFQSABackend: {e}")
            self.sweep_source = None
            self._sdr_initialized = False
            return False
    
    def deinitialize_sdr(self):
        """Деинициализирует SDR устройство."""
        if not self._sdr_initialized:
            return
            
        try:
            if self.is_running:
                self.stop_sweep()
            
            if self.sweep_source:
                self.sweep_source = None
            
            self._sdr_initialized = False
            self.log.info("HackRFQSABackend deinitialized")
            
        except Exception as e:
            self.log.error(f"Error deinitializing SDR: {e}")
    
    def start_sweep(self, start_hz: float = None, stop_hz: float = None, 
                    bin_hz: float = None, dwell_ms: float = None):
        """Запускает sweep с заданными параметрами."""
        if not C_LIB_AVAILABLE:
            error_msg = "HackRFQSABackend not available"
            self.log.error(error_msg)
            self.sweep_error.emit(error_msg)
            return
        
        # Инициализируем SDR если нужно
        if not self._sdr_initialized:
            self.log.info("SDR not initialized, initializing now...")
            if not self.initialize_sdr():
                error_msg = "Failed to initialize SDR"
                self.log.error(error_msg)
                self.sweep_error.emit(error_msg)
                return
        
        # Проверяем не запущен ли уже sweep
        if self.is_running:
            self.log.warning("Sweep already running. Stopping first...")
            self.stop_sweep()
            time.sleep(0.1)
        
        # Обновляем параметры
        if start_hz is not None:
            self.start_hz = start_hz
        if stop_hz is not None:
            self.stop_hz = stop_hz
        if bin_hz is not None:
            self.bin_hz = bin_hz
        if dwell_ms is not None:
            self.dwell_ms = dwell_ms
            
        self.log.info(f"Starting sweep: {self.start_hz/1e6:.1f}-{self.stop_hz/1e6:.1f} MHz, "
                     f"bin: {self.bin_hz/1e3:.0f} kHz")
        
        try:
            # Создаем конфигурацию
            from panorama.drivers.base import SweepConfig
            config = SweepConfig(
                freq_start_hz=int(self.start_hz),
                freq_end_hz=int(self.stop_hz),
                bin_hz=int(self.bin_hz),
                lna_db=24,
                vga_db=20,
                amp_on=False
            )
            
            # Запускаем через backend
            self.sweep_source.start(config)
            self.is_running = True
            
            self.log.info("Master sweep started successfully")
            
        except Exception as e:
            self.log.error(f"Failed to start sweep: {e}")
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}")
            self.sweep_error.emit(str(e))
    
    def stop_sweep(self):
        """Останавливает sweep."""
        if not self._sdr_initialized or not self.sweep_source:
            return
        
        try:
            self.sweep_source.stop()
            self.is_running = False
            self.log.info("Master sweep stopped")
            
        except Exception as e:
            self.log.error(f"Failed to stop sweep: {e}")
            self.is_running = False
    
    def _on_full_sweep(self, freqs_hz, power_dbm):
        """Обрабатывает полный sweep от backend."""
        try:
            self.log.info(f"Received full sweep: {len(freqs_hz)} points, "
                         f"range {freqs_hz[0]/1e6:.1f}-{freqs_hz[-1]/1e6:.1f} MHz")
            
            # Сохраняем последний спектр
            self.last_spectrum = (freqs_hz, power_dbm)
            
            # Эмитим для GUI (только один раз!)
            self.full_sweep_ready.emit(freqs_hz, power_dbm)
            
            # Детектируем пики
            peaks = self.peak_detector.detect_peaks_in_spectrum(freqs_hz, power_dbm)
            for peak in peaks:
                # Обновляем watchlist
                if peak.id not in self.watchlist:
                    self.watchlist[peak.id] = peak
                    self.peak_detected.emit(peak)
                    self.log.info(f"New peak detected: {peak.f_peak/1e6:.1f} MHz, SNR: {peak.snr_db:.1f} dB")
                else:
                    # Обновляем существующий пик
                    self.watchlist[peak.id].last_seen = time.time()
                    self.watchlist[peak.id].snr_db = peak.snr_db
            
        except Exception as e:
            self.log.error(f"Error processing full sweep: {e}")
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}")
    
    def _on_error(self, msg):
        """Обрабатывает ошибку от backend."""
        self.log.error(f"Backend error: {msg}")
        self.sweep_error.emit(msg)
    
    def _on_started(self):
        """Обрабатывает сигнал запуска от backend."""
        self.log.info("Backend started")
    
    def _on_finished(self, code):
        """Обрабатывает сигнал завершения от backend."""
        self.log.info(f"Backend finished with code: {code}")
        self.is_running = False
    
    def _update_peak_detection(self):
        """Обновляет детекцию пиков."""
        if not self.is_running or not self.last_spectrum:
            return
            
        try:
            current_time = time.time()
            
            # Удаляем устаревшие пики (старше 30 секунд)
            expired_peaks = []
            for peak_id, peak in self.watchlist.items():
                if current_time - peak.last_seen > 30.0:
                    expired_peaks.append(peak_id)
                    peak.status = "EXPIRED"
                elif current_time - peak.last_seen > 5.0:
                    peak.status = "QUIET"
                else:
                    peak.status = "ACTIVE"
            
            for peak_id in expired_peaks:
                del self.watchlist[peak_id]
                
        except Exception as e:
            self.log.error(f"Error in peak detection update: {e}")
    
    def get_watchlist(self) -> List[DetectedPeak]:
        """Возвращает текущий watchlist."""
        return list(self.watchlist.values())
    
    def get_sweep_status(self) -> Dict:
        """Возвращает статус sweep."""
        return {
            'is_running': self.is_running,
            'start_hz': self.start_hz,
            'stop_hz': self.stop_hz,
            'bin_hz': self.bin_hz,
            'dwell_ms': self.dwell_ms,
            'watchlist_count': len(self.watchlist),
            'active_peaks': len([p for p in self.watchlist.values() if p.status == "ACTIVE"]),
            'c_library_available': C_LIB_AVAILABLE
        }
    
    def cleanup(self):
        """Очищает ресурсы."""
        try:
            if self.is_running:
                self.stop_sweep()
            
            if hasattr(self, 'peak_detection_timer'):
                self.peak_detection_timer.stop()
            
            self.deinitialize_sdr()
            
            self.log.info("MasterSweepController cleanup completed")
            
        except Exception as e:
            self.log.error(f"Error during cleanup: {e}")
    
    # Методы совместимости
    def is_sdr_available(self):
        return C_LIB_AVAILABLE
    
    def is_sdr_initialized(self):
        return self._sdr_initialized
    
    def enumerate_devices(self):
        """Сканирует доступные устройства."""
        try:
            from panorama.drivers.hrf_backend import enumerate_devices
            return enumerate_devices()
        except Exception as e:
            self.log.error(f"Failed to enumerate devices: {e}")
            return []