"""
Master sweep controller для управления полным сканированием спектра.
Работает с переработанным HackRFQSABackend.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

# Импортируем переработанный backend
try:
    from panorama.drivers.hrf_backend import HackRFQSABackend
    C_LIB_AVAILABLE = True
except ImportError:
    HackRFQSABackend = None
    C_LIB_AVAILABLE = False
    print("WARNING: HackRFQSABackend not available. Check libhackrf_qsa.so")


@dataclass
class DetectedPeak:
    """Обнаруженный пик сигнала."""
    id: str
    f_peak: float      # Центральная частота пика (Гц)
    power_dbm: float   # Мощность пика
    snr_db: float      # SNR в дБ
    bin_hz: float      # Ширина бина
    t0: float          # Время обнаружения
    last_seen: float   # Последнее время наблюдения
    span_user: float   # Пользовательская ширина полосы (Гц)
    status: str        # Статус: ACTIVE, QUIET, EXPIRED


class SimplePeakDetector:
    """Упрощенный детектор пиков для полного спектра."""
    
    def __init__(self, min_snr_db: float = 15.0, min_peak_distance_mhz: float = 1.0):
        self.min_snr_db = min_snr_db
        self.min_peak_distance_hz = min_peak_distance_mhz * 1e6
        
    def detect_peaks(self, freqs_hz: np.ndarray, powers_dbm: np.ndarray) -> List[DetectedPeak]:
        """Обнаруживает пики в полном спектре."""
        peaks = []
        
        if len(powers_dbm) < 10:
            return peaks
        
        # Оцениваем шумовой пол как медиану нижних 30% значений
        sorted_powers = np.sort(powers_dbm)
        noise_floor = np.median(sorted_powers[:len(sorted_powers)//3])
        
        # Находим локальные максимумы с использованием скользящего окна
        window_size = 21  # Нечетное число для симметричного окна
        half_window = window_size // 2
        
        for i in range(half_window, len(powers_dbm) - half_window):
            # Проверяем, является ли точка локальным максимумом
            window = powers_dbm[i-half_window:i+half_window+1]
            if powers_dbm[i] == np.max(window):
                # Вычисляем SNR
                snr = powers_dbm[i] - noise_floor
                
                if snr >= self.min_snr_db:
                    # Проверяем расстояние до других пиков
                    freq = freqs_hz[i]
                    too_close = False
                    
                    for peak in peaks:
                        if abs(freq - peak.f_peak) < self.min_peak_distance_hz:
                            # Если новый пик сильнее, заменяем старый
                            if powers_dbm[i] > peak.power_dbm:
                                peaks.remove(peak)
                            else:
                                too_close = True
                            break
                    
                    if not too_close:
                        peaks.append(DetectedPeak(
                            id=f"peak_{freq:.0f}",
                            f_peak=freq,
                            power_dbm=powers_dbm[i],
                            snr_db=snr,
                            bin_hz=freqs_hz[1] - freqs_hz[0] if len(freqs_hz) > 1 else 200e3,
                            t0=time.time(),
                            last_seen=time.time(),
                            span_user=2e6,
                            status="ACTIVE"
                        ))
        
        # Сортируем по мощности (сильные сигналы первыми)
        peaks.sort(key=lambda p: p.power_dbm, reverse=True)
        
        return peaks


class MasterSweepController(QObject):
    """Контроллер Master sweep для полного сканирования спектра."""
    
    # Сигналы
    full_sweep_ready = pyqtSignal(object, object)  # (freqs_hz, power_dbm)
    peak_detected = pyqtSignal(object)             # DetectedPeak
    sweep_error = pyqtSignal(str)                  # Ошибка sweep
    sweep_progress = pyqtSignal(float)             # Прогресс покрытия (0-100%)
    
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.log = logger
        
        # Проверяем доступность библиотеки
        if not C_LIB_AVAILABLE:
            self.log.error("HackRFQSABackend not available")
            self.sweep_source = None
        else:
            self.sweep_source = None
            self.log.info("HackRFQSABackend available")
        
        self.peak_detector = SimplePeakDetector()
        self.is_running = False
        self._sdr_initialized = False
        
        # Параметры sweep по умолчанию (полный диапазон)
        self.start_hz = 50e6      # 50 МГц
        self.stop_hz = 6000e6     # 6 ГГц  
        self.bin_hz = 200e3       # 200 кГц
        
        # Таймер для обновления детекции пиков
        self.peak_detection_timer = QTimer()
        self.peak_detection_timer.timeout.connect(self._detect_peaks)
        self.peak_detection_timer.setInterval(2000)  # Каждые 2 секунды
        
        # Хранилище для watchlist
        self.watchlist: Dict[str, DetectedPeak] = {}
        
        # Последний полный спектр
        self.last_freqs = None
        self.last_spectrum = None
        
        # Статистика
        self.sweep_count = 0
        self.start_time = None
    
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
                    bin_hz: float = None):
        """Запускает полное сканирование спектра."""
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
            time.sleep(0.5)
        
        # Обновляем параметры
        if start_hz is not None:
            self.start_hz = start_hz
        if stop_hz is not None:
            self.stop_hz = stop_hz
        if bin_hz is not None:
            self.bin_hz = bin_hz
            
        self.log.info(f"Starting full spectrum sweep: {self.start_hz/1e6:.1f}-{self.stop_hz/1e6:.1f} MHz, "
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
            
            # Сбрасываем статистику
            self.sweep_count = 0
            self.start_time = time.time()
            self.watchlist.clear()
            
            # Запускаем через backend
            self.sweep_source.start(config)
            self.is_running = True
            
            # Запускаем детектор пиков
            self.peak_detection_timer.start()
            
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
            self.peak_detection_timer.stop()
            self.sweep_source.stop()
            self.is_running = False
            self.log.info("Master sweep stopped")
            
        except Exception as e:
            self.log.error(f"Failed to stop sweep: {e}")
            self.is_running = False
    
    def _on_full_sweep(self, freqs_hz, power_dbm):
        """Обрабатывает полный спектр от backend."""
        try:
            self.sweep_count += 1
            
            # Сохраняем последний спектр
            self.last_freqs = freqs_hz
            self.last_spectrum = power_dbm
            
            # Вычисляем покрытие (процент не-NaN значений)
            valid_data = power_dbm[power_dbm > -119]  # Значения выше шумового пола
            coverage = (len(valid_data) / len(power_dbm)) * 100 if len(power_dbm) > 0 else 0
            
            # Эмитим прогресс
            self.sweep_progress.emit(coverage)
            
            # Логируем статистику
            if self.sweep_count % 10 == 0:
                elapsed = time.time() - self.start_time if self.start_time else 0
                rate = self.sweep_count / elapsed if elapsed > 0 else 0
                self.log.info(f"Sweep #{self.sweep_count}: coverage={coverage:.1f}%, "
                             f"rate={rate:.1f} sweeps/s, "
                             f"range={freqs_hz[0]/1e6:.1f}-{freqs_hz[-1]/1e6:.1f} MHz")
            
            # Эмитим полный спектр для GUI
            self.full_sweep_ready.emit(freqs_hz, power_dbm)
            
        except Exception as e:
            self.log.error(f"Error processing full sweep: {e}")
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}")
    
    def _detect_peaks(self):
        """Детектирует пики в последнем спектре."""
        if not self.is_running or self.last_freqs is None or self.last_spectrum is None:
            return
        
        try:
            # Детектируем пики
            peaks = self.peak_detector.detect_peaks(self.last_freqs, self.last_spectrum)
            
            current_time = time.time()
            
            # Обновляем watchlist
            new_peaks = []
            for peak in peaks[:20]:  # Ограничиваем топ-20 пиками
                if peak.id not in self.watchlist:
                    self.watchlist[peak.id] = peak
                    new_peaks.append(peak)
                    self.peak_detected.emit(peak)
                else:
                    # Обновляем существующий пик
                    self.watchlist[peak.id].last_seen = current_time
                    self.watchlist[peak.id].power_dbm = peak.power_dbm
                    self.watchlist[peak.id].snr_db = peak.snr_db
            
            # Удаляем устаревшие пики (не видели более 30 секунд)
            expired_peaks = []
            for peak_id, peak in self.watchlist.items():
                if current_time - peak.last_seen > 30.0:
                    expired_peaks.append(peak_id)
                    peak.status = "EXPIRED"
                elif current_time - peak.last_seen > 10.0:
                    peak.status = "QUIET"
                else:
                    peak.status = "ACTIVE"
            
            for peak_id in expired_peaks:
                del self.watchlist[peak_id]
            
            # Логируем новые пики
            if new_peaks:
                self.log.info(f"Detected {len(new_peaks)} new peaks:")
                for peak in new_peaks[:5]:  # Логируем топ-5
                    self.log.info(f"  - {peak.f_peak/1e6:.1f} MHz: {peak.power_dbm:.1f} dBm (SNR: {peak.snr_db:.1f} dB)")
                
        except Exception as e:
            self.log.error(f"Error in peak detection: {e}")
    
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
        self.peak_detection_timer.stop()
    
    def get_watchlist(self) -> List[DetectedPeak]:
        """Возвращает текущий watchlist."""
        return list(self.watchlist.values())
    
    def get_sweep_status(self) -> Dict:
        """Возвращает статус sweep."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        rate = self.sweep_count / elapsed if elapsed > 0 else 0
        
        return {
            'is_running': self.is_running,
            'start_hz': self.start_hz,
            'stop_hz': self.stop_hz,
            'bin_hz': self.bin_hz,
            'sweep_count': self.sweep_count,
            'sweep_rate': rate,
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