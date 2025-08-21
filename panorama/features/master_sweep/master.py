"""
Master sweep controller for real-time spectrum analysis and peak detection.
Uses C library through CFFI for HackRF sweep operations.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from pathlib import Path

# Импортируем C библиотеку через CFFI
try:
    from panorama.drivers.hrf_backend import HackRFMaster
    C_LIB_AVAILABLE = True
except ImportError:
    HackRFMaster = None
    C_LIB_AVAILABLE = False
    print("WARNING: HackRF Master C library not available. Install and build it first.")


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
        self.psd_buffer = {}  # Буфер PSD для накопления данных
        
    def add_sweep_tile(self, tile: SweepTile):
        """Добавляет новую sweep плитку в буфер."""
        if tile.f_start not in self.psd_buffer:
            self.psd_buffer[tile.f_start] = []
        
        self.psd_buffer[tile.f_start].append({
            'powers': tile.power,
            't0': tile.t0,
            'bin_hz': tile.bin_hz
        })
        
        # Ограничиваем размер буфера
        if len(self.psd_buffer[tile.f_start]) > 10:
            self.psd_buffer[tile.f_start].pop(0)
    
    def detect_peaks(self) -> List[DetectedPeak]:
        """Обнаруживает пики в накопленных данных."""
        peaks = []
        
        for f_start, tiles in self.psd_buffer.items():
            if not tiles:
                continue
                
            # Усредняем мощности по времени (EMA)
            latest_tile = tiles[-1]
            avg_powers = np.array(latest_tile['powers'])
            
            # Простое усреднение (можно заменить на EMA)
            for tile in tiles[:-1]:
                avg_powers = 0.7 * avg_powers + 0.3 * np.array(tile['powers'])
            
            # Поиск локальных максимумов
            local_maxima = self._find_local_maxima(avg_powers)
            
            # Фильтрация по SNR и расстоянию
            for peak_idx in local_maxima:
                peak_power = avg_powers[peak_idx]
                noise_level = self._estimate_noise_level(avg_powers, peak_idx)
                snr_db = peak_power - noise_level
                
                if snr_db >= self.min_snr_db:
                    # Проверяем расстояние до других пиков
                    if self._check_peak_distance(peak_idx, peaks, latest_tile['bin_hz']):
                        f_peak = f_start + peak_idx * latest_tile['bin_hz']
                        peaks.append(DetectedPeak(
                            id=f"peak_{f_peak:.0f}",
                            f_peak=f_peak,
                            snr_db=snr_db,
                            bin_hz=latest_tile['bin_hz'],
                            t0=latest_tile['t0'],
                            last_seen=latest_tile['t0'],
                            span_user=2e6,  # По умолчанию 2 МГц
                            status="ACTIVE"
                        ))
        
        return peaks
    
    def _find_local_maxima(self, powers: np.ndarray) -> List[int]:
        """Находит индексы локальных максимумов."""
        maxima = []
        for i in range(1, len(powers) - 1):
            if powers[i] > powers[i-1] and powers[i] > powers[i+1]:
                maxima.append(i)
        return maxima
    
    def _estimate_noise_level(self, powers: np.ndarray, peak_idx: int) -> float:
        """Оценивает уровень шума вокруг пика."""
        # Исключаем область пика и берем медиану
        start_idx = max(0, peak_idx - 5)
        end_idx = min(len(powers), peak_idx + 6)
        
        noise_powers = []
        for i in range(len(powers)):
            if i < start_idx or i >= end_idx:
                noise_powers.append(powers[i])
        
        if noise_powers:
            return np.median(noise_powers)
        return np.mean(powers)
    
    def _check_peak_distance(self, peak_idx: int, existing_peaks: List[DetectedPeak], bin_hz: float) -> bool:
        """Проверяет минимальное расстояние до существующих пиков."""
        for peak in existing_peaks:
            peak_bin = (peak.f_peak - peak.f_peak % bin_hz) / bin_hz
            if abs(peak_idx - peak_bin) < self.min_peak_distance_bins:
                return False
        return True


class MasterSweepController(QObject):
    """Контроллер Master sweep для управления HackRF и детекции пиков."""
    
    # Сигналы
    sweep_tile_received = pyqtSignal(object)  # SweepTile
    peak_detected = pyqtSignal(object)        # DetectedPeak
    sweep_error = pyqtSignal(str)             # Ошибка sweep
    # Внутренние сигналы для безопасной доставки данных из CFFI-потока
    _sweep_data_received = pyqtSignal(object)  # dict payload from C callback
    _peak_data_received = pyqtSignal(object)   # dict payload from C callback
    
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.log = logger
        
        # Проверяем доступность C библиотеки
        if not C_LIB_AVAILABLE:
            self.log.error("HackRF Master C library not available")
            self.sweep_source = None
        else:
            # НЕ инициализируем SDR здесь - только проверяем доступность
            self.sweep_source = None  # Будет инициализирован позже
            self.log.info("HackRF Master C library available (will be initialized when needed)")
        
        self.peak_detector = PeakDetector()
        self.is_running = False
        self._sdr_initialized = False  # Флаг инициализации SDR
        
        # Параметры sweep
        self.start_hz = 24e6      # 24 МГц
        self.stop_hz = 6e9        # 6 ГГц
        self.bin_hz = 200e3       # 200 кГц
        self.dwell_ms = 100       # 100 мс
        self.step_hz = 200e3      # Шаг sweep
        self.avg_count = 1        # Количество усреднений
        
        # Таймер для обновления детекции пиков — Master только обнаруживает и публикует пики
        self.peak_detection_timer = QTimer()
        self.peak_detection_timer.timeout.connect(self._update_peak_detection)
        self.peak_detection_timer.start(500)  # Каждые 500 мс
        
        # Таймер для опроса данных от C библиотеки (альтернатива callback)
        self.data_polling_timer = QTimer()
        self.data_polling_timer.timeout.connect(self._poll_data_from_c_library)
        self.data_polling_timer.start(50)  # Каждые 50 мс для быстрого отклика
        
        # Буфер для watchlist
        self.watchlist: Dict[str, DetectedPeak] = {}
        
        # Проксируем данные из CFFI-потока в GUI-поток через сигналы
        self._sweep_data_received.connect(self._on_sweep_tile_received)
        self._peak_data_received.connect(self._on_peak_received)
    
    def initialize_sdr(self):
        """Инициализирует SDR устройство. Вызывается при необходимости."""
        if self._sdr_initialized:
            self.log.info("SDR already initialized")
            return True
            
        if not C_LIB_AVAILABLE:
            self.log.error("HackRF Master C library not available")
            return False
            
        try:
            # Инициализируем C библиотеку
            self.sweep_source = HackRFMaster()
            self._sdr_initialized = True
            
            # Проверяем data reader
            if self.sweep_source.data_reader:
                self.log.info("Data reader is available for polling data")
            else:
                self.log.warning("Data reader is not available")
            
            self.log.info("HackRF Master C library initialized successfully")
            return True
            
        except Exception as e:
            self.log.error(f"Failed to initialize HackRF Master C library: {e}")
            self.sweep_source = None
            self._sdr_initialized = False
            return False
    
    def deinitialize_sdr(self):
        """Деинициализирует SDR устройство."""
        if not self._sdr_initialized:
            return
            
        try:
            # Останавливаем sweep если запущен
            if self.is_running:
                self.stop_sweep()
            
            # Очищаем ресурсы
            if self.sweep_source:
                self.sweep_source = None
            
            self._sdr_initialized = False
            self.log.info("HackRF Master C library deinitialized")
            
        except Exception as e:
            self.log.error(f"Error deinitializing SDR: {e}")
    
    def is_sdr_available(self):
        """Проверяет доступность SDR библиотеки."""
        return C_LIB_AVAILABLE
    
    def is_sdr_initialized(self):
        """Проверяет инициализирован ли SDR."""
        return self._sdr_initialized
    
    def _setup_callbacks(self):
        """Настраивает callback функции для C библиотеки (больше не используется)."""
        self.log.info("Callbacks are no longer used - using data reader polling instead")
        self.log.info("Data will be retrieved via data_reader.get_last_sweep_tile() every 50ms")
        
        # Проверяем что data reader доступен
        if self.sweep_source and self.sweep_source.data_reader:
            self.log.info("Data reader is available and ready for polling")
        else:
            self.log.warning("Data reader is not available - data polling may not work")
    
    def start_sweep(self, start_hz: float = None, stop_hz: float = None, 
                    bin_hz: float = None, dwell_ms: float = None):
        """Запускает sweep с заданными параметрами."""
        if not C_LIB_AVAILABLE:
            error_msg = "HackRF Master C library not available"
            self.log.error(error_msg)
            self.sweep_error.emit(error_msg)
            return
        
        # Принудительно инициализируем SDR
        self.log.info(f"Starting sweep - SDR initialized: {self._sdr_initialized}")
        if not self._sdr_initialized:
            self.log.info("SDR not initialized, initializing now...")
            if not self.initialize_sdr():
                error_msg = "Failed to initialize SDR"
                self.log.error(error_msg)
                self.sweep_error.emit(error_msg)
                return
            else:
                self.log.info("SDR initialized successfully")
        
        # Проверяем не запущен ли уже sweep
        if self.is_running:
            self.log.warning("Sweep already running. Stopping first...")
            self.stop_sweep()
            # Даем время на остановку
            time.sleep(0.1)
        
        if start_hz is not None:
            self.start_hz = start_hz
        if stop_hz is not None:
            self.stop_hz = stop_hz
        if bin_hz is not None:
            self.bin_hz = bin_hz
        if dwell_ms is not None:
            self.dwell_ms = dwell_ms
            
        self.log.info(f"Sweep parameters: {self.start_hz/1e6:.1f}-{self.stop_hz/1e6:.1f} MHz, "
                     f"bin: {self.bin_hz/1e3:.0f} kHz, dwell: {self.dwell_ms} ms")
        
        try:
            # Проверяем sweep_source
            if not self.sweep_source:
                error_msg = "Sweep source not available"
                self.log.error(error_msg)
                self.sweep_error.emit(error_msg)
                return
            
            self.log.info(f"Starting sweep via sweep_source: {self.sweep_source}")
            
            # Запускаем sweep через C библиотеку
            self.sweep_source.start_sweep(
                start_hz=self.start_hz,
                stop_hz=self.stop_hz,
                bin_hz=self.bin_hz,
                dwell_ms=self.dwell_ms,
                step_hz=self.step_hz,
                avg_count=self.avg_count,
                min_snr_db=10.0,  # Минимальный SNR для детекции
                min_peak_distance_bins=2  # Минимальное расстояние между пиками
            )
            
            self.is_running = True
            self.log.info(f"Master sweep started: {self.start_hz/1e6:.1f}-{self.stop_hz/1e6:.1f} MHz, "
                         f"bin: {self.bin_hz/1e3:.0f} kHz, dwell: {self.dwell_ms} ms")
            
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
            # Проверяем состояние через C библиотеку
            if self.sweep_source.is_running():
                self.sweep_source.stop_sweep()
                # Ждем полной остановки
                for _ in range(10):  # Максимум 1 секунда
                    if not self.sweep_source.is_running():
                        break
                    time.sleep(0.1)
            
            self.is_running = False
            self.log.info("Master sweep stopped")
            
        except Exception as e:
            self.log.error(f"Failed to stop sweep: {e}")
            # Принудительно сбрасываем флаг
            self.is_running = False
    
    def _on_sweep_tile_received(self, tile_data: Dict):
        """Обрабатывает sweep tile от C библиотеки."""
        try:
            # Детальное логирование для отладки
            self.log.info(f"Raw tile data received: {tile_data}")
            self.log.info(f"Tile data type: {type(tile_data)}")
            self.log.info(f"Tile data keys: {list(tile_data.keys()) if isinstance(tile_data, dict) else 'Not a dict'}")
            
            # Проверяем структуру данных
            if not isinstance(tile_data, dict):
                self.log.error(f"Invalid tile data type: {type(tile_data)}, expected dict")
                return
                
            required_keys = ['f_start', 'bin_hz', 'count', 'power', 't0']
            missing_keys = [key for key in required_keys if key not in tile_data]
            if missing_keys:
                self.log.error(f"Missing required keys in tile data: {missing_keys}")
                return
            
            # Проверяем типы данных
            if not isinstance(tile_data['power'], (list, tuple)):
                self.log.error(f"Invalid power data type: {type(tile_data['power'])}, expected list")
                return
                
            if not isinstance(tile_data['count'], int):
                self.log.error(f"Invalid count data type: {type(tile_data['count'])}, expected int")
                return
            
            # Создаем SweepTile
            tile = SweepTile(
                f_start=tile_data['f_start'],
                bin_hz=tile_data['bin_hz'],
                count=tile_data['count'],
                power=tile_data['power'],
                t0=tile_data['t0']
            )
            
            # Передаем плитку детектору
            self.peak_detector.add_sweep_tile(tile)
            
            # Эмитим сигнал (без проверки receivers)
            self.log.info(f"Emitting sweep_tile_received signal")
            self.sweep_tile_received.emit(tile)
            self.log.info(f"Sweep tile emitted to GUI: {tile.f_start/1e6:.1f} MHz")
            
        except Exception as e:
            self.log.error(f"Error processing sweep tile: {e}")
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}")
    
    def _on_peak_received(self, peak_data: Dict):
        """Обрабатывает peak данные от C библиотеки."""
        try:
            # Детальное логирование для отладки
            self.log.info(f"Raw peak data received: {peak_data}")
            self.log.info(f"Peak data type: {type(peak_data)}")
            self.log.info(f"Peak data keys: {list(peak_data.keys()) if isinstance(peak_data, dict) else 'Not a dict'}")
            
            # Проверяем структуру данных
            if not isinstance(peak_data, dict):
                self.log.error(f"Invalid peak data type: {type(peak_data)}, expected dict")
                return
                
            required_keys = ['f_peak', 'snr_db', 'bin_hz', 't0', 'status']
            missing_keys = [key for key in required_keys if key not in peak_data]
            if missing_keys:
                self.log.error(f"Missing required keys in peak data: {missing_keys}")
                return
            
            # Создаем DetectedPeak
            peak = DetectedPeak(
                id=f"peak_{peak_data['f_peak']:.0f}",
                f_peak=peak_data['f_peak'],
                snr_db=peak_data['snr_db'],
                bin_hz=peak_data['bin_hz'],
                t0=peak_data['t0'],
                last_seen=peak_data['t0'],
                span_user=2e6,  # По умолчанию 2 МГц
                status="ACTIVE"
            )
            
            # Эмитим сигнал (без проверки receivers)
            self.log.info(f"Emitting peak_detected signal")
            self.peak_detected.emit(peak)
            self.log.info(f"Peak detected and emitted to GUI: {peak.f_peak/1e6:.1f} MHz")
            
        except Exception as e:
            self.log.error(f"Error processing peak data: {e}")
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}")
    
    def _on_sweep_error(self, error_msg: str):
        """Обрабатывает ошибку от C библиотеки."""
        self.log.error(f"Sweep error from C library: {error_msg}")
        self.sweep_error.emit(error_msg)
    
    def _update_peak_detection(self):
        """Обновляет детекцию пиков (fallback для Python детектора)."""
        if not self.is_running or not self.sweep_source:
            return
            
        try:
            # Получаем новые пики от Python детектора (fallback)
            new_peaks = self.peak_detector.detect_peaks()
            
            # Обновляем watchlist
            current_time = time.time()
            
            for peak in new_peaks:
                if peak.id in self.watchlist:
                    # Обновляем существующий пик
                    existing = self.watchlist[peak.id]
                    existing.last_seen = current_time
                    existing.snr_db = peak.snr_db
                    
                    # Проверяем статус
                    if existing.snr_db < 3.0:  # Порог SNR
                        if current_time - existing.last_seen > 3.0:  # 3 секунды
                            existing.status = "QUIET"
                    else:
                        existing.status = "ACTIVE"
                else:
                    # Добавляем новый пик
                    self.watchlist[peak.id] = peak
                    self.peak_detected.emit(peak)
                    self.log.info(f"New peak detected by Python detector: {peak.f_peak/1e6:.1f} MHz, SNR: {peak.snr_db:.1f} dB")
            
            # Удаляем устаревшие пики (старше 30 секунд)
            expired_peaks = []
            for peak_id, peak in self.watchlist.items():
                if current_time - peak.last_seen > 30.0:
                    expired_peaks.append(peak_id)
            
            for peak_id in expired_peaks:
                del self.watchlist[peak_id]
                
        except Exception as e:
            self.log.error(f"Error in peak detection update: {e}")
    
    def get_watchlist(self) -> List[DetectedPeak]:
        """Возвращает текущий watchlist."""
        return list(self.watchlist.values())
    
    def set_peak_span(self, peak_id: str, span_hz: float):
        """Устанавливает пользовательскую ширину полосы для пика."""
        if peak_id in self.watchlist:
            self.watchlist[peak_id].span_user = span_hz
            self.log.info(f"Set span for peak {peak_id}: {span_hz/1e6:.1f} MHz")
    
    def get_sweep_status(self) -> Dict:
        """Возвращает статус sweep."""
        status = {
            'is_running': self.is_running,
            'start_hz': self.start_hz,
            'stop_hz': self.stop_hz,
            'bin_hz': self.bin_hz,
            'dwell_ms': self.dwell_ms,
            'watchlist_count': len(self.watchlist),
            'active_peaks': len([p for p in self.watchlist.values() if p.status == "ACTIVE"]),
            'c_library_available': C_LIB_AVAILABLE
        }
        
        # Добавляем статистику от C библиотеки если доступна
        if self.sweep_source and hasattr(self.sweep_source, 'get_stats'):
            try:
                c_stats = self.sweep_source.get_stats()
                status.update({
                    'c_sweep_count': c_stats.get('sweep_count', 0),
                    'c_peak_count': c_stats.get('peak_count', 0),
                    'c_last_sweep_time': c_stats.get('last_sweep_time', 0),
                    'c_avg_sweep_time': c_stats.get('avg_sweep_time', 0),
                    'c_error_count': c_stats.get('error_count', 0)
                })
            except Exception as e:
                self.log.error(f"Failed to get C library stats: {e}")
        
        return status
    
    def get_capabilities(self) -> Dict:
        """Возвращает возможности системы."""
        capabilities = {
            'c_library_available': C_LIB_AVAILABLE,
            'python_detector_available': True
        }
        
        if self.sweep_source and hasattr(self.sweep_source, 'get_frequency_range_min'):
            try:
                capabilities.update({
                    'frequency_range_min': self.sweep_source.get_frequency_range_min(),
                    'frequency_range_max': self.sweep_source.get_frequency_range_max(),
                    'max_bin_count': self.sweep_source.get_max_bin_count(),
                    'max_bandwidth': self.sweep_source.get_max_bandwidth()
                })
            except Exception as e:
                self.log.error(f"Failed to get C library capabilities: {e}")
        
        return capabilities
    
    def cleanup(self):
        """Очищает ресурсы."""
        try:
            # Останавливаем sweep если запущен
            if self.is_running:
                self.stop_sweep()
            
            # Останавливаем таймер
            if hasattr(self, 'peak_detection_timer'):
                self.peak_detection_timer.stop()
            
            # Очищаем C библиотеку
            if self.sweep_source:
                self.sweep_source.cleanup()
                self.sweep_source = None
            
            self.log.info("MasterSweepController cleanup completed")
            
        except Exception as e:
            self.log.error(f"Error during cleanup: {e}")

    def get_sdr_info(self):
        """Получает информацию об SDR устройствах без инициализации."""
        if not C_LIB_AVAILABLE:
            return {"error": "HackRF Master C library not available"}
        
        try:
            # Создаем временный объект для получения информации
            temp_sdr = HackRFMaster()
            info = temp_sdr.get_device_info()
            # Уничтожаем временный объект
            del temp_sdr
            return info
        except Exception as e:
            return {"error": f"Failed to get SDR info: {e}"}
    
    def enumerate_devices(self):
        """Сканирует доступные SDR устройства без инициализации."""
        if not C_LIB_AVAILABLE:
            return []
        
        try:
            # Если у нас уже есть инициализированный sweep_source, используем его
            if self.sweep_source and hasattr(self.sweep_source, 'enumerate_devices'):
                return self.sweep_source.enumerate_devices()
            
            # Иначе вызываем статический метод C библиотеки напрямую
            # БЕЗ создания нового экземпляра HackRFMaster
            from cffi import FFI
            ffi = FFI()
            
            # Минимальное определение для enumerate
            ffi.cdef("""
                typedef struct {
                    char serial[64];
                } hackrf_devinfo_t;
                
                int hackrf_master_enumerate(void* out_list, int max_count);
            """, override=True)
            
            # Загружаем библиотеку напрямую
            lib_path = Path(__file__).parent.parent.parent / "drivers" / "hackrf_master" / "build" / "libhackrf_master.so"
            if not lib_path.exists():
                return []
                
            lib = ffi.dlopen(str(lib_path))
            
            # Вызываем enumerate
            maxn = 16
            arr = ffi.new("hackrf_devinfo_t[]", maxn)
            n = int(lib.hackrf_master_enumerate(arr, maxn))
            
            devices = []
            for i in range(n):
                serial = ffi.string(arr[i].serial).decode("utf-8")
                if serial and serial != "default":
                    devices.append(serial)
            
            return devices
            
        except Exception as e:
            self.log.error(f"Failed to scan devices: {e}")
            return []

    def _poll_data_from_c_library(self):
        """Опрашивает C библиотеку на наличие новых данных (альтернатива callback)."""
        if not self._sdr_initialized or not self.sweep_source or not self.sweep_source.data_reader:
            return
            
        try:
            # Получаем sweep данные
            sweep_tile = self.sweep_source.data_reader.get_last_sweep_tile()
            if sweep_tile:
                self.log.info(f"Got sweep tile from data reader: {sweep_tile['f_start']/1e6:.1f} MHz")
                self._sweep_data_received.emit(sweep_tile)
            
            # Получаем peak данные
            peak_data = self.sweep_source.data_reader.get_last_peak()
            if peak_data:
                self.log.info(f"Got peak from data reader: {peak_data['f_peak']/1e6:.1f} MHz")
                self._peak_data_received.emit(peak_data)
            
            # Получаем error сообщения
            error_message = self.sweep_source.data_reader.get_last_error_message()
            if error_message:
                self.log.info(f"Got error from data reader: {error_message}")
                self.sweep_error.emit(error_message)
                
        except Exception as e:
            self.log.error(f"Error polling data from C library: {e}")
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}")
