"""
Slave SDR controller for RSSI measurements and band power analysis.
Supports various SDR devices through SoapySDR interface.
"""

import os
# ОТКЛЮЧАЕМ AVAHI В SOAPYSDR ДО ВСЕХ ИМПОРТОВ
# Это предотвращает ошибки "avahi_service_browser_new() failed: Bad state"
os.environ['SOAPY_SDR_DISABLE_AVAHI'] = '1'

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex
import threading

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False
    SoapySDR = None


@dataclass
class RSSIMeasurement:
    """Результат измерения RSSI для полосы частот."""
    slave_id: str
    center_hz: float      # Центральная частота (Гц)
    span_hz: float        # Ширина полосы (Гц)
    band_rssi_dbm: float  # RSSI полосы (дБм)
    band_noise_dbm: float # Уровень шума (дБм)
    snr_db: float         # SNR (дБ)
    n_samples: int        # Количество сэмплов
    ts: float             # Временная метка
    flags: Dict           # Флаги (clip, agc, etc.)


@dataclass
class MeasurementWindow:
    """Окно измерения для slave."""
    center: float         # Центральная частота (Гц)
    span: float           # Ширина полосы (Гц)
    dwell_ms: int         # Время измерения (мс)
    epoch: float          # Эпоха измерения


class SlaveSDR(QObject):
    """Slave SDR для измерения RSSI в заданной полосе частот."""
    
    # Сигналы
    measurement_complete = pyqtSignal(object)  # RSSIMeasurement
    measurement_error = pyqtSignal(str)        # Ошибка измерения
    status_changed = pyqtSignal(str)          # Изменение статуса
    
    def __init__(self, slave_id: str, uri: str, logger: logging.Logger):
        super().__init__()
        self.slave_id = slave_id
        self.uri = uri
        self.log = logger
        
        # SDR устройство
        self.sdr = None
        self.rx_stream = None
        self.is_initialized = False
        
        # Параметры
        self.sample_rate = 8e6        # 8 Мс/с по умолчанию
        self.gain = 20.0              # Усиление (дБ)
        self.frequency = 2.4e9        # Частота (Гц)
        self.bandwidth = 2.5e6        # Полоса (Гц)
        
        # Калибровка
        self.k_cal_db = 0.0           # Калибровочный коэффициент
        
        # Состояние
        self.is_measuring = False
        self.current_window = None
        
        # Мьютекс для потокобезопасности
        self._mutex = QMutex()
        
        # Инициализация SDR
        self._init_sdr()
    
    def _init_sdr(self):
        """Инициализирует SDR устройство."""
        if not SOAPY_AVAILABLE:
            self.log.error("SoapySDR not available")
            return
            
        try:
            self.sdr = SoapySDR.Device(self.uri)
            
            # Настройка RX
            self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
            self.sdr.setGain(SOAPY_SDR_RX, 0, self.gain)
            self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.frequency)
            self.sdr.setBandwidth(SOAPY_SDR_RX, 0, self.bandwidth)
            
            # Создание RX потока
            self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
            self.sdr.activateStream(self.rx_stream)
            
            self.is_initialized = True
            self.status_changed.emit("READY")
            self.log.info(f"Slave {self.slave_id} initialized: {self.uri}")
            
        except Exception as e:
            self.log.error(f"Failed to initialize slave {self.slave_id}: {e}")
            self.status_changed.emit("ERROR")
    
    def measure_band_rssi(self, center_hz: float, span_hz: float, 
                          dwell_ms: int, k_cal_db: float = None) -> RSSIMeasurement:
        """Измеряет RSSI в заданной полосе частот."""
        if not self.is_initialized:
            raise RuntimeError(f"Slave {self.slave_id} not initialized")
        
        if self.is_measuring:
            raise RuntimeError(f"Slave {self.slave_id} already measuring")
        
        # Обновляем калибровку
        if k_cal_db is not None:
            self.k_cal_db = k_cal_db
        
        # Проверяем параметры
        if span_hz > self.sample_rate * 0.8:
            raise ValueError(f"Span {span_hz/1e6:.1f} MHz too large for sample rate {self.sample_rate/1e6:.1f} MHz")
        
        try:
            self._mutex.lock()
            self.is_measuring = True
            self.current_window = MeasurementWindow(
                center=center_hz,
                span=span_hz,
                dwell_ms=dwell_ms,
                epoch=time.time()
            )
            self._mutex.unlock()
            
            # Настраиваем SDR на центральную частоту
            self.sdr.setFrequency(SOAPY_SDR_RX, 0, center_hz)
            # Обеспечим полосу окна не больше доступной
            if span_hz > self.bandwidth:
                self.bandwidth = min(self.sample_rate * 0.8, span_hz)
                try:
                    self.sdr.setBandwidth(SOAPY_SDR_RX, 0, self.bandwidth)
                except Exception:
                    pass
            
            # Вычисляем количество сэмплов
            n_samples = int(self.sample_rate * dwell_ms / 1000)
            
            # Читаем IQ данные
            iq_data = self._read_iq_samples(n_samples)
            
            # Вычисляем PSD и RSSI (среднеквадратичный в диапазоне)
            measurement = self._calculate_rssi(iq_data, center_hz, span_hz, n_samples)
            
            # Сбрасываем состояние
            self._mutex.lock()
            self.is_measuring = False
            self.current_window = None
            self._mutex.unlock()
            
            # Эмитим результат
            self.measurement_complete.emit(measurement)
            
            return measurement
            
        except Exception as e:
            self._mutex.lock()
            self.is_measuring = False
            self.current_window = None
            self._mutex.unlock()
            
            error_msg = f"Measurement failed on slave {self.slave_id}: {e}"
            self.log.error(error_msg)
            self.measurement_error.emit(error_msg)
            raise
    
    def _read_iq_samples(self, n_samples: int) -> np.ndarray:
        """Читает IQ сэмплы из SDR."""
        buffer = np.empty(n_samples, dtype=np.complex64)
        samples_read = 0
        
        while samples_read < n_samples:
            try:
                # Читаем данные
                result = self.sdr.readStream(self.rx_stream, buffer[samples_read:], timeoutUs=1000000)
                
                if result.ret > 0:
                    samples_read += result.ret
                elif result.ret == -1:  # SOAPY_SDR_TIMEOUT
                    self.log.warning(f"Timeout reading from slave {self.slave_id}")
                    break
                else:
                    self.log.error(f"Error reading from slave {self.slave_id}: {result.ret}")
                    break
                    
            except Exception as e:
                self.log.error(f"Exception reading from slave {self.slave_id}: {e}")
                break
        
        if samples_read < n_samples:
            # Заполняем оставшиеся сэмплы нулями
            buffer[samples_read:] = 0
        
        return buffer
    
    def _calculate_rssi(self, iq_data: np.ndarray, center_hz: float, 
                        span_hz: float, n_samples: int) -> RSSIMeasurement:
        """Вычисляет RSSI для заданной полосы частот."""
        try:
            # Вычисляем PSD (Welch метод)
            f, psd = self._welch_psd(iq_data, self.sample_rate)
            
            # Находим индексы для заданной полосы
            span_half = span_hz / 2
            band_mask = (f >= center_hz - span_half) & (f <= center_hz + span_half)
            
            if not np.any(band_mask):
                raise ValueError("No frequency bins in specified band")
            
            # Вычисляем мощность в полосе
            band_power = np.mean(psd[band_mask])
            band_power_linear = 10**(band_power / 10)  # Переводим в линейную шкалу
            
            # Оцениваем уровень шума (соседние частоты)
            noise_mask = ~band_mask
            if np.any(noise_mask):
                noise_power = np.median(psd[noise_mask])
                noise_power_linear = 10**(noise_power / 10)
            else:
                noise_power = band_power - 20  # Примерная оценка
                noise_power_linear = 10**(noise_power / 10)
            
            # Вычисляем SNR
            if noise_power_linear > 0:
                snr_linear = band_power_linear / noise_power_linear
                snr_db = 10 * np.log10(snr_linear)
            else:
                snr_db = 0
            
            # Применяем калибровку
            band_rssi_dbm = 10 * np.log10(band_power_linear) + self.k_cal_db
            band_noise_dbm = 10 * np.log10(noise_power_linear) + self.k_cal_db
            
            # Проверяем на клиппинг
            iq_magnitude = np.abs(iq_data)
            clip_threshold = 0.9 * np.max(iq_magnitude)
            is_clipped = np.any(iq_magnitude > clip_threshold)
            
            # Создаем результат
            measurement = RSSIMeasurement(
                slave_id=self.slave_id,
                center_hz=center_hz,
                span_hz=span_hz,
                band_rssi_dbm=band_rssi_dbm,
                band_noise_dbm=band_noise_dbm,
                snr_db=snr_db,
                n_samples=n_samples,
                ts=time.time(),
                flags={
                    'clip': is_clipped,
                    'agc': False,  # TODO: добавить поддержку AGC
                    'valid': snr_db >= 3.0
                }
            )
            
            return measurement
            
        except Exception as e:
            self.log.error(f"Error calculating RSSI: {e}")
            raise
    
    def _welch_psd(self, data: np.ndarray, fs: float, 
                   nperseg: int = None, noverlap: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Вычисляет PSD методом Уэлча."""
        if nperseg is None:
            nperseg = min(256, len(data) // 4)
        if noverlap is None:
            noverlap = nperseg // 2
        
        # Простая реализация PSD (можно заменить на scipy.signal.welch)
        n_fft = nperseg
        hop = nperseg - noverlap
        
        psd_sum = np.zeros(n_fft // 2 + 1)
        n_windows = 0
        
        for i in range(0, len(data) - nperseg + 1, hop):
            window = data[i:i + n_fft]
            # Применяем окно Хэмминга
            window = window * np.hamming(n_fft)
            
            # FFT
            fft_result = np.fft.fft(window, n_fft)
            psd = np.abs(fft_result[:n_fft//2 + 1])**2
            
            psd_sum += psd
            n_windows += 1
        
        if n_windows > 0:
            psd_avg = psd_sum / n_windows
        else:
            psd_avg = psd_sum
        
        # Нормализация и перевод в дБ
        psd_db = 10 * np.log10(psd_avg + 1e-12)
        
        # Частотная ось
        freqs = np.fft.fftfreq(n_fft, 1/fs)[:n_fft//2 + 1]
        
        return freqs, psd_db
    
    def get_capabilities(self) -> Dict:
        """Возвращает возможности SDR."""
        if not self.is_initialized:
            return {}
        
        try:
            # Получаем диапазоны частот
            freq_range = self.sdr.getFrequencyRange(SOAPY_SDR_RX, 0)
            sample_rates = self.sdr.listSampleRates(SOAPY_SDR_RX, 0)
            gains = self.sdr.listGains(SOAPY_SDR_RX, 0)
            
            return {
                'frequency_range': freq_range,
                'sample_rates': sample_rates,
                'gains': gains,
                'max_bandwidth': self.sample_rate * 0.8,
                'uri': self.uri
            }
        except Exception as e:
            self.log.error(f"Error getting capabilities: {e}")
            return {}
    
    def get_status(self) -> Dict:
        """Возвращает текущий статус slave."""
        return {
            'slave_id': self.slave_id,
            'is_initialized': self.is_initialized,
            'is_measuring': self.is_measuring,
            'frequency': self.frequency,
            'sample_rate': self.sample_rate,
            'gain': self.gain,
            'k_cal_db': self.k_cal_db,
            'current_window': self.current_window
        }
    
    def close(self):
        """Закрывает SDR соединение."""
        try:
            if self.rx_stream:
                self.sdr.deactivateStream(self.rx_stream)
                self.sdr.closeStream(self.rx_stream)
            
            if self.sdr:
                self.sdr = None
            
            self.is_initialized = False
            self.log.info(f"Slave {self.slave_id} closed")
            
        except Exception as e:
            self.log.error(f"Error closing slave {self.slave_id}: {e}")


class SlaveManager(QObject):
    """Менеджер для управления множественными slave SDR."""
    
    # Сигналы
    all_measurements_complete = pyqtSignal(list)  # Список RSSIMeasurement
    measurement_error = pyqtSignal(str)           # Ошибка измерения
    
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.log = logger
        self.slaves: Dict[str, SlaveSDR] = {}
        # Синхронизация измерений
        self.measurement_barrier: Optional[threading.Barrier] = None
        self._results: List[RSSIMeasurement] = []
        self._results_lock = threading.Lock()
        self._expected_count: int = 0
        
    def add_slave(self, slave_id: str, uri: str) -> bool:
        """Добавляет новый slave SDR."""
        try:
            slave = SlaveSDR(slave_id, uri, self.log)
            self.slaves[slave_id] = slave
            
            # Подключаем сигналы
            slave.measurement_complete.connect(self._on_measurement_complete)
            slave.measurement_error.connect(self._on_measurement_error)
            
            if not slave.is_initialized:
                # Инициализация не удалась — очищаем и сообщаем об ошибке
                try:
                    slave.close()
                except Exception:
                    pass
                del self.slaves[slave_id]
                self.log.error(f"Add slave failed (not initialized): {slave_id} ({uri})")
                return False

            self.log.info(f"Added slave: {slave_id} ({uri})")
            return True
            
        except Exception as e:
            self.log.error(f"Failed to add slave {slave_id}: {e}")
            return False
    
    def remove_slave(self, slave_id: str):
        """Удаляет slave SDR."""
        if slave_id in self.slaves:
            try:
                self.slaves[slave_id].close()
                del self.slaves[slave_id]
                self.log.info(f"Removed slave: {slave_id}")
            except Exception as e:
                self.log.error(f"Error removing slave {slave_id}: {e}")
    
    def measure_all_bands(self, windows: List[MeasurementWindow], 
                          k_cal_db: Dict[str, float]) -> bool:
        """Асинхронно запускает измерение на всех slave одновременно (не блокирует GUI)."""
        if not self.slaves:
            self.log.warning("No slaves available")
            return False
        
        try:
            active = [s for s in self.slaves.values() if s.is_initialized]
            if not active:
                self.log.warning("No initialized slaves available")
                return False

            task_count = len(active) * len(windows)
            self.measurement_barrier = threading.Barrier(task_count)
            with self._results_lock:
                self._results = []
                self._expected_count = task_count

            threads: List[threading.Thread] = []
            for slave in active:
                k_cal = k_cal_db.get(slave.slave_id, 0.0)
                for window in windows:
                    t = threading.Thread(target=self._measure_slave_band, args=(slave, window, k_cal), daemon=True)
                    t.start()
                    threads.append(t)

            # Фоновая присмотр за завершением — эмитим всё, что получили
            def _wait_and_emit():
                for t in threads:
                    t.join()
                with self._results_lock:
                    results_copy = list(self._results)
                self.all_measurements_complete.emit(results_copy)
                self.log.info(f"Completed measurements: {len(results_copy)}/{self._expected_count}")

            threading.Thread(target=_wait_and_emit, daemon=True).start()
            return True
            
        except Exception as e:
            self.log.error(f"Error in measure_all_bands: {e}")
            return False
    
    def _measure_slave_band(self, slave: SlaveSDR, window: MeasurementWindow, k_cal_db: float):
        """Измеряет полосу на конкретном slave."""
        try:
            # Ждем барьера для синхронного старта всех задач
            if self.measurement_barrier is not None:
                self.measurement_barrier.wait()
            
            # Выполняем измерение
            measurement = slave.measure_band_rssi(
                center_hz=window.center,
                span_hz=window.span,
                dwell_ms=window.dwell_ms,
                k_cal_db=k_cal_db
            )
            # Сохраняем результат и, при необходимости, эмитим общий сигнал
            with self._results_lock:
                self._results.append(measurement)
                if self._expected_count > 0 and len(self._results) >= self._expected_count:
                    results_copy = list(self._results)
                    self.all_measurements_complete.emit(results_copy)
        except Exception as e:
            self.log.error(f"Error measuring on slave {slave.slave_id}: {e}")
    
    def _on_measurement_complete(self, measurement: RSSIMeasurement):
        """Агрегирует результаты от слейвов и эмитит общий список при полном наборе."""
        with self._results_lock:
            self._results.append(measurement)
            if self._expected_count > 0 and len(self._results) >= self._expected_count:
                results_copy = list(self._results)
                self.all_measurements_complete.emit(results_copy)
    
    def _on_measurement_error(self, error_msg: str):
        """Обрабатывает ошибку измерения."""
        self.measurement_error.emit(error_msg)
    
    def get_slave_status(self) -> Dict[str, Dict]:
        """Возвращает статус всех slave."""
        return {slave_id: slave.get_status() for slave_id, slave in self.slaves.items()}
    
    def enumerate_soapy_devices(self) -> List[Dict[str, str]]:
        """Перечисляет доступные SDR устройства через SoapySDR.
        Возвращает список словарей с ключами: 'driver', 'serial', 'label', 'uri'.
        """
        devices: List[Dict[str, str]] = []
        if not SOAPY_AVAILABLE:
            return devices
        try:
            # Отключаем Avahi для предотвращения ошибок при поиске сетевых устройств
            import os
            old_env = os.environ.get('SOAPY_SDR_DISABLE_AVAHI')
            os.environ['SOAPY_SDR_DISABLE_AVAHI'] = '1'
            
            results = SoapySDR.Device.enumerate()
            
            # Восстанавливаем старое значение
            if old_env is not None:
                os.environ['SOAPY_SDR_DISABLE_AVAHI'] = old_env
            else:
                del os.environ['SOAPY_SDR_DISABLE_AVAHI']
            
            for info in results:
                drv = info.get('driver', '') if hasattr(info, 'get') else ''
                ser = info.get('serial', '') if hasattr(info, 'get') else ''
                label = info.get('label', '') if hasattr(info, 'get') else ''
                
                # Пропускаем пустые или некорректные устройства
                if not drv or drv == '':
                    continue
                
                # Пропускаем аудио устройства (не SDR)
                if drv.lower() in ['audio', 'pulse', 'alsa', 'jack']:
                    continue
                
                # НЕ пропускаем устройства без серийного номера - HackRF может не иметь серийника в некоторых случаях
                # if not ser:
                #     continue
                
                # Сформируем URI/args строку
                parts = []
                if drv:
                    parts.append(f"driver={drv}")
                if ser:
                    parts.append(f"serial={ser}")
                uri = ",".join(parts) if parts else drv
                
                devices.append({
                    'driver': drv,
                    'serial': ser,
                    'label': label,
                    'uri': uri or (f"driver={drv}" if drv else ''),
                })
        except Exception as e:
            print(f"Error enumerating SoapySDR devices: {e}")
            return devices
        return devices
    
    def close_all(self):
        """Закрывает все slave соединения."""
        for slave in self.slaves.values():
            try:
                slave.close()
            except Exception as e:
                self.log.error(f"Error closing slave: {e}")
        
        self.slaves.clear()
        self.log.info("All slaves closed")
