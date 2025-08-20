"""
Core orchestrator for coordinating Master sweep and Slave SDR operations.
Manages measurement windows, RSSI collection, and trilateration triggers.
"""

import time
import logging
import threading
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QMutex
from collections import defaultdict, deque

from panorama.features.master_sweep.master import DetectedPeak
from panorama.features.slave_sdr.slave import RSSIMeasurement, SlaveManager, MeasurementWindow
from panorama.features.trilateration.engine import RSSITrilaterationEngine, TrilaterationResult


@dataclass
class MeasurementTask:
    """Задача измерения для slave."""
    id: str
    peak: DetectedPeak
    window: MeasurementWindow
    status: str  # PENDING, RUNNING, COMPLETED, FAILED
    created_at: float
    completed_at: Optional[float] = None
    measurements: List[RSSIMeasurement] = None
    
    def __post_init__(self):
        if self.measurements is None:
            self.measurements = []


class Orchestrator(QObject):
    """Основной оркестратор системы."""
    
    # Сигналы
    task_created = pyqtSignal(object)      # MeasurementTask
    task_completed = pyqtSignal(object)    # MeasurementTask
    task_failed = pyqtSignal(object)       # MeasurementTask
    target_detected = pyqtSignal(object)   # TrilaterationResult
    
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.log = logger
        
        # Компоненты системы
        self.master_controller = None
        self.slave_manager = None
        self.trilateration_engine = None
        
        # Состояние системы
        self.is_running = False
        self.auto_mode = True  # Автоматический режим с Master
        self.manual_mode = False  # Ручной режим без Master
        
        # Управление задачами
        self.tasks: Dict[str, MeasurementTask] = {}
        self.task_queue = deque()
        self.completed_tasks = deque(maxlen=100)
        
        # Параметры измерений
        self.global_span_hz = 2e6  # 2 МГц по умолчанию
        self.global_dwell_ms = 150  # 150 мс по умолчанию
        self.min_snr_threshold = 3.0  # Минимальный SNR для валидности
        
        # Синхронизация
        self._mutex = QMutex()
        self._measurement_barrier = None
        
        # Таймеры
        self.task_cleanup_timer = QTimer()
        self.task_cleanup_timer.timeout.connect(self._cleanup_old_tasks)
        self.task_cleanup_timer.start(5000)  # Каждые 5 секунд
        
        # Статистика
        self.stats = {
            'tasks_created': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'targets_detected': 0,
            'last_update': time.time()
        }
    
    def set_master_controller(self, master_controller):
        """Устанавливает контроллер Master."""
        self.master_controller = master_controller
        
        if self.master_controller:
            # Подключаем сигналы
            self.master_controller.peak_detected.connect(self._on_peak_detected)
            self.log.info("Master controller connected")
    
    def set_slave_manager(self, slave_manager: SlaveManager):
        """Устанавливает менеджер Slave."""
        self.slave_manager = slave_manager
        
        if self.slave_manager:
            # Подключаем сигналы
            self.slave_manager.all_measurements_complete.connect(self._on_all_measurements_complete)
            self.slave_manager.measurement_error.connect(self._on_measurement_error)
            self.log.info("Slave manager connected")
    
    def set_trilateration_engine(self, trilateration_engine: RSSITrilaterationEngine):
        """Устанавливает движок трилатерации."""
        self.trilateration_engine = trilateration_engine
        
        if self.trilateration_engine:
            # Подключаем сигналы
            self.trilateration_engine.target_update.connect(self._on_target_update)
            self.trilateration_engine.trilateration_error.connect(self._on_trilateration_error)
            self.log.info("Trilateration engine connected")
    
    def start(self):
        """Запускает оркестратор."""
        if self.is_running:
            return
        
        self.is_running = True
        self.log.info("Orchestrator started")
        
        # Запускаем обработку задач
        self._process_task_queue()
    
    def stop(self):
        """Останавливает оркестратор."""
        self.is_running = False
        self.log.info("Orchestrator stopped")
    
    def set_auto_mode(self, enabled: bool):
        """Устанавливает автоматический режим."""
        self.auto_mode = enabled
        self.manual_mode = not enabled
        
        if enabled:
            self.log.info("Auto mode enabled")
        else:
            self.log.info("Manual mode enabled")
    
    def set_global_parameters(self, span_hz: float, dwell_ms: int):
        """Устанавливает глобальные параметры измерений."""
        self.global_span_hz = span_hz
        self.global_dwell_ms = dwell_ms
        self.log.info(f"Global parameters: span={span_hz/1e6:.1f} MHz, dwell={dwell_ms} ms")
    
    def create_manual_measurement(self, center_hz: float, span_hz: float = None, dwell_ms: int = None):
        """Создает ручное измерение без Master."""
        if not self.slave_manager:
            self.log.error("Slave manager not available")
            return
        
        # Используем переданные параметры или глобальные
        span = span_hz if span_hz is not None else self.global_span_hz
        dwell = dwell_ms if dwell_ms is not None else self.global_dwell_ms
        
        # Создаем виртуальный пик
        virtual_peak = DetectedPeak(
            id=f"manual_{center_hz:.0f}",
            f_peak=center_hz,
            snr_db=20.0,  # Высокий SNR для ручного режима
            bin_hz=span,
            t0=time.time(),
            last_seen=time.time(),
            span_user=span,
            status="ACTIVE"
        )
        
        # Создаем задачу
        self._create_measurement_task(virtual_peak)
    
    def _on_peak_detected(self, peak: DetectedPeak):
        """Обрабатывает обнаружение нового пика от Master."""
        if not self.auto_mode:
            return
        
        self.log.info(f"Peak detected: {peak.f_peak/1e6:.1f} MHz, SNR: {peak.snr_db:.1f} dB")
        
        # Создаем задачу измерения
        self._create_measurement_task(peak)
    
    def _create_measurement_task(self, peak: DetectedPeak):
        """Создает новую задачу измерения."""
        try:
            self._mutex.lock()
            
            # Создаем окно измерения
            window = MeasurementWindow(
                center=peak.f_peak,
                span=peak.span_user,
                dwell_ms=self.global_dwell_ms,
                epoch=time.time()
            )
            
            # Создаем задачу
            task = MeasurementTask(
                id=f"task_{peak.id}_{int(time.time())}",
                peak=peak,
                window=window,
                status="PENDING",
                created_at=time.time()
            )
            
            # Добавляем в очередь
            self.tasks[task.id] = task
            self.task_queue.append(task.id)
            
            self.stats['tasks_created'] += 1
            self.stats['last_update'] = time.time()
            
            # Эмитим сигнал
            self.task_created.emit(task)
            
            self.log.info(f"Created measurement task: {task.id}")
            
        except Exception as e:
            self.log.error(f"Error creating measurement task: {e}")
        finally:
            self._mutex.unlock()
    
    def _process_task_queue(self):
        """Обрабатывает очередь задач."""
        if not self.is_running:
            return
        
        try:
            self._mutex.lock()
            
            # Берем задачи из очереди
            while self.task_queue and self.slave_manager:
                task_id = self.task_queue.popleft()
                task = self.tasks.get(task_id)
                
                if not task or task.status != "PENDING":
                    continue
                
                # Проверяем доступность slave
                if len(self.slave_manager.slaves) < 3:
                    self.log.warning(f"Insufficient slaves for task {task_id}: {len(self.slave_manager.slaves)} < 3")
                    task.status = "FAILED"
                    self.task_failed.emit(task)
                    continue
                
                # Запускаем измерение
                self._execute_measurement_task(task)
                
        except Exception as e:
            self.log.error(f"Error processing task queue: {e}")
        finally:
            self._mutex.unlock()
        
        # Планируем следующую обработку
        if self.is_running:
            QTimer.singleShot(100, self._process_task_queue)
    
    def _execute_measurement_task(self, task: MeasurementTask):
        """Выполняет задачу измерения."""
        try:
            # Обновляем статус
            task.status = "RUNNING"
            
            # Создаем окна для всех slave
            windows = [task.window]
            
            # Получаем калибровочные коэффициенты
            k_cal_db = {}
            for slave_id in self.slave_manager.slaves.keys():
                # TODO: Загружать из файла калибровки
                k_cal_db[slave_id] = 0.0
            
            # Запускаем измерение на всех slave
            success = self.slave_manager.measure_all_bands(windows, k_cal_db)
            
            if not success:
                task.status = "FAILED"
                self.task_failed.emit(task)
                self.stats['tasks_failed'] += 1
                return
            
            self.log.info(f"Started measurement for task: {task.id}")
            
        except Exception as e:
            self.log.error(f"Error executing measurement task {task.id}: {e}")
            task.status = "FAILED"
            self.task_failed.emit(task)
            self.stats['tasks_failed'] += 1
    
    def _on_all_measurements_complete(self, measurements: List[RSSIMeasurement]):
        """Обрабатывает завершение всех измерений."""
        try:
            # Группируем измерения по задачам
            measurements_by_task = self._group_measurements_by_task(measurements)
            
            for task_id, task_measurements in measurements_by_task.items():
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    task.measurements = task_measurements
                    task.status = "COMPLETED"
                    task.completed_at = time.time()
                    
                    # Проверяем валидность измерений
                    valid_measurements = self._validate_measurements(task_measurements)
                    
                    if len(valid_measurements) >= 3:
                        # Запускаем трилатерацию
                        self._trigger_trilateration(valid_measurements)
                    else:
                        self.log.warning(f"Task {task_id}: insufficient valid measurements ({len(valid_measurements)} < 3)")
                    
                    # Эмитим сигнал завершения
                    self.task_completed.emit(task)
                    self.stats['tasks_completed'] += 1
                    
                    # Перемещаем в завершенные
                    self.completed_tasks.append(task)
                    
        except Exception as e:
            self.log.error(f"Error processing completed measurements: {e}")
    
    def _group_measurements_by_task(self, measurements: List[RSSIMeasurement]) -> Dict[str, List[RSSIMeasurement]]:
        """Группирует измерения по задачам."""
        grouped = defaultdict(list)
        
        for measurement in measurements:
            # Ищем задачу по частоте и полосе
            for task_id, task in self.tasks.items():
                if (abs(measurement.center_hz - task.window.center) < 1e3 and  # 1 кГц допуск
                    abs(measurement.span_hz - task.window.span) < 1e3):
                    grouped[task_id].append(measurement)
                    break
        
        return grouped
    
    def _validate_measurements(self, measurements: List[RSSIMeasurement]) -> List[RSSIMeasurement]:
        """Валидирует RSSI измерения."""
        valid_measurements = []
        
        for measurement in measurements:
            # Проверяем SNR
            if measurement.snr_db < self.min_snr_threshold:
                continue
            
            # Проверяем флаги
            if measurement.flags.get('clip', False):
                continue
            
            if not measurement.flags.get('valid', True):
                continue
            
            valid_measurements.append(measurement)
        
        return valid_measurements
    
    def _trigger_trilateration(self, measurements: List[RSSIMeasurement]):
        """Запускает трилатерацию."""
        if not self.trilateration_engine:
            self.log.warning("Trilateration engine not available")
            return
        
        try:
            # Вычисляем позицию
            result = self.trilateration_engine.calculate_position(measurements)
            
            if result:
                self.log.info(f"Trilateration result: ({result.x:.1f}, {result.y:.1f}), confidence: {result.confidence:.2f}")
                self.stats['targets_detected'] += 1
            else:
                self.log.warning("Trilateration failed")
                
        except Exception as e:
            self.log.error(f"Error in trilateration: {e}")
    
    def _on_measurement_error(self, error_msg: str):
        """Обрабатывает ошибку измерения."""
        self.log.error(f"Measurement error: {error_msg}")
    
    def _on_target_update(self, result: TrilaterationResult):
        """Обрабатывает обновление цели."""
        self.target_detected.emit(result)
    
    def _on_trilateration_error(self, error_msg: str):
        """Обрабатывает ошибку трилатерации."""
        self.log.error(f"Trilateration error: {error_msg}")
    
    def _cleanup_old_tasks(self):
        """Очищает старые завершенные задачи."""
        try:
            current_time = time.time()
            cutoff_time = current_time - 300  # 5 минут
            
            # Удаляем старые задачи
            old_task_ids = []
            for task_id, task in self.tasks.items():
                if (task.status in ["COMPLETED", "FAILED"] and 
                    task.completed_at and 
                    task.completed_at < cutoff_time):
                    old_task_ids.append(task_id)
            
            for task_id in old_task_ids:
                del self.tasks[task_id]
            
            if old_task_ids:
                self.log.info(f"Cleaned up {len(old_task_ids)} old tasks")
                
        except Exception as e:
            self.log.error(f"Error cleaning up old tasks: {e}")
    
    def get_task_status(self) -> Dict:
        """Возвращает статус задач."""
        try:
            self._mutex.lock()
            
            status = {
                'total_tasks': len(self.tasks),
                'pending_tasks': len([t for t in self.tasks.values() if t.status == "PENDING"]),
                'running_tasks': len([t for t in self.tasks.values() if t.status == "RUNNING"]),
                'completed_tasks': len([t for t in self.tasks.values() if t.status == "COMPLETED"]),
                'failed_tasks': len([t for t in self.tasks.values() if t.status == "FAILED"]),
                'queue_length': len(self.task_queue),
                'completed_history': len(self.completed_tasks)
            }
            
            return status
            
        finally:
            self._mutex.unlock()
    
    def get_system_status(self) -> Dict:
        """Возвращает общий статус системы."""
        return {
            'is_running': self.is_running,
            'auto_mode': self.auto_mode,
            'manual_mode': self.manual_mode,
            'global_span_hz': self.global_span_hz,
            'global_dwell_ms': self.global_dwell_ms,
            'min_snr_threshold': self.min_snr_threshold,
            'stats': self.stats.copy(),
            'master_connected': self.master_controller is not None,
            'slave_connected': self.slave_manager is not None,
            'trilateration_connected': self.trilateration_engine is not None
        }
    
    def get_active_tasks(self) -> List[MeasurementTask]:
        """Возвращает активные задачи."""
        try:
            self._mutex.lock()
            return [task for task in self.tasks.values() if task.status in ["PENDING", "RUNNING"]]
        finally:
            self._mutex.unlock()
    
    def cancel_task(self, task_id: str) -> bool:
        """Отменяет задачу."""
        try:
            self._mutex.lock()
            
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status in ["PENDING", "RUNNING"]:
                    task.status = "FAILED"
                    self.task_failed.emit(task)
                    self.log.info(f"Cancelled task: {task_id}")
                    return True
            
            return False
            
        finally:
            self._mutex.unlock()
