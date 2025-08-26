# -*- coding: utf-8 -*-
"""
Core orchestrator: формирует окна измерений для slaves по событиям Master,
собирает RMS RSSI и запускает трилатерацию.
"""

from __future__ import annotations
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque, defaultdict

from PyQt5.QtCore import QObject, pyqtSignal, QMutex, QTimer

from panorama.features.spectrum.master import MasterSweepController
from panorama.features.spectrum.model import DetectedPeak
from panorama.features.slave_sdr.slave import (
    RSSIMeasurement,
    MeasurementWindow,
    SlaveManager,
)
from panorama.features.trilateration.engine import (
    RSSITrilaterationEngine,
    TrilaterationResult,
)


@dataclass
class MeasurementTask:
    id: str
    peak: DetectedPeak
    window: MeasurementWindow
    status: str  # PENDING|RUNNING|COMPLETED|FAILED|CANCELLED
    created_at: float
    completed_at: Optional[float] = None
    measurements: List[RSSIMeasurement] = None

    def __post_init__(self):
        if self.measurements is None:
            self.measurements = []


class Orchestrator(QObject):
    # события для UI
    task_created = pyqtSignal(object)     # MeasurementTask
    task_completed = pyqtSignal(object)   # MeasurementTask
    task_failed = pyqtSignal(object)      # MeasurementTask
    target_update = pyqtSignal(object)    # TrilaterationResult

    # алиас на старое имя, которое ожидает UI
    target_detected = pyqtSignal(object)  # TrilaterationResult

    # статус/ошибки для статус-бара
    status_changed = pyqtSignal(dict)     # dict от get_system_status()
    error = pyqtSignal(str)

    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.log = logger

        self.master_controller: Optional[MasterSweepController] = None
        self.slave_manager: Optional[SlaveManager] = None
        self.trilateration_engine: Optional[RSSITrilaterationEngine] = None

        self.is_running = True
        self.auto_mode = True
        self.manual_mode = False

        # ширина окна ±W по умолчанию (т.е. span_hz — полная ширина)
        self.global_span_hz = 5e6
        self.global_dwell_ms = 150
        self.min_snr_threshold = 3.0

        self.tasks: Dict[str, MeasurementTask] = {}
        self.queue: deque[str] = deque()
        self.completed: deque[MeasurementTask] = deque(maxlen=100)

        self._mutex = QMutex()

        self._cleanup_timer = QTimer(self)
        self._cleanup_timer.timeout.connect(self._cleanup_old)
        self._cleanup_timer.start(5000)

        # прокинем alias-сигнал
        self.target_update.connect(self._emit_target_detected)

        # сообщим UI стартовый статус
        self._emit_status()

    # ───────────────────────── wiring ─────────────────────────

    def set_master_controller(self, master: MasterSweepController):
        self.master_controller = master
        if master:
            master.peak_detected.connect(self._on_peak_detected)
            self.log.info("Master controller connected")
            self._emit_status()

    def set_slave_manager(self, sm: SlaveManager):
        self.slave_manager = sm
        if sm:
            sm.all_measurements_complete.connect(self._on_all_measurements_complete)
            sm.measurement_error.connect(self._on_measurement_error)
            self.log.info("Slave manager connected")
            self._emit_status()

    def set_trilateration_engine(self, eng: RSSITrilaterationEngine):
        self.trilateration_engine = eng
        if eng:
            eng.target_update.connect(self._on_target_update)
            eng.trilateration_error.connect(self._on_trilateration_error)
            self.log.info("Trilateration engine connected")
            self._emit_status()

    # ─────────────────────── public API (UI) ───────────────────────

    def start(self):
        """Включает автопроцесс постановки задач (мастер ⇒ слейвы ⇒ трилатерация)."""
        self.is_running = True
        self.auto_mode = True
        self.manual_mode = False
        self.log.info("Orchestrator started (auto mode)")
        self._emit_status()

    def stop(self):
        """Останавливает автоматическую постановку задач (очередь не чистится)."""
        self.is_running = False
        self.log.info("Orchestrator stopped")
        self._emit_status()

    def shutdown(self):
        """Полная остановка и очистка очереди."""
        self.stop()
        try:
            # отмена запланированного
            while self.queue:
                tid = self.queue.popleft()
                t = self.tasks.get(tid)
                if t and t.status == "PENDING":
                    t.status = "CANCELLED"
            self._emit_status()
        except Exception as e:
            self._emit_error(f"Shutdown error: {e}")

    def is_active(self) -> bool:
        return bool(self.is_running)

    def set_auto_mode(self, enabled: bool):
        self.auto_mode = bool(enabled)
        self.manual_mode = not self.auto_mode
        self.log.info("Auto mode enabled" if self.auto_mode else "Manual mode enabled")
        self._emit_status()

    def set_global_parameters(self, span_hz: float, dwell_ms: int):
        self.global_span_hz = float(span_hz)
        self.global_dwell_ms = int(dwell_ms)
        self.log.info(f"Global parameters: span={span_hz/1e6:.2f} MHz, dwell={dwell_ms} ms")
        self._emit_status()

    def create_manual_measurement(self, center_hz: float,
                                  span_hz: Optional[float] = None,
                                  dwell_ms: Optional[int] = None):
        """
        Ручная постановка окна без участия мастера.
        """
        if not self.slave_manager:
            self._emit_error("Slave manager not available")
            return
        span = float(span_hz) if span_hz is not None else self.global_span_hz
        dwell = int(dwell_ms) if dwell_ms is not None else self.global_dwell_ms
        peak = DetectedPeak.from_peak(center_hz, snr_db=20.0, span_hz=span)  # высокий SNR для ручного режима
        self._enqueue_task(peak, span, dwell)

    # ─────────────────────── master → orchestrator ───────────────────────

    def _on_peak_detected(self, peak: DetectedPeak):
        """
        Получаем пик от Master. Формируем окно ±W (W = span/2) и ставим измерение на slaves.
        """
        if not self.is_running or not self.auto_mode:
            return
        self._enqueue_task(peak, self.global_span_hz, self.global_dwell_ms)

    # ─────────────────────── постановка и запуск ───────────────────────

    def _enqueue_task(self, peak: DetectedPeak, span_hz: float, dwell_ms: int):
        if not self.slave_manager:
            self.log.warning("No slave manager: skip task")
            self._emit_error("No slave manager: skip task")
            return
        center = float(peak.f_peak)
        window = MeasurementWindow(center=center, span=span_hz, dwell_ms=dwell_ms, epoch=time.time())
        task_id = f"{int(center)}_{int(span_hz)}_{int(window.epoch)}"
        task = MeasurementTask(
            id=task_id, peak=peak, window=window, status="PENDING",
            created_at=time.time()
        )
        self.tasks[task_id] = task
        self.queue.append(task_id)
        self.task_created.emit(task)
        self.log.info(f"Task queued: f={center/1e6:.3f} MHz, span={span_hz/1e6:.3f} MHz, dwell={dwell_ms} ms")
        self._emit_status()
        self._process_queue()

    def _process_queue(self):
        if not self.queue or not self.slave_manager:
            return
        # Берём одну задачу и запускаем параллельно всем активным slave.
        task_id = self.queue.popleft()
        task = self.tasks.get(task_id)
        if not task:
            return
        task.status = "RUNNING"
        ok = False
        try:
            ok = self.slave_manager.measure_all_bands(
                windows=[task.window],
                k_cal_db={},  # при необходимости: словарь калибровок по каждому slave
            )
        except Exception as e:
            self._emit_error(f"Failed to start measurement on slaves: {e}")

        if not ok:
            task.status = "FAILED"
            self.task_failed.emit(task)
            self.log.error("Failed to start measurement on slaves")
        self._emit_status()

    # ─────────────────────── slave → orchestrator ───────────────────────

    def _on_all_measurements_complete(self, results: List[RSSIMeasurement]):
        try:
            grouped: Dict[str, List[RSSIMeasurement]] = defaultdict(list)
            for r in results:
                # сопоставляем с задачей по (center, span) в пределах 1 кГц
                for tid, t in self.tasks.items():
                    if (abs(r.center_hz - t.window.center) < 1e3 and
                            abs(r.span_hz - t.window.span) < 1e3 and
                            t.status == "RUNNING"):
                        grouped[tid].append(r)
                        break

            for tid, meas_list in grouped.items():
                task = self.tasks.get(tid)
                if not task:
                    continue
                task.measurements = meas_list
                task.status = "COMPLETED"
                task.completed_at = time.time()
                self.task_completed.emit(task)
                self.completed.append(task)
                self.log.info(f"Task completed: {tid} with {len(meas_list)} measurements")

                # триггерим трилатерацию только если >= 3 slaves отдали валидные данные
                valid = [m for m in meas_list if m.snr_db >= self.min_snr_threshold and m.flags.get("valid", True)]
                if len(valid) >= 3 and self.trilateration_engine:
                    try:
                        result: Optional[TrilaterationResult] = self.trilateration_engine.calculate_position(valid)
                        if result:
                            self.target_update.emit(result)  # alias уйдёт автоматически
                            self.log.info(
                                f"Trilateration: ({result.x:.1f},{result.y:.1f}) conf={result.confidence:.2f}"
                            )
                    except Exception as e:
                        self._emit_error(f"Trilateration error: {e}")

            # после обработки — запускаем следующую задачу, если есть
            self._process_queue()
            self._emit_status()

        except Exception as e:
            self._emit_error(f"Error in _on_all_measurements_complete: {e}")

    def _on_measurement_error(self, msg: str):
        self._emit_error(f"Measurement error: {msg}")

    # ─────────────────────── trilateration → orchestrator ───────────────────────

    def _on_target_update(self, result: TrilaterationResult):
        # дублируем наружу (UI слушает target_update, а также alias target_detected)
        self.target_update.emit(result)

    def _on_trilateration_error(self, msg: str):
        self._emit_error(f"Trilateration error: {msg}")

    # ───────────────────────── service ─────────────────────────

    def _cleanup_old(self):
        try:
            now = time.time()
            drop: List[str] = []
            for tid, t in self.tasks.items():
                if t.status in ("COMPLETED", "FAILED", "CANCELLED") and t.completed_at and now - t.completed_at > 300:
                    drop.append(tid)
            for tid in drop:
                del self.tasks[tid]
            if drop:
                self.log.info(f"Cleanup tasks: {len(drop)}")
        except Exception as e:
            self._emit_error(f"Cleanup error: {e}")

    # ───────────────────────── utils ─────────────────────────

    def get_system_status(self) -> Dict:
        """
        Статус, который периодически опрашивает UI.
        Важно: ключ 'is_running' должен присутствовать.
        """
        return {
            "is_running": self.is_running,
            "auto_mode": self.auto_mode,
            "manual_mode": self.manual_mode,
            "global_span_hz": self.global_span_hz,
            "global_dwell_ms": self.global_dwell_ms,
            "master_connected": self.master_controller is not None,
            "slave_connected": self.slave_manager is not None,
            "trilateration_connected": self.trilateration_engine is not None,
            "queue_len": len(self.queue),
            "tasks_total": len(self.tasks),
        }
    
    def get_active_tasks(self) -> List[MeasurementTask]:
        """
        Возвращает список активных задач для отображения в UI.
        """
        return list(self.tasks.values())

    def _emit_status(self):
        try:
            self.status_changed.emit(self.get_system_status())
        except Exception as e:
            # не мешаем работе из-за статуса
            self.log.debug(f"status_changed emit failed: {e}")

    def _emit_error(self, msg: str):
        self.log.error(msg)
        try:
            self.error.emit(str(msg))
        except Exception:
            pass

    def _emit_target_detected(self, result: TrilaterationResult):
        """Пробрасываем alias-сигнал, которого ждёт старый UI."""
        try:
            self.target_detected.emit(result)
        except Exception:
            pass
