# -*- coding: utf-8 -*-
"""
Core orchestrator: формирует окна измерений для slaves по событиям Master,
собирает RMS RSSI и запускает трилатерацию.
"""

from __future__ import annotations
import time
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from collections import deque, defaultdict

from PyQt5.QtCore import QObject, pyqtSignal, QMutex, QTimer
import numpy as np

from panorama.features.spectrum.model import DetectedPeak
from panorama.features.slave_controller.slave import (
    RSSIMeasurement,
    MeasurementWindow,
    SlaveManager,
)
from panorama.features.trilateration import (
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

        self.master_controller: Optional[object] = None
        self.slave_manager: Optional[SlaveManager] = None
        self.trilateration_engine: Optional[RSSITrilaterationEngine] = None
        
        # Добавляем координатор трилатерации
        self.trilateration_coordinator = None

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

        # Соответствие центра окна (Hz) -> peak_id из watchlist
        self._center_to_peak_id: Dict[float, str] = {}

        self._cleanup_timer = QTimer(self)
        self._cleanup_timer.timeout.connect(self._cleanup_old)
        self._cleanup_timer.start(5000)

        # Live-буфер последних RMS для UI (range_str -> {slave_id: (rssi, ts)})
        self._live_rssi: Dict[str, Dict[str, tuple]] = {}

        # Периодическое обновление измерений watchlist (постоянные замеры)
        self.measure_interval_ms = 1000  # увеличили интервал до 1 секунды для стабильности
        self._last_measure_at: Dict[str, float] = {}
        self._measure_timer = QTimer(self)
        self._measure_timer.timeout.connect(self._tick_watchlist_measurements)

        # прокинем alias-сигнал
        self.target_update.connect(self._emit_target_detected)

        # сообщим UI стартовый статус
        self._emit_status()

    # ───────────────────────── wiring ─────────────────────────

    def set_master_controller(self, master: object):
        self.master_controller = master
        if master:
            master.peak_detected.connect(self._on_peak_detected)
            self.log.info("Master controller connected")
            self._emit_status()

    def set_slave_manager(self, sm: SlaveManager):
        self.slave_manager = sm
        if sm:
            sm.all_measurements_complete.connect(self._on_all_measurements_complete)
            # Реалтайм-апдейты (каждое измерение)
            try:
                sm.measurement_progress.connect(self._on_measurement_progress)
            except Exception:
                pass
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
    
    def set_trilateration_coordinator(self, coordinator):
        """Устанавливает координатор трилатерации."""
        self.trilateration_coordinator = coordinator
    
    def _get_peak_id_for_freq(self, center_hz: float) -> Optional[str]:
        """Извлекает peak_id из watchlist по частоте."""
        # Сначала пробуем прямое сопоставление из карты
        # с допуском по частоте в 1 кГц
        best_peak_id: Optional[str] = None
        best_delta = 1e9
        for c, pid in list(self._center_to_peak_id.items()):
            d = abs(c - center_hz)
            if d < best_delta:
                best_delta = d
                best_peak_id = pid
        if best_peak_id is not None and best_delta <= 1e3:
            return best_peak_id
        
        # Если не найдено в карте, ищем в watchlist напрямую с увеличенным допуском
        if self.trilateration_coordinator and hasattr(self.trilateration_coordinator, 'peak_manager'):
            for entry in self.trilateration_coordinator.peak_manager.watchlist.values():
                delta = abs(entry.center_freq_hz - center_hz)
                if delta < best_delta and delta <= 5e3:  # 5 кГц допуск
                    best_delta = delta
                    best_peak_id = entry.peak_id
            if best_peak_id is not None:
                return best_peak_id
                
        return None

    # ─────────────────────── public API (UI) ───────────────────────

    def start(self):
        """Включает автопроцесс постановки задач (мастер ⇒ слейвы ⇒ трилатерация)."""
        self.is_running = True
        self.auto_mode = True
        self.manual_mode = False
        self.log.info("Orchestrator started (auto mode)")
        self._emit_status()
        # Запускаем периодический цикл измерений
        try:
            if not self._measure_timer.isActive():
                self._measure_timer.start(self.measure_interval_ms)
        except Exception:
            pass

    def stop(self):
        """Останавливает автоматическую постановку задач (очередь не чистится)."""
        self.is_running = False
        self.log.info("Orchestrator stopped")
        self._emit_status()
        try:
            if self._measure_timer.isActive():
                self._measure_timer.stop()
        except Exception:
            pass

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

    # ─────────────────────── watchlist → orchestrator ───────────────────────

    def enqueue_watchlist_task(self, task: Dict[str, float]):
        """
        Принимает задачу от PeakWatchlistManager: {peak_id, center_freq_hz, span_hz, ...}
        Ставит окно на измерение всеми слейвами.
        """
        if not self.slave_manager:
            self._emit_error("No slave manager: skip watchlist task")
            return
        try:
            center = float(task.get('center_freq_hz'))
            span = float(task.get('span_hz', self.global_span_hz))
            dwell = int(self.global_dwell_ms)
            peak_id = str(task.get('peak_id', f"peak_{int(center)}"))

            window = MeasurementWindow(center=center, span=span, dwell_ms=dwell, epoch=time.time())
            task_id = f"{peak_id}_{int(center)}_{int(span)}_{int(window.epoch)}"
            # Формируем суррогат DetectedPeak для совместимости (минимальный набор полей)
            dp = DetectedPeak(freq_hz=center, snr_db=0.0, power_dbm=0.0, band_hz=span / 2.0, idx=0,
                              id=peak_id, f_peak=center, bin_hz=span, t0=window.epoch, last_seen=window.epoch,
                              span_user=span, status="ACTIVE")
            mt = MeasurementTask(id=task_id, peak=dp, window=window, status="PENDING", created_at=time.time())
            self.tasks[task_id] = mt
            self.queue.append(task_id)
            # Запоминаем соответствие частоты → peak_id
            self._center_to_peak_id[center] = peak_id
            self.task_created.emit(mt)
            self.log.info(f"Watchlist task queued: {peak_id} f={center/1e6:.3f} MHz span={span/1e6:.3f} MHz")
            self._emit_status()
            self._process_queue()
        except Exception as e:
            self._emit_error(f"enqueue_watchlist_task error: {e}")

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

    def _tick_watchlist_measurements(self):
        """Периодически переизмеряет все активные диапазоны в watchlist."""
        if not self.is_running:
            return
        if not self.slave_manager:
            return
        if not self.trilateration_coordinator:
            return
        try:
            pm = self.trilateration_coordinator.peak_manager
            now = time.time()
            # Обходим все активные записи watchlist (без дублей) и ставим задачи, если пришло время
            seen_ids: set = set()
            for entry in list(pm.watchlist.values()):
                if entry.peak_id in seen_ids:
                    continue
                seen_ids.add(entry.peak_id)
                peak_id = entry.peak_id
                last_ts = self._last_measure_at.get(peak_id, 0.0)
                if (now - last_ts) * 1000.0 < self.measure_interval_ms:
                    continue
                self._last_measure_at[peak_id] = now
                try:
                    # ДИНАМИЧЕСКИЙ per-slave центр: для каждого слейва хотим окно ±halfspan вокруг
                    # текущего центра, который поддерживается менеджером пиков (обновляется от мастера)
                    # Здесь ставим задачу как обычно: центр = entry.center_freq_hz, span = entry.span_hz
                    # А вычисление F_max/F_centroid происходит на стороне PeakWatchlistManager и обновит entry центр
                    # перед следующей итерацией. Этого достаточно, чтобы окна «ехали» за сигналом.
                    self.enqueue_watchlist_task({
                        'peak_id': peak_id,
                        'center_freq_hz': float(entry.center_freq_hz),
                        'span_hz': float(entry.span_hz)
                    })
                except Exception:
                    pass
        except Exception as e:
            self._emit_error(f"tick_watchlist_measurements error: {e}")

    # ─────────────────────── slave → orchestrator ───────────────────────

    def _on_all_measurements_complete(self, results: List[RSSIMeasurement]):
        try:
            # Live-обновление RSSI в watchlist для UI сразу по каждому измерению (убираем дубликат)
            grouped: Dict[str, List[RSSIMeasurement]] = defaultdict(list)
            for r in results:
                # сопоставляем с задачей по (center, span) - увеличиваем допуск до 5 кГц
                for tid, t in self.tasks.items():
                    if (abs(r.center_hz - t.window.center) < 5e3 and
                            abs(r.span_hz - t.window.span) < 5e3 and
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

                # Консольный вывод RMS от каждого слейва (по завершению задачи)
                try:
                    for m in meas_list:
                        self.log.info(
                            f"RMS {m.slave_id}: {m.band_rssi_dbm:.1f} dBm (noise {m.band_noise_dbm:.1f} dBm, "
                            f"SNR {m.snr_db:.1f} dB) @ {m.center_hz/1e6:.1f} MHz"
                        )
                    
                except Exception:
                    pass

                # Всегда обновляем RMS в watchlist для UI (даже при низком SNR)
                try:
                    if self.trilateration_coordinator:
                        for m in meas_list:
                            peak_id_any = self._get_peak_id_for_freq(m.center_hz)
                            if peak_id_any:
                                self.log.debug(f"Updating RSSI for peak_id={peak_id_any}, slave={m.slave_id}, rssi={m.band_rssi_dbm:.1f} dBm")
                                self.trilateration_coordinator.peak_manager.update_rssi_measurement(
                                    peak_id_any, m.slave_id, m.band_rssi_dbm
                                )
                            else:
                                self.log.warning(f"No peak_id found for freq {m.center_hz/1e6:.1f} MHz from slave {m.slave_id}")
                                # Попробуем найти с большим допуском
                                for entry in list(self.trilateration_coordinator.peak_manager.watchlist.values()):
                                    if abs(m.center_hz - entry.center_freq_hz) < 5e3:  # 5 кГц допуск
                                        self.log.info(f"Found close match: peak_id={entry.peak_id} for freq {m.center_hz/1e6:.1f} MHz")
                                        self.trilateration_coordinator.peak_manager.update_rssi_measurement(
                                            entry.peak_id, m.slave_id, m.band_rssi_dbm
                                        )
                                        break
                except Exception as e:
                    self.log.error(f"Error updating RSSI measurements: {e}")

                # триггерим трилатерацию только если >= 3 slaves отдали валидные данные
                valid = [m for m in meas_list if m.snr_db >= self.min_snr_threshold and m.flags.get("valid", True)]
                if len(valid) >= 3 and self.trilateration_engine:
                    try:
                        # Преобразуем список измерений в словарь RSSI для движка
                        rssi_dict: Dict[str, float] = {m.slave_id: float(m.band_rssi_dbm) for m in valid}

                        # Фильтруем по известным станциям, чтобы не спамить ошибками движка
                        try:
                            known_stations = set(getattr(self.trilateration_engine, 'get_station_positions')())
                        except Exception:
                            known_stations = set(getattr(self.trilateration_engine, 'stations', {}).keys())
                        rssi_known = {sid: val for sid, val in rssi_dict.items() if sid in known_stations}
                        if len(rssi_known) < 3:
                            # Недостаточно известных станций — пропускаем расчёт, избежав ошибок в логе
                            raise RuntimeError("skip_trilateration_insufficient_known")
                        # Извлекаем peak_id по частоте (берём из первого валидного измерения)
                        freq_hz = float(valid[0].center_hz)
                        freq_mhz = freq_hz / 1e6
                        peak_id_guess = self._get_peak_id_for_freq(freq_hz) or f"peak_{int(freq_mhz)}MHz"
                        result: Optional[TrilaterationResult] = self.trilateration_engine.calculate_position(
                            rssi_known, peak_id=peak_id_guess, freq_mhz=freq_mhz
                        )
                        if result:
                            self.target_update.emit(result)  # alias уйдёт автоматически
                            self.log.info(
                                f"Trilateration: ({result.x:.1f},{result.y:.1f}) conf={result.confidence:.2f}"
                            )
                    except Exception as e:
                        if str(e) != "skip_trilateration_insufficient_known":
                            self._emit_error(f"Trilateration error: {e}")
                
                # Передаем в координатор трилатерации только валидные измерения (для трекинга)
                if self.trilateration_coordinator:
                    for measurement in valid:
                        peak_id = self._get_peak_id_for_freq(measurement.center_hz)
                        if peak_id:
                            self.trilateration_coordinator.add_slave_rssi_measurement(
                                slave_id=measurement.slave_id,
                                peak_id=peak_id,
                                center_freq_hz=measurement.center_hz,
                                rssi_rms_dbm=measurement.band_rssi_dbm
                            )

                # Гарантируем непрерывность: если таймер ещё не поставил следующее окно,
                # а последний запуск по этому peak_id был давно — ставим одно окно сразу
                try:
                    any_pid = None
                    any_center = None
                    for m in meas_list:
                        pid = self._get_peak_id_for_freq(m.center_hz)
                        if pid:
                            any_pid = pid
                            any_center = float(m.center_hz)
                            break
                    if any_pid and self.trilateration_coordinator:
                        now = time.time()
                        last = self._last_measure_at.get(any_pid, 0.0)
                        if (now - last) * 1000.0 >= max(500.0, self.measure_interval_ms * 0.8):
                            pm = self.trilateration_coordinator.peak_manager
                            entry = pm.watchlist.get(any_pid)
                            if entry:
                                self._last_measure_at[any_pid] = now
                                self.enqueue_watchlist_task({
                                    'peak_id': any_pid,
                                    'center_freq_hz': float(entry.center_freq_hz),
                                    'span_hz': float(entry.span_hz)
                                })
                except Exception:
                    pass

            # после обработки — запускаем следующую задачу, если есть
            self._process_queue()
            self._emit_status()

        except Exception as e:
            self._emit_error(f"Error in _on_all_measurements_complete: {e}")

    def _on_measurement_progress(self, m: RSSIMeasurement):
        """Онлайновое обновление RSSI и baseline для UI/веб-таблицы."""
        try:
            # Пробуем привязать к существующему peak_id по частоте
            peak_id_any = self._get_peak_id_for_freq(m.center_hz)
            if self.trilateration_coordinator and peak_id_any:
                self.trilateration_coordinator.peak_manager.update_rssi_measurement(
                    peak_id_any, m.slave_id, m.band_rssi_dbm
                )
            # Кладём в live-буфер для мгновенного снапшота в таблицу
            try:
                rng = f"{(m.center_hz - m.span_hz/2)/1e6:.1f}-{(m.center_hz + m.span_hz/2)/1e6:.1f}"
                if rng not in self._live_rssi:
                    self._live_rssi[rng] = {}
                self._live_rssi[rng][m.slave_id] = (float(m.band_rssi_dbm), float(time.time()))
            except Exception:
                pass
            # Пингуем статус, чтобы веб-таблица получила свежие baseline по каждому слейву
            self._emit_status()
            # Реалтайм вывод RMS от слейва в консоль
            try:
                self.log.info(
                    f"RMS {m.slave_id}: {m.band_rssi_dbm:.1f} dBm (noise {m.band_noise_dbm:.1f} dBm, "
                    f"SNR {m.snr_db:.1f} dB) @ {m.center_hz/1e6:.1f} MHz"
                )
            except Exception:
                pass
        except Exception as e:
            self.log.debug(f"measurement_progress handler error: {e}")

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
        status = {
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
        # Добавим пер-слейв baseline в статус
        try:
            if self.slave_manager:
                baselines: Dict[str, float] = {}
                for sid, sl in self.slave_manager.slaves.items():
                    baselines[sid] = float(getattr(sl, 'noise_baseline_dbm', -90.0))
                status["noise_baseline_dbm"] = baselines
        except Exception:
            pass
        return status
    
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

    # ───────────────────────── UI snapshot for SlavesView ─────────────────────────
    def get_ui_snapshot(self) -> Dict:
        """
        Возвращает снимок данных для ImprovedSlavesView:
          - watchlist: [{id, freq, span, rssi_1, rssi_2, rssi_3, updated}]
          - tasks:     [{id, range, status, progress, time, priority, timestamp}]
          - rssi_measurements: [{range, slave_id, rssi_rms}]
        """
        snapshot: Dict[str, Any] = {}

        # Watchlist из координатора (если подключен)
        try:
            wl_ui = []
            if self.trilateration_coordinator is not None:
                pm = self.trilateration_coordinator.peak_manager
                for entry in pm.watchlist.values():
                    # entry: WatchlistEntry
                    rssi_vals = entry.rssi_measurements if entry.rssi_measurements else {}
                    # Нормализуем ключи слейвов к виду slave0/slave1/slave2
                    rssi_vals_norm: Dict[str, float] = {}
                    for sid_orig, val in rssi_vals.items():
                        sid = str(sid_orig).strip().lower()
                        if sid.startswith('slave'):
                            digits = ''.join(ch for ch in sid if ch.isdigit())
                            if digits:
                                try:
                                    idx = int(digits)
                                except Exception:
                                    idx = 0
                                sid = f"slave{idx}"
                            else:
                                sid = "slave0"
                        rssi_vals_norm[sid] = val

                    # Предпочтительный порядок отображения и резервный фоллбек на первые три измерения
                    preferred = ['slave0', 'slave1', 'slave2']
                    order: List[str] = [k for k in preferred if k in rssi_vals_norm]
                    for k in rssi_vals_norm.keys():
                        if k not in order:
                            order.append(k)
                    vals: List[Optional[float]] = [rssi_vals_norm.get(order[i]) if i < len(order) else None for i in range(3)]
                    # Рассчитываем halfspan и количество бинов
                    halfspan = entry.span_hz / 2.0 / 1e6  # halfspan в МГц
                    # Примерно оцениваем количество бинов (зависит от разрешения спектра)
                    # Предполагаем разрешение ~10 кГц на бин
                    total_bins = max(1, int(entry.span_hz / 10e3))
                    
                    # Оценка уверенности на основе SNR относительно текущего baseline
                    try:
                        baseline_med = float(np.median(list(pm.baseline_history))) if len(pm.baseline_history) > 0 else -90.0
                    except Exception:
                        baseline_med = -90.0
                    max_rssi = None
                    for v in rssi_vals_norm.values():
                        try:
                            fv = float(v)
                            max_rssi = fv if max_rssi is None else max(max_rssi, fv)
                        except Exception:
                            pass
                    est_conf = 0.0
                    if max_rssi is not None:
                        snr_est = max_rssi - baseline_med
                        # нормируем к [0..1] относительно порога и 20 дБ запаса
                        est_conf = max(0.0, min(1.0, (snr_est - self.min_snr_threshold) / 20.0))

                    # Добавляем baseline данные для каждого slave
                    baseline_data = {}
                    if self.slave_manager:
                        for sid, slave in self.slave_manager.slaves.items():
                            baseline_data[sid] = getattr(slave, 'noise_baseline_dbm', -90.0)
                    
                    # Динамические центры поддиапазонов для каждого slave (F_max или F_centroid + halfspan)
                    sub_centers = {}
                    user_halfspan = 2.5  # MHz - из настроек детектора
                    for i, slave_id in enumerate(['slave0', 'slave1', 'slave2']):
                        if vals[i] is not None:
                            # Используем текущий центр entry как базу (он уже обновляется в PeakWatchlistManager)
                            sub_centers[slave_id] = entry.center_freq_hz / 1e6  # MHz
                    
                    wl_ui.append({
                        'id': entry.peak_id,
                        'freq': entry.center_freq_hz / 1e6,
                        'span': (entry.freq_stop_hz - entry.freq_start_hz) / 1e6,
                        'halfspan': halfspan,
                        'rms_1': vals[0],
                        'rms_2': vals[1], 
                        'rms_3': vals[2],
                        'confidence': est_conf,
                        'total_bins': total_bins,
                        'bins_used_1': total_bins if vals[0] is not None else 0,
                        'bins_used_2': total_bins if vals[1] is not None else 0,
                        'bins_used_3': total_bins if vals[2] is not None else 0,
                        'timestamp_1': time.strftime('%H:%M:%S', time.localtime(entry.last_update)),
                        'timestamp_2': time.strftime('%H:%M:%S', time.localtime(entry.last_update)),
                        'timestamp_3': time.strftime('%H:%M:%S', time.localtime(entry.last_update)),
                        'updated': time.strftime('%H:%M:%S', time.localtime(entry.last_update)),
                        # Новые поля для GUI
                        'baseline': baseline_data,
                        'sub_centers': sub_centers,
                        'sub_halfspan': user_halfspan
                    })
            snapshot['watchlist'] = wl_ui
        except Exception as e:
            self._emit_error(f"get_ui_snapshot: watchlist error: {e}")

        # Задачи
        try:
            tasks_ui = []
            for t in self.tasks.values():
                rng = f"{(t.window.center - t.window.span/2)/1e6:.1f}-{(t.window.center + t.window.span/2)/1e6:.1f}"
                tasks_ui.append({
                    'id': t.id,
                    'range': rng,
                    'status': t.status,
                    'progress': 100 if t.status == 'COMPLETED' else (50 if t.status == 'RUNNING' else 0),
                    'time': time.strftime('%H:%M:%S', time.localtime(t.created_at)),
                    'priority': 'NORMAL',
                    'timestamp': t.created_at
                })
            snapshot['tasks'] = tasks_ui
        except Exception as e:
            self._emit_error(f"get_ui_snapshot: tasks error: {e}")

        # RSSI измерения из завершённых задач + live-буфер прогресса
        try:
            measurements_ui = []
            # live: отдаём последние измеренные значения без тайм-окна, чтобы не было «паузы»
            try:
                for rng, by_slave in self._live_rssi.items():
                    for sid, (val, ts) in by_slave.items():
                        measurements_ui.append({'range': rng, 'slave_id': sid, 'rssi_rms': val})
            except Exception:
                pass

            # completed tasks snapshot
            for t in self.tasks.values():
                if not t.measurements:
                    continue
                rng = f"{(t.window.center - t.window.span/2)/1e6:.1f}-{(t.window.center + t.window.span/2)/1e6:.1f}"
                for m in t.measurements:
                    measurements_ui.append({
                        'range': rng,
                        'slave_id': m.slave_id,
                        'rssi_rms': m.band_rssi_dbm
                    })
            snapshot['rssi_measurements'] = measurements_ui
        except Exception as e:
            self._emit_error(f"get_ui_snapshot: rssi error: {e}")

        return snapshot
