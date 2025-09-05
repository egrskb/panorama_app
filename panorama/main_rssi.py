#!/usr/bin/env python3
"""
ПАНОРАМА RSSI - Система трилатерации по RSSI в реальном времени.
Основное приложение для координации Master sweep и Slave SDR операций.
"""

import sys
import logging
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTableWidgetItem, QMessageBox, QFileDialog, QStatusBar
import numpy as np


# Импортируем наши модули
from panorama.core.status_manager import SystemStatusManager
from panorama.core.config_manager import ConfigurationManager
from panorama.core.components_manager import ComponentsManager
from panorama.core.error_handler import ErrorHandler, safe_method
from panorama.ui.main_ui_manager import MainUIManager
# from panorama.ui.theme_manager import ThemeManager  # removed

# Импортируем модули для диалогов
from panorama.features.settings.manager_improved import ImprovedDeviceManagerDialog
from panorama.ui import DetectorSettingsDialog, DetectorSettings
from panorama.drivers.hackrf.hackrf_slaves.hackrf_slave_wrapper import HackRFSlaveDevice
from panorama.features.calibration.settings_dialog import CalibrationSettingsDialog, CalibrationDialogResult


class PanoramaAppWindow(QMainWindow):
    """Главное окно приложения ПАНОРАМА RSSI."""
    
    def __init__(self):
        super().__init__()
        # Устанавливаем иконку окна (левый верхний угол)
        try:
            from PyQt5.QtGui import QIcon
            from pathlib import Path
            icon_path = Path(__file__).parent / "resources" / "icons" / "logo.png"
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
        except Exception:
            pass
        
        # Настройка логирования
        self._setup_logging()
        
        # Менеджер тем удален; тема управляется через выпадающий список в тулбаре
        
        # Обработчик ошибок
        self.error_handler = ErrorHandler(self.log, self)
        
        # Менеджер конфигурации
        self.config_manager = ConfigurationManager(self, self.log)
        
        # Менеджер компонентов
        self.components_manager = ComponentsManager(self.config_manager, self.log)

        # Кэш последних значений уверенности по peak_id для тест_таблицы
        self._last_confidence_by_peak = {}
        
        # Инициализация компонентов
        success = self.components_manager.initialize_all_components()
        if not success:
            QMessageBox.critical(self, "Ошибка инициализации", 
                               "Не удалось инициализировать компоненты системы")
        
        # Получаем ссылки на компоненты
        self._setup_component_references()
        
        # Создание UI менеджера
        self.ui_manager = MainUIManager(self, self.orchestrator, self.log)
        
        # Настройка UI
        self.ui_manager.setup_main_ui()
        try:
            if not self.statusBar():
                self.setStatusBar(QStatusBar())
        except Exception:
            pass
        
        # Получаем ссылки на основные виджеты через UI менеджер
        self.map_view = self.ui_manager.get_map_view()
        self.spectrum_view = self.ui_manager.get_spectrum_view()
        self.slaves_view = self.ui_manager.get_slaves_view()
        
        # Подключение сигналов
        self._connect_signals()
        
        # Подключаем сигналы UI менеджера к методам главного окна
        self._connect_ui_signals()
        
        # Подключаем трилатерацию к спектру (после создания UI)
        self._connect_trilateration()
        # Синхронизируем позиции станций в движке трилатерации из конфигурации
        try:
            self._sync_trilateration_stations_from_config()
        except Exception:
            pass
        
        # Обновляем карту с текущими настройками
        if hasattr(self, 'sdr_settings') and self.sdr_settings:
            self.ui_manager.update_stations_from_config(self.sdr_settings)
        
        # Загрузка калибровки
        self._load_calibration()
        
        # Проверяем наличие конфигурации Master (БЕЗ инициализации SDR)
        self.master_ready = self.config_manager.is_master_configured()
        
        # Обновляем состояние кнопок в зависимости от наличия конфигурации Master
        self._update_ui_for_master_status()
        
        # Менеджер статуса системы
        self.status_manager = SystemStatusManager(self, update_interval_ms=1000)
        self.status_manager.status_updated.connect(self._on_status_updated)
        self._setup_status_callbacks()
        
        # Устанавливаем фиксированный заголовок окна
        self.setWindowTitle("PANORAMA")
        


        self.log.info("ПАНОРАМА RSSI initialized")
    
    def _setup_logging(self):
        """Настраивает систему логирования."""
        self.log = logging.getLogger("panorama_rssi")
        self.log.setLevel(logging.INFO)
        
        # Создаем handler для вывода в консоль
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Форматтер
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        self.log.addHandler(console_handler)
    
    def _setup_status_callbacks(self):
        """Настраивает callbacks для обновления статуса системы."""
        # Регистрируем callbacks для получения статуса компонентов
        if self.master_controller:
            self.status_manager.register_status_callback(
                'master_running', 
                lambda: bool(getattr(self.master_controller, 'is_running', False))
            )
        
        if self.orchestrator:
            self.status_manager.register_status_callback(
                'orchestrator_running',
                lambda: self.orchestrator.is_running if hasattr(self.orchestrator, 'is_running') else False
            )
        
        if self.slave_manager:
            self.status_manager.register_status_callback(
                'n_slaves',
                lambda: len(self.slave_manager.slaves) if self.slave_manager else 0
            )
    
    def _on_status_updated(self, status: dict):
        """Обрабатывает обновление статуса системы."""
        try:
            # Заголовок окна остается фиксированным как "ПАНОРАМА RSSI"
            pass
            
        except Exception as e:
            self.log.error(f"Error updating UI from status: {e}")
    
    def _setup_component_references(self):
        """Устанавливает ссылки на компоненты из менеджера компонентов."""
        self.calibration_manager = self.components_manager.calibration_manager
        self.trilateration_engine = self.components_manager.trilateration_engine
        self.slave_manager = self.components_manager.slave_manager
        self.master_controller = self.components_manager.master_controller
        self.orchestrator = self.components_manager.orchestrator
        self.peak_watchlist_manager = self.components_manager.peak_watchlist_manager
        self.trilateration_coordinator = self.components_manager.trilateration_coordinator
        
        # Устанавливаем сохранённые настройки для совместимости
        self.sdr_settings = self.config_manager.get_full_config()
    

    def _update_ui_for_master_status(self):
        """Обновляет состояние UI в зависимости от наличия конфигурации Master устройства."""
        try:
            if hasattr(self, 'toolbar_orch_action'):
                if self.master_ready:
                    self.toolbar_orch_action.setEnabled(True)
                    self.toolbar_orch_action.setToolTip("Master устройство настроено. Можно запускать спектр.")
                else:
                    self.toolbar_orch_action.setEnabled(False)
                    self.toolbar_orch_action.setToolTip("Master устройство не настроено")
            
            # Показываем сообщение о статусе Master
            if not self.master_ready:
                QMessageBox.information(self, "Настройка Master устройства", 
                    "Master устройство не настроено.\n\n"
                    "Для работы спектра необходимо:\n"
                    "1. Перейти в Настройки → Диспетчер устройств\n"
                    "2. Выбрать HackRF устройство как Master\n"
                    "3. Сохранить конфигурацию\n\n"
                    "После настройки перезапустите приложение.")
            
        except Exception as e:
            self.log.error(f"Error updating UI for master status: {e}")
    
    

    
    
    def _connect_signals(self):
        """Подключает сигналы компонентов."""
        try:
            # Сигналы Master
            if self.master_controller:
                self.master_controller.peak_detected.connect(self._on_peak_detected)
                self.master_controller.sweep_error.connect(self._on_sweep_error)
            # Сигналы оркестратора
            if self.orchestrator:
                self.orchestrator.task_created.connect(self._on_task_created)
                self.orchestrator.task_completed.connect(self._on_task_completed)
                self.orchestrator.task_failed.connect(self._on_task_failed)
                self.orchestrator.target_detected.connect(self._on_target_detected)
            # Сигналы трилатерации
            if self.trilateration_engine:
                self.trilateration_engine.target_update.connect(self._on_target_update)
                self.trilateration_engine.trilateration_error.connect(self._on_trilateration_error)
            # Сигналы slave
            if self.slave_manager:
                self.slave_manager.measurement_error.connect(self._on_measurement_error)
                # Live-лог в тестовую таблицу
                try:
                    self.slave_manager.measurement_progress.connect(self._on_measurement_progress_for_test_table)
                except Exception:
                    pass
                
            # Сигналы WatchlistView
            if hasattr(self, 'watchlist_view') and self.watchlist_view:
                self.watchlist_view.task_cancelled.connect(self._on_task_cancelled)
                self.watchlist_view.task_retried.connect(self._on_task_retried)
            
            # Сигналы SlavesView
            if hasattr(self, 'slaves_view') and self.slaves_view:
                self.slaves_view.send_to_map.connect(self._on_slave_target_to_map)
                self.slaves_view.task_selected.connect(self._on_slave_task_selected)
                self.slaves_view.watchlist_updated.connect(self._on_slave_watchlist_updated)
                
            self.log.info("All signals connected successfully")
        except Exception as e:
            self.log.error(f"Error connecting signals: {e}")

    def _on_measurement_progress_for_test_table(self, m):
        """Прокидывает live-измерения в тестовую таблицу (QTableView+pandas)."""
        try:
            if not hasattr(self.ui_manager, 'test_table') or self.ui_manager.test_table is None:
                return
            # Определяем peak_id по частоте исходного окна
            try:
                peak_id = self.orchestrator._get_peak_id_for_freq(float(m.center_hz)) if self.orchestrator else ''
            except Exception:
                peak_id = ''
            # Локальный центр, если присутствует
            try:
                local_center_hz = float(getattr(m, 'flags', {}).get('local_center_hz')) if hasattr(m, 'flags') else None
            except Exception:
                local_center_hz = None
            center_mhz = (local_center_hz if local_center_hz is not None else float(m.center_hz)) / 1e6
            halfspan_mhz = float(m.span_hz) / 2e6
            entry = {
                'time': time.strftime('%H:%M:%S', time.localtime(getattr(m, 'ts', time.time()))),
                'peak_id': str(peak_id),
                'slave': str(m.slave_id),
                'center_mhz': center_mhz,
                'halfspan_mhz': halfspan_mhz,
                'rms_dbm': float(m.band_rssi_dbm),
                'noise_dbm': float(getattr(m, 'band_noise_dbm', 0.0)),
                'snr_db': float(getattr(m, 'snr_db', 0.0)),
                'confidence': float(self._last_confidence_by_peak.get(peak_id, 0.0)) if hasattr(self, '_last_confidence_by_peak') else ''
            }
            self.ui_manager.test_table.add_log_entry(entry)
        except Exception:
            pass
    
    def _connect_ui_signals(self):
        """Подключает сигналы UI менеджера к методам главного окна."""
        try:
            # Подключаем сигналы UI менеджера к методам главного окна
            self.ui_manager.load_calibration_requested.connect(self._load_calibration_dialog)
            self.ui_manager.save_calibration_requested.connect(self._save_calibration_dialog)
            self.ui_manager.calibration_settings_requested.connect(self._show_calibration_settings)
            self.ui_manager.detector_settings_requested.connect(self._show_detector_settings)
            self.ui_manager.refresh_slaves_requested.connect(self._refresh_slaves_data)
            self.ui_manager.export_slaves_requested.connect(self._export_slaves_state)
            self.ui_manager.clear_slaves_requested.connect(self._clear_slaves_data)
            self.ui_manager.about_requested.connect(self._show_about)
            self.ui_manager.device_manager_requested.connect(self._open_device_manager)
            # Прогресс калибровки
            if self.slave_manager:
                self.slave_manager.calibration_progress.connect(self._on_calibration_progress)
                self.slave_manager.calibration_finished.connect(self._on_calibration_finished)
            
            self.log.info("UI signals connected successfully")
        except Exception as e:
            self.log.error(f"Error connecting UI signals: {e}")

    def _open_settings(self):
        # Устаревший диалог — переадресация на новый диспетчер
        self._open_device_manager()

    def _open_device_manager(self):
        current = {
            'master': {
                'nickname': self.config_manager.get_master_config().get('nickname', 'Master'),
                'serial': self.config_manager.get_master_config().get('serial', ''),
                'pos': [0.0, 0.0, 0.0],
            },
            'slaves': self.config_manager.get_slaves_config()
        }
        dlg = ImprovedDeviceManagerDialog(self, current)
        # Передаем Master контроллер для сканирования устройств
        if hasattr(dlg, 'set_master_controller'):
            dlg.set_master_controller(self.master_controller)
        def _on_conf(data: dict):
            # Apply to runtime через конфигурационный менеджер
            self.config_manager.update_configuration(data)
            self.sdr_settings = self.config_manager.get_full_config()
            
            # Master
            try:
                master_config = self.config_manager.get_master_config()
                serial = master_config.get('serial')
                if self.master_controller and getattr(self.master_controller, 'sweep_source', None):
                    if not serial:
                        self.master_controller.stop_sweep()
                        self.master_controller.sweep_source.set_serial(None)
                    else:
                        self.master_controller.sweep_source.set_serial(serial)
            except Exception:
                pass
            
            # Rebuild slaves через менеджер компонентов
            try:
                self.components_manager.refresh_slaves_configuration()
                try:
                    self._update_slave_table()
                except Exception:
                    pass
            except Exception:
                pass
            # Update map stations from new config
            try:
                self.ui_manager.update_stations_from_config(self.sdr_settings)
            except Exception:
                pass
            # После изменения конфигурации обновляем позиции станций в трилатерации
            try:
                self._sync_trilateration_stations_from_config()
            except Exception:
                pass
        def _on_slaves_available(slaves_data: list):
            # Обновляем координаты слейвов во вкладке
            if hasattr(self, 'slaves_view') and self.slaves_view:
                if hasattr(self.slaves_view, 'update_available_devices'):
                    self.slaves_view.update_available_devices(slaves_data)
                    self.log.info(f"Обновлены доступные устройства для слейвов: {len(slaves_data)} устройств")
        
        def _on_devices_for_coordinates(devices_data: list):
            """Обрабатывает устройства для координатной таблицы."""
            try:
                if hasattr(self, 'slaves_view') and self.slaves_view:
                    # Используем прямой метод синхронизации координатной таблицы
                    if hasattr(self.slaves_view, 'update_coordinates_from_manager'):
                        self.slaves_view.update_coordinates_from_manager(devices_data)
                    else:
                        # Fallback на старый метод
                        self.slaves_view.update_available_devices(devices_data)
                    self.log.info(f"Обновлена координатная таблица: {len(devices_data)} устройств")
            except Exception as e:
                self.log.error(f"Ошибка обновления координатной таблицы: {e}")
        
        dlg.devicesConfigured.connect(_on_conf)
        dlg.slavesAvailable.connect(_on_slaves_available)
        dlg.devicesForCoordinatesTable.connect(_on_devices_for_coordinates)
        dlg.exec_()

    def _sync_trilateration_stations_from_config(self):
        """Синхронизирует позиции станций в движке трилатерации из текущей конфигурации.
        Приводит идентификаторы к slave0/slave1/slave2..., чтобы совпадали с измерениями."""
        try:
            config = self.sdr_settings or self.config_manager.get_full_config()
            slaves_cfg = (config or {}).get('slaves', [])
            positions = {}
            # Гарантируем наличие slave0 в (0,0,0)
            positions['slave0'] = (0.0, 0.0, 0.0)
            for idx, s in enumerate(slaves_cfg, start=1):
                pos = s.get('pos', [0.0, 0.0, 0.0])
                x = float(pos[0]) if len(pos) > 0 else 0.0
                y = float(pos[1]) if len(pos) > 1 else 0.0
                z = float(pos[2]) if len(pos) > 2 else 0.0
                positions[f'slave{idx}'] = (x, y, z)
            if hasattr(self.trilateration_coordinator, 'set_slave_positions'):
                self.trilateration_coordinator.set_slave_positions(positions)
                self.log.info(f"Трилатерация: обновлены позиции станций ({len(positions)} шт.)")
        except Exception as e:
            self.log.error(f"Не удалось синхронизировать позиции станций: {e}")
    
    def _load_calibration(self):
        """Загружает калибровку по умолчанию."""
        try:
            # Калибровка уже загружена в _init_components
            self.log.info("Calibration loaded successfully")
        except Exception as e:
            self.log.error(f"Error loading calibration: {e}")
    
    def _add_slave(self):
        """Добавляет новый slave SDR."""
        try:
            uri = self.slave_uri_edit.currentText()
            if not uri:
                QMessageBox.warning(self, "Предупреждение", "Введите URI для SDR")
                return
            
            # Генерируем ID для slave
            slave_id = f"slave{len(self.slave_manager.slaves) + 1:02d}"
            
            # Добавляем slave
            success = self.slave_manager.add_slave(slave_id, uri)
            
            if success:
                self._update_slave_table()
                self.log.info(f"Added slave: {slave_id} ({uri})")
            else:
                QMessageBox.critical(self, "Ошибка", f"Не удалось добавить slave: {uri}")
            
        except Exception as e:
            self.log.error(f"Error adding slave: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при добавлении slave: {e}")
    
    def _update_slave_table(self):
        """Обновляет таблицу slave."""
        try:
            # Если таблицы нет (упрощённый UI), выходим без ошибок
            if not hasattr(self, 'slave_table') or self.slave_table is None:
                return
            slaves = self.slave_manager.get_slave_status()
            
            self.slave_table.setRowCount(len(slaves))
            
            for row, (slave_id, status) in enumerate(slaves.items()):
                # ID
                self.slave_table.setItem(row, 0, QTableWidgetItem(slave_id))
                
                # URI
                self.slave_table.setItem(row, 1, QTableWidgetItem(status.get('uri', 'N/A')))
                
                # Статус
                status_text = "READY" if status.get('is_initialized') else "ERROR"
                self.slave_table.setItem(row, 2, QTableWidgetItem(status_text))
                
                # Кнопка удаления
                remove_btn = QPushButton("Удалить")
                remove_btn.clicked.connect(lambda checked, sid=slave_id: self._remove_slave(sid))
                self.slave_table.setCellWidget(row, 3, remove_btn)
            
            self.status_manager.set_status('n_slaves', len(slaves))
            
        except Exception as e:
            self.log.error(f"Error updating slave table: {e}")
    
    def _remove_slave(self, slave_id: str):
        """Удаляет slave SDR."""
        try:
            self.slave_manager.remove_slave(slave_id)
            self._update_slave_table()
            self.log.info(f"Removed slave: {slave_id}")
            
        except Exception as e:
            self.log.error(f"Error removing slave {slave_id}: {e}")
    

    
    def _start_orchestrator(self):
        """Запускает оркестратор."""
        try:
            # Проверяем наличие конфигурации Master устройства
            if not self.master_ready:
                QMessageBox.warning(self, "Master устройство не настроено", 
                    "Для запуска спектра необходимо настроить Master устройство.\n\n"
                    "Перейдите в Настройки → Диспетчер устройств")
                return
            
            # Инициализируем SDR если нужно
            if self.master_controller and not self.master_controller.is_sdr_initialized():
                self.log.info("Initializing SDR for orchestrator...")
                if not self.master_controller.initialize_sdr():
                    raise RuntimeError("Failed to initialize SDR")
                self.log.info("SDR initialized successfully")
            
            # Устанавливаем глобальные параметры по умолчанию
            # (конкретные значения настраиваются через диалог настроек детектора)
            span_hz = 2.0 * 1e6  # 2 MHz по умолчанию
            dwell_ms = 150  # 150 ms по умолчанию
            
            self.orchestrator.set_global_parameters(span_hz, dwell_ms)
            
            # Запускаем оркестратор
            self.orchestrator.start()
            
            # Кнопки управления оркестратором находятся в тулбаре
            self.toolbar_orch_action.setText('⏹ Контроль')
            
            self.status_manager.set_status('orchestrator_running', True)
            self.log.info("Orchestrator started")
            
        except Exception as e:
            self.log.error(f"Error starting orchestrator: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить оркестратор: {e}")
    
    def _stop_orchestrator(self):
        """Останавливает оркестратор."""
        try:
            # Безопасно останавливаем мастер, чтобы не осталось потоков/коллбеков
            try:
                if self.master_controller:
                    self.master_controller.stop_sweep()
            except Exception:
                pass
            self.orchestrator.stop()
            
            # Деинициализируем SDR после остановки
            if self.master_controller and self.master_controller.is_sdr_initialized():
                self.log.info("Deinitializing SDR after orchestrator stop...")
                self.master_controller.deinitialize_sdr()
                self.log.info("SDR deinitialized successfully")
            
            # Кнопки управления оркестратором находятся в тулбаре
            self.toolbar_orch_action.setText('▶ Контроль')
            
            self.status_manager.set_status('orchestrator_running', False)
            self.log.info("Orchestrator stopped")
            
        except Exception as e:
            self.log.error(f"Error stopping orchestrator: {e}")
    
    def _toggle_orchestrator(self):
        """Переключает состояние оркестратора."""
        if self.status_manager.get_status('orchestrator_running'):
            self._stop_orchestrator()
        else:
            self._start_orchestrator()
    
    @safe_method("Peak detection handling", default_return=None)
    def _on_peak_detected(self, peak):
        """Обрабатывает обнаружение пика."""
        try:
            self.log.info(f"Пик: {peak.f_peak/1e6:.1f} MHz, SNR: {peak.snr_db:.1f} dB")
            
            # Обновляем спектр
            if hasattr(self.spectrum_view, 'add_peak'):
                self.spectrum_view.add_peak(peak.f_peak, peak.snr_db)
                
        except Exception as e:
            self.log.error(f"Error handling peak: {e}")
    
    def _on_sweep_error(self, error_msg):
        """Обрабатывает ошибку sweep."""
        self.log.info(f"Ошибка sweep: {error_msg}")
    
    def _on_task_created(self, task):
        """Обрабатывает создание задачи."""
        try:
            self.log.info(f"Задача создана: {task.id}")
            self._update_tasks_table()
            
        except Exception as e:
            self.log.error(f"Error handling task created: {e}")
    
    def _on_task_completed(self, task):
        """Обрабатывает завершение задачи."""
        try:
            self.log.info(f"Задача завершена: {task.id}")
            self._update_tasks_table()
            
        except Exception as e:
            self.log.error(f"Error handling task completed: {e}")
    
    def _on_task_failed(self, task):
        """Обрабатывает ошибку задачи."""
        try:
            self.log.info(f"Задача провалена: {task.id}")
            self._update_tasks_table()
            
        except Exception as e:
            self.log.error(f"Error handling task failed: {e}")
    
    @safe_method("Target detection handling", default_return=None)
    def _on_target_detected(self, target):
        """Обрабатывает обнаружение цели."""
        try:
            self.log.info(f"Цель: {target.center_hz/1e6:.1f} MHz, "
                               f"({target.x:.1f}, {target.y:.1f}), доверие: {target.confidence:.2f}")
            
            # Обновляем карту
            if hasattr(self.map_view, 'add_target'):
                self.map_view.add_target(target.x, target.y, target.confidence)
            
            # Обновляем таблицу целей
            self._update_targets_table()
            
            current_targets = self.status_manager.get_status('n_targets')
            self.status_manager.set_status('n_targets', current_targets + 1)
            
        except Exception as e:
            self.log.error(f"Error handling target: {e}")
    
    def _on_target_update(self, target):
        """Обрабатывает обновление цели."""
        try:
            # Кэшируем уверенность для тест_таблицы
            try:
                peak_id = getattr(target, 'peak_id', '')
                conf = float(getattr(target, 'confidence', 0.0))
                if peak_id:
                    self._last_confidence_by_peak[peak_id] = conf
                    # Онлайн-апдейт для всех строк этого peak в тест_таблице
                    if hasattr(self.ui_manager, 'test_table') and self.ui_manager.test_table is not None:
                        try:
                            self.ui_manager.test_table.update_confidence_for_peak(peak_id, conf)
                        except Exception:
                            pass
            except Exception:
                pass
            # Обновляем карту
            if hasattr(self.map_view, 'update_target'):
                self.map_view.update_target(target.x, target.y, target.confidence)
            
            # Обновляем таблицу целей
            self._update_targets_table()
            
        except Exception as e:
            self.log.error(f"Error handling target update: {e}")
    
    def _on_trilateration_error(self, error_msg):
        """Обрабатывает ошибку трилатерации."""
        self.log.info(f"Ошибка трилатерации: {error_msg}")

    def _on_trilateration_target_detected(self, result):
        """Адаптер: приводит результат трилатерации к dict для карты и UI."""
        try:
            payload = {
                'id': getattr(result, 'peak_id', 'Unknown'),
                'freq': float(getattr(result, 'freq_mhz', 0.0)),
                'x': float(getattr(result, 'x', 0.0)),
                'y': float(getattr(result, 'y', 0.0)),
                'confidence': float(getattr(result, 'confidence', 0.0)),
            }
            if hasattr(self, 'map_view') and self.map_view:
                if hasattr(self.map_view, 'add_target_from_detector'):
                    self.map_view.add_target_from_detector(payload)
            # Дублируем в SlavesView если нужно
            if hasattr(self, 'slaves_view') and self.slaves_view:
                if hasattr(self.slaves_view, 'add_transmitter'):
                    # SlavesView ожидает объект с атрибутами; создадим лёгкий адаптер
                    class _R:
                        pass
                    r = _R()
                    r.peak_id = payload['id']
                    r.freq_mhz = payload['freq']
                    r.x = payload['x']
                    r.y = payload['y']
                    r.confidence = payload['confidence']
                    self.slaves_view.add_transmitter(r)
        except Exception as e:
            self.log.error(f"_on_trilateration_target_detected error: {e}")

    def _on_trilateration_target_updated(self, result):
        """Адаптер для обновления цели (dict к SlavesView и карте)."""
        try:
            payload = {
                'id': getattr(result, 'peak_id', 'Unknown'),
                'freq': float(getattr(result, 'freq_mhz', 0.0)),
                'x': float(getattr(result, 'x', 0.0)),
                'y': float(getattr(result, 'y', 0.0)),
                'confidence': float(getattr(result, 'confidence', 0.0)),
            }
            if hasattr(self, 'map_view') and self.map_view:
                if hasattr(self.map_view, 'add_target_from_detector'):
                    self.map_view.add_target_from_detector(payload)
            if hasattr(self, 'slaves_view') and self.slaves_view:
                if hasattr(self.slaves_view, 'update_transmitter_position'):
                    self.slaves_view.update_transmitter_position(payload)
        except Exception as e:
            self.log.error(f"_on_trilateration_target_updated error: {e}")
    
    def _on_measurement_error(self, error_msg):
        """Обрабатывает ошибку измерения."""
        self.log.info(f"Ошибка измерения: {error_msg}")
    
    def _on_task_cancelled(self, task_id: str):
        """Обрабатывает отмену задачи."""
        try:
            self.log.info(f"Задача отменена: {task_id}")
            # Обновляем статус задачи в оркестраторе
            if self.orchestrator and task_id in self.orchestrator.tasks:
                self.orchestrator.tasks[task_id].status = "CANCELLED"
                self._update_tasks_table()
                
        except Exception as e:
            self.log.error(f"Error handling task cancelled: {e}")
    
    def _on_task_retried(self, task_id: str):
        """Обрабатывает повторение задачи."""
        try:
            self.log.info(f"Задача повторяется: {task_id}")
            # Создаем новую задачу на основе старой
            if self.orchestrator and task_id in self.orchestrator.tasks:
                old_task = self.orchestrator.tasks[task_id]
                # Создаем новую задачу с теми же параметрами
                self.orchestrator._enqueue_task(
                    old_task.peak, 
                    old_task.window.span, 
                    old_task.window.dwell_ms
                )
                self._update_tasks_table()
                
        except Exception as e:
            self.log.error(f"Error handling task retry: {e}")
    
    def _update_tasks_table(self):
        """Обновляет таблицу задач."""
        try:
            if not hasattr(self, 'tasks_table') or self.tasks_table is None:
                return
            tasks = self.orchestrator.get_active_tasks()
            
            self.tasks_table.setRowCount(len(tasks))
            
            for row, task in enumerate(tasks):
                # ID
                self.tasks_table.setItem(row, 0, QTableWidgetItem(task.id))
                
                # Частота
                freq_mhz = task.peak.f_peak / 1e6
                self.tasks_table.setItem(row, 1, QTableWidgetItem(f"{freq_mhz:.1f}"))
                
                # Статус
                self.tasks_table.setItem(row, 2, QTableWidgetItem(task.status))
                
                # Время создания
                created_time = time.strftime("%H:%M:%S", time.localtime(task.created_at))
                self.tasks_table.setItem(row, 3, QTableWidgetItem(created_time))
                
                # Время завершения
                if task.completed_at:
                    completed_time = time.strftime("%H:%M:%S", time.localtime(task.completed_at))
                    self.tasks_table.setItem(row, 4, QTableWidgetItem(completed_time))
                else:
                    self.tasks_table.setItem(row, 4, QTableWidgetItem(""))
            
        except Exception as e:
            self.log.error(f"Error updating tasks table: {e}")
    
    def _update_targets_table(self):
        """Обновляет таблицу целей."""
        try:
            if not hasattr(self, 'targets_table') or self.targets_table is None:
                return
            targets = self.trilateration_engine.get_latest_results()
            
            self.targets_table.setRowCount(len(targets))
            
            for row, target in enumerate(targets):
                # Частота
                freq_mhz = target.center_hz / 1e6
                self.targets_table.setItem(row, 0, QTableWidgetItem(f"{freq_mhz:.1f}"))
                
                # Координаты
                self.targets_table.setItem(row, 1, QTableWidgetItem(f"{target.x:.1f}"))
                self.targets_table.setItem(row, 2, QTableWidgetItem(f"{target.y:.1f}"))
                
                # Доверие
                self.targets_table.setItem(row, 3, QTableWidgetItem(f"{target.confidence:.2f}"))
                
                # Возраст
                age_sec = target.age_ms / 1000
                self.targets_table.setItem(row, 4, QTableWidgetItem(f"{age_sec:.1f}"))
                
                # Количество станций
                self.targets_table.setItem(row, 5, QTableWidgetItem(str(target.n_stations)))
            
        except Exception as e:
            self.log.error(f"Error updating targets table: {e}")
    
    def _load_calibration_dialog(self):
        """Показывает диалог загрузки калибровки."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Загрузить калибровку", "", "JSON files (*.json)"
            )
            
            if file_path:
                success = self.calibration_manager.import_profile(file_path)
                if success:
                    QMessageBox.information(self, "Успех", "Калибровка загружена успешно")
                    self._setup_trilateration()
                else:
                    QMessageBox.critical(self, "Ошибка", "Не удалось загрузить калибровку")
                    
        except Exception as e:
            self.log.error(f"Error in load calibration dialog: {e}")
    
    def _save_calibration_dialog(self):
        """Показывает диалог сохранения калибровки."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить калибровку", "", "JSON files (*.json)"
            )
            
            if file_path:
                if self.calibration_manager.current_profile:
                    success = self.calibration_manager.export_profile(
                        self.calibration_manager.current_profile.name, file_path
                    )
                    if success:
                        QMessageBox.information(self, "Успех", "Калибровка сохранена успешно")
                    else:
                        QMessageBox.critical(self, "Ошибка", "Не удалось сохранить калибровку")
                        
        except Exception as e:
            self.log.error(f"Error in save calibration dialog: {e}")
    
    def _show_calibration_settings(self):
        """Показывает настройки калибровки и запускает процесс по запросу."""
        try:
            dlg = CalibrationSettingsDialog(self)
            params_holder = {'ok': False, 'params': None}
            def _on_apply(res: CalibrationDialogResult):
                params_holder['ok'] = True
                params_holder['params'] = res
            dlg.settingsApplied.connect(_on_apply)
            dlg.exec_()
            if not params_holder['ok']:
                return
            self.statusBar().showMessage("Калибровка: старт...")
            # Готовим master устройство для калибровки (через Slave wrapper API)
            master_device = None
            master_serial = None
            try:
                master_cfg = self.config_manager.get_master_config()
                master_serial = master_cfg.get('serial') if master_cfg else None
            except Exception:
                master_serial = None
            try:
                # Останавливаем Master sweep, если он запущен, чтобы освободить устройство
                if self.spectrum_view and getattr(self.spectrum_view, '_source', None):
                    adapter = self.spectrum_view._source
                    if getattr(adapter, 'running', False):
                        adapter.stop()
            except Exception:
                pass
            try:
                master_device = HackRFSlaveDevice(serial=master_serial or "", logger=self.log)
                if master_device.open():
                    # Configure master device before any spectrum/measurement calls
                    try:
                        from panorama.drivers.hackrf.hackrf_slaves.hackrf_slave_wrapper import SlaveConfig
                        # базовые значения; частота не критична, т.к. выставляется в измерениях
                        cfg = SlaveConfig(
                            center_freq_hz=int(100e6),
                            sample_rate_hz=8_000_000,
                            lna_gain=16,
                            vga_gain=20,
                            amp_enable=False,
                            window_type=1,
                            dc_offset_correction=True,
                            iq_balance_correction=True,
                            freq_offset_hz=0.0,
                            calibration_db=0.0,
                        )
                        master_device.configure(cfg)
                    except Exception as e:
                        self.log.warning(f"Master device configure failed (will try anyway): {e}")
                else:
                    master_device = None
            except Exception as e:
                self.log.error(f"Master device open error: {e}")
                master_device = None
            ok = False
            if self.slave_manager and hasattr(self.slave_manager, 'calibrate_all_slaves'):
                # Применяем настройки калибратора
                try:
                    if getattr(self.slave_manager, 'calibrator', None):
                        cal = self.slave_manager.calibrator
                        cal.set_targets(params_holder['params'].targets_mhz,
                                        params_holder['params'].dwell_ms,
                                        params_holder['params'].search_span_khz,
                                        params_holder['params'].amplitude_tolerance_db,
                                        params_holder['params'].sync_timeout_sec)
                except Exception:
                    pass
                ok = self.slave_manager.calibrate_all_slaves(master_device)
            # Закрываем временно открытое master устройство
            try:
                if master_device:
                    master_device.close()
            except Exception:
                pass
            self.statusBar().showMessage("Калибровка завершена" if ok else "Калибровка завершена с ошибками", 5000)
        except Exception as e:
            self.statusBar().showMessage(f"Ошибка калибровки: {e}", 5000)
            self.log.error(f"Error showing calibration settings: {e}")

    def _on_calibration_progress(self, info: dict):
        try:
            stage = info.get('stage', '')
            cur = info.get('current', 0)
            total = info.get('total', 0)
            serial = info.get('serial', '')
            if stage == 'start_device':
                self.statusBar().showMessage(f"Калибровка {cur}/{total}: {serial}...")
            elif stage == 'device_done':
                ok = info.get('ok', False)
                self.statusBar().showMessage(f"Готово {cur}/{total}: {serial} ({'OK' if ok else 'ERR'})")
        except Exception:
            pass

    def _on_calibration_finished(self, ok: bool):
        try:
            self.statusBar().showMessage("Калибровка: успех" if ok else "Калибровка: ошибки", 5000)
        except Exception:
            pass

    def _show_detector_settings(self):
        """Показывает диалог настроек детектора."""
        try:
            dlg = DetectorSettingsDialog(self)
            def _on_changed(s: DetectorSettings):
                # Применяем некоторые глобальные параметры к оркестратору
                try:
                    if self.orchestrator:
                        self.orchestrator.set_global_parameters(span_hz=s.rms_halfspan_mhz * 2e6,  # Полная ширина = 2 × halfspan
                                                                dwell_ms=int(s.watchlist_dwell_ms))
                        # Применяем интервал обновления RMS (сек)
                        try:
                            self.orchestrator.set_measure_interval_sec(float(getattr(s, 'measurement_interval_sec', 1.0)))
                        except Exception:
                            pass
                    # Обновляем параметры видео‑детектора (С) для мастера
                    try:
                        if hasattr(self.ui_manager, 'spectrum_view') and self.ui_manager.spectrum_view:
                            adapter = getattr(self.ui_manager.spectrum_view, '_source', None)
                            if adapter and hasattr(adapter, 'set_video_detector_params'):
                                adapter.set_video_detector_params(s)
                    except Exception:
                        pass
                except Exception:
                    pass
                # Прокидываем режим центра в SlaveManager на лету
                try:
                    if self.slave_manager and hasattr(self.slave_manager, 'set_center_mode'):
                        self.slave_manager.set_center_mode(getattr(s, 'center_mode', 'fmax'))
                except Exception:
                    pass
            dlg.settingsChanged.connect(_on_changed)
            dlg.exec_()
        except Exception as e:
            self.log.error(f"Detector settings dialog error: {e}")
    
    def _on_slave_target_to_map(self, target_data: dict):
        """Обрабатывает сигнал отправки данных на карту от slaves_view."""
        try:
            if hasattr(self, 'map_view') and self.map_view:
                data_type = target_data.get('type', 'target')
                
                if data_type in ('update_slaves_coordinates', 'update_devices_coordinates'):
                    # Обновляем координаты слейвов на карте
                    # Поддерживаем оба формата ключей: 'slaves' и 'devices'
                    slaves_data = target_data.get('slaves') or target_data.get('devices') or []
                    
                    # Преобразуем в формат для карты
                    config_slaves = []
                    try:
                        # Сначала опорное как slave0
                        ref = next((s for s in slaves_data if s.get('is_reference')), None)
                        if ref is not None:
                            config_slaves.append({
                                'nickname': 'slave0',
                                'pos': [0.0, 0.0, 0.0],
                                'is_reference': True
                            })
                        # Остальные начиная с slave1
                        idx = 1
                        for s in slaves_data:
                            if s is ref:
                                continue
                            config_slaves.append({
                                'nickname': f'slave{idx}',
                                'pos': [float(s.get('x', 0.0)), float(s.get('y', 0.0)), float(s.get('z', 0.0))],
                                'is_reference': False
                            })
                            idx += 1
                    except Exception:
                        pass
                    
                    # Обновляем карту
                    if hasattr(self.map_view, 'update_stations_from_config'):
                        self.map_view.update_stations_from_config({'slaves': config_slaves})
                        self.log.info(f"Координаты слейвов обновлены на карте: {len(config_slaves)} устройств")
                    
                elif data_type == 'slaves_layout':
                    # Показываем расположение слейвов на карте
                    slaves_data = target_data.get('slaves', [])
                    for slave in slaves_data:
                        if hasattr(self.map_view, 'add_station_marker'):
                            self.map_view.add_station_marker(
                                slave['id'], 
                                slave['x'], 
                                slave['y'], 
                                slave['id'] == 'slave0'  # is_reference
                            )
                    self.log.info(f"Показано расположение {len(slaves_data)} слейвов на карте")
                
                else:
                    # Обычная цель/передатчик
                    if hasattr(self.map_view, 'add_target_from_detector'):
                        self.map_view.add_target_from_detector(target_data)
                        self.log.info(f"Цель от слейва отправлена на карту: {target_data.get('id', 'Unknown')}")
                    else:
                        self.log.warning("Метод add_target_from_detector не доступен в map_view")
            else:
                self.log.warning("Map view не доступен")
        except Exception as e:
            self.log.error(f"Ошибка отправки данных на карту: {e}")
    
    def _on_slave_task_selected(self, task_id: str):
        """Обрабатывает сигнал выбора задачи от slaves_view."""
        try:
            self.log.info(f"Выбрана задача от слейва: {task_id}")
            # TODO: Реализовать логику выбора задачи
        except Exception as e:
            self.log.error(f"Ошибка обработки выбора задачи: {e}")
    
    def _on_slave_watchlist_updated(self, watchlist_data: list):
        """Обрабатывает сигнал обновления watchlist от slaves_view."""
        try:
            watchlist_count = len(watchlist_data)
            prev_count = getattr(self, '_last_watchlist_count', 0)
            
            # Логируем только при реальных изменениях
            if prev_count != watchlist_count:
                self.log.info(f"Watchlist обновлен: {watchlist_count} элементов (было: {prev_count})")
                self._last_watchlist_count = watchlist_count
                
                # Автоматическое управление оркестратором и slaves
                self._auto_manage_orchestrator(watchlist_count, prev_count)
                
        except Exception as e:
            self.log.error(f"Ошибка обработки обновления watchlist: {e}")
    
    def _auto_manage_orchestrator(self, current_count: int, prev_count: int):
        """Автоматически управляет оркестратором в зависимости от состояния watchlist."""
        try:
            # Проверяем что компоненты доступны
            if not hasattr(self, 'components_manager') or not self.components_manager:
                return
                
            orchestrator = self.components_manager.orchestrator
            slave_manager = self.components_manager.slave_manager
            
            if not orchestrator or not slave_manager:
                return
            
            # Проверяем доступность slaves
            available_slaves = sum(1 for slave in slave_manager.slaves.values() if slave.is_initialized) if slave_manager else 0
            
            # Логика управления:
            # Первый пик в watchlist (0 -> 1+) = запуск slaves и оркестратора
            if prev_count == 0 and current_count > 0:
                if available_slaves > 0:
                    self.log.info(f"🚀 Первый пик обнаружен - запускаем оркестратор с {available_slaves} slaves")
                    
                    # Запускаем оркестратор если он не запущен
                    if not orchestrator.is_running:
                        orchestrator.start()
                        self.log.info("Orchestrator started")
                    
                    # Уведомляем о начале активной фазы
                    if available_slaves == 1:
                        self.log.info(f"📡 Начинаем мониторинг {current_count} целей (shared device mode - timeouts expected)")
                    else:
                        self.log.info(f"📡 Начинаем мониторинг {current_count} целей с {available_slaves} slaves")
                else:
                    self.log.warning(f"🚀 Первый пик обнаружен, но нет доступных slaves для измерений")
                    self.log.info(f"📊 Обнаружено {current_count} целей - работаем только в режиме обнаружения (без RSSI)")
                
            # Watchlist стал пустым (1+ -> 0) = остановка slaves и оркестратора  
            elif prev_count > 0 and current_count == 0:
                self.log.info("⏹️  Все пики исчезли - останавливаем оркестратор")
                
                # Останавливаем оркестратор
                if orchestrator.is_running:
                    orchestrator.stop()
                    self.log.info("Orchestrator stopped")
                
                # Уведомляем об окончании активной фазы
                self.log.info("📡 Мониторинг завершен - система в режиме ожидания")
                
            # Изменение количества целей в активной фазе
            elif prev_count > 0 and current_count > 0:
                if available_slaves > 0:
                    self.log.info(f"📊 Обновление watchlist: {current_count} активных целей, {available_slaves} slaves")
                else:
                    self.log.info(f"📊 Обновление watchlist: {current_count} активных целей (только обнаружение)")
                
        except Exception as e:
            self.log.error(f"Ошибка автоуправления оркестратором: {e}")
    
    def _on_watchlist_updated_from_peak_manager(self, watchlist_entries: list):
        """Обрабатывает обновления watchlist от PeakWatchlistManager для автоуправления."""
        try:
            # Конвертируем WatchlistEntry в простой формат для совместимости
            watchlist_data = []
            for entry in watchlist_entries:
                watchlist_data.append({
                    'id': entry.peak_id,
                    'freq': entry.center_freq_hz / 1e6,  # МГц
                    'span': entry.span_hz / 1e6,         # МГц
                    'rssi': entry.rssi_measurements or {},
                    'updated': entry.last_update
                })
            
            # Вызываем основную логику автоуправления
            self._on_slave_watchlist_updated(watchlist_data)

            # Подсветка диапазонов на спектре/водопаде: строим список (start_mhz, stop_mhz)
            try:
                if hasattr(self.ui_manager, 'spectrum_view') and self.ui_manager.spectrum_view:
                    ranges = []
                    for entry in watchlist_entries:
                        try:
                            # Предпочитаем границы кластера
                            f0c = float(getattr(entry, 'cluster_start_hz', 0.0)) * 1e-6
                            f1c = float(getattr(entry, 'cluster_end_hz', 0.0)) * 1e-6
                            if f1c > f0c and f1c > 0 and f0c > 0:
                                f0, f1 = f0c, f1c
                            else:
                                f0 = float(entry.freq_start_hz) * 1e-6
                                f1 = float(entry.freq_stop_hz) * 1e-6
                        except Exception:
                            # fallback на центр+span
                            c = float(entry.center_freq_hz) * 1e-6
                            w = float(entry.span_hz) * 1e-6 * 0.5
                            f0, f1 = c - w, c + w
                        ranges.append((f0, f1))
                    self.ui_manager.spectrum_view.set_highlight_ranges(ranges)
            except Exception as _e:
                self.log.debug(f"Highlight update error: {_e}")
            
        except Exception as e:
            self.log.error(f"Ошибка обработки watchlist от peak_manager: {e}")
    
    @safe_method("Master spectrum for slaves", default_return=None)
    def _on_master_spectrum_for_slaves(self, freqs, dbm):
        """Передает спектральные данные от Master всем slaves для Virtual Slave режима."""
        try:
            if not hasattr(self, 'components_manager') or not self.components_manager:
                return
            
            slave_manager = self.components_manager.slave_manager
            if not slave_manager or not slave_manager.slaves:
                return
                
            # Конвертируем в numpy arrays если нужно
            if not isinstance(freqs, np.ndarray):
                freqs = np.array(freqs, dtype=np.float64)
            if not isinstance(dbm, np.ndarray):
                dbm = np.array(dbm, dtype=np.float32)
            
            # Передаем данные всем slaves
            for slave in slave_manager.slaves.values():
                try:
                    slave.update_spectrum_from_master(freqs, dbm)
                except Exception as e:
                    self.log.debug(f"Error updating spectrum for slave {slave.slave_id}: {e}")
                    
        except Exception as e:
            self.log.error(f"Error processing master spectrum for slaves: {e}")
    
    def _refresh_slaves_data(self):
        """Обновляет данные слейвов."""
        try:
            if hasattr(self, 'slaves_view') and self.slaves_view:
                # Вызываем метод обновления данных в slaves_view
                if hasattr(self.slaves_view, 'manual_refresh'):
                    self.slaves_view.manual_refresh()
                self.log.info("Данные слейвов обновлены")
            else:
                self.log.warning("Slaves view не доступен")
        except Exception as e:
            self.log.error(f"Ошибка обновления данных слейвов: {e}")
    
    def _export_slaves_state(self):
        """Экспортирует состояние слейвов в файл."""
        try:
            if hasattr(self, 'slaves_view') and self.slaves_view:
                # Вызываем метод экспорта в slaves_view
                if hasattr(self.slaves_view, 'export_current_state'):
                    self.slaves_view.export_current_state()
                    self.log.info("Состояние слейвов экспортировано")
                else:
                    self.log.warning("Метод экспорта не доступен в slaves_view")
            else:
                self.log.warning("Slaves view не доступен")
        except Exception as e:
            self.log.error(f"Ошибка экспорта состояния слейвов: {e}")
    
    def _clear_slaves_data(self):
        """Очищает данные слейвов."""
        try:
            if hasattr(self, 'slaves_view') and self.slaves_view:
                # Вызываем метод очистки в slaves_view
                if hasattr(self.slaves_view, 'clear_all_data'):
                    self.slaves_view.clear_all_data()
                    self.log.info("Данные слейвов очищены")
                else:
                    self.log.warning("Метод очистки не доступен в slaves_view")
            else:
                self.log.warning("Slaves view не доступен")
        except Exception as e:
            self.log.error(f"Ошибка очистки данных слейвов: {e}")
    
    def _show_about(self):
        """Показывает информацию о программе."""
        QMessageBox.about(self, "О программе", 
                         "ПАНОРАМА RSSI v1.0\n\n"
                         "Система трилатерации по RSSI в реальном времени\n"
                         "Поддерживает множественные SDR станции\n\n"
                         "© 2024 ПАНОРАМА Team")
    
    # Тема управляется через combo в тулбаре (MainUIManager)
    
    def _connect_trilateration(self):
        """Подключает систему трилатерации к спектру."""
        # Когда приходят данные от Master
        self.spectrum_view.newRowReady.connect(
            self.trilateration_coordinator.process_master_spectrum
        )
        
        # Сохраняем адаптер мастера, чтобы координатор мог вызывать C‑детектор
        try:
            adapter = getattr(self.spectrum_view, '_source', None)
            if adapter is not None:
                self.trilateration_coordinator.master_adapter = adapter
        except Exception:
            pass

        # Virtual-slave режим отключён: не подключаем спектр Master к слейвам
        # Пересылка задач watchlist в оркестратор для измерений слейвами
        try:
            self.trilateration_coordinator.peak_manager.watchlist_task_ready.connect(
                self.orchestrator.enqueue_watchlist_task
            )
            
            # Подключаем автоматическое управление оркестратором при изменении watchlist
            self.trilateration_coordinator.peak_manager.watchlist_updated.connect(
                self._on_watchlist_updated_from_peak_manager
            )
        except Exception as e:
            self.log.error(f"Failed to connect watchlist tasks: {e}")
        
        # Позиции трёх слейвов (slave0 — опорный)
        self.trilateration_coordinator.set_slave_positions({
            'slave0': (0.0, 0.0, 0.0),
            'slave1': (10.0, 0.0, 0.0),
            'slave2': (0.0, 10.0, 0.0)
        })
        
        # Подключаем к карте через адаптер: преобразуем объект в dict
        try:
            self.trilateration_coordinator.target_detected.disconnect()
        except Exception:
            pass
        self.trilateration_coordinator.target_detected.connect(
            self._on_trilateration_target_detected
        )
        # И в UI слейвов — добавлять передатчики
        try:
            if hasattr(self, 'slaves_view') and self.slaves_view:
                # через адаптеры, чтобы формат был dict для update_transmitter_position
                self.trilateration_coordinator.target_detected.connect(
                    self._on_trilateration_target_detected
                )
                if hasattr(self.trilateration_coordinator, 'target_updated'):
                    self.trilateration_coordinator.target_updated.connect(
                        self._on_trilateration_target_updated
                    )
        except Exception:
            pass
        # TODO: Добавить метод update_target_position в MapLibreWidget
        # self.trilateration_coordinator.target_updated.connect(
        #     self.map_view.update_target_position
        # )
        # Подписка карты на живой список слейвов
        try:
            if self.slave_manager and hasattr(self.slave_manager, 'slaves_updated'):
                def _on_slaves_updated(status: dict):
                    # Преобразуем в список станций
                    stations = []
                    for sid in status.keys():
                        pos = (0.0, 0.0, 0.0)
                        for cfg in self.sdr_settings.get('slaves', []):
                            cid = cfg.get('nickname') or cfg.get('label') or cfg.get('serial')
                            if cid == sid:
                                p = cfg.get('pos', [0.0, 0.0, 0.0])
                                if len(p) >= 2:
                                    pos = (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0)
                                break
                        stations.append({'id': sid, 'x': pos[0], 'y': pos[1], 'z': pos[2]})
                    # Обновляем карту
                    try:
                        self.map_view.update_stations_from_config({'slaves': [{'nickname': s['id'], 'pos': [s['x'], s['y'], s['z']]} for s in stations]})
                    except Exception:
                        pass
                    # Проброс статусов слейвов в веб-таблицу измерений (для обновления baseline)
                    try:
                        if hasattr(self, 'slaves_view') and self.slaves_view and hasattr(self.slaves_view, 'web_table_widget') and self.slaves_view.web_table_widget:
                            self.slaves_view.web_table_widget.update_slaves_info(status)
                    except Exception:
                        pass
                self.slave_manager.slaves_updated.connect(_on_slaves_updated)
                # Инициализация: отразить отсутствие слейвов
                self.slave_manager.slaves_updated.emit(self.slave_manager.get_slave_status())
        except Exception:
            pass
        # Запуск координатора
        try:
            self.trilateration_coordinator.start()
        except Exception as e:
            self.log.error(f"Failed to start TrilaterationCoordinator: {e}")
    
    def closeEvent(self, event):
        """Обрабатывает закрытие приложения."""
        try:
            # Останавливаем менеджер статуса
            if hasattr(self, 'status_manager'):
                self.status_manager.stop()
            
            # Останавливаем все компоненты через менеджер компонентов
            if hasattr(self, 'components_manager'):
                self.components_manager.cleanup_all_components()
            
            self.log.info("Application closed")
            event.accept()
            
        except Exception as e:
            self.log.error(f"Error during application close: {e}")
            event.accept()


def main():
    """Главная функция приложения."""
    # WSL/без GPU: отключаем аппаратное ускорение для QtWebEngine/Qt OpenGL
    try:
        import os
        os.environ.setdefault("QT_OPENGL", "software")
        os.environ.setdefault("QTWEBENGINE_DISABLE_GPU", "1")
        os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-software-rasterizer --in-process-gpu")
        os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
        from PyQt5 import QtCore
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL)
    except Exception:
        pass
    app = QApplication(sys.argv)
    
    # Настройка приложения
    app.setApplicationName("ПАНОРАМА RSSI")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("ПАНОРАМА Team")
    
    # Тема: по умолчанию оставляем стандартную; управляется из тулбара
    
    # Создание главного окна
    window = PanoramaAppWindow()
    try:
        from PyQt5.QtGui import QIcon
        from pathlib import Path
        icon_path = Path(__file__).parent / "resources" / "icons" / "logo.png"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
            window.setWindowIcon(QIcon(str(icon_path)))
    except Exception:
        pass
    window.show()
    
    # Запуск приложения
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
