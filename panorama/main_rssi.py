#!/usr/bin/env python3
"""
ПАНОРАМА RSSI - Система трилатерации по RSSI в реальном времени.
Основное приложение для координации Master sweep и Slave SDR операций.
"""

import sys
import logging
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTableWidgetItem, QMessageBox, QFileDialog

# Импортируем наши модули
from panorama.core.status_manager import SystemStatusManager
from panorama.core.config_manager import ConfigurationManager
from panorama.core.components_manager import ComponentsManager
from panorama.core.error_handler import ErrorHandler, safe_method
from panorama.ui.main_ui_manager import MainUIManager

# Импортируем модули для диалогов
from panorama.features.settings.manager_improved import ImprovedDeviceManagerDialog
from panorama.features.detector.settings_dialog import DetectorSettingsDialog, DetectorSettings


class RSSIPanoramaMainWindow(QMainWindow):
    """Главное окно приложения ПАНОРАМА RSSI."""
    
    def __init__(self):
        super().__init__()
        
        # Настройка логирования
        self._setup_logging()
        
        # Обработчик ошибок
        self.error_handler = ErrorHandler(self.log, self)
        
        # Менеджер конфигурации
        self.config_manager = ConfigurationManager(self, self.log)
        
        # Менеджер компонентов
        self.components_manager = ComponentsManager(self.config_manager, self.log)
        
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
            # Обновляем заголовок окна
            title = self.status_manager.format_status_title()
            self.setWindowTitle(title)
            
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
        dlg.devicesConfigured.connect(_on_conf)
        dlg.exec_()
    
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
        """Показывает настройки калибровки."""
        # TODO: Реализовать диалог настроек калибровки
        QMessageBox.information(self, "Информация", "Настройки калибровки будут добавлены в следующей версии")

    def _show_detector_settings(self):
        """Показывает диалог настроек детектора."""
        try:
            dlg = DetectorSettingsDialog(self)
            def _on_changed(s: DetectorSettings):
                # Применяем некоторые глобальные параметры к оркестратору
                try:
                    if self.orchestrator:
                        self.orchestrator.set_global_parameters(span_hz=s.watchlist_span_mhz * 1e6,
                                                                dwell_ms=int(s.watchlist_dwell_ms))
                except Exception:
                    pass
            dlg.settingsChanged.connect(_on_changed)
            dlg.exec_()
        except Exception as e:
            self.log.error(f"Detector settings dialog error: {e}")
    
    def _on_slave_target_to_map(self, target_data: dict):
        """Обрабатывает сигнал отправки цели на карту от slaves_view."""
        try:
            if hasattr(self, 'map_view') and self.map_view:
                # Отправляем цель на карту
                if hasattr(self.map_view, 'add_target_from_detector'):
                    self.map_view.add_target_from_detector(target_data)
                    self.log.info(f"Цель от слейва отправлена на карту: {target_data.get('id', 'Unknown')}")
                else:
                    self.log.warning("Метод add_target_from_detector не доступен в map_view")
            else:
                self.log.warning("Map view не доступен")
        except Exception as e:
            self.log.error(f"Ошибка отправки цели на карту: {e}")
    
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
            # Логируем только при реальных изменениях, а не каждые 2 секунды
            if not hasattr(self, '_last_watchlist_count') or self._last_watchlist_count != len(watchlist_data):
                self.log.info(f"Watchlist обновлен от слейва: {len(watchlist_data)} элементов")
                self._last_watchlist_count = len(watchlist_data)
            # TODO: Реализовать логику обновления watchlist
        except Exception as e:
            self.log.error(f"Ошибка обработки обновления watchlist: {e}")
    
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
    
    def _connect_trilateration(self):
        """Подключает систему трилатерации к спектру."""
        # Когда приходят данные от Master
        self.spectrum_view.newRowReady.connect(
            self.trilateration_coordinator.process_master_spectrum
        )
        # Пересылка задач watchlist в оркестратор для измерений слейвами
        try:
            self.trilateration_coordinator.peak_manager.watchlist_task_ready.connect(
                self.orchestrator.enqueue_watchlist_task
            )
        except Exception as e:
            self.log.error(f"Failed to connect watchlist tasks: {e}")
        
        # Обновления позиций slave из настроек
        self.trilateration_coordinator.set_slave_positions({
            'slave1': (10.0, 0.0, 0.0),
            'slave2': (0.0, 10.0, 0.0),
            'slave3': (-10.0, 0.0, 0.0)
        })
        
        # Подключаем к карте
        self.trilateration_coordinator.target_detected.connect(
            self.map_view.add_target_from_detector
        )
        # И в UI слейвов — добавлять передатчики
        try:
            if hasattr(self, 'slaves_view') and self.slaves_view:
                self.trilateration_coordinator.target_detected.connect(
                    self.slaves_view.add_transmitter
                )
                # Обновление трекинга — позиция и уверенность
                if hasattr(self.trilateration_coordinator, 'target_updated'):
                    self.trilateration_coordinator.target_updated.connect(
                        self.slaves_view.update_transmitter_position
                    )
        except Exception:
            pass
        # TODO: Добавить метод update_target_position в OpenLayersMapWidget
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
    
    # Создание главного окна
    window = RSSIPanoramaMainWindow()
    window.show()
    
    # Запуск приложения
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
