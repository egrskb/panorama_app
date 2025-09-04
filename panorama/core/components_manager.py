"""
Менеджер компонентов для ПАНОРАМА RSSI.
"""

import logging
from typing import Optional, Any
from panorama.features.slave_controller.slave import SlaveManager
from panorama.features.trilateration import RSSITrilaterationEngine
from panorama.features.slave_controller.orchestrator import Orchestrator
from panorama.features.calibration.manager import CalibrationManager
from panorama.features.detector.peak_watchlist_manager import PeakWatchlistManager
from panorama.features.trilateration.coordinator import TrilaterationCoordinator
from panorama.features.detector.settings_dialog import (
    load_detector_settings, apply_settings_to_watchlist_manager
)


class ComponentsManager:
    """Менеджер для инициализации и управления компонентами системы."""
    
    def __init__(self, config_manager, logger: Optional[logging.Logger] = None):
        self.config_manager = config_manager
        self.log = logger or logging.getLogger(__name__)
        
        # Основные компоненты
        self.calibration_manager: Optional[CalibrationManager] = None
        self.trilateration_engine: Optional[RSSITrilaterationEngine] = None
        self.slave_manager: Optional[SlaveManager] = None
        self.master_controller: Optional[Any] = None  # MasterController не импортируется напрямую
        self.orchestrator: Optional[Orchestrator] = None
        self.peak_watchlist_manager: Optional[PeakWatchlistManager] = None
        self.trilateration_coordinator: Optional[TrilaterationCoordinator] = None
        
    def initialize_all_components(self) -> bool:
        """Инициализирует все компоненты системы."""
        try:
            self._init_calibration_manager()
            self._init_trilateration_engine()
            self._init_slave_manager()
            self._init_orchestrator()
            self._init_peak_managers()
            self._configure_orchestrator()
            self._setup_trilateration_engine()
            
            self.log.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.log.error(f"Error initializing components: {e}")
            return False
    
    def _init_calibration_manager(self):
        """Инициализирует менеджер калибровки."""
        self.calibration_manager = CalibrationManager(self.log)
    
    def _init_trilateration_engine(self):
        """Инициализирует движок трилатерации."""
        self.trilateration_engine = RSSITrilaterationEngine()
    
    def _init_slave_manager(self):
        """Инициализирует менеджер slave SDR."""
        self.slave_manager = SlaveManager(self.log)
        self._init_slaves_from_config()
    
    def _init_slaves_from_config(self):
        """Инициализирует слейвы из конфигурации."""
        if not self.slave_manager:
            return
        
        # Очищаем текущие слейвы
        for sid in list(self.slave_manager.slaves.keys()):
            self.slave_manager.remove_slave(sid)
        
        # Добавляем слейвы из конфига (не более трёх)
        slaves_config = self.config_manager.get_slaves_config()[:3]
        for idx, slave_config in enumerate(slaves_config):
            # Жестко нормализуем ID слейвов к 'slave0', 'slave1', 'slave2'
            slave_id = f"slave{idx}"
            uri = self.config_manager.get_slave_uri(slave_config)
            
            if uri:
                ok = self.slave_manager.add_slave(slave_id, uri)
                if not ok:
                    self.log.error(f"Failed to init slave {slave_id} with uri={uri}")
    
    def _init_orchestrator(self):
        """Инициализирует оркестратор."""
        self.orchestrator = Orchestrator(self.log)
    
    def _init_peak_managers(self):
        """Инициализирует менеджеры пиков и координатор трилатерации."""
        self.peak_watchlist_manager = PeakWatchlistManager()
        self.trilateration_coordinator = TrilaterationCoordinator()
        
        # Устанавливаем span из UI по умолчанию
        self.trilateration_coordinator.set_user_span(5.0)  # По умолчанию 5 МГц
        
        # Применяем сохранённые настройки детектора
        try:
            det_settings = load_detector_settings()
            if det_settings:
                apply_settings_to_watchlist_manager(det_settings, self.trilateration_coordinator.peak_manager)
                if self.orchestrator:
                    self.orchestrator.set_global_parameters(
                        span_hz=det_settings.rms_halfspan_mhz * 2e6,  # Полная ширина = 2 × halfspan
                        dwell_ms=int(det_settings.watchlist_dwell_ms)
                    )
                self.trilateration_coordinator.set_user_span(float(det_settings.rms_halfspan_mhz))
        except Exception as e:
            self.log.warning(f"Failed to load detector settings: {e}")
    
    def _configure_orchestrator(self):
        """Настраивает оркестратор с компонентами."""
        if not self.orchestrator:
            return
        
        # Подключаем компоненты к оркестратору
        self.orchestrator.set_master_controller(self.master_controller)
        self.orchestrator.set_slave_manager(self.slave_manager)
        self.orchestrator.set_trilateration_engine(self.trilateration_engine)
        
        # Подключаем координатор трилатерации к оркестратору
        if self.trilateration_coordinator:
            self.orchestrator.set_trilateration_coordinator(self.trilateration_coordinator)
    
    def _setup_trilateration_engine(self):
        """Настраивает движок трилатерации."""
        if not self.trilateration_engine or not self.calibration_manager:
            return
        
        try:
            # Получаем параметры калибровки
            cal_params = self.calibration_manager.get_calibration_parameters()
            
            if cal_params:
                self.trilateration_engine.set_path_loss_exponent(cal_params['path_loss_exponent'])
                self.trilateration_engine.set_reference_parameters(
                    cal_params['reference_distance'],
                    cal_params['reference_power']
                )
            
            # Очищаем все станции
            self.trilateration_engine.stations.clear()
            
            # Добавляем slaves в трилатерацию (Master исключен - только спектр)
            # ВАЖНО: нормализуем идентификаторы к 'slave0/1/2' чтобы совпадать с теми,
            # которые используются в измерениях и оркестраторе
            slave_positions = self.config_manager.get_slave_positions()
            for idx, (_orig_id, pos) in enumerate(list(slave_positions.items())[:3]):
                x, y, z = pos
                norm_id = f"slave{idx}"
                self.trilateration_engine.add_station(norm_id, float(x), float(y), float(z), 0.0)
            
            stations_count = len(self.trilateration_engine.get_station_positions())
            self.log.info(f"Trilateration engine configured with {stations_count} stations (Master excluded - spectrum only)")
            
        except Exception as e:
            self.log.error(f"Error setting up trilateration: {e}")
    
    def get_component(self, component_name: str):
        """Получает компонент по имени."""
        return getattr(self, component_name, None)
    
    def cleanup_all_components(self):
        """Очищает все компоненты при закрытии приложения."""
        try:
            if self.trilateration_coordinator:
                try:
                    self.trilateration_coordinator.stop()
                except Exception:
                    pass
                    
            if self.master_controller and hasattr(self.master_controller, 'cleanup'):
                try:
                    self.master_controller.stop_sweep()
                    self.master_controller.cleanup()
                except Exception:
                    pass
            
            if self.orchestrator:
                try:
                    self.orchestrator.stop()
                except Exception:
                    pass
            
            if self.slave_manager:
                try:
                    self.slave_manager.close_all()
                except Exception:
                    pass
                    
            self.log.info("All components cleaned up")
            
        except Exception as e:
            self.log.error(f"Error during components cleanup: {e}")
    
    def refresh_slaves_configuration(self):
        """Обновляет конфигурацию слейвов."""
        try:
            self._init_slaves_from_config()
            self._setup_trilateration_engine()
            self.log.info("Slaves configuration refreshed")
        except Exception as e:
            self.log.error(f"Error refreshing slaves configuration: {e}")