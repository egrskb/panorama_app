"""
Менеджер конфигурации для ПАНОРАМА RSSI.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from PyQt5.QtCore import QObject, pyqtSignal

from panorama.features.settings.storage import load_sdr_settings, save_sdr_settings


class ConfigurationManager(QObject):
    """Менеджер для управления конфигурацией системы."""
    
    config_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None, logger: Optional[logging.Logger] = None):
        super().__init__(parent)
        
        self.log = logger or logging.getLogger(__name__)
        self._config = {}
        self.load_configuration()
    
    def load_configuration(self) -> bool:
        """Загружает конфигурацию из файла."""
        try:
            self._config = load_sdr_settings() or {}
            self.config_updated.emit(self._config.copy())
            self.log.info("Configuration loaded successfully")
            return True
        except Exception as e:
            self.log.error(f"Error loading configuration: {e}")
            return False
    
    def save_configuration(self) -> bool:
        """Сохраняет конфигурацию в файл."""
        try:
            save_sdr_settings(self._config)
            self.log.info("Configuration saved successfully")
            return True
        except Exception as e:
            self.log.error(f"Error saving configuration: {e}")
            return False
    
    def update_configuration(self, new_config: Dict[str, Any]) -> None:
        """Обновляет конфигурацию."""
        self._config.update(new_config)
        self.config_updated.emit(self._config.copy())
    
    def get_master_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию master устройства."""
        return self._config.get('master', {})
    
    def get_slaves_config(self) -> List[Dict[str, Any]]:
        """Возвращает конфигурацию slave устройств."""
        return self._config.get('slaves', [])
    
    def get_slave_config(self, slave_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает конфигурацию конкретного slave устройства."""
        for slave in self.get_slaves_config():
            if (slave.get('nickname') == slave_id or 
                slave.get('label') == slave_id or 
                slave.get('serial') == slave_id):
                return slave
        return None
    
    def is_master_configured(self) -> bool:
        """Проверяет, настроено ли master устройство."""
        master_config = self.get_master_config()
        return bool(master_config.get('serial') or master_config.get('uri'))
    
    def get_slave_positions(self) -> Dict[str, tuple]:
        """Возвращает позиции всех slave устройств."""
        positions = {}
        for slave in self.get_slaves_config():
            slave_id = slave.get('nickname') or slave.get('label') or slave.get('serial')
            if slave_id:
                pos = slave.get('pos', [0.0, 0.0, 0.0])
                if len(pos) >= 2:
                    positions[slave_id] = (float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0)
        return positions
    
    def validate_master_config(self) -> tuple[bool, str]:
        """Валидирует конфигурацию master устройства."""
        master_config = self.get_master_config()
        
        if not master_config:
            return False, "Master device not configured"
        
        serial = master_config.get('serial', '')
        if not serial or len(serial) < 16:
            return False, f"Invalid master serial: {serial}"
        
        return True, "Master configuration is valid"
    
    def get_slave_uri(self, slave_config: Dict[str, Any]) -> str:
        """Извлекает URI для slave устройства из его конфигурации."""
        uri = slave_config.get('uri') or slave_config.get('soapy')
        
        if not uri and slave_config.get('driver'):
            uri = f"driver={slave_config.get('driver')}"
        
        if not uri and slave_config.get('serial'):
            uri = f"serial={slave_config.get('serial')}"
        
        return uri or ""
    
    def generate_slave_id(self, slave_config: Dict[str, Any], index: int = 1) -> str:
        """Генерирует ID для slave устройства."""
        return (slave_config.get('nickname') or 
                slave_config.get('label') or 
                slave_config.get('serial') or 
                f"slave{index:02d}")
    
    def get_full_config(self) -> Dict[str, Any]:
        """Возвращает полную конфигурацию."""
        return self._config.copy()