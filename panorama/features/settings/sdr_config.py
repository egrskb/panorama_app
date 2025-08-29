"""
Менеджер для сохранения/загрузки параметров SDR в JSON
Файл: panorama/features/settings/sdr_config.py
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class SDRConfig:
    """Конфигурация параметров SDR."""
    # Частотный диапазон
    freq_start_mhz: float = 50.0
    freq_stop_mhz: float = 6000.0
    bin_khz: float = 5.0  # 5 кГц по умолчанию для высокого разрешения
    
    # Параметры усиления
    lna_db: int = 24
    vga_db: int = 20
    amp_on: bool = False
    
    # Параметры отображения
    waterfall_rows: int = 100
    
    # Сглаживание
    smoothing_enabled: bool = True
    smoothing_window: int = 7
    ema_enabled: bool = True
    ema_alpha: float = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь для JSON."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SDRConfig':
        """Создает из словаря."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SDRConfigManager:
    """Менеджер конфигурации SDR."""
    
    DEFAULT_CONFIG_PATH = Path.home() / ".panorama" / "sdr_config.json"
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = self.load() or SDRConfig()
    
    def load(self) -> Optional[SDRConfig]:
        """Загружает конфигурацию из JSON."""
        if not self.config_path.exists():
            print(f"[SDRConfig] Файл конфигурации не найден: {self.config_path}")
            return None
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            config = SDRConfig.from_dict(data)
            print(f"[SDRConfig] Загружена конфигурация из {self.config_path}")
            return config
            
        except Exception as e:
            print(f"[SDRConfig] Ошибка загрузки конфигурации: {e}")
            return None
    
    def save(self, config: Optional[SDRConfig] = None) -> bool:
        """Сохраняет конфигурацию в JSON."""
        if config:
            self.config = config
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
            
            print(f"[SDRConfig] Конфигурация сохранена в {self.config_path}")
            return True
            
        except Exception as e:
            print(f"[SDRConfig] Ошибка сохранения конфигурации: {e}")
            return False
    
    def update_from_gui(self, **kwargs) -> None:
        """Обновляет параметры из GUI."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    

    

