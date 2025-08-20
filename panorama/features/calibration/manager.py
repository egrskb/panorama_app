"""
Calibration manager for SDR stations.
Handles loading, saving, and applying calibration parameters.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal


@dataclass
class StationCalibration:
    """Калибровочные параметры SDR станции."""
    station_id: str
    k_cal_db: float          # Калибровочный коэффициент (дБ)
    pos_x: float             # X координата (м)
    pos_y: float             # Y координата (м)
    pos_z: float             # Z координата (м)
    description: str          # Описание станции
    last_calibrated: str     # Дата последней калибровки
    temperature: float        # Температура при калибровке (°C)


@dataclass
class CalibrationProfile:
    """Профиль калибровки для системы."""
    name: str
    description: str
    path_loss_exponent: float    # Показатель затухания
    reference_distance: float    # Опорное расстояние (м)
    reference_power: float       # Опорная мощность (дБм)
    stations: Dict[str, StationCalibration]
    created_at: str
    updated_at: str


class CalibrationManager(QObject):
    """Менеджер калибровки SDR станций."""
    
    # Сигналы
    calibration_loaded = pyqtSignal(str)      # Имя профиля
    calibration_saved = pyqtSignal(str)       # Имя профиля
    calibration_error = pyqtSignal(str)       # Ошибка калибровки
    
    def __init__(self, logger: logging.Logger, config_dir: str = None):
        super().__init__()
        self.log = logger
        
        # Директория конфигурации
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path.home() / ".panorama" / "calibration"
        
        # Создаем директорию если не существует
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Текущий профиль
        self.current_profile: Optional[CalibrationProfile] = None
        
        # Список доступных профилей
        self.available_profiles: List[str] = []
        
        # Загружаем список профилей
        self._load_profile_list()
        
        # Загружаем профиль по умолчанию
        self._load_default_profile()
    
    def _load_profile_list(self):
        """Загружает список доступных профилей."""
        try:
            profile_files = list(self.config_dir.glob("*.json"))
            self.available_profiles = [f.stem for f in profile_files]
            self.log.info(f"Found {len(self.available_profiles)} calibration profiles: {self.available_profiles}")
        except Exception as e:
            self.log.error(f"Error loading profile list: {e}")
            self.available_profiles = []
    
    def _load_default_profile(self):
        """Загружает профиль по умолчанию."""
        if "default" in self.available_profiles:
            self.load_profile("default")
        elif self.available_profiles:
            self.load_profile(self.available_profiles[0])
        else:
            self._create_default_profile()
    
    def _create_default_profile(self):
        """Создает профиль по умолчанию."""
        try:
            # Создаем станции по умолчанию
            stations = {
                "slave01": StationCalibration(
                    station_id="slave01",
                    k_cal_db=-33.2,
                    pos_x=0.0,
                    pos_y=0.0,
                    pos_z=0.0,
                    description="Default Slave 1",
                    last_calibrated="2024-01-01",
                    temperature=25.0
                ),
                "slave02": StationCalibration(
                    station_id="slave02",
                    k_cal_db=-31.8,
                    pos_x=100.0,
                    pos_y=0.0,
                    pos_z=0.0,
                    description="Default Slave 2",
                    last_calibrated="2024-01-01",
                    temperature=25.0
                ),
                "slave03": StationCalibration(
                    station_id="slave03",
                    k_cal_db=-34.5,
                    pos_x=50.0,
                    pos_y=86.6,
                    pos_z=0.0,
                    description="Default Slave 3",
                    last_calibrated="2024-01-01",
                    temperature=25.0
                )
            }
            
            # Создаем профиль
            profile = CalibrationProfile(
                name="default",
                description="Default calibration profile",
                path_loss_exponent=2.7,
                reference_distance=1.0,
                reference_power=-30.0,
                stations=stations,
                created_at="2024-01-01",
                updated_at="2024-01-01"
            )
            
            # Сохраняем профиль
            self.save_profile(profile)
            
            # Загружаем как текущий
            self.current_profile = profile
            self.log.info("Created and loaded default calibration profile")
            
        except Exception as e:
            self.log.error(f"Error creating default profile: {e}")
    
    def load_profile(self, profile_name: str) -> bool:
        """Загружает профиль калибровки."""
        try:
            profile_path = self.config_dir / f"{profile_name}.json"
            
            if not profile_path.exists():
                self.log.error(f"Profile {profile_name} not found")
                return False
            
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
            
            # Создаем объект профиля
            stations = {}
            for station_id, station_data in profile_data.get('stations', {}).items():
                stations[station_id] = StationCalibration(**station_data)
            
            profile = CalibrationProfile(
                name=profile_data['name'],
                description=profile_data['description'],
                path_loss_exponent=profile_data['path_loss_exponent'],
                reference_distance=profile_data['reference_distance'],
                reference_power=profile_data['reference_power'],
                stations=stations,
                created_at=profile_data['created_at'],
                updated_at=profile_data['updated_at']
            )
            
            self.current_profile = profile
            self.log.info(f"Loaded calibration profile: {profile_name}")
            
            # Эмитим сигнал
            self.calibration_loaded.emit(profile_name)
            
            return True
            
        except Exception as e:
            error_msg = f"Error loading profile {profile_name}: {e}"
            self.log.error(error_msg)
            self.calibration_error.emit(error_msg)
            return False
    
    def save_profile(self, profile: CalibrationProfile) -> bool:
        """Сохраняет профиль калибровки."""
        try:
            # Обновляем время изменения
            profile.updated_at = self._get_current_time()
            
            # Преобразуем в словарь
            profile_data = asdict(profile)
            
            # Сохраняем в файл
            profile_path = self.config_dir / f"{profile.name}.json"
            
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=2, ensure_ascii=False)
            
            # Обновляем список профилей
            if profile.name not in self.available_profiles:
                self.available_profiles.append(profile.name)
            
            self.log.info(f"Saved calibration profile: {profile.name}")
            
            # Эмитим сигнал
            self.calibration_saved.emit(profile.name)
            
            return True
            
        except Exception as e:
            error_msg = f"Error saving profile {profile.name}: {e}"
            self.log.error(error_msg)
            self.calibration_error.emit(error_msg)
            return False
    
    def create_profile(self, name: str, description: str = "") -> Optional[CalibrationProfile]:
        """Создает новый профиль калибровки."""
        try:
            if name in self.available_profiles:
                self.log.error(f"Profile {name} already exists")
                return None
            
            # Создаем пустой профиль
            profile = CalibrationProfile(
                name=name,
                description=description,
                path_loss_exponent=2.7,
                reference_distance=1.0,
                reference_power=-30.0,
                stations={},
                created_at=self._get_current_time(),
                updated_at=self._get_current_time()
            )
            
            # Сохраняем профиль
            if self.save_profile(profile):
                return profile
            
            return None
            
        except Exception as e:
            self.log.error(f"Error creating profile {name}: {e}")
            return None
    
    def delete_profile(self, profile_name: str) -> bool:
        """Удаляет профиль калибровки."""
        try:
            if profile_name == "default":
                self.log.error("Cannot delete default profile")
                return False
            
            profile_path = self.config_dir / f"{profile_name}.json"
            
            if profile_path.exists():
                profile_path.unlink()
                
                # Удаляем из списка
                if profile_name in self.available_profiles:
                    self.available_profiles.remove(profile_name)
                
                # Если удаляемый профиль был текущим, загружаем профиль по умолчанию
                if (self.current_profile and 
                    self.current_profile.name == profile_name):
                    self._load_default_profile()
                
                self.log.info(f"Deleted calibration profile: {profile_name}")
                return True
            
            return False
            
        except Exception as e:
            self.log.error(f"Error deleting profile {profile_name}: {e}")
            return False
    
    def add_station(self, station_id: str, k_cal_db: float, pos_x: float, 
                    pos_y: float, pos_z: float = 0.0, description: str = "") -> bool:
        """Добавляет станцию в текущий профиль."""
        if not self.current_profile:
            self.log.error("No current profile")
            return False
        
        try:
            station = StationCalibration(
                station_id=station_id,
                k_cal_db=k_cal_db,
                pos_x=pos_x,
                pos_y=pos_y,
                pos_z=pos_z,
                description=description or f"Station {station_id}",
                last_calibrated=self._get_current_time(),
                temperature=25.0
            )
            
            self.current_profile.stations[station_id] = station
            
            # Сохраняем профиль
            return self.save_profile(self.current_profile)
            
        except Exception as e:
            self.log.error(f"Error adding station {station_id}: {e}")
            return False
    
    def remove_station(self, station_id: str) -> bool:
        """Удаляет станцию из текущего профиля."""
        if not self.current_profile:
            self.log.error("No current profile")
            return False
        
        try:
            if station_id in self.current_profile.stations:
                del self.current_profile.stations[station_id]
                
                # Сохраняем профиль
                return self.save_profile(self.current_profile)
            
            return False
            
        except Exception as e:
            self.log.error(f"Error removing station {station_id}: {e}")
            return False
    
    def update_station_calibration(self, station_id: str, **kwargs) -> bool:
        """Обновляет калибровку станции."""
        if not self.current_profile:
            self.log.error("No current profile")
            return False
        
        try:
            if station_id not in self.current_profile.stations:
                self.log.error(f"Station {station_id} not found")
                return False
            
            station = self.current_profile.stations[station_id]
            
            # Обновляем поля
            for key, value in kwargs.items():
                if hasattr(station, key):
                    setattr(station, key, value)
            
            # Обновляем время калибровки
            station.last_calibrated = self._get_current_time()
            
            # Сохраняем профиль
            return self.save_profile(self.current_profile)
            
        except Exception as e:
            self.log.error(f"Error updating station {station_id}: {e}")
            return False
    
    def get_station_calibration(self, station_id: str) -> Optional[StationCalibration]:
        """Возвращает калибровку станции."""
        if not self.current_profile:
            return None
        
        return self.current_profile.stations.get(station_id)
    
    def get_all_stations(self) -> Dict[str, StationCalibration]:
        """Возвращает все станции текущего профиля."""
        if not self.current_profile:
            return {}
        
        return self.current_profile.stations.copy()
    
    def get_station_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Возвращает позиции всех станций."""
        if not self.current_profile:
            return {}
        
        positions = {}
        for station_id, station in self.current_profile.stations.items():
            positions[station_id] = (station.pos_x, station.pos_y, station.pos_z)
        
        return positions
    
    def get_calibration_parameters(self) -> Dict:
        """Возвращает параметры калибровки."""
        if not self.current_profile:
            return {}
        
        return {
            'path_loss_exponent': self.current_profile.path_loss_exponent,
            'reference_distance': self.current_profile.reference_distance,
            'reference_power': self.current_profile.reference_power
        }
    
    def set_calibration_parameters(self, **kwargs) -> bool:
        """Устанавливает параметры калибровки."""
        if not self.current_profile:
            self.log.error("No current profile")
            return False
        
        try:
            # Обновляем параметры
            for key, value in kwargs.items():
                if hasattr(self.current_profile, key):
                    setattr(self.current_profile, key, value)
            
            # Сохраняем профиль
            return self.save_profile(self.current_profile)
            
        except Exception as e:
            self.log.error(f"Error setting calibration parameters: {e}")
            return False
    
    def export_profile(self, profile_name: str, export_path: str) -> bool:
        """Экспортирует профиль в указанный путь."""
        try:
            # Загружаем профиль
            if not self.load_profile(profile_name):
                return False
            
            # Экспортируем
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.current_profile), f, indent=2, ensure_ascii=False)
            
            self.log.info(f"Exported profile {profile_name} to {export_path}")
            return True
            
        except Exception as e:
            self.log.error(f"Error exporting profile {profile_name}: {e}")
            return False
    
    def import_profile(self, import_path: str) -> bool:
        """Импортирует профиль из указанного пути."""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
            
            # Проверяем структуру
            required_fields = ['name', 'stations', 'path_loss_exponent', 'reference_distance', 'reference_power']
            for field in required_fields:
                if field not in profile_data:
                    self.log.error(f"Missing required field: {field}")
                    return False
            
            # Создаем профиль
            stations = {}
            for station_id, station_data in profile_data.get('stations', {}).items():
                stations[station_id] = StationCalibration(**station_data)
            
            profile = CalibrationProfile(
                name=profile_data['name'],
                description=profile_data.get('description', ''),
                path_loss_exponent=profile_data['path_loss_exponent'],
                reference_distance=profile_data['reference_distance'],
                reference_power=profile_data['reference_power'],
                stations=stations,
                created_at=profile_data.get('created_at', self._get_current_time()),
                updated_at=self._get_current_time()
            )
            
            # Сохраняем профиль
            return self.save_profile(profile)
            
        except Exception as e:
            self.log.error(f"Error importing profile from {import_path}: {e}")
            return False
    
    def get_profile_info(self, profile_name: str) -> Optional[Dict]:
        """Возвращает информацию о профиле."""
        try:
            profile_path = self.config_dir / f"{profile_name}.json"
            
            if not profile_path.exists():
                return None
            
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
            
            return {
                'name': profile_data['name'],
                'description': profile_data.get('description', ''),
                'n_stations': len(profile_data.get('stations', {})),
                'created_at': profile_data.get('created_at', ''),
                'updated_at': profile_data.get('updated_at', ''),
                'path_loss_exponent': profile_data.get('path_loss_exponent', 2.7),
                'reference_distance': profile_data.get('reference_distance', 1.0),
                'reference_power': profile_data.get('reference_power', -30.0)
            }
            
        except Exception as e:
            self.log.error(f"Error getting profile info for {profile_name}: {e}")
            return None
    
    def _get_current_time(self) -> str:
        """Возвращает текущее время в строковом формате."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")
    
    def get_status(self) -> Dict:
        """Возвращает статус менеджера калибровки."""
        return {
            'current_profile': self.current_profile.name if self.current_profile else None,
            'available_profiles': self.available_profiles,
            'n_stations': len(self.current_profile.stations) if self.current_profile else 0,
            'config_dir': str(self.config_dir)
        }
