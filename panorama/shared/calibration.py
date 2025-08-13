from __future__ import annotations
from typing import Optional, Tuple, Dict, List
import numpy as np
import os


def load_calibration_csv(path: str) -> Dict[Tuple[int, int, int], List[Tuple[float, float]]]:
    """
    Загружает CSV калибровки формата: freq_mhz,lna_db,vga_db,amp,offset_db
    
    Возвращает словарь: {(lna, vga, amp): [(freq_mhz, offset_db), ...]}
    """
    if not os.path.isfile(path):
        return {}
    
    profiles = {}
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split(",")
                if len(parts) < 5:
                    continue
                
                try:
                    freq_mhz = float(parts[0].strip())
                    lna_db = int(parts[1].strip())
                    vga_db = int(parts[2].strip())
                    amp = int(parts[3].strip())
                    offset_db = float(parts[4].strip())
                except (ValueError, IndexError):
                    continue
                
                key = (lna_db, vga_db, amp)
                if key not in profiles:
                    profiles[key] = []
                profiles[key].append((freq_mhz, offset_db))
    
    except Exception:
        return {}
    
    # Сортируем точки по частоте для каждого профиля
    for key in profiles:
        profiles[key].sort(key=lambda x: x[0])
    
    return profiles


def get_calibration_lut(profiles: Dict, lna: int, vga: int, amp: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Получает LUT для конкретных усилений.
    
    Возвращает: (freq_mhz_array, offset_db_array) или None
    """
    key = (lna, vga, amp)
    if key not in profiles or not profiles[key]:
        return None
    
    points = profiles[key]
    freq_mhz = np.array([p[0] for p in points], dtype=np.float64)
    offset_db = np.array([p[1] for p in points], dtype=np.float64)
    
    return freq_mhz, offset_db


def apply_calibration(freqs_hz: np.ndarray, power_dbm: np.ndarray,
                      lut: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Применяет калибровку к измеренным значениям.
    
    Args:
        freqs_hz: Массив частот в Гц
        power_dbm: Массив мощностей в дБм
        lut: Кортеж (freq_mhz, offset_db) или None
    
    Returns:
        Скорректированные значения power_dbm
    """
    if lut is None:
        return power_dbm
    
    freq_mhz_lut, offset_db_lut = lut
    freq_mhz = freqs_hz / 1e6
    
    # Интерполируем оффсеты для наших частот
    offsets = np.interp(
        freq_mhz, 
        freq_mhz_lut, 
        offset_db_lut,
        left=offset_db_lut[0],
        right=offset_db_lut[-1]
    ).astype(np.float32)
    
    return power_dbm + offsets


# Для обратной совместимости со старым кодом
def load_lut(path: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Загружает простой LUT freq_hz,offset_db (старый формат)."""
    if not path or not os.path.isfile(path):
        return None
    
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                try:
                    freq = float(parts[0])
                    offs = float(parts[1])
                    data.append((freq, offs))
                except ValueError:
                    continue
    except Exception:
        return None
    
    if not data:
        return None
    
    arr = np.asarray(data, dtype=np.float64)
    freq_hz = arr[:, 0]
    off_db = arr[:, 1]
    
    # Сортируем по частоте
    idx = np.argsort(freq_hz)
    return freq_hz[idx], off_db[idx]


def apply_lut(freqs_hz: np.ndarray, power_dbm: np.ndarray,
              lut: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Применяет простой LUT (обратная совместимость)."""
    return apply_calibration(freqs_hz, power_dbm, lut)