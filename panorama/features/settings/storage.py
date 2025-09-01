from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

SETTINGS_PATH = Path.home() / ".panorama" / "device_settings.json"
DETECTOR_SETTINGS_PATH = Path.home() / ".panorama" / "signal_processing_settings.json"


def load_sdr_settings() -> Dict[str, Any]:
    try:
        if SETTINGS_PATH.exists():
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            print(f"DEBUG: Loaded config: {json.dumps(data, indent=2)}")

            # Очищаем фиктивные устройства из сохраненной конфигурации
            if 'master' in data:
                master_serial = data['master'].get('serial', '')
                if not master_serial or len(master_serial) != 32:
                    print(f"DEBUG: Removing invalid master serial: {master_serial}")
                    data['master'] = {}

            if 'slaves' in data:
                valid_slaves = []
                for slave in data['slaves']:
                    slave_serial = slave.get('serial', '')
                    if slave_serial and len(slave_serial) >= 16:  # Минимальная длина для реального серийника
                        valid_slaves.append(slave)
                    else:
                        print(f"DEBUG: Removing invalid slave: {slave}")
                data['slaves'] = valid_slaves

            return data
    except Exception as e:
        print(f"DEBUG: Error loading config: {e}")

    # Возвращаем пустую конфигурацию вместо дефолтной
    return {
        "master": {},
        "slaves": []
    }


def save_sdr_settings(data: Dict[str, Any]) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def load_detector_settings() -> Dict[str, Any]:
    """Загружает настройки детектора из файла."""
    try:
        if DETECTOR_SETTINGS_PATH.exists():
            data = json.loads(DETECTOR_SETTINGS_PATH.read_text(encoding="utf-8"))
            print(f"DEBUG: Loaded detector settings: {json.dumps(data, indent=2)}")
            return data
    except Exception as e:
        print(f"DEBUG: Error loading detector settings: {e}")

    # Возвращаем дефолтные настройки детектора
    return {
        "coverage_threshold": 0.85,  # Уменьшено с 0.95 для лучшего покрытия
        "snr_threshold_db": 10.0,
        "min_peak_bins": 3,
        "min_peak_distance_bins": 5,
        "peak_band_hz": 5e6,
        "smoothing_enabled": True,   # Включено по умолчанию
        "smoothing_window": 7,
        "ema_enabled": True,         # Включено по умолчанию
        "ema_alpha": 0.3,
        # RMS параметры для трилатерации
        "rms_halfspan_hz": 2500000,  # 2.5 МГц по умолчанию
        "rms_halfspan_limits_hz": [2000000, 6000000],  # Пределы 2-6 МГц
        "recentering_trigger_fraction": 0.3,
        "cluster": {
            "min_bw_hz": 1500000,
            "max_bw_hz": 12000000,
            "merge_if_gap_hz": 2000000
        }
    }


def save_detector_settings(data: Dict[str, Any]) -> None:
    """Сохраняет настройки детектора в файл."""
    try:
        DETECTOR_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        DETECTOR_SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"DEBUG: Saved detector settings: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"DEBUG: Error saving detector settings: {e}")


def get_coverage_threshold() -> float:
    """Получает значение coverage_threshold из настроек детектора."""
    settings = load_detector_settings()
    return float(settings.get("coverage_threshold", 0.95))


# interpolation setting removed
