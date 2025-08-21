from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

SETTINGS_PATH = Path.home() / ".panorama" / "sdr_settings.json"


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
