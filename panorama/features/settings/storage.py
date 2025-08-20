from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

SETTINGS_PATH = Path.home() / ".panorama" / "sdr_settings.json"


def load_sdr_settings() -> Dict[str, Any]:
    try:
        if SETTINGS_PATH.exists():
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "master": {"nickname": "Master", "serial": "", "pos": [0.0, 0.0, 0.0]},
        "slaves": []  # [{nickname, uri, pos:[x,y,z], driver, serial, label}]
    }


def save_sdr_settings(data: Dict[str, Any]) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
