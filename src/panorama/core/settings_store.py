import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

from PyQt5 import QtCore

APP_DIR = {
    "win32": Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "PANORAMA",
}.get(sys.platform, Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "PANORAMA")
USER_PATH = APP_DIR / "user.json"


def _ensure_dir():
    APP_DIR.mkdir(parents=True, exist_ok=True)


def _load_qsettings() -> Dict[str, Any]:
    settings = QtCore.QSettings("PANORAMA", "PANORAMA")
    data: Dict[str, Any] = {}
    for key in settings.allKeys():
        data[key] = settings.value(key)
    return data


def load_user() -> Dict[str, Any]:
    """Load user state from JSON, migrating from QSettings if needed."""
    _ensure_dir()
    if USER_PATH.exists():
        with USER_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    data = _load_qsettings()
    save_user(data)
    return data


def save_user(data: Dict[str, Any]) -> None:
    _ensure_dir()
    with USER_PATH.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def update_field(path: Iterable[str] | str, value: Any) -> None:
    data = load_user()
    if isinstance(path, str):
        path = path.split(".")
    d = data
    for p in path[:-1]:
        d = d.setdefault(p, {})
    d[path[-1]] = value
    save_user(data)


def get(path: Iterable[str] | str, default: Any = None) -> Any:
    data = load_user()
    if isinstance(path, str):
        path = path.split(".")
    d = data
    for p in path:
        if isinstance(d, dict) and p in d:
            d = d[p]
        else:
            return default
    return d
