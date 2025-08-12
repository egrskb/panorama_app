from __future__ import annotations
from typing import Dict, Any
import os, pathlib

try:
    import yaml  # type: ignore
except Exception:  # PyYAML может и не быть — это ок
    yaml = None  # noqa: N816


# Дефолты «из коробки»
_DEFAULTS: Dict[str, Any] = {
    "spectrum": {
        "freq_start_hz": 2_400_000_000,
        "freq_end_hz":   2_480_000_000,
        "bin_hz":        200_000,
        "lna_db":        24,
        "vga_db":        20,
        "amp_on":        False,
        "coverage_threshold": 0.95,
        "colormap": "turbo",
    }
}


def _config_home() -> pathlib.Path:
    # ~/.config/panorama
    base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    p = pathlib.Path(base) / "panorama"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml_defaults() -> Dict[str, Any]:
    """
    Читает ~/.config/panorama/panorama.yaml (или .yml) и возвращает dict с дефолтами.
    Если файла нет или нет PyYAML — возвращает {}.
    """
    cfg_dir = _config_home()
    for name in ("panorama.yaml", "panorama.yml"):
        path = cfg_dir / name
        if path.exists() and yaml is not None:
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
    return {}


def merged_defaults() -> Dict[str, Any]:
    """Сливает базовые дефолты с YAML (YAML имеет приоритет)."""
    y = load_yaml_defaults()
    out = {**_DEFAULTS}
    for k, v in y.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out
