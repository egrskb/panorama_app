import json
import logging
from pathlib import Path
from copy import deepcopy

LOGGER = logging.getLogger("panorama")

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
DEFAULT_PATH = CONFIG_DIR / "app.default.json"
SCHEMA_PATH = CONFIG_DIR / "schema" / "app.schema.json"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_default() -> dict:
    return _load_json(DEFAULT_PATH)


def load_user_override() -> dict:
    user_path = Path.home() / ".config" / "PANORAMA" / "user.json"
    if user_path.exists():
        with user_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _merge_dicts(a: dict, b: dict) -> dict:
    res = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(res.get(k), dict):
            res[k] = _merge_dicts(res[k], v)
        else:
            res[k] = deepcopy(v)
    return res


def merged_config() -> dict:
    base = load_default()
    override = load_user_override()
    cfg = _merge_dicts(base, override)
    try:
        validate_with_schema(cfg)
    except Exception as exc:  # pragma: no cover - validation may not be available
        LOGGER.warning("config validation failed: %s", exc)
    return cfg


def validate_with_schema(data: dict) -> None:
    try:
        import jsonschema
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("jsonschema not available: %s", exc)
        return
    schema = _load_json(SCHEMA_PATH)
    jsonschema.validate(data, schema)
