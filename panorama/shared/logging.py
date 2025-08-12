from __future__ import annotations
import logging, os, pathlib, sys
from logging.handlers import RotatingFileHandler

def _state_dir() -> pathlib.Path:
    # ~/.local/state/panorama/logs  (Linux); иначе — ./logs
    base = os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state"))
    p = pathlib.Path(base) / "panorama" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p

def setup_logging(name: str = "panorama") -> logging.Logger:
    lvl_name = os.environ.get("PANORAMA_LOG", "INFO").upper()
    level = getattr(logging, lvl_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    # файл с ротацией
    log_path = _state_dir() / "panorama.log"
    fh = RotatingFileHandler(str(log_path), maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # консоль (полезно при запуске из терминала)
    sh = logging.StreamHandler(stream=sys.stderr)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("Logging initialized → %s", log_path)
    return logger
