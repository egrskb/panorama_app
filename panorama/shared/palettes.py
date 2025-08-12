from __future__ import annotations
import numpy as np

# Набор простых градиентов на случай отсутствия matplotlib
PRESET_GRADIENTS = {
    "viridis": ["#440154", "#30678D", "#35B779", "#FDE725"],
    "plasma":  ["#0D0887", "#6A00A8", "#CB4778", "#F0F921"],
    "inferno": ["#000004", "#420A68", "#932667", "#F1605D", "#FCFFA4"],
    "magma":   ["#000004", "#3B0F70", "#8C2981", "#F46D43", "#FBFDBF"],
    "turbo":   ["#30123B", "#4145AB", "#2AB0C5", "#7AD151", "#F9E721"],
    "gray":    ["#000000", "#FFFFFF"],
}

def _hex_to_rgb(hex_str: str) -> np.ndarray:
    h = hex_str.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)

def _grad_lut_hex(colors: list[str], n: int = 256) -> np.ndarray:
    stops = np.linspace(0.0, 1.0, num=len(colors), dtype=np.float32)
    target = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    rgb = np.stack([_hex_to_rgb(c) for c in colors], axis=0)  # (k, 3)
    out = np.empty((n, 4), dtype=np.uint8)
    for ch in range(3):
        out[:, ch] = np.clip(np.interp(target, stops, rgb[:, ch]), 0, 255).astype(np.uint8)
    out[:, 3] = 255  # alpha
    return out

def _matplotlib_lut(name: str, n: int) -> np.ndarray | None:
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(name, n)
        arr = (cmap(np.linspace(0, 1, n)) * 255.0).astype(np.uint8)  # (n,4) RGBA
        return arr
    except Exception:
        return None

def get_colormap(name: str = "viridis", n: int = 256) -> np.ndarray:
    """
    Возвращает LUT (n,4) uint8 RGBA. Сначала пытается взять из matplotlib,
    если нет — строит из PRESET_GRADIENTS.
    """
    name = (name or "viridis").lower()
    lut = _matplotlib_lut(name, n)
    if lut is not None:
        return lut
    colors = PRESET_GRADIENTS.get(name, PRESET_GRADIENTS["viridis"])
    return _grad_lut_hex(colors, n)
