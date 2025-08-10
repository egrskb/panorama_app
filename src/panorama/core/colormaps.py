import numpy as np

# Default gradients used if matplotlib is unavailable
PRESET_GRADIENTS = {
    "qsa": [
        "#0b0030", "#1c006b", "#3c00a8", "#6f02b3", "#a13c93",
        "#d06367", "#ef8d3e", "#f7c545", "#ffffbf",
    ],
    "plasma": ["#0d0887","#5b02a3","#8f0da4","#bb3787","#e16462","#fca636","#f0f921"],
    "inferno": ["#000003","#1f0c48","#550f6d","#88226a","#b73759","#e1642c","#fca50a","#fcffa4"],
    "magma":   ["#000004","#1b0c41","#4f116f","#822681","#b73779","#e35933","#f98e09","#fbfdbf"],
    "turbo":   ["#30123b","#3f1d9a","#2780ff","#1bbbe3","#1edeaa","#5be65f","#c2d923","#ffb000","#ff5800"],
    "gray":    ["#000000","#ffffff"],
}


def _lut_from_hex_gradient(hex_list, n=512):
    cols = []
    for h in hex_list:
        h = h.strip()
        if h.startswith("#") and len(h) == 7:
            r = int(h[1:3], 16)
            g = int(h[3:5], 16)
            b = int(h[5:7], 16)
            cols.append((r, g, b))
    if len(cols) < 2:
        cols = [(0, 0, 0), (255, 255, 255)]
    cols = np.array(cols, dtype=np.float32)
    stops = np.linspace(0.0, 1.0, len(cols))
    xi = np.linspace(0.0, 1.0, int(n))
    r = np.interp(xi, stops, cols[:, 0])
    g = np.interp(xi, stops, cols[:, 1])
    b = np.interp(xi, stops, cols[:, 2])
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def get_colormap(name="qsa", n=512):
    """Return LUT (``N``x3) for pyqtgraph."""
    name = (name or "qsa").lower()
    if name in ("plasma", "inferno", "magma", "viridis", "cividis", "turbo"):
        try:
            import matplotlib.cm as cm

            cmap = cm.get_cmap("turbo" if name == "turbo" else name)
            arr = (cmap(np.linspace(0.0, 1.0, int(n)))[:, :3] * 255.0).astype(np.uint8)
            return arr
        except Exception:
            pass
    hex_list = PRESET_GRADIENTS.get(name, PRESET_GRADIENTS["qsa"])
    return _lut_from_hex_gradient(hex_list, n)
