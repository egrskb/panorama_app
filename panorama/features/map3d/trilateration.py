from __future__ import annotations
import numpy as np

def trilaterate_2d(p1, r1, p2, r2, p3, r3):
    """
    Заглушка-решатель (Евклид), возвращает (x,y) или None, если система плохая.
    В реальном проекте сюда придёт RSSI->distance и MLE/LSQ.
    """
    x1,y1 = p1; x2,y2 = p2; x3,y3 = p3
    A = np.array([
        [2*(x2-x1), 2*(y2-y1)],
        [2*(x3-x1), 2*(y3-y1)],
    ], dtype=float)
    b = np.array([
        r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2,
        r1**2 - r3**2 - x1**2 + x3**2 - y1**2 + y3**2,
    ], dtype=float)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return float(sol[0]), float(sol[1])
    except Exception:
        return None
