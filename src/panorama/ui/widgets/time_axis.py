import pyqtgraph as pg


class TimeAxis(pg.AxisItem):
    """Time axis for waterfall plot."""

    def __init__(self, orientation: str = "left"):
        super().__init__(orientation=orientation)
        self.dt_est = 0.3

    def tickStrings(self, values, scale, spacing):  # pragma: no cover - GUI
        return [f"{-v * self.dt_est:.0f}" for v in values]
