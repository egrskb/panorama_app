import importlib.util
import pytest

# Avoid mixed Qt bindings (PySide6 vs PyQt5) in CI/WSL causing aborts
if importlib.util.find_spec("PySide6") is not None:
    pytest.skip("Skipping MapView tests due to PySide6 present (mixing with PyQt5 causes abort)", allow_module_level=True)

from panorama.features.map3d.map2d import MapView


def test_map2d_update_stations(qtbot):
    mv = MapView()
    qtbot.addWidget(mv)
    cfg = {
        'slaves': [
            {'nickname': 'S1', 'pos': [5.0, 5.0, 0.0]},
            {'nickname': 'S2', 'pos': [10.0, 0.0, 0.0]},
            {'nickname': 'S3', 'pos': [0.0, 10.0, 0.0]},
        ]
    }
    mv.update_stations_from_config(cfg)
    # Первый slave фиксирован в (0,0,0)
    s1 = mv.stations.get('slave_1')
    assert s1 is not None
    assert s1.position == (0.0, 0.0, 0.0)

