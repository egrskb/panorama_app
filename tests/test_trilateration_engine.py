import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from panorama.features.trilateration.engine import (
    TrilaterationEngine,
    SynchronizedMeasurement,
)

def rssi_from_distance(engine, dist):
    return engine.REFERENCE_POWER_DBM - 10 * engine.PATH_LOSS_EXPONENT * np.log10(dist / engine.REFERENCE_DISTANCE_M)

def test_calculate_position_with_altitudes():
    engine = TrilaterationEngine()
    master = (0.0, 0.0, 1.0)
    slave1 = (10.0, 0.0, 2.0)
    slave2 = (0.0, 10.0, 3.0)
    engine.set_device_positions(master, slave1, slave2, "m", "s1", "s2")

    target = np.array([3.0, 4.0, 5.0])
    d_master = np.linalg.norm(target - np.array(master))
    d_slave1 = np.linalg.norm(target - np.array(slave1))
    d_slave2 = np.linalg.norm(target - np.array(slave2))

    measurement = SynchronizedMeasurement(
        timestamp=0.0,
        freq_mhz=100.0,
        master_power=rssi_from_distance(engine, d_master),
        slave1_power=rssi_from_distance(engine, d_slave1),
        slave2_power=rssi_from_distance(engine, d_slave2),
    )

    pos = engine._calculate_position(measurement)
    assert pos is not None
    assert np.allclose([pos.x, pos.y, pos.z], target, atol=1e-2)
