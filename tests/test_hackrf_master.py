import os
import pytest


def _import_wrapper():
    try:
        from panorama.drivers.hackrf_master.hackrf_master_wrapper import HackRFMaster
        return HackRFMaster
    except FileNotFoundError:
        pytest.skip("libhackrf_master.so not found; build it with ./build_hackrf_master.sh")
    except Exception as e:
        pytest.skip(f"HackRF wrapper not importable: {e}")


def test_hackrf_wrapper_enumerate_and_probe():
    try:
        from panorama.drivers.hrf_backend import HackRFMaster
    except Exception:
        HackRFMaster = _import_wrapper()
    hw = HackRFMaster()
    serials = hw.enumerate_devices()
    assert isinstance(serials, list)

    # Try probe default device; if not present, skip HW-dependent part
    ok = hw.probe()
    if not ok:
        pytest.skip("No HackRF device available for probe")

    # Stats call should work without running sweep
    stats = hw.get_stats()
    assert isinstance(stats, dict)
    assert 'sweep_count' in stats


@pytest.mark.skipif(os.environ.get('RUN_HW_TESTS') != '1', reason="Set RUN_HW_TESTS=1 to run hardware start/stop test")
def test_hackrf_start_stop_smoke():
    try:
        from panorama.drivers.hrf_backend import HackRFMaster
    except Exception:
        HackRFMaster = _import_wrapper()
    hw = HackRFMaster()
    if not hw.probe():
        pytest.skip("No HackRF device available for start/stop")

    # Minimal sweep (200 kHz band) to reduce risk
    start = int(24e6)
    stop = int(24e6 + 200e3)
    bin_hz = int(200e3)
    dwell_ms = 50
    hw.start_sweep(start_hz=start, stop_hz=stop, bin_hz=bin_hz, dwell_ms=dwell_ms)
    assert hw.is_running() is True
    hw.stop_sweep()
    assert hw.is_running() is False

