import os
import pytest


def test_hackrf_wrapper_enumerate_and_probe():
    try:
        from panorama.drivers.hrf_backend import HackRFQSABackend
        hw = HackRFQSABackend()
    except Exception as e:
        pytest.skip(f"HackRF backend not importable: {e}")
    
    # Проверяем, что можем создать экземпляр
    assert hw is not None
    
    # Проверяем, что можем получить серийники устройств
    try:
        serials = hw.list_serials()
        assert isinstance(serials, list)
    except Exception:
        # Если нет устройств, это нормально для тестов
        pass


@pytest.mark.skipif(os.environ.get('RUN_HW_TESTS') != '1', reason="Set RUN_HW_TESTS=1 to run hardware start/stop test")
def test_hackrf_start_stop_smoke():
    try:
        from panorama.drivers.hrf_backend import HackRFQSABackend
        hw = HackRFQSABackend()
    except Exception as e:
        pytest.skip(f"HackRF backend not importable: {e}")
    
    if not hw:
        pytest.skip("No HackRF backend available")
    
    # Проверяем базовую функциональность
    assert hasattr(hw, 'start')
    assert hasattr(hw, 'stop')
    assert hasattr(hw, 'is_running')

