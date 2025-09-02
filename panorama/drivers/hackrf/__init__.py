"""
HackRF driver package with master and slave implementations
"""

from .hrf_backend import (
    HackRFQSABackend, HackRFSlaveDevice, 
    get_slave_device_count, list_slave_devices,
    HackRFSlaveError, HackRFSlaveDeviceError, HackRFSlaveConfigError,
    HACKRF_SLAVE_WINDOW_HAMMING, HACKRF_SLAVE_WINDOW_HANN,
    HACKRF_SLAVE_WINDOW_BLACKMAN, HACKRF_SLAVE_WINDOW_KAISER
)

__all__ = [
    'HackRFQSABackend', 'HackRFSlaveDevice', 
    'get_slave_device_count', 'list_slave_devices',
    'HackRFSlaveError', 'HackRFSlaveDeviceError', 'HackRFSlaveConfigError',
    'HACKRF_SLAVE_WINDOW_HAMMING', 'HACKRF_SLAVE_WINDOW_HANN',
    'HACKRF_SLAVE_WINDOW_BLACKMAN', 'HACKRF_SLAVE_WINDOW_KAISER'
]