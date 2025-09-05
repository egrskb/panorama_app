# -*- coding: utf-8 -*-
"""
CFFI wrapper for hackrf_slave C library.
Provides Python interface for HackRF slave device control and IQ data processing.
"""

from __future__ import annotations
import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import threading
import time

try:
    import cffi
    CFFI_AVAILABLE = True
except ImportError:
    CFFI_AVAILABLE = False


# Exception hierarchy for hackrf_slave errors
class HackRFSlaveError(Exception):
    """Base exception for HackRF slave errors."""
    pass

class HackRFSlaveDeviceError(HackRFSlaveError):
    """Device-related errors."""
    pass

class HackRFSlaveConfigError(HackRFSlaveError):
    """Configuration errors."""
    pass

class HackRFSlaveCaptureError(HackRFSlaveError):
    """Data capture errors."""
    pass

class HackRFSlaveProcessingError(HackRFSlaveError):
    """Signal processing errors."""
    pass

class HackRFSlaveTimeoutError(HackRFSlaveError):
    """Timeout errors."""
    pass


@dataclass
class SlaveConfig:
    """Configuration for HackRF slave device."""
    center_freq_hz: int
    sample_rate_hz: int
    lna_gain: int = 14
    vga_gain: int = 20
    amp_enable: bool = False
    window_type: int = 1  # HAMMING
    dc_offset_correction: bool = True
    iq_balance_correction: bool = True
    freq_offset_hz: float = 0.0


@dataclass 
class RSSIMeasurement:
    """RSSI measurement result from slave."""
    center_freq_hz: float
    span_hz: float
    rssi_dbm: float
    noise_floor_dbm: float
    snr_db: float
    timestamp: float
    sample_count: int


class HackRFSlaveDevice:
    """CFFI wrapper for HackRF slave device."""
    
    def __init__(self, serial: str = "", logger: Optional[logging.Logger] = None):
        if not CFFI_AVAILABLE:
            raise HackRFSlaveError("CFFI not available - install with: pip install cffi")
            
        self.serial = serial
        self.logger = logger or logging.getLogger(__name__)
        self._ffi = None
        self._lib = None
        self._device_handle = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        
        self._load_library()
    
    def _load_library(self):
        """Load the hackrf_slave CFFI library."""
        try:
            # Find the shared library
            lib_path = Path(__file__).parent / "build" / "libhackrf_slave.so"
            if not lib_path.exists():
                # Try alternative paths
                alt_paths = [
                    Path(__file__).parent / "libhackrf_slave.so",
                    Path(__file__).parent.parent / "build" / "libhackrf_slave.so"
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        lib_path = alt_path
                        break
                else:
                    raise HackRFSlaveError(f"hackrf_slave library not found. Build it with: make -f Makefile.linux")
            
            # Create FFI instance
            self._ffi = cffi.FFI()
            
            # Define C interface based on hackrf_slave.h
            self._ffi.cdef("""
                
                // Device management
                typedef struct hackrf_slave_device hackrf_slave_device_t;
                
                // Configuration structure
                typedef struct {
                    uint32_t sample_rate;
                    uint32_t lna_gain;
                    uint32_t vga_gain;
                    bool amp_enable;
                    uint32_t bandwidth_hz;
                    double calibration_db;
                    bool dc_offset_correction;
                    double frequency_offset_hz;
                    bool iq_balance_correction;
                    uint32_t filter_window_type;
                    double filter_beta;
                    bool spectral_smoothing;
                    uint32_t smoothing_factor;
                } hackrf_slave_config_t;
                
                // RSSI measurement structure
                typedef struct {
                    char slave_id[64];
                    double center_hz;
                    double span_hz;
                    double band_rssi_dbm;
                    double band_noise_dbm;
                    double snr_db;
                    uint32_t n_samples;
                    double timestamp;
                    bool valid;
                } hackrf_slave_rssi_measurement_t;
                
                // RMS measurement structure
                typedef struct {
                    char slave_id[64];
                    char target_id[64];
                    double center_hz;
                    double halfspan_hz;
                    double guard_hz;
                    double rssi_rms_dbm;
                    double noise_floor_dbm;
                    double snr_db;
                    uint32_t n_samples;
                    double timestamp;
                    bool valid;
                } hackrf_slave_rms_measurement_t;
                
                // Core functions
                const char* hackrf_slave_last_error(void);
                int hackrf_slave_device_count(void);
                int hackrf_slave_get_serial(int index, char* serial_out, int max_len);
                hackrf_slave_device_t* hackrf_slave_open(const char* serial);
                void hackrf_slave_close(hackrf_slave_device_t* device);
                int hackrf_slave_configure(hackrf_slave_device_t* device, const hackrf_slave_config_t* config);
                int hackrf_slave_set_id(hackrf_slave_device_t* device, const char* slave_id);
                int hackrf_slave_is_ready(hackrf_slave_device_t* device);
                
                // Measurement functions
                int hackrf_slave_measure_rssi(hackrf_slave_device_t* device, double center_hz, double span_hz, 
                                             uint32_t dwell_ms, hackrf_slave_rssi_measurement_t* measurement);
                int hackrf_slave_measure_target_rms(hackrf_slave_device_t* device, const char* target_id, 
                                                   double center_hz, double halfspan_hz, double guard_hz,
                                                   uint32_t dwell_ms, hackrf_slave_rms_measurement_t* measurement);
                int hackrf_slave_get_spectrum(hackrf_slave_device_t* device, double center_hz, double span_hz,
                                             uint32_t dwell_ms, double* freqs_out, float* powers_out, 
                                             int max_points);
                int hackrf_slave_get_config(hackrf_slave_device_t* device, hackrf_slave_config_t* config_out);
            """)
            
            # Load the library
            self._lib = self._ffi.dlopen(str(lib_path))
            
            # Library is ready to use (no global init needed)
            self.logger.info(f"HackRF slave library loaded: {lib_path}")
            
            # Define constants manually since CFFI can't access them from dlopen library
            self.HACKRF_SLAVE_SUCCESS = 0
            self.HACKRF_SLAVE_ERROR_DEVICE = -1
            self.HACKRF_SLAVE_ERROR_CONFIG = -2
            self.HACKRF_SLAVE_ERROR_CAPTURE = -3
            self.HACKRF_SLAVE_ERROR_PROCESSING = -4
            self.HACKRF_SLAVE_ERROR_TIMEOUT = -5
            
        except Exception as e:
            raise HackRFSlaveError(f"Failed to load hackrf_slave library: {e}")
    
    def open(self) -> bool:
        """Open the HackRF device."""
        try:
            with self._lock:
                if self._device_handle:
                    return True
                    
                serial_cstr = self._ffi.new("char[]", self.serial.encode('utf-8')) if self.serial else self._ffi.NULL
                self._device_handle = self._lib.hackrf_slave_open(serial_cstr)
                
                if self._device_handle == self._ffi.NULL:
                    error_msg = self._ffi.string(self._lib.hackrf_slave_last_error()).decode('utf-8')
                    # Проверяем если устройство уже используется как master
                    if "not found" in error_msg.lower():
                        self.logger.warning(f"HackRF device {self.serial} not found - may be in use as master or not connected")
                    else:
                        self.logger.warning(f"HackRF device {self.serial} unavailable: {error_msg}")
                    raise HackRFSlaveDeviceError(f"Failed to open HackRF device {self.serial}: {error_msg}")
                
                self.logger.info(f"Opened HackRF slave device: {self.serial}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to open HackRF slave: {e}")
            return False
    
    def close(self):
        """Close the HackRF device."""
        try:
            with self._lock:
                if self._device_handle:
                    self._lib.hackrf_slave_close(self._device_handle)
                    self._device_handle = None
                    self.logger.info(f"Closed HackRF slave device: {self.serial}")
        except Exception as e:
            self.logger.error(f"Error closing HackRF slave: {e}")
    
    def configure(self, config: SlaveConfig) -> bool:
        """Configure the HackRF slave device."""
        try:
            with self._lock:
                if not self._device_handle:
                    raise HackRFSlaveDeviceError("Device not opened")
                
                # Create C config structure
                c_config = self._ffi.new("hackrf_slave_config_t*")
                c_config.sample_rate = config.sample_rate_hz
                c_config.lna_gain = config.lna_gain
                c_config.vga_gain = config.vga_gain
                c_config.amp_enable = config.amp_enable
                c_config.bandwidth_hz = 2500000  # Default 2.5 MHz
                c_config.calibration_db = 0.0
                c_config.dc_offset_correction = config.dc_offset_correction
                c_config.frequency_offset_hz = config.freq_offset_hz
                c_config.iq_balance_correction = config.iq_balance_correction
                c_config.filter_window_type = config.window_type
                c_config.filter_beta = 8.6  # Kaiser window beta
                c_config.spectral_smoothing = False
                c_config.smoothing_factor = 1
                
                ret = self._lib.hackrf_slave_configure(self._device_handle, c_config)
                if ret != self.HACKRF_SLAVE_SUCCESS:
                    error_msg = self._ffi.string(self._lib.hackrf_slave_last_error()).decode('utf-8')
                    raise HackRFSlaveConfigError(f"Configuration failed: {error_msg} ({ret})")
                
                self.logger.info(f"Configured HackRF slave: sr={config.sample_rate_hz/1e6:.1f}MS/s, lna={config.lna_gain}dB, vga={config.vga_gain}dB")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to configure HackRF slave: {e}")
            return False
    
    def start_capture(self) -> bool:
        """Start data capture (not needed for this API - measurements handle capture internally)."""
        self._running = True
        return True
    
    def stop_capture(self):
        """Stop data capture (not needed for this API)."""
        self._running = False
    
    def measure_rssi(self, center_hz: float, span_hz: float, duration_sec: float = 1.0) -> Optional[RSSIMeasurement]:
        """Measure RSSI in the specified frequency range."""
        try:
            with self._lock:
                if not self._device_handle:
                    self.logger.error("Device not opened for RSSI measurement")
                    raise HackRFSlaveDeviceError("Device not opened")
                
                # Create result structure
                result = self._ffi.new("hackrf_slave_rssi_measurement_t*")
                dwell_ms = int(duration_sec * 1000)
                
                self.logger.debug(f"Starting RSSI measurement: f={center_hz/1e6:.1f}MHz, span={span_hz/1e6:.1f}MHz, dwell={dwell_ms}ms")
                
                ret = self._lib.hackrf_slave_measure_rssi(
                    self._device_handle, 
                    center_hz, 
                    span_hz, 
                    dwell_ms,
                    result
                )
                
                self.logger.debug(f"C library returned: {ret} (success={self.HACKRF_SLAVE_SUCCESS})")
                
                if ret != self.HACKRF_SLAVE_SUCCESS:
                    error_msg = self._ffi.string(self._lib.hackrf_slave_last_error()).decode('utf-8')
                    if ret == self.HACKRF_SLAVE_ERROR_TIMEOUT:
                        self.logger.warning(f"RSSI measurement timeout - device may be busy with master operations")
                    else:
                        self.logger.error(f"RSSI measurement failed with code {ret}: {error_msg}")
                    raise HackRFSlaveProcessingError(f"RSSI measurement failed: {error_msg} ({ret})")
                
                self.logger.debug(f"Measurement valid: {result.valid}")
                if not result.valid:
                    self.logger.error("RSSI measurement returned invalid result")
                    raise HackRFSlaveProcessingError("RSSI measurement returned invalid result")
                
                # Convert to Python dataclass
                measurement = RSSIMeasurement(
                    center_freq_hz=result.center_hz,
                    span_hz=result.span_hz,
                    rssi_dbm=result.band_rssi_dbm,
                    noise_floor_dbm=result.band_noise_dbm,
                    snr_db=result.snr_db,
                    timestamp=result.timestamp,
                    sample_count=result.n_samples
                )
                
                self.logger.info(f"RSSI measurement successful: f={center_hz/1e6:.1f}MHz, rssi={measurement.rssi_dbm:.1f}dBm, snr={measurement.snr_db:.1f}dB")
                return measurement
                
        except Exception as e:
            self.logger.error(f"RSSI measurement failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def get_spectrum(self, center_hz: float, span_hz: float, dwell_ms: int = 400, max_points: int = 8192) -> Tuple[list, list]:
        """Acquire spectrum in the specified band.
        Returns (freqs_hz, powers_dbm) lists.
        """
        try:
            with self._lock:
                if not self._device_handle:
                    raise HackRFSlaveDeviceError("Device not opened")

                freqs_array = self._ffi.new("double[]", int(max_points))
                powers_array = self._ffi.new("float[]", int(max_points))

                self.logger.debug(f"Requesting spectrum: f={center_hz/1e6:.1f}MHz, span={span_hz/1e6:.1f}MHz, dwell={int(dwell_ms)}ms, max_points={int(max_points)}")

                ret = self._lib.hackrf_slave_get_spectrum(
                    self._device_handle,
                    float(center_hz),
                    float(span_hz),
                    int(dwell_ms),
                    freqs_array,
                    powers_array,
                    int(max_points)
                )

                if int(ret) < 0:
                    # Fetch last error from the library if available
                    try:
                        error_msg = self._ffi.string(self._lib.hackrf_slave_last_error()).decode('utf-8')
                    except Exception:
                        error_msg = "Unknown error"
                    raise HackRFSlaveProcessingError(f"Spectrum acquisition failed: {error_msg} ({ret})")

                num_points = int(ret)
                freqs = [float(freqs_array[i]) for i in range(num_points)]
                powers = [float(powers_array[i]) for i in range(num_points)]
                return freqs, powers
        except Exception as e:
            self.logger.error(f"Spectrum acquisition failed: {e}")
            return [], []
    
    def is_streaming(self) -> bool:
        """Check if device is currently streaming."""
        try:
            with self._lock:
                if not self._device_handle:
                    return False
                return bool(self._lib.hackrf_slave_is_ready(self._device_handle))
        except:
            return False
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_capture()
        self.close()
    
    def __del__(self):
        try:
            self.close()
        except:
            pass


def list_devices() -> List[str]:
    """List available HackRF devices."""
    if not CFFI_AVAILABLE:
        return []
    
    try:
        # This would need to be implemented in the C library
        # For now, return empty list - devices will be discovered via SoapySDR
        return []
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to list HackRF slave devices: {e}")
        return []


# Window type constants (should match C library)
WINDOW_RECTANGULAR = 0
WINDOW_HAMMING = 1
WINDOW_HANN = 2
WINDOW_BLACKMAN = 3
WINDOW_KAISER = 4