#!/usr/bin/env python3
"""
Error handling and edge case management for RMS-based trilateration system.
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from panorama.shared.rms_utils import RMSMeasurement, validate_band_coverage
from panorama.shared.rms_logger import get_rms_logger


class RMSErrorHandler:
    """Handles edge cases and errors in RMS measurement system."""
    
    def __init__(self):
        self.logger = get_rms_logger()
        self.min_bins_threshold = 3
        self.max_retries = 1
        self.nan_inf_count = 0
        self.insufficient_bins_count = 0
        self.tune_failure_count = 0
        
    def validate_spectrum_data(self, freq_hz: np.ndarray, power_dbm: np.ndarray, 
                              target_id: str = "") -> Tuple[bool, str]:
        """
        Validates spectrum data for NaN/Inf values.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Check for NaN values
            nan_mask = np.isnan(power_dbm)
            inf_mask = np.isinf(power_dbm)
            
            nan_count = np.count_nonzero(nan_mask)
            inf_count = np.count_nonzero(inf_mask)
            
            if nan_count > 0 or inf_count > 0:
                self.nan_inf_count += 1
                error_msg = f"Spectrum contains {nan_count} NaN and {inf_count} Inf values"
                self.logger.log_edge_case("NAN_INF_SPECTRUM", error_msg, target_id)
                
                # Clean the data by replacing with noise floor
                if nan_count > 0:
                    noise_floor = np.nanmedian(power_dbm) - 10.0  # 10dB below median
                    power_dbm[nan_mask] = noise_floor
                
                if inf_count > 0:
                    finite_values = power_dbm[np.isfinite(power_dbm)]
                    if len(finite_values) > 0:
                        replacement_value = np.max(finite_values)
                        power_dbm[inf_mask] = replacement_value
                
                # If too many bad values, reject
                bad_fraction = (nan_count + inf_count) / len(power_dbm)
                if bad_fraction > 0.5:
                    return False, f"Too many bad values: {bad_fraction:.2%}"
                
                self.logger.log_edge_case("SPECTRUM_CLEANED", 
                                        f"Replaced {nan_count} NaN, {inf_count} Inf values", 
                                        target_id)
            
            return True, ""
            
        except Exception as e:
            return False, f"Spectrum validation error: {e}"
    
    def validate_band_coverage(self, freq_hz: np.ndarray, center_hz: float, 
                             halfspan_hz: float, guard_hz: float = 0,
                             target_id: str = "") -> Tuple[bool, str]:
        """
        Validates that frequency array covers required band.
        
        Returns:
            (is_adequate, error_message)
        """
        try:
            if not validate_band_coverage(freq_hz, center_hz, halfspan_hz, guard_hz):
                f_min, f_max = freq_hz[0], freq_hz[-1]
                required_min = center_hz - halfspan_hz - guard_hz
                required_max = center_hz + halfspan_hz + guard_hz
                
                error_msg = (f"Insufficient coverage: available={f_min/1e6:.1f}-{f_max/1e6:.1f} MHz, "
                           f"required={required_min/1e6:.1f}-{required_max/1e6:.1f} MHz")
                
                self.logger.log_edge_case("INSUFFICIENT_COVERAGE", error_msg, target_id)
                return False, error_msg
            
            return True, ""
            
        except Exception as e:
            return False, f"Coverage validation error: {e}"
    
    def validate_measurement_bins(self, bins_used: int, target_id: str = "") -> Tuple[bool, str]:
        """
        Validates minimum number of bins for reliable RMS calculation.
        
        Returns:
            (is_sufficient, error_message)
        """
        if bins_used < self.min_bins_threshold:
            self.insufficient_bins_count += 1
            error_msg = f"Insufficient bins: {bins_used} < {self.min_bins_threshold}"
            self.logger.log_edge_case("INSUFFICIENT_BINS", error_msg, target_id)
            return False, error_msg
        
        return True, ""
    
    def handle_tune_failure(self, slave_id: str, center_hz: float, 
                           attempt: int = 0, target_id: str = "") -> bool:
        """
        Handles SDR tuning failures with retry logic.
        
        Returns:
            True if should retry, False if should abort
        """
        self.tune_failure_count += 1
        
        error_msg = f"Tune failure on {slave_id} to {center_hz/1e6:.3f} MHz (attempt {attempt + 1})"
        self.logger.log_edge_case("TUNE_FAILURE", error_msg, target_id)
        
        if attempt < self.max_retries:
            self.logger.log_edge_case("RETRY_TUNE", f"Retrying tune on {slave_id}", target_id)
            return True
        else:
            self.logger.log_edge_case("ABORT_TUNE", f"Max retries reached for {slave_id}", target_id)
            return False
    
    def handle_measurement_failure(self, slave_id: str, error: Exception, 
                                 attempt: int = 0, target_id: str = "") -> bool:
        """
        Handles measurement failures with retry logic.
        
        Returns:
            True if should retry, False if should abort
        """
        error_msg = f"Measurement failure on {slave_id}: {error}"
        self.logger.log_edge_case("MEASUREMENT_FAILURE", error_msg, target_id)
        
        if attempt < self.max_retries:
            self.logger.log_edge_case("RETRY_MEASUREMENT", f"Retrying measurement on {slave_id}", target_id)
            return True
        else:
            self.logger.log_edge_case("ABORT_MEASUREMENT", f"Max retries reached for {slave_id}", target_id)
            return False
    
    def check_target_timeout(self, target_last_update: float, timeout_sec: float, 
                           target_id: str) -> bool:
        """
        Checks if target has timed out and should be removed.
        
        Returns:
            True if target should be removed
        """
        current_time = time.time()
        age_sec = current_time - target_last_update
        
        if age_sec > timeout_sec:
            self.logger.log_target_timeout(target_id, target_last_update, timeout_sec)
            return True
        
        return False
    
    def validate_trilateration_measurements(self, measurements: Dict[str, float],
                                          min_slaves: int = 3, target_id: str = "") -> Tuple[bool, str]:
        """
        Validates measurements for trilateration.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Check minimum number of slaves
            if len(measurements) < min_slaves:
                error_msg = f"Insufficient slaves for trilateration: {len(measurements)} < {min_slaves}"
                self.logger.log_edge_case("INSUFFICIENT_SLAVES", error_msg, target_id)
                return False, error_msg
            
            # Check for reasonable RSSI values
            rssi_values = list(measurements.values())
            
            # Check for extreme values
            for slave_id, rssi in measurements.items():
                if rssi > 0:  # RSSI should be negative in dBm
                    error_msg = f"Invalid RSSI from {slave_id}: {rssi} dBm (should be negative)"
                    self.logger.log_edge_case("INVALID_RSSI", error_msg, target_id)
                    return False, error_msg
                
                if rssi < -150:  # Too weak signal
                    error_msg = f"RSSI too weak from {slave_id}: {rssi} dBm"
                    self.logger.log_edge_case("WEAK_RSSI", error_msg, target_id)
                    # Don't return False - just warn
            
            # Check for unreasonable power differences
            if len(rssi_values) >= 2:
                max_rssi = max(rssi_values)
                min_rssi = min(rssi_values)
                power_spread = max_rssi - min_rssi
                
                if power_spread > 60:  # More than 60dB difference
                    error_msg = f"Large power spread: {power_spread:.1f} dB"
                    self.logger.log_edge_case("LARGE_POWER_SPREAD", error_msg, target_id)
                    # Don't fail - could be legitimate
            
            return True, ""
            
        except Exception as e:
            return False, f"Measurement validation error: {e}"
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Returns error statistics for monitoring."""
        return {
            "nan_inf_count": self.nan_inf_count,
            "insufficient_bins_count": self.insufficient_bins_count,
            "tune_failure_count": self.tune_failure_count
        }
    
    def reset_statistics(self):
        """Resets error counters."""
        self.nan_inf_count = 0
        self.insufficient_bins_count = 0
        self.tune_failure_count = 0
        self.logger.logger.info("Error statistics reset")


# Global error handler instance
_error_handler: Optional[RMSErrorHandler] = None


def get_error_handler() -> RMSErrorHandler:
    """Get or create the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = RMSErrorHandler()
    return _error_handler