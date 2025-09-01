#!/usr/bin/env python3
"""
RMS calculation utilities for PANORAMA trilateration system.
Implements band RMS calculation according to technical specification.
"""

from __future__ import annotations
import numpy as np
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class RMSMeasurement:
    """RMS measurement result."""
    target_id: str
    sdr_id: str
    center_hz: float
    halfspan_hz: float
    bins_used: int
    rssi_rms_dbm: float
    timestamp: float


class EMAFilter:
    """Exponential Moving Average filter for RMS smoothing."""
    
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self.state: Dict[tuple, float] = {}  # (target_id, sdr_id) -> current_value
    
    def update(self, key: tuple, value: float) -> float:
        """Updates EMA state and returns smoothed value."""
        if key not in self.state:
            self.state[key] = value
        else:
            self.state[key] = self.alpha * value + (1.0 - self.alpha) * self.state[key]
        return self.state[key]
    
    def get(self, key: tuple) -> Optional[float]:
        """Gets current EMA value without updating."""
        return self.state.get(key)
    
    def clear(self, key: Optional[tuple] = None):
        """Clears EMA state for key or all keys if None."""
        if key is None:
            self.state.clear()
        else:
            self.state.pop(key, None)


def compute_band_rssi_dbm(freq_hz: np.ndarray, power_dbm: np.ndarray, 
                         center_hz: float, halfspan_hz: float) -> Optional[float]:
    """
    Computes RMS power in dBm over frequency band [center_hz ± halfspan_hz].
    
    Args:
        freq_hz: Frequency array in Hz
        power_dbm: Power spectral density in dBm
        center_hz: F_max - center frequency of the band
        halfspan_hz: Half-width of the band around center
        
    Returns:
        RMS power in dBm or None if insufficient bins
    """
    try:
        # Define frequency band
        f_start = center_hz - halfspan_hz
        f_stop = center_hz + halfspan_hz
        
        # Select bins within the band
        mask = (freq_hz >= f_start) & (freq_hz <= f_stop)
        bins_in_band = np.count_nonzero(mask)
        
        # Need at least 3 bins for reliable RMS calculation
        if bins_in_band < 3:
            return None
            
        # Extract power values in the band
        band_powers_dbm = power_dbm[mask]
        
        # Convert dBm to milliwatts
        band_powers_mw = 10.0 ** (band_powers_dbm / 10.0)
        
        # Calculate mean power in linear scale
        mean_power_mw = np.mean(band_powers_mw)
        
        # Convert back to dBm
        rms_dbm = 10.0 * np.log10(mean_power_mw + 1e-20)  # Add small epsilon to avoid log(0)
        
        return float(rms_dbm)
        
    except Exception as e:
        logging.error(f"compute_band_rssi_dbm error: {e}")
        return None


def validate_band_coverage(freq_hz: np.ndarray, center_hz: float, 
                          halfspan_hz: float, guard_hz: float = 0) -> bool:
    """
    Validates that the frequency array covers the required band with optional guard.
    
    Args:
        freq_hz: Available frequency range
        center_hz: Target center frequency
        halfspan_hz: Required halfspan around center
        guard_hz: Additional guard band (optional)
        
    Returns:
        True if coverage is adequate
    """
    if len(freq_hz) < 2:
        return False
        
    f_min, f_max = freq_hz[0], freq_hz[-1]
    required_min = center_hz - halfspan_hz - guard_hz
    required_max = center_hz + halfspan_hz + guard_hz
    
    return f_min <= required_min and f_max >= required_max


def create_target_json(target_id: str, center_hz: float, halfspan_hz: float, 
                      guard_hz: float, bw_hz: float, confidence: float) -> Dict[str, Any]:
    """
    Creates target JSON for slave tasks according to specification.
    
    Args:
        target_id: Target identifier (e.g., "det-1724920000")
        center_hz: F_max frequency
        halfspan_hz: RMS halfspan
        guard_hz: Guard frequency for slave scanning
        bw_hz: Estimated bandwidth from cluster
        confidence: Detection confidence
        
    Returns:
        Target task JSON
    """
    return {
        "id": target_id,
        "center_hz": center_hz,
        "halfspan_hz": halfspan_hz,
        "guard_hz": guard_hz,
        "bw_hz": bw_hz,
        "confidence": confidence,
        "ts": time.time()
    }


def should_recenter_target(old_center_hz: float, new_center_hz: float, 
                          halfspan_hz: float, trigger_fraction: float = 0.3) -> bool:
    """
    Determines if target should be recentered based on F_max drift.
    
    Args:
        old_center_hz: Current center frequency
        new_center_hz: New F_max from master
        halfspan_hz: RMS halfspan
        trigger_fraction: Recentering trigger as fraction of halfspan
        
    Returns:
        True if recentering is needed
    """
    delta_hz = abs(new_center_hz - old_center_hz)
    threshold_hz = trigger_fraction * halfspan_hz
    return delta_hz > threshold_hz


class RMSCalculator:
    """Main RMS calculation class with EMA smoothing."""
    
    def __init__(self, ema_alpha: float = 0.7, min_bins: int = 3):
        self.ema_filter = EMAFilter(ema_alpha)
        self.min_bins = min_bins
        self.logger = logging.getLogger(__name__)
    
    def measure_target_rms(self, freq_hz: np.ndarray, power_dbm: np.ndarray,
                          target: Dict[str, Any], sdr_id: str) -> Optional[RMSMeasurement]:
        """
        Measures RMS for a target with EMA smoothing.
        
        Args:
            freq_hz: Frequency array
            power_dbm: Power array in dBm  
            target: Target definition with center_hz, halfspan_hz
            sdr_id: SDR identifier
            
        Returns:
            RMS measurement or None if failed
        """
        try:
            target_id = str(target.get("id", "unknown"))
            center_hz = float(target.get("center_hz", 0))
            halfspan_hz = float(target.get("halfspan_hz", 2.5e6))
            
            # Calculate raw RMS
            raw_rms = compute_band_rssi_dbm(freq_hz, power_dbm, center_hz, halfspan_hz)
            if raw_rms is None:
                return None
            
            # Apply EMA smoothing
            ema_key = (target_id, sdr_id)
            smoothed_rms = self.ema_filter.update(ema_key, raw_rms)
            
            # Count bins used
            f_start = center_hz - halfspan_hz
            f_stop = center_hz + halfspan_hz
            mask = (freq_hz >= f_start) & (freq_hz <= f_stop)
            bins_used = int(np.count_nonzero(mask))
            
            return RMSMeasurement(
                target_id=target_id,
                sdr_id=sdr_id,
                center_hz=center_hz,
                halfspan_hz=halfspan_hz,
                bins_used=bins_used,
                rssi_rms_dbm=smoothed_rms,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"RMS measurement error: {e}")
            return None
    
    def clear_target(self, target_id: str):
        """Clears EMA state for all SDRs measuring this target."""
        keys_to_remove = [k for k in self.ema_filter.state.keys() if k[0] == target_id]
        for key in keys_to_remove:
            self.ema_filter.clear(key)