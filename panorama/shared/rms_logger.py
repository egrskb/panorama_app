#!/usr/bin/env python3
"""
Logging and CSV dump functionality for RMS-based trilateration system.
"""

from __future__ import annotations
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from panorama.shared.rms_utils import RMSMeasurement


@dataclass 
class LogEntry:
    """Log entry for RMS system events."""
    timestamp: float
    level: str
    component: str
    message: str
    target_id: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None


class RMSLogger:
    """Enhanced logger for RMS trilateration system."""
    
    def __init__(self, log_level: str = "INFO", csv_dump_enabled: bool = False, 
                 csv_path: str = "output/band_rms.csv"):
        self.logger = logging.getLogger("panorama.rms")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # CSV dump configuration
        self.csv_dump_enabled = csv_dump_enabled
        self.csv_path = Path(csv_path)
        
        if self.csv_dump_enabled:
            self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers."""
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file exists and has headers
            file_exists = self.csv_path.exists()
            
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header if file is new
                if not file_exists or self.csv_path.stat().st_size == 0:
                    writer.writerow([
                        'timestamp', 'target_id', 'sdr_id', 'center_hz', 
                        'halfspan_hz', 'bins_used', 'rssi_rms_dbm'
                    ])
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize CSV: {e}")
            self.csv_dump_enabled = False
    
    def log_master_detect(self, center_hz: float, bw_hz: float, peak_dbm: float, 
                         confidence: float):
        """Log master detection event."""
        self.logger.info(
            f"MASTER.DETECT: F_max={center_hz/1e6:.3f} MHz, "
            f"BW={bw_hz/1e6:.1f} MHz, peak={peak_dbm:.1f} dBm, conf={confidence:.2f}"
        )
    
    def log_watchlist_event(self, action: str, target_id: str, center_hz: float, 
                           halfspan_hz: float, reason: str = ""):
        """Log watchlist management events."""
        self.logger.info(
            f"WATCHLIST.{action.upper()}: {target_id} @ {center_hz/1e6:.3f} MHz "
            f"±{halfspan_hz/1e6:.1f} MHz {reason}"
        )
    
    def log_slave_measurement(self, measurement: RMSMeasurement, dwell_ms: int = 0, 
                            retries: int = 0):
        """Log slave RMS measurement."""
        self.logger.info(
            f"SLAVE.MEAS: {measurement.target_id} on {measurement.sdr_id} = "
            f"{measurement.rssi_rms_dbm:.1f} dBm, bins={measurement.bins_used}, "
            f"dwell={dwell_ms}ms, retries={retries}"
        )
        
        # CSV dump if enabled
        if self.csv_dump_enabled:
            self._write_csv_measurement(measurement)
    
    def log_trilateration_input(self, target_id: str, measurements: Dict[str, float]):
        """Log trilateration inputs."""
        rssi_str = ", ".join([f"{sdr}={rssi:.1f}" for sdr, rssi in measurements.items()])
        delta_12 = list(measurements.values())[0] - list(measurements.values())[1] if len(measurements) >= 2 else 0
        delta_13 = list(measurements.values())[0] - list(measurements.values())[2] if len(measurements) >= 3 else 0
        
        self.logger.info(
            f"TRILAT.INPUT: {target_id} RMS=[{rssi_str}] dBm, "
            f"ΔP_12={delta_12:.1f} dB, ΔP_13={delta_13:.1f} dB"
        )
    
    def log_trilateration_result(self, target_id: str, x: float, y: float, z: float,
                               error: float, confidence: float):
        """Log trilateration result."""
        self.logger.info(
            f"TRILAT.RESULT: {target_id} pos=({x:.1f}, {y:.1f}, {z:.1f}) m, "
            f"error={error:.2f}, conf={confidence:.2f}"
        )
    
    def log_trilateration_error(self, target_id: str, error_msg: str):
        """Log trilateration error."""
        self.logger.error(f"TRILAT.ERROR: {target_id} - {error_msg}")
    
    def log_recentering(self, target_id: str, old_center: float, new_center: float, 
                       halfspan: float):
        """Log target recentering."""
        delta = abs(new_center - old_center)
        trigger_threshold = 0.3 * halfspan
        
        self.logger.info(
            f"WATCHLIST.RECENTER: {target_id} "
            f"{old_center/1e6:.3f} → {new_center/1e6:.3f} MHz "
            f"(Δ={delta/1e6:.3f} MHz > {trigger_threshold/1e6:.3f} MHz)"
        )
    
    def log_target_timeout(self, target_id: str, last_update: float, timeout_sec: float):
        """Log target timeout and removal."""
        age_sec = time.time() - last_update
        self.logger.info(
            f"WATCHLIST.TIMEOUT: {target_id} aged {age_sec:.1f}s > {timeout_sec:.1f}s"
        )
    
    def log_edge_case(self, case: str, details: str, target_id: str = ""):
        """Log edge case handling."""
        target_str = f" ({target_id})" if target_id else ""
        self.logger.warning(f"EDGE.{case.upper()}{target_str}: {details}")
    
    def _write_csv_measurement(self, measurement: RMSMeasurement):
        """Write RMS measurement to CSV file."""
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    measurement.timestamp,
                    measurement.target_id,
                    measurement.sdr_id,
                    measurement.center_hz,
                    measurement.halfspan_hz,
                    measurement.bins_used,
                    measurement.rssi_rms_dbm
                ])
        except Exception as e:
            self.logger.error(f"CSV write failed: {e}")
    
    def enable_csv_dump(self, enabled: bool = True):
        """Enable or disable CSV dumping."""
        self.csv_dump_enabled = enabled
        if enabled:
            self._initialize_csv()
            self.logger.info(f"CSV dump enabled: {self.csv_path}")
        else:
            self.logger.info("CSV dump disabled")
    
    def set_csv_path(self, path: str):
        """Change CSV dump file path."""
        self.csv_path = Path(path)
        if self.csv_dump_enabled:
            self._initialize_csv()
            self.logger.info(f"CSV path changed to: {self.csv_path}")


# Global logger instance
_rms_logger: Optional[RMSLogger] = None


def get_rms_logger(log_level: str = "INFO", csv_dump: bool = False, 
                   csv_path: str = "output/band_rms.csv") -> RMSLogger:
    """Get or create the global RMS logger instance."""
    global _rms_logger
    if _rms_logger is None:
        _rms_logger = RMSLogger(log_level, csv_dump, csv_path)
    return _rms_logger


def configure_rms_logging(config: Dict[str, Any]):
    """Configure RMS logging from config dictionary."""
    log_level = config.get("debug", {}).get("log_level", "INFO")
    csv_dump = config.get("debug", {}).get("band_rms_csv_dump", False)
    csv_path = config.get("debug", {}).get("band_rms_csv_path", "output/band_rms.csv")
    
    logger = get_rms_logger(log_level, csv_dump, csv_path)
    logger.logger.info("RMS logging configured from config")
    return logger