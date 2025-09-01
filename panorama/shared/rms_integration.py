#!/usr/bin/env python3
"""
Integration helper for RMS-based trilateration system.
Demonstrates how all components work together according to the specification.
"""

from __future__ import annotations
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from panorama.shared.rms_utils import create_target_json, should_recenter_target
from panorama.shared.rms_logger import configure_rms_logging
from panorama.shared.rms_error_handler import get_error_handler
from panorama.features.settings.storage import load_detector_settings, save_detector_settings


class RMSSystemIntegrator:
    """Integrates all RMS system components according to specification."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the RMS system with configuration."""
        self.config = self._load_config(config_path)
        self.logger = configure_rms_logging(self.config)
        self.error_handler = get_error_handler()
        
        # System components (to be set by main application)
        self.master_controller = None
        self.slave_manager = None
        self.trilateration_engine = None
        self.watchlist_manager = None
        
        # Active targets dictionary: {target_id: target_dict}
        self.active_targets: Dict[str, Dict[str, Any]] = {}
        
        self.logger.logger.info("RMS System Integrator initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"Loaded RMS config from {config_path}")
                return config
        except Exception as e:
            print(f"Failed to load config from {config_path}: {e}")
        
        # Return default configuration
        return {
            "detector": {
                "bin_hz": 5000,
                "window_master": "hann",
                "ema_alpha": 0.6,
                "dwell_ms": 500,
                "threshold": {
                    "k_sigma_overview": 4.5,
                    "k_sigma_confirm": 3.5,
                    "prominence_db": 6.0,
                    "abs_floor_dbm": -75.0,
                    "edge_guard_hz": 500000
                },
                "cluster": {
                    "min_bw_hz": 1500000,
                    "max_bw_hz": 12000000,
                    "merge_if_gap_hz": 2000000
                },
                "rms_halfspan_hz": 2500000,
                "recentering_trigger_fraction": 0.3
            },
            "slaves": {
                "dwell_ms": 400,
                "guard_hz": 1000000
            },
            "trilateration": {
                "power_metric": "band_rssi_dbm",
                "ema_alpha_time": 0.7,
                "target_timeout_sec": 5,
                "path_loss_n": 2.2
            },
            "debug": {
                "band_rms_csv_dump": False,
                "band_rms_csv_path": "output/band_rms.csv",
                "log_level": "INFO"
            }
        }
    
    def on_master_detect(self, center_hz: float, bw_hz: float, peak_dbm: float, 
                        confidence: float) -> str:
        """
        Handle master detection event - create or update target.
        
        Args:
            center_hz: F_max frequency (center of detected cluster)
            bw_hz: Estimated bandwidth from cluster analysis
            peak_dbm: Peak power in dBm
            confidence: Detection confidence
            
        Returns:
            Target ID
        """
        # Generate target ID
        target_id = f"det-{int(time.time() * 1000)}"
        
        # Get parameters from config
        halfspan_hz = self.config["detector"]["rms_halfspan_hz"]
        guard_hz = self.config["slaves"]["guard_hz"]
        
        # Log the detection
        self.logger.log_master_detect(center_hz, bw_hz, peak_dbm, confidence)
        
        # Create target JSON
        target = create_target_json(
            target_id, center_hz, halfspan_hz, guard_hz, bw_hz, confidence
        )
        
        # Check if we need to recenter an existing target
        recentered = False
        for existing_id, existing_target in self.active_targets.items():
            existing_center = existing_target["center_hz"]
            if should_recenter_target(
                existing_center, center_hz, halfspan_hz,
                self.config["detector"]["recentering_trigger_fraction"]
            ):
                # Recenter existing target
                self.logger.log_recentering(existing_id, existing_center, center_hz, halfspan_hz)
                existing_target["center_hz"] = center_hz
                existing_target["bw_hz"] = bw_hz
                existing_target["confidence"] = confidence
                existing_target["ts"] = time.time()
                
                # Send update to slaves
                self._send_target_to_slaves(existing_target)
                recentered = True
                target_id = existing_id  # Use existing ID
                break
        
        if not recentered:
            # Add new target
            self.active_targets[target_id] = target
            self.logger.log_watchlist_event("add", target_id, center_hz, halfspan_hz, 
                                          f"confidence={confidence:.2f}")
            
            # Send to slaves
            self._send_target_to_slaves(target)
        
        return target_id
    
    def _send_target_to_slaves(self, target: Dict[str, Any]):
        """Send target to slave manager for RMS measurement."""
        if self.slave_manager:
            try:
                # Use the new target-based RMS measurement
                measurements = self.slave_manager.measure_target_rms_all(
                    [target], self.config["slaves"]["dwell_ms"]
                )
                
                if measurements:
                    self.logger.logger.debug(f"Received {len(measurements)} RMS measurements for target {target['id']}")
                    self._process_rms_measurements(measurements)
                    
            except Exception as e:
                self.logger.log_edge_case("SLAVE_TASK_FAILED", str(e), target["id"])
    
    def _process_rms_measurements(self, measurements: List[Any]):
        """Process RMS measurements from slaves."""
        # Group by target ID
        targets: Dict[str, List] = {}
        for measurement in measurements:
            target_id = measurement.target_id
            if target_id not in targets:
                targets[target_id] = []
            targets[target_id].append(measurement)
        
        # Process each target
        for target_id, target_measurements in targets.items():
            # Log measurements
            for measurement in target_measurements:
                self.logger.log_slave_measurement(measurement)
            
            # Validate measurements for trilateration
            rssi_dict = {m.sdr_id: m.rssi_rms_dbm for m in target_measurements}
            
            is_valid, error_msg = self.error_handler.validate_trilateration_measurements(
                rssi_dict, min_slaves=3, target_id=target_id
            )
            
            if is_valid and self.trilateration_engine:
                # Log trilateration inputs
                self.logger.log_trilateration_input(target_id, rssi_dict)
                
                # Calculate position
                try:
                    result = self.trilateration_engine.calculate_position_from_rms_measurements(
                        target_measurements
                    )
                    
                    if result:
                        self.logger.log_trilateration_result(
                            target_id, result.x, result.y, result.z, 
                            result.error, result.confidence
                        )
                    else:
                        self.logger.log_trilateration_error(target_id, "Failed to calculate position")
                        
                except Exception as e:
                    self.logger.log_trilateration_error(target_id, str(e))
            else:
                self.logger.log_trilateration_error(target_id, error_msg)
    
    def cleanup_expired_targets(self):
        """Remove expired targets based on timeout."""
        timeout_sec = self.config["trilateration"]["target_timeout_sec"]
        current_time = time.time()
        
        expired_targets = []
        for target_id, target in self.active_targets.items():
            last_update = target.get("ts", 0)
            
            if self.error_handler.check_target_timeout(last_update, timeout_sec, target_id):
                expired_targets.append(target_id)
        
        # Remove expired targets
        for target_id in expired_targets:
            del self.active_targets[target_id]
            self.logger.log_watchlist_event("remove", target_id, 0, 0, "timeout")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "active_targets": len(self.active_targets),
            "targets": list(self.active_targets.keys()),
            "config": {
                "rms_halfspan_mhz": self.config["detector"]["rms_halfspan_hz"] / 1e6,
                "guard_mhz": self.config["slaves"]["guard_hz"] / 1e6,
                "target_timeout_sec": self.config["trilateration"]["target_timeout_sec"],
                "csv_dump_enabled": self.config["debug"]["band_rms_csv_dump"]
            },
            "error_stats": self.error_handler.get_error_statistics(),
            "components": {
                "master_connected": self.master_controller is not None,
                "slave_manager_connected": self.slave_manager is not None,
                "trilateration_connected": self.trilateration_engine is not None
            }
        }
    
    def update_rms_halfspan(self, halfspan_mhz: float):
        """Update RMS halfspan parameter and apply to active targets."""
        halfspan_hz = halfspan_mhz * 1e6
        self.config["detector"]["rms_halfspan_hz"] = halfspan_hz
        
        # Update all active targets
        for target in self.active_targets.values():
            target["halfspan_hz"] = halfspan_hz
        
        # Save to detector settings
        detector_settings = load_detector_settings()
        detector_settings["rms_halfspan_hz"] = halfspan_hz
        save_detector_settings(detector_settings)
        
        self.logger.logger.info(f"RMS halfspan updated to {halfspan_mhz:.1f} MHz")
    
    def enable_csv_dump(self, enabled: bool = True, path: Optional[str] = None):
        """Enable/disable CSV dumping."""
        self.config["debug"]["band_rms_csv_dump"] = enabled
        if path:
            self.config["debug"]["band_rms_csv_path"] = path
        
        self.logger.enable_csv_dump(enabled)
        if path:
            self.logger.set_csv_path(path)


def create_pseudocode_example():
    """
    Demonstrates the pseudocode from specification section 12.
    """
    print("=== RMS System Pseudocode Example ===")
    
    # Simulate master detect event
    integrator = RMSSystemIntegrator()
    
    # Example: Master detects signal at 5.653 GHz
    target_id = integrator.on_master_detect(
        center_hz=5653000000,  # F_max
        bw_hz=6000000,         # Cluster bandwidth
        peak_dbm=-45.0,        # Peak power
        confidence=0.85        # Detection confidence
    )
    
    print(f"Created target: {target_id}")
    print(f"Target details: {integrator.active_targets[target_id]}")
    
    # Show system status
    status = integrator.get_system_status()
    print(f"System status: {json.dumps(status, indent=2)}")


if __name__ == "__main__":
    create_pseudocode_example()