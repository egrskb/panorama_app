#!/usr/bin/env python3
"""
MapLibre GL JS widget for PANORAMA application.
Provides interactive map with drag-and-drop SDR positioning and real-time target tracking.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import time

from PyQt5.QtCore import QObject, QUrl, pyqtSignal, pyqtSlot
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton


class MapInterface(QObject):
    """Interface for communication between Python and JavaScript."""
    
    # Signals to emit to other parts of the application
    sdr_position_changed = pyqtSignal(str, float, float, float)  # id, x, y, z
    target_selected = pyqtSignal(str)  # target_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    @pyqtSlot(str)
    def updateSDRPositions(self, positions_json: str):
        """Called from JavaScript when SDR positions are updated via drag."""
        try:
            positions = json.loads(positions_json)
            for sdr_id, coords in positions.items():
                x, y, z = coords.get('x', 0), coords.get('y', 0), coords.get('z', 0)
                self.sdr_position_changed.emit(sdr_id, x, y, z)
                print(f"[MapLibre] SDR {sdr_id} moved to ({x:.1f}, {y:.1f}, {z:.1f})")
        except Exception as e:
            print(f"[MapLibre] Error updating SDR positions: {e}")
    
    @pyqtSlot(str)
    def selectTarget(self, target_id: str):
        """Called from JavaScript when target is selected."""
        self.target_selected.emit(target_id)
    
    @pyqtSlot(str)
    def logMessage(self, message: str):
        """Called from JavaScript for logging."""
        print(f"[MapLibre JS] {message}")


class MapLibreWidget(QWidget):
    """
    MapLibre GL JS widget for interactive mapping with SDR positioning.
    Features:
    - Drag-and-drop SDR repositioning
    - Real-time target tracking
    - Coordinate synchronization
    - Export functionality
    """
    
    # Signals
    sdr_position_changed = pyqtSignal(str, float, float, float)
    target_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State
        self.sdr_stations: Dict[str, Dict[str, Any]] = {}
        self.targets: Dict[str, Dict[str, Any]] = {}
        self.is_ready = False
        
        # Create UI
        self._setup_ui()
        
        # Create web interface
        self._setup_web_interface()
        
    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel("🗺️ PANORAMA Interactive Map")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # Quick actions
        btn_center = QPushButton("📍 Center on SDRs")
        btn_center.clicked.connect(self.center_on_sdrs)
        header_layout.addWidget(btn_center)
        
        btn_clear = QPushButton("🗑️ Clear Targets")
        btn_clear.clicked.connect(self.clear_targets)
        header_layout.addWidget(btn_clear)
        
        layout.addLayout(header_layout)
        
        # Web view
        self.web_view = QWebEngineView()
        self.web_view.setMinimumHeight(600)
        layout.addWidget(self.web_view)
        
        # Status bar
        self.status_label = QLabel("Loading map...")
        self.status_label.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 5px;")
        layout.addWidget(self.status_label)
    
    def _setup_web_interface(self):
        """Setup the web engine interface."""
        # Create web channel for Python-JS communication
        self.web_channel = QWebChannel()
        self.map_interface = MapInterface()
        
        # Connect signals
        self.map_interface.sdr_position_changed.connect(self.sdr_position_changed)
        self.map_interface.target_selected.connect(self.target_selected)
        
        self.web_channel.registerObject("mapInterface", self.map_interface)
        
        # Setup web page
        self.web_page = self.web_view.page()
        self.web_page.setWebChannel(self.web_channel)
        
        # Load map HTML
        map_file = Path(__file__).parent / "maplibre_map.html"
        if map_file.exists():
            self.web_view.load(QUrl.fromLocalFile(str(map_file)))
        else:
            self.status_label.setText("❌ Map file not found")
            return
        
        # Setup page callbacks
        self.web_page.loadFinished.connect(self._on_page_loaded)
    
    def _on_page_loaded(self, success: bool):
        """Called when the web page is loaded."""
        if success:
            self.is_ready = True
            self.status_label.setText("✅ Map ready - Drag SDRs to reposition them")
            print("[MapLibre] Map loaded successfully")
        else:
            self.status_label.setText("❌ Failed to load map")
            print("[MapLibre] Failed to load map")
    
    def _execute_js(self, script: str):
        """Execute JavaScript code safely."""
        if self.is_ready:
            self.web_page.runJavaScript(script)
        else:
            print(f"[MapLibre] Map not ready, queued JS: {script[:50]}...")
    
    def add_or_update_sdr(self, sdr_id: str, x: float, y: float, z: float = 0.0,
                          is_reference: bool = False, role: str = "measurement"):
        """
        Add or update an SDR station on the map.
        
        Args:
            sdr_id: SDR identifier
            x, y, z: Local coordinates in meters
            is_reference: Whether this is the reference station
            role: Station role description
        """
        sdr_data = {
            'id': sdr_id,
            'x': x,
            'y': y,
            'z': z,
            'is_reference': is_reference,
            'role': role
        }
        
        self.sdr_stations[sdr_id] = sdr_data
        
        js_data = json.dumps(sdr_data)
        script = f"if (window.mapAPI) {{ window.mapAPI.addOrUpdateSDR({js_data}); }}"
        self._execute_js(script)
        
        print(f"[MapLibre] Updated SDR {sdr_id}: ({x:.1f}, {y:.1f}, {z:.1f}) {'(reference)' if is_reference else ''}")
    
    def add_or_update_target(self, target_id: str, x: float, y: float, z: float = 0.0,
                           frequency: Optional[float] = None, confidence: Optional[float] = None,
                           target_type: str = "detected", label: Optional[str] = None):
        """
        Add or update a target on the map.
        
        Args:
            target_id: Target identifier
            x, y, z: Local coordinates in meters
            frequency: Signal frequency in MHz
            confidence: Detection confidence (0-1)
            target_type: 'detected' or 'tracked'
            label: Custom label for the target
        """
        target_data = {
            'id': target_id,
            'x': x,
            'y': y,
            'z': z,
            'frequency': frequency,
            'confidence': f"{confidence*100:.1f}%" if confidence is not None else None,
            'type': target_type,
            'label': label or f"Target {frequency:.1f}MHz" if frequency else f"Target {target_id}",
            'timestamp': time.strftime('%H:%M:%S')
        }
        
        self.targets[target_id] = target_data
        
        js_data = json.dumps(target_data)
        script = f"if (window.mapAPI) {{ window.mapAPI.addOrUpdateTarget({js_data}); }}"
        self._execute_js(script)
        
        print(f"[MapLibre] Updated target {target_id}: ({x:.1f}, {y:.1f}) {frequency}MHz conf={confidence}")
    
    def update_sdr_positions(self, positions: Dict[str, Tuple[float, float, float]]):
        """
        Update multiple SDR positions at once.
        
        Args:
            positions: Dictionary of {sdr_id: (x, y, z)}
        """
        for sdr_id, (x, y, z) in positions.items():
            is_reference = self.sdr_stations.get(sdr_id, {}).get('is_reference', False)
            role = self.sdr_stations.get(sdr_id, {}).get('role', 'measurement')
            self.add_or_update_sdr(sdr_id, x, y, z, is_reference, role)
    
    def center_on_sdrs(self):
        """Center the map view on all SDR stations."""
        script = "if (window.mapAPI) { window.mapAPI.centerOnSDRs(); }"
        self._execute_js(script)
    
    def clear_targets(self):
        """Clear all targets from the map."""
        self.targets.clear()
        script = "if (window.mapAPI) { window.mapAPI.clearTargets(); }"
        self._execute_js(script)
    
    def set_connection_status(self, connected: bool):
        """Update the connection status indicator."""
        script = f"if (window.mapAPI) {{ window.mapAPI.updateConnectionStatus({str(connected).lower()}); }}"
        self._execute_js(script)
    
    def export_map_data(self) -> Dict[str, Any]:
        """
        Export current map data.
        
        Returns:
            Dictionary with SDR stations and targets data
        """
        return {
            'sdr_stations': self.sdr_stations,
            'targets': self.targets,
            'timestamp': time.time()
        }
    
    def load_map_data(self, data: Dict[str, Any]):
        """
        Load map data from exported data.
        
        Args:
            data: Dictionary with map data
        """
        # Load SDR stations
        if 'sdr_stations' in data:
            for sdr_data in data['sdr_stations'].values():
                self.add_or_update_sdr(
                    sdr_data['id'],
                    sdr_data['x'], 
                    sdr_data['y'], 
                    sdr_data.get('z', 0),
                    sdr_data.get('is_reference', False),
                    sdr_data.get('role', 'measurement')
                )
        
        # Load targets
        if 'targets' in data:
            for target_data in data['targets'].values():
                self.add_or_update_target(
                    target_data['id'],
                    target_data['x'],
                    target_data['y'],
                    target_data.get('z', 0),
                    target_data.get('frequency'),
                    target_data.get('confidence'),
                    target_data.get('type', 'detected'),
                    target_data.get('label')
                )
    
    def handle_data_from_app(self, data: Dict[str, Any]):
        """
        Handle data updates from the main application.
        
        Args:
            data: Data dictionary with type and payload
        """
        try:
            data_type = data.get('type')
            payload = data.get('data', data)
            
            if data_type == 'sdr_update' or data_type == 'update_devices_coordinates':
                # Handle SDR position updates
                devices = payload.get('devices', [])
                for device in devices:
                    self.add_or_update_sdr(
                        device.get('id', 'unknown'),
                        device.get('x', 0),
                        device.get('y', 0),
                        device.get('z', 0),
                        device.get('is_reference', False),
                        device.get('role', 'measurement')
                    )
            
            elif data_type == 'target' or data_type == 'target_update':
                # Handle target updates
                self.add_or_update_target(
                    payload.get('id', f'target_{time.time()}'),
                    payload.get('x', 0),
                    payload.get('y', 0),
                    payload.get('z', 0),
                    payload.get('freq_mhz', payload.get('frequency')),
                    payload.get('confidence'),
                    'tracked' if 'confidence' in payload else 'detected'
                )
            
            elif data_type == 'stations_update':
                # Handle station updates
                stations = payload.get('stations', [])
                for station in stations:
                    self.add_or_update_sdr(
                        station.get('id', 'unknown'),
                        station.get('x', 0),
                        station.get('y', 0),
                        station.get('z', 0),
                        station.get('is_reference', False)
                    )
            
            else:
                print(f"[MapLibre] Unknown data type: {data_type}")
                
        except Exception as e:
            print(f"[MapLibre] Error handling app data: {e}")
    
    def get_current_sdrs(self) -> Dict[str, Dict[str, Any]]:
        """Get current SDR stations data."""
        return self.sdr_stations.copy()
    
    def get_current_targets(self) -> Dict[str, Dict[str, Any]]:
        """Get current targets data."""
        return self.targets.copy()
    
    def update_stations_from_config(self, config: dict):
        """Update stations from configuration (backward compatibility)."""
        try:
            # Extract station information from config
            stations = config.get('stations', [])
            
            for station in stations:
                station_id = station.get('id', 'unknown')
                x = station.get('x', 0.0)
                y = station.get('y', 0.0)
                z = station.get('z', 0.0)
                is_reference = station.get('is_reference', False)
                role = station.get('role', 'measurement')
                
                # Add or update the SDR station
                self.add_or_update_sdr(
                    station_id, x, y, z, 
                    is_reference=is_reference, 
                    role=role
                )
            
            print(f"[MapLibre] Updated {len(stations)} stations from config")
            
        except Exception as e:
            print(f"[MapLibre] Error updating stations from config: {e}")


# Convenience function for creating the widget
def create_map_widget(parent=None) -> MapLibreWidget:
    """Create a configured MapLibre widget."""
    widget = MapLibreWidget(parent)
    
    # Example SDRs (will be replaced by actual data)
    widget.add_or_update_sdr("slave0", 0, 0, 0, is_reference=True, role="reference")
    widget.add_or_update_sdr("slave1", 50, 0, 0, role="measurement")
    widget.add_or_update_sdr("slave2", 25, 43.3, 0, role="measurement")
    
    return widget


if __name__ == "__main__":
    # Test the widget
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create and show widget
    widget = create_map_widget()
    widget.show()
    widget.resize(1000, 700)
    
    # Add some test data
    widget.add_or_update_target("test1", 25, 15, 0, 2400.0, 0.8, "detected")
    widget.add_or_update_target("test2", 40, 30, 0, 5800.0, 0.95, "tracked")
    
    sys.exit(app.exec_())