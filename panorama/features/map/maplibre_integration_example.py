#!/usr/bin/env python3
"""
Example integration of MapLibre widget with PANORAMA application.
Shows how to connect the map with the slave controller and trilateration system.
"""

import sys
from typing import Dict, Any
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer

from maplibre_widget import MapLibreWidget


class PanoramaMapIntegration(QMainWindow):
    """Example integration of MapLibre map with PANORAMA systems."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("PANORAMA - MapLibre Integration")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create map widget
        self.map_widget = MapLibreWidget()
        layout.addWidget(self.map_widget)
        
        # Connect signals
        self.map_widget.sdr_position_changed.connect(self.on_sdr_moved)
        self.map_widget.target_selected.connect(self.on_target_selected)
        
        # Setup example data
        self.setup_example_sdrs()
        
        # Timer for simulating target updates
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.simulate_target_updates)
        self.simulation_timer.start(5000)  # Every 5 seconds
        
        self.target_counter = 0
    
    def setup_example_sdrs(self):
        """Setup example SDR configuration."""
        # Reference SDR (slave0) at origin
        self.map_widget.add_or_update_sdr(
            "slave0", 0, 0, 0, 
            is_reference=True, 
            role="reference"
        )
        
        # Measurement SDRs in triangular formation
        self.map_widget.add_or_update_sdr(
            "slave1", 100, 0, 0,
            is_reference=False,
            role="measurement" 
        )
        
        self.map_widget.add_or_update_sdr(
            "slave2", 50, 86.6, 0,  # ~60 degree angle
            is_reference=False,
            role="measurement"
        )
        
        self.map_widget.add_or_update_sdr(
            "slave3", -50, 86.6, 0,
            is_reference=False,
            role="measurement"
        )
        
        print("[Integration] Setup complete - 4 SDR stations configured")
        print("[Integration] Drag the measurement SDRs to reposition them")
        
        # Center map on SDRs
        self.map_widget.center_on_sdrs()
    
    def on_sdr_moved(self, sdr_id: str, x: float, y: float, z: float):
        """Handle SDR position changes from drag-and-drop."""
        print(f"[Integration] SDR {sdr_id} moved to ({x:.1f}, {y:.1f}, {z:.1f})")
        
        # Here you would update your trilateration engine with new positions
        # Example:
        # if hasattr(self, 'trilateration_engine'):
        #     self.trilateration_engine.add_station(sdr_id, x, y, z)
        
        # Update slaves view coordinates table
        # if hasattr(self, 'slaves_view'):
        #     self.slaves_view.update_sdr_coordinates(sdr_id, x, y, z)
    
    def on_target_selected(self, target_id: str):
        """Handle target selection."""
        print(f"[Integration] Target {target_id} selected")
    
    def simulate_target_updates(self):
        """Simulate incoming target detection updates."""
        import random
        
        self.target_counter += 1
        target_id = f"target_{self.target_counter}"
        
        # Random position within SDR triangle
        x = random.uniform(-30, 80)
        y = random.uniform(10, 70)
        z = random.uniform(0, 20)  # Height
        
        # Random frequency and confidence
        frequency = random.uniform(2400, 5800)
        confidence = random.uniform(0.6, 0.95)
        
        # Determine type based on confidence
        target_type = "tracked" if confidence > 0.8 else "detected"
        
        self.map_widget.add_or_update_target(
            target_id, x, y, z,
            frequency=frequency,
            confidence=confidence,
            target_type=target_type
        )
        
        print(f"[Integration] Simulated target: {target_id} at ({x:.1f}, {y:.1f}) "
              f"{frequency:.1f}MHz conf={confidence:.2f}")
    
    def handle_orchestrator_data(self, data: Dict[str, Any]):
        """
        Handle data from the orchestrator (like the existing slaves view).
        This method shows how to integrate with the existing PANORAMA system.
        """
        # This would be called by the main application when new data arrives
        self.map_widget.handle_data_from_app(data)
    
    def update_trilateration_result(self, result):
        """
        Handle trilateration results.
        Convert TrilaterationResult to map target.
        """
        if hasattr(result, 'peak_id') and hasattr(result, 'x'):
            self.map_widget.add_or_update_target(
                result.peak_id,
                result.x,
                result.y,
                getattr(result, 'z', 0),
                frequency=getattr(result, 'freq_mhz', None),
                confidence=getattr(result, 'confidence', None),
                target_type="tracked"
            )


def main():
    """Run the integration example."""
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = PanoramaMapIntegration()
    window.show()
    
    print("=" * 60)
    print("PANORAMA MapLibre Integration Example")
    print("=" * 60)
    print("Features:")
    print("• Drag measurement SDRs to reposition them")
    print("• Real-time target simulation every 5 seconds")  
    print("• Click on SDRs/targets for information")
    print("• Use map controls in the top-right panel")
    print("=" * 60)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()