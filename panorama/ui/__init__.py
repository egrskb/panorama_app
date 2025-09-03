from panorama.features.spectrum.view import SpectrumView  # re-export
from panorama.features.watchlist.view import ImprovedSlavesView  # re-export
from panorama.features.map.openlayers_widget_v10 import OpenLayersMapWidget  # re-export
from panorama.features.detector.settings_dialog import DetectorSettingsDialog, DetectorSettings  # re-export

__all__ = [
    "SpectrumView",
    "ImprovedSlavesView",
    "OpenLayersMapWidget",
    "DetectorSettingsDialog",
    "DetectorSettings",
]
"""
UI модули для ПАНОРАМА RSSI.
"""

from .main_ui_manager import MainUIManager

__all__ = ['MainUIManager']