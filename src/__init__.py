"""
Static Camera Object Tracking Package

A real-time object tracking system using YOLOv11 with built-in tracking capabilities.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .detector_with_tracking import YOLODetectorWithTracking
from .yolo_tracker import YOLOTracker
from .utils import setup_logging, print_system_info

__all__ = [
    "YOLODetectorWithTracking",
    "YOLOTracker",
    "setup_logging",
    "print_system_info"
]
