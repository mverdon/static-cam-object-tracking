"""
Static Camera Object Tracking Package

A real-time object tracking system using YOLOv11 with CUDA and TensorRT acceleration.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .detector import YOLODetector
from .tracker import ObjectTracker, Track
from .utils import setup_logging, print_system_info

__all__ = [
    "YOLODetector",
    "ObjectTracker",
    "Track",
    "setup_logging",
    "print_system_info"
]
