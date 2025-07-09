"""
Configuration file for the object tracking system.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
VIDEOS_DIR = PROJECT_ROOT / "videos"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
VIDEOS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# YOLOv11 model configuration
YOLO_MODEL = "yolo11n.pt"  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Tracking configuration
TRACKER_TYPE = "bytetrack"  # Options: bytetrack, botsort
MAX_DISAPPEARED = 30
MAX_DISTANCE = 50

# Video processing configuration
TARGET_FPS = 30
FRAME_SKIP = 1  # Process every nth frame
RESIZE_WIDTH = 1280  # Set to None to keep original size
RESIZE_HEIGHT = 720  # Set to None to keep original size

# Hardware acceleration
USE_CUDA = True
USE_TENSORRT = True
TENSORRT_PRECISION = "fp16"  # Options: fp32, fp16, int8

# Output configuration
OUTPUT_FORMAT = "mp4"
OUTPUT_CODEC = "mp4v"
SHOW_CONFIDENCE = True
SHOW_TRACK_ID = True
SHOW_CLASS_NAME = True

# Colors for different object classes (BGR format)
CLASS_COLORS = {
    0: (255, 0, 0),      # person - red
    1: (0, 255, 0),      # bicycle - green
    2: (0, 0, 255),      # car - blue
    3: (255, 255, 0),    # motorcycle - cyan
    5: (255, 0, 255),    # bus - magenta
    7: (0, 255, 255),    # truck - yellow
}

# Default color for other classes
DEFAULT_COLOR = (128, 128, 128)  # gray

# Logging configuration
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = True
LOG_FILE = PROJECT_ROOT / "tracking.log"
