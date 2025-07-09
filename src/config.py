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
YOLO_MODEL = "yolo11s.pt"  # Using yolo11s for better accuracy on horses
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for better horse detection
IOU_THRESHOLD = 0.45

# Tracking configuration
TRACKER_TYPE = "bytetrack"  # Options: bytetrack, botsort
MAX_DISAPPEARED = 50
MAX_DISTANCE = 200

# Horse-specific tracking configuration
HORSE_MIN_AREA = 1000  # Minimum area (pixels) for a valid horse detection
HORSE_MAX_AREA = 50000  # Maximum area (pixels) for a valid horse detection
HORSE_MIN_CONFIDENCE = 0.3  # Minimum confidence for horse detections
TRAJECTORY_LENGTH = 50  # Number of points to keep in trajectory (for movement analysis)
MOVEMENT_SMOOTHING = True  # Enable trajectory smoothing for horses

# Video processing configuration
TARGET_FPS = 30
FRAME_SKIP = 1  # Process every nth frame
RESIZE_WIDTH = 1280  # Set to None to keep original size
RESIZE_HEIGHT = 720  # Set to None to keep original size

# Hardware acceleration
USE_CUDA = True
USE_TENSORRT = True
TENSORRT_PRECISION = "fp16"  # Options: fp32, fp16, int8
CUDA_DEVICE = 0  # GPU device index to use (0, 1, 2, etc.)

# Class filtering configuration
FILTER_CLASSES = [17]  # Only detect horses (class 17). Set to None or [] to detect all classes
HORSE_CLASS_ID = 17  # YOLO class ID for horses

# Masking configuration
USE_MASK = True  # Enable automatic mask detection and application
APPLY_MASK_TO_FRAME = True  # Apply mask to frame before detection (recommended)
SHOW_MASK_OVERLAY = False  # Show mask overlay on output video

# Output configuration
OUTPUT_FORMAT = "mp4"
OUTPUT_CODEC = "mp4v"
SHOW_CONFIDENCE = True
SHOW_TRACK_ID = True
SHOW_CLASS_NAME = True

# Colors for different object classes (BGR format)
CLASS_COLORS = {
    0: (255, 0, 0),      # person - red
    1: (0, 165, 255),    # bicycle - orange
    2: (0, 0, 255),      # car - blue
    3: (255, 255, 0),    # motorcycle - cyan
    5: (255, 0, 255),    # bus - magenta
    7: (0, 255, 255),    # truck - yellow
    17: (0, 255, 0),     # horse - green (primary focus)
}

# Default color for other classes
DEFAULT_COLOR = (128, 128, 128)  # gray

# Logging configuration
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = True
LOG_FILE = PROJECT_ROOT / "tracking.log"
