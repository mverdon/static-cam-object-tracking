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
YOLO_MODEL = "yolo11x.pt"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45

# Tracking configuration
TRACKER_TYPE = "bytetrack"  # Options: bytetrack, botsort

# YOLO Built-in Tracker Configuration
# These map to the YOLO tracker YAML parameters
YOLO_TRACKER_CONFIG = {
    "bytetrack": {
        "track_high_thresh": 0.25,   # threshold for the first association
        "track_low_thresh": 0.05,    # threshold for the second association
        "new_track_thresh": 0.25,    # threshold for init new track
        "track_buffer": 300,         # buffer to calculate the time when to remove tracks
        "match_thresh": 0.8,        # threshold for matching tracks
        "fuse_score": True,         # Whether to fuse confidence scores with the iou distances
    },
    "botsort": {
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.6,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.8,
        "with_reid": False,
    }
}

# Video processing configuration
TARGET_FPS = None  # Set to None to use original video FPS, or specify a number to override (can also use --fps argument)
FRAME_SKIP = 1  # Process every nth frame
RESIZE_WIDTH = None  # Set to None to keep original size
RESIZE_HEIGHT = None  # Set to None to keep original size

# Hardware acceleration
USE_CUDA = True
USE_TENSORRT = True
TENSORRT_PRECISION = "fp16"  # Options: fp32, fp16, int8
CUDA_DEVICE = 0  # GPU device index to use (0, 1, 2, etc.)

# Class filtering configuration
FILTER_CLASSES = [17]  # Only detect horses (class 17). Set to None or [] to detect all classes
OBJECT_CLASS_ID = 17  # YOLO class ID for horses

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

# Track-specific video output configuration
ENABLE_TRACK_VIDEOS = False  # Enable separate video files for each track
TRACK_VIDEO_CROP_RATIO = 1.5  # Ratio to expand the bounding box for cropping (1.0 = exact bbox, 2.0 = double size)
TRACK_VIDEO_MIN_SIZE = (128, 72)  # Minimum crop size to ensure readability (16:9 aspect ratio)
TRACK_VIDEO_MAX_SIZE = (1280, 720)  # Fixed output size for all track videos (16:9 aspect ratio)
TRACK_VIDEO_FORMAT = "mp4"
TRACK_VIDEO_CODEC = "mp4v"
TRACK_VIDEOS_DIR = OUTPUTS_DIR / "track_outputs"  # Directory for track-specific videos

# Ensure track videos directory exists
TRACK_VIDEOS_DIR.mkdir(exist_ok=True)

# Logging configuration
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = True
LOG_FILE = PROJECT_ROOT / "tracking.log"
