"""
Utility functions for the object tracking system.
"""

import logging
import cv2
import numpy as np
import math
from typing import List, Tuple, Optional
from pathlib import Path
import torch

from . import config

logger = logging.getLogger(__name__)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.LOG_FILE) if config.LOG_TO_FILE else logging.NullHandler()
        ]
    )


def resize_frame(frame: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.

    Args:
        frame: Input frame
        width: Target width (optional)
        height: Target height (optional)

    Returns:
        Resized frame
    """
    if width is None and height is None:
        return frame

    h, w = frame.shape[:2]

    if width is not None and height is not None:
        # Resize to exact dimensions
        return cv2.resize(frame, (width, height))
    elif width is not None:
        # Resize based on width, maintain aspect ratio
        aspect_ratio = h / w
        new_height = int(width * aspect_ratio)
        return cv2.resize(frame, (width, new_height))
    else:
        # Resize based on height, maintain aspect ratio
        aspect_ratio = w / h
        new_width = int(height * aspect_ratio)
        return cv2.resize(frame, (new_width, height)) # type: ignore


def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    info = {
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }

    cap.release()
    return info


def create_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """
    Create a video writer for output.

    Args:
        output_path: Path for output video
        fps: Frames per second
        width: Frame width
        height: Frame height

    Returns:
        OpenCV VideoWriter object
    """
    fourcc = cv2.VideoWriter.fourcc(*config.OUTPUT_CODEC)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise ValueError(f"Could not create video writer for: {output_path}")

    return writer


def validate_video_file(video_path: str) -> bool:
    """
    Validate if a video file can be opened and read.

    Args:
        video_path: Path to the video file

    Returns:
        True if valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    except Exception:
        return False


def get_available_video_files(directory: str) -> List[str]:
    """
    Get list of available video files in a directory.

    Args:
        directory: Directory to search

    Returns:
        List of video file paths
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    video_files = []

    directory_path = Path(directory)
    if not directory_path.exists():
        return video_files

    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            if validate_video_file(str(file_path)):
                video_files.append(str(file_path))

    return sorted(video_files)


def list_cuda_devices():
    """List available CUDA devices."""
    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        return

    device_count = torch.cuda.device_count()
    print(f"\nFound {device_count} CUDA device(s):")

    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_props = torch.cuda.get_device_properties(i)
        memory_gb = device_props.total_memory / (1024**3)
        print(f"  Device {i}: {device_name}")
        print(f"    Memory: {memory_gb:.1f} GB")
        print(f"    Compute Capability: {device_props.major}.{device_props.minor}")
    print()


def print_system_info():
    """Print system information including CUDA devices."""
    print("System Information:")
    print("==================")

    # CUDA information
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current CUDA device: {current_device}")
        list_cuda_devices()
    else:
        print("CUDA available: No")

    # OpenCV information
    print(f"OpenCV version: {cv2.__version__}")
    print()


def load_video_mask(video_path: str) -> Optional[np.ndarray]:
    """
    Load mask image for a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Mask array (binary, 0-255) or None if no mask found
    """
    try:
        video_path_obj = Path(video_path)
        mask_path = video_path_obj.parent / f"{video_path_obj.stem}_mask.png"

        if not mask_path.exists():
            logger.info(f"No mask file found: {mask_path}")
            return None

        # Load mask as grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"Failed to load mask file: {mask_path}")
            return None

        # Normalize to binary (0 or 255)
        # Black/transparent areas (≤127) are masked out (0)
        # White areas (>127) are valid detection areas (255)
        mask = np.where(mask < 127, 0, 255).astype(np.uint8)
        logger.info(f"Loaded mask: {mask_path} (shape: {mask.shape}, range: {mask.min()}-{mask.max()})")
        return mask

    except Exception as e:
        logger.error(f"Error loading mask for {video_path}: {e}")
        return None


def apply_mask_to_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to frame, setting masked areas to black.

    Args:
        frame: Input frame (BGR)
        mask: Binary mask (0 = mask out, 255 = keep)

    Returns:
        Masked frame
    """
    try:
        # Resize mask to match frame if needed
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # Convert mask to 3-channel for broadcasting
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Apply mask (keep areas where mask is white/255)
        masked_frame = cv2.bitwise_and(frame, mask_3ch)

        return masked_frame

    except Exception as e:
        logger.error(f"Error applying mask to frame: {e}")
        return frame
