"""
Utility functions for the object tracking system.
"""

import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from . import config
from .tracker import Track

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
        return cv2.resize(frame, (new_width, height))


def draw_tracks(frame: np.ndarray, tracks: List[Track], class_names: dict) -> np.ndarray:
    """
    Draw tracking results on the frame.

    Args:
        frame: Input frame
        tracks: List of active tracks
        class_names: Dictionary mapping class IDs to names

    Returns:
        Frame with tracking visualization
    """
    output_frame = frame.copy()

    for track in tracks:
        if track.disappeared > 0:
            continue

        x1, y1, x2, y2 = map(int, track.bbox)
        class_id = track.class_id
        track_id = track.track_id
        confidence = track.confidence

        # Get color for this class
        color = config.CLASS_COLORS.get(class_id, config.DEFAULT_COLOR)

        # Draw bounding box
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        label_parts = []

        if config.SHOW_TRACK_ID:
            label_parts.append(f"ID:{track_id}")

        if config.SHOW_CLASS_NAME and class_id in class_names:
            label_parts.append(class_names[class_id])

        if config.SHOW_CONFIDENCE:
            label_parts.append(f"{confidence:.2f}")

        label = " ".join(label_parts)

        # Draw label background
        if label:
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                output_frame,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                output_frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        # Draw trajectory if available
        if len(track.positions) > 1:
            points = [(int(x), int(y)) for x, y in track.positions]
            for i in range(1, len(points)):
                # Fade the trajectory
                alpha = i / len(points)
                thickness = max(1, int(2 * alpha))
                cv2.line(output_frame, points[i-1], points[i], color, thickness)

    return output_frame


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
    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_CODEC)
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


def print_system_info():
    """Print system information for debugging."""
    import torch

    logger.info("=== System Information ===")
    logger.info(f"OpenCV version: {cv2.__version__}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    try:
        import tensorrt as trt
        logger.info(f"TensorRT version: {trt.__version__}")
    except ImportError:
        logger.info("TensorRT not available")

    logger.info("==========================")
