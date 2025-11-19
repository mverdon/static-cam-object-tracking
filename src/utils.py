"""
Utility functions for the object tracking system.
"""

import logging
import cv2
import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import torch

from . import config

logger = logging.getLogger(__name__)


class TrackVideoManager:
    """Manages separate video files for each tracked object."""

    def __init__(self, output_dir: str, fps: float, crop_ratio: float = 1.5):
        """
        Initialize the track video manager.

        Args:
            output_dir: Directory to save track videos
            fps: Frames per second for track videos
            crop_ratio: Ratio to expand bounding box for cropping
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.crop_ratio = crop_ratio
        self.track_writers: Dict[int, cv2.VideoWriter] = {}
        self.track_info: Dict[int, Dict[str, Any]] = {}
        self.track_pos_buffer: Dict[int, List[Tuple[int, int]]] = {}
        self.track_centers: Dict[int, Tuple[float, float]] = {}

        # Fixed output video dimensions (all track videos will have this size)
        self.output_width = config.TRACK_VIDEO_MAX_SIZE[0]
        self.output_height = config.TRACK_VIDEO_MAX_SIZE[1]
        self.output_aspect_ratio = self.output_width / self.output_height

        logger.info(f"TrackVideoManager initialized: {self.output_dir}, crop_ratio={crop_ratio}")
        logger.info(f"Fixed output size: {self.output_width}x{self.output_height}")

    def _calculate_crop_region_and_padding(self, bbox: Tuple[float, float, float, float],
                                          frame_shape: Tuple[int, int]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Calculate the crop region around a bounding box with fixed aspect ratio.

        This method ensures consistent video dimensions by:
        1. Using bbox center and height to determine crop size
        2. Maintaining the output video aspect ratio
        3. Calculating padding needed for out-of-bounds areas

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            frame_shape: Frame shape (height, width)

        Returns:
            Tuple of (crop_region, padding):
            - crop_region: (x1, y1, x2, y2) - actual crop coordinates (may be out of bounds)
            - padding: (top, bottom, left, right) - padding needed for out-of-bounds areas
        """
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame_shape[:2]

        # Calculate center and bbox dimensions
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        bbox_height = y2 - y1

        # Calculate crop dimensions based on bbox height and crop ratio
        crop_height = bbox_height * self.crop_ratio
        crop_width = crop_height * self.output_aspect_ratio

        # Ensure minimum size based on bbox height
        min_crop_height = max(crop_height, config.TRACK_VIDEO_MIN_SIZE[1])
        min_crop_width = min_crop_height * self.output_aspect_ratio

        # Use the larger of calculated or minimum dimensions
        final_crop_height = 480
        final_crop_width = 854

        # Calculate crop bounds centered on bbox center
        half_width = final_crop_width / 2
        half_height = final_crop_height / 2

        crop_x1 = int(center_x - half_width)
        crop_y1 = int(center_y - half_height)
        crop_x2 = int(center_x + half_width)
        crop_y2 = int(center_y + half_height)

        # Calculate padding needed for out-of-bounds areas
        pad_left = max(0, -crop_x1)
        pad_top = max(0, -crop_y1)
        pad_right = max(0, crop_x2 - frame_width)
        pad_bottom = max(0, crop_y2 - frame_height)

        return (crop_x1, crop_y1, crop_x2, crop_y2), (pad_top, pad_bottom, pad_left, pad_right)

    def _get_track_filename(self, track_id: int, class_name: str) -> str:
        """Generate filename for a track video."""
        return f"track_{track_id:04d}_{class_name}.{config.TRACK_VIDEO_FORMAT}"

    def add_track_frame(self, track: Dict[str, Any], frame: np.ndarray):
        """
        Add a frame for a specific track with fixed output dimensions.

        Args:
            track: Track information dictionary
            frame: Full frame
        """
        track_id = track['track_id']
        class_name = track['class_name']
        bbox = track['bbox']

        # Update track position buffer
        if track_id not in self.track_pos_buffer:
            self.track_pos_buffer[track_id] = []
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.track_pos_buffer[track_id].append((center_x, center_y))

        # Limit buffer size
        if len(self.track_pos_buffer[track_id]) > config.TRACK_POS_BUFFER_SIZE:
            self.track_pos_buffer[track_id].pop(0)

        # Calculate center of mass (average position) for the track
        if len(self.track_pos_buffer[track_id]) > 0:
            avg_x = sum(pos[0] for pos in self.track_pos_buffer[track_id]) / len(self.track_pos_buffer[track_id])
            avg_y = sum(pos[1] for pos in self.track_pos_buffer[track_id]) / len(self.track_pos_buffer[track_id])
            self.track_centers[track_id] = (avg_x, avg_y)

        # Adjust bbox to center around average position if available
        if track_id in self.track_centers:
            center_x, center_y = self.track_centers[track_id]
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            bbox = (
                center_x - bbox_width / 2,
                center_y - bbox_height / 2,
                center_x + bbox_width / 2,
                center_y + bbox_height / 2
            )

        # Calculate crop region and padding
        (crop_x1, crop_y1, crop_x2, crop_y2), (pad_top, pad_bottom, pad_left, pad_right) = \
            self._calculate_crop_region_and_padding(bbox, frame.shape)

        # Initialize writer for this track if not exists
        if track_id not in self.track_writers:
            filename = self._get_track_filename(track_id, class_name)
            output_path = self.output_dir / filename

            # Store track info with fixed dimensions
            self.track_info[track_id] = {
                'class_name': class_name,
                'filename': filename,
                'path': str(output_path),
                'frame_count': 0,
                'crop_width': self.output_width,
                'crop_height': self.output_height
            }

            # Create video writer with fixed output dimensions
            fourcc = cv2.VideoWriter.fourcc(*config.TRACK_VIDEO_CODEC)
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.fps,
                (self.output_width, self.output_height)
            )

            if not writer.isOpened():
                logger.error(f"Failed to create video writer for track {track_id}: {output_path}")
                return

            self.track_writers[track_id] = writer
            logger.info(f"Created video writer for track {track_id}: {filename}, "
                       f"size={self.output_width}x{self.output_height}")

        # Create the cropped and padded frame
        if track_id in self.track_writers:
            # Step 1: Crop the frame (may include out-of-bounds areas)
            frame_height, frame_width = frame.shape[:2]

            # Clamp crop coordinates to frame boundaries
            crop_x1_clamped = max(0, crop_x1)
            crop_y1_clamped = max(0, crop_y1)
            crop_x2_clamped = min(frame_width, crop_x2)
            crop_y2_clamped = min(frame_height, crop_y2)

            # Extract the available portion of the crop
            cropped_frame = frame[crop_y1_clamped:crop_y2_clamped, crop_x1_clamped:crop_x2_clamped]

            # Step 2: Add black padding for out-of-bounds areas
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                cropped_frame = cv2.copyMakeBorder(
                    cropped_frame,
                    pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)  # Black padding
                )

            # Step 3: Resize to fixed output dimensions
            if cropped_frame.shape[1] != self.output_width or cropped_frame.shape[0] != self.output_height:
                cropped_frame = cv2.resize(cropped_frame, (self.output_width, self.output_height))

            # Step 4: Write the frame
            self.track_writers[track_id].write(cropped_frame)
            self.track_info[track_id]['frame_count'] += 1

    def close_all(self):
        """Close all track video writers and log summary."""
        logger.info("Closing track video writers...")

        for track_id, writer in self.track_writers.items():
            writer.release()
            track_info = self.track_info[track_id]
            logger.info(f"Track {track_id} ({track_info['class_name']}): "
                       f"{track_info['frame_count']} frames -> {track_info['filename']}")

        self.track_writers.clear()
        logger.info(f"Generated {len(self.track_info)} track videos in {self.output_dir}")


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
        print(f"PyTorch version: {torch.__version__}")
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
