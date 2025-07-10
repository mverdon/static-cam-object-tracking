"""
YOLOv11 detector with built-in tracking support.
Uses YOLO's native ByteTrack and BotsSort implementations.
"""

import logging
import torch
import cv2
from ultralytics import YOLO
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import tempfile
import yaml

from . import config

logger = logging.getLogger(__name__)


def tensor_to_numpy(tensor_or_array):
    """Convert tensor or array to numpy array."""
    if tensor_or_array is None:
        return None

    # If it's already a numpy array, return as-is
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array

    # If it's a tensor, convert to numpy
    if hasattr(tensor_or_array, 'cpu'):
        return tensor_or_array.cpu().numpy()

    # If it has numpy() method, use it
    if hasattr(tensor_or_array, 'numpy'):
        return tensor_or_array.numpy()

    # Otherwise, convert to numpy array
    return np.array(tensor_or_array)


def safe_astype(array, dtype):
    """Safely convert array to specified dtype."""
    if array is None:
        return None
    return array.astype(dtype)


class YOLODetectorWithTracking:
    """YOLOv11 detector with built-in tracking capabilities."""

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL,
        confidence: float = config.CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.IOU_THRESHOLD,
        use_cuda: bool = config.USE_CUDA,
        use_tensorrt: bool = config.USE_TENSORRT,
        cuda_device: int = config.CUDA_DEVICE,
        tracker_type: str = config.TRACKER_TYPE,
    ):
        """
        Initialize the YOLOv11 detector with tracking.

        Args:
            model_path: Path to the YOLO model file
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            use_cuda: Whether to use CUDA acceleration
            use_tensorrt: Whether to use TensorRT optimization
            cuda_device: CUDA device index to use
            tracker_type: Type of tracker to use ('bytetrack' or 'botsort')
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.use_cuda = use_cuda
        self.use_tensorrt = use_tensorrt
        self.cuda_device = cuda_device
        self.tracker_type = tracker_type.lower()

        # Validate tracker type
        if self.tracker_type not in ['bytetrack', 'botsort']:
            logger.warning(f"Unknown tracker type '{tracker_type}'. Using 'bytetrack'.")
            self.tracker_type = 'bytetrack'

        # Check CUDA availability
        if self.use_cuda and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.use_cuda = False

        # Validate CUDA device
        if self.use_cuda and self.cuda_device >= torch.cuda.device_count():
            logger.warning(f"CUDA device {self.cuda_device} not available. Using device 0.")
            self.cuda_device = 0

        # Set the CUDA device
        if self.use_cuda:
            torch.cuda.set_device(self.cuda_device)
            logger.info(f"Using CUDA device {self.cuda_device}: {torch.cuda.get_device_name(self.cuda_device)}")

        # Load model
        self.model = self._load_model(model_path)

        # Create custom tracker config if needed
        self.tracker_config_path = self._create_tracker_config()

        # Initialize tracking state
        self.tracking_initialized = False
        self.active_tracks = []

        logger.info(f"YOLODetectorWithTracking initialized with {self.tracker_type} tracker")

    def _load_model(self, model_path: str) -> YOLO:
        """Load the YOLO model."""
        try:
            # First check if TensorRT engine exists and should be used
            if self.use_tensorrt and self.use_cuda:
                model_name = model_path.replace('.pt', '')
                tensorrt_path = config.MODELS_DIR / f"{model_name}.engine"

                if tensorrt_path.exists():
                    logger.info(f"Loading existing TensorRT engine: {tensorrt_path}")
                    model = YOLO(str(tensorrt_path))
                    return model

            # Load PyTorch model if no TensorRT engine exists
            model_file = config.MODELS_DIR / model_path
            if not model_file.exists():
                logger.info(f"Model {model_path} not found locally. Downloading...")
                model = YOLO(model_path)  # This will download the model
                # Save to models directory
                model.save(str(model_file))
            else:
                model = YOLO(str(model_file))

            # Move to GPU if available (only for PyTorch models)
            if self.use_cuda:
                model.to(f'cuda:{self.cuda_device}')
                logger.info(f"Model loaded on CUDA device {self.cuda_device}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise

    def _create_tracker_config(self) -> str:
        """Create a custom tracker configuration file."""
        try:
            # Use the built-in tracker configs directly
            return f"{self.tracker_type}.yaml"

        except Exception as e:
            logger.warning(f"Failed to create custom tracker config: {e}")
            logger.info(f"Using default {self.tracker_type}.yaml config")
            return f"{self.tracker_type}.yaml"

    def detect_and_track(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Detect objects and track them using YOLO's built-in tracking.

        Args:
            frame: Input frame
            mask: Optional mask to apply to frame

        Returns:
            List of tracked objects with format:
            [
                {
                    'track_id': int,
                    'class_id': int,
                    'confidence': float,
                    'bbox': (x1, y1, x2, y2),
                    'center': (cx, cy),
                    'class_name': str
                },
                ...
            ]
        """
        # Apply mask if provided
        if mask is not None:
            if config.APPLY_MASK_TO_FRAME:
                # Resize mask to match frame dimensions if needed
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Apply mask to frame before detection
                frame = np.where(mask[..., None] == 0, 0, frame)

        # Run YOLO tracking
        try:
            results = self.model.track(
                frame,
                tracker=self.tracker_config_path,
                persist=True,
                conf=self.confidence,
                iou=self.iou_threshold,
                device=self.cuda_device if self.use_cuda else 'cpu',
                verbose=False,
                classes=config.FILTER_CLASSES  # Filter for specific classes
            )

            # Process results
            tracked_objects = []
            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes

                    # Get track IDs (may be None if no tracks)
                    track_ids = safe_astype(tensor_to_numpy(boxes.id), int) if boxes.id is not None else None

                    # Get detection data using utility function
                    xyxy = tensor_to_numpy(boxes.xyxy)
                    confs = tensor_to_numpy(boxes.conf)
                    classes = safe_astype(tensor_to_numpy(boxes.cls), int)

                    # Ensure arrays are not None
                    if xyxy is None or confs is None or classes is None:
                        return []

                    # Create tracked objects
                    for i, (bbox, conf, cls) in enumerate(zip(xyxy, confs, classes)):
                        # Skip if confidence is too low
                        if conf < self.confidence:
                            continue

                        # Get track ID (use detection index if no track ID available)
                        track_id = track_ids[i] if track_ids is not None else i

                        # Calculate center
                        x1, y1, x2, y2 = bbox
                        center = ((x1 + x2) / 2, (y1 + y2) / 2)

                        # Get class name
                        class_name = self.model.names.get(int(cls), f"class_{cls}")

                        tracked_objects.append({
                            'track_id': track_id,
                            'class_id': cls,
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2),
                            'center': center,
                            'class_name': class_name
                        })

            self.active_tracks = tracked_objects
            return tracked_objects

        except Exception as e:
            logger.error(f"Error in detect_and_track: {e}")
            return []

    def detect(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Tuple[int, float, float, float, float, float]]:
        """
        Detect objects without tracking (for backward compatibility).

        Args:
            frame: Input frame
            mask: Optional mask to apply to frame

        Returns:
            List of detections in format: [(class_id, confidence, x1, y1, x2, y2), ...]
        """
        # Apply mask if provided
        if mask is not None:
            if config.APPLY_MASK_TO_FRAME:
                frame = np.where(mask[..., None] == 0, 0, frame)

        try:
            results = self.model.predict(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                device=self.cuda_device if self.use_cuda else 'cpu',
                verbose=False,
                classes=config.FILTER_CLASSES
            )

            detections = []
            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes

                    # Get detection data using utility function
                    xyxy = tensor_to_numpy(boxes.xyxy)
                    confs = tensor_to_numpy(boxes.conf)
                    classes = safe_astype(tensor_to_numpy(boxes.cls), int)

                    # Ensure arrays are not None
                    if xyxy is None or confs is None or classes is None:
                        return []

                    for bbox, conf, cls in zip(xyxy, confs, classes):
                        if conf >= self.confidence:
                            x1, y1, x2, y2 = bbox
                            detections.append((cls, conf, x1, y1, x2, y2))

            return detections

        except Exception as e:
            logger.error(f"Error in detect: {e}")
            return []

    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """Get currently active tracks."""
        return self.active_tracks

    def get_class_names(self) -> Dict[int, str]:
        """Get class names dictionary."""
        return self.model.names

    def warm_up(self):
        """Warm up the model with a dummy inference."""
        try:
            logger.info("Warming up model...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

            # Warm up detection
            self.model.predict(dummy_frame, verbose=False)

            # Warm up tracking
            self.model.track(dummy_frame, tracker=self.tracker_config_path, persist=True, verbose=False)

            logger.info("Model warmed up successfully")

        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def __del__(self):
        """Clean up temporary files."""
        if hasattr(self, 'tracker_config_path') and self.tracker_config_path.startswith('/tmp'):
            try:
                Path(self.tracker_config_path).unlink(missing_ok=True)
            except:
                pass
