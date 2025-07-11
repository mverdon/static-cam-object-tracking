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
import os
import time

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
                else:
                    # TensorRT engine doesn't exist, need to convert from PyTorch model
                    logger.info(f"TensorRT engine not found. Converting from PyTorch model...")

                    # First load the PyTorch model
                    model_file = config.MODELS_DIR / model_path
                    if not model_file.exists():
                        logger.info(f"Model {model_path} not found locally. Downloading...")
                        model = YOLO(model_path)  # This will download the model
                        # Save to models directory
                        model.save(str(model_file))
                    else:
                        model = YOLO(str(model_file))

                    # Move to GPU for conversion
                    if self.use_cuda:
                        model.to(f'cuda:{self.cuda_device}')
                        logger.info(f"Model loaded on CUDA device {self.cuda_device}")

                    # Convert to TensorRT using the improved conversion method
                    tensorrt_model = self._convert_to_tensorrt(model, model_file, tensorrt_path)

                    if tensorrt_model is not None:
                        return tensorrt_model
                    else:
                        logger.info("Falling back to PyTorch model")
                        return model

            # Load PyTorch model if TensorRT is disabled or not available
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
        """        # Apply mask if provided
        if mask is not None:
            if config.APPLY_MASK_TO_FRAME:
                # Resize mask to match frame dimensions if needed
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Ensure mask is 2D (height, width) for proper broadcasting
                if len(mask.shape) == 3 and mask.shape[2] == 1:
                    mask = mask[:, :, 0]

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

    def _convert_to_tensorrt(self, model: YOLO, model_file: Path, tensorrt_path: Path) -> Optional[YOLO]:
        """
        Convert PyTorch model to TensorRT engine with comprehensive error handling.

        Args:
            model: Loaded PyTorch YOLO model
            model_file: Path to the PyTorch model file
            tensorrt_path: Target path for the TensorRT engine

        Returns:
            YOLO model with TensorRT engine if successful, None if failed
        """
        logger.info(f"Converting PyTorch model to TensorRT engine...")
        logger.info(f"This may take several minutes on first run...")
        logger.info(f"Target precision: {config.TENSORRT_PRECISION}")

        try:
            # Validate TensorRT prerequisites
            if not self._validate_tensorrt_environment():
                return None

            # Ensure model is on GPU
            model.to(f'cuda:{self.cuda_device}')

            # Determine optimization settings
            precision_settings = {
                'fp32': {'half': False, 'int8': False},
                'fp16': {'half': True, 'int8': False},
                'int8': {'half': False, 'int8': True}
            }

            precision = config.TENSORRT_PRECISION.lower()
            if precision not in precision_settings:
                logger.warning(f"Invalid TensorRT precision '{precision}'. Using fp16.")
                precision = 'fp16'

            settings = precision_settings[precision]

            # Create export parameters
            export_params = {
                'format': 'engine',
                'half': settings['half'],
                'int8': settings['int8'],
                'device': self.cuda_device,
                'workspace': 4,  # GB
                'verbose': False,
                'imgsz': 640,  # Standard YOLO input size
                'batch': 1,    # Optimize for batch size 1
                'simplify': True,
                'opset': 17,   # ONNX opset version
            }

            # Add INT8 calibration if needed
            if settings['int8']:
                logger.info("INT8 quantization enabled - this may take longer but provides better performance")
                # INT8 quantization will use automatic calibration

            logger.info(f"Export parameters: {export_params}")

            # Perform conversion with timeout
            conversion_start = time.time()

            # Export to TensorRT format
            exported_path = model.export(**export_params)

            conversion_time = time.time() - conversion_start
            logger.info(f"TensorRT conversion completed in {conversion_time:.2f} seconds")

            # Handle the exported file location
            exported_engine = Path(exported_path) if exported_path else model_file.with_suffix('.engine')

            # Ensure the engine is in the correct location
            if exported_engine.exists():
                if exported_engine != tensorrt_path:
                    # Move to target location
                    if tensorrt_path.exists():
                        tensorrt_path.unlink()
                    exported_engine.rename(tensorrt_path)

                # Validate the created engine
                if self._validate_tensorrt_engine(tensorrt_path):
                    logger.info(f"TensorRT engine successfully created: {tensorrt_path}")
                    logger.info(f"Engine file size: {tensorrt_path.stat().st_size / (1024*1024):.2f} MB")

                    # Load and return the TensorRT model
                    tensorrt_model = YOLO(str(tensorrt_path))
                    return tensorrt_model
                else:
                    logger.error("TensorRT engine validation failed")
                    return None
            else:
                logger.error("TensorRT engine file was not created")
                return None

        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}")
            logger.info("This can happen due to:")
            logger.info("  - Insufficient GPU memory")
            logger.info("  - Incompatible model architecture")
            logger.info("  - TensorRT version compatibility issues")
            logger.info("  - CUDA driver issues")
            return None

    def _validate_tensorrt_environment(self) -> bool:
        """Validate that TensorRT environment is properly set up."""
        try:
            # Check TensorRT availability
            import tensorrt as trt
            logger.info(f"TensorRT version: {trt.__version__}")

            # Check CUDA device properties
            device_props = torch.cuda.get_device_properties(self.cuda_device)
            logger.info(f"GPU: {device_props.name}")
            logger.info(f"GPU Memory: {device_props.total_memory / (1024**3):.1f} GB")
            logger.info(f"GPU Compute Capability: {device_props.major}.{device_props.minor}")

            # Check minimum memory requirements (at least 2GB for TensorRT conversion)
            if device_props.total_memory < 2 * (1024**3):
                logger.warning("GPU has less than 2GB memory - TensorRT conversion may fail")
                return False

            # Check compute capability (minimum 6.0 for modern TensorRT)
            compute_capability = device_props.major + device_props.minor * 0.1
            if compute_capability < 6.0:
                logger.warning(f"GPU compute capability {compute_capability} may not support TensorRT")
                return False

            return True

        except ImportError:
            logger.error("TensorRT not installed or not available")
            return False
        except Exception as e:
            logger.error(f"TensorRT environment validation failed: {e}")
            return False

    def _validate_tensorrt_engine(self, engine_path: Path) -> bool:
        """Validate that the TensorRT engine is properly created and loadable."""
        try:
            # Check file exists and has reasonable size
            if not engine_path.exists():
                logger.error("Engine file does not exist")
                return False

            file_size = engine_path.stat().st_size
            if file_size < 1024 * 1024:  # Less than 1MB is suspicious
                logger.error(f"Engine file too small: {file_size} bytes")
                return False

            # Try to load the engine with YOLO to verify it's valid
            try:
                test_model = YOLO(str(engine_path))
                logger.info(f"Engine validation successful")
                return True
            except Exception as e:
                logger.error(f"Failed to load TensorRT engine: {e}")
                return False

        except Exception as e:
            logger.error(f"Engine validation failed: {e}")
            return False

    def __del__(self):
        """Clean up temporary files."""
        if hasattr(self, 'tracker_config_path') and self.tracker_config_path.startswith('/tmp'):
            try:
                Path(self.tracker_config_path).unlink(missing_ok=True)
            except:
                pass
