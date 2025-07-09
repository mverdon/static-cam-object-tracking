"""
YOLOv11 detector wrapper with CUDA and TensorRT support.
"""

import logging
import torch
from ultralytics import YOLO
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from . import config

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLOv11 object detector with hardware acceleration support."""

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL,
        confidence: float = config.CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.IOU_THRESHOLD,
        use_cuda: bool = config.USE_CUDA,
        use_tensorrt: bool = config.USE_TENSORRT,
    ):
        """
        Initialize the YOLOv11 detector.

        Args:
            model_path: Path to the YOLO model file
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            use_cuda: Whether to use CUDA acceleration
            use_tensorrt: Whether to use TensorRT optimization
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.use_cuda = use_cuda
        self.use_tensorrt = use_tensorrt

        # Check CUDA availability
        if self.use_cuda and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.use_cuda = False

        # Load model
        self.model = self._load_model(model_path)

        # Optimize with TensorRT if requested
        if self.use_tensorrt and self.use_cuda:
            self._optimize_with_tensorrt()

    def _load_model(self, model_path: str) -> YOLO:
        """Load the YOLO model."""
        try:
            # Check if model exists locally, otherwise it will be downloaded
            model_file = config.MODELS_DIR / model_path
            if not model_file.exists():
                logger.info(f"Model {model_path} not found locally. Downloading...")
                model = YOLO(model_path)  # This will download the model
                # Save to models directory
                model.save(str(model_file))
            else:
                model = YOLO(str(model_file))

            # Move to GPU if available
            if self.use_cuda:
                model.to('cuda')
                logger.info("Model loaded on CUDA device")
            else:
                logger.info("Model loaded on CPU")

            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise

    def _optimize_with_tensorrt(self):
        """Optimize model with TensorRT."""
        try:
            logger.info("Optimizing model with TensorRT...")

            # Export to TensorRT format
            tensorrt_path = config.MODELS_DIR / f"{config.YOLO_MODEL.replace('.pt', '')}_tensorrt.engine"

            if not tensorrt_path.exists():
                self.model.export(
                    format='engine',
                    half=config.TENSORRT_PRECISION == 'fp16',
                    int8=config.TENSORRT_PRECISION == 'int8',
                    device=0 if self.use_cuda else 'cpu',
                    workspace=4,  # 4GB workspace
                    verbose=False
                )
                logger.info("TensorRT optimization completed")
            else:
                logger.info("Using existing TensorRT engine")

        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}. Continuing with standard model.")
            self.use_tensorrt = False

    def detect(self, frame: np.ndarray) -> List[Tuple[int, float, float, float, float, float]]:
        """
        Perform object detection on a frame.

        Args:
            frame: Input frame as numpy array (BGR format)

        Returns:
            List of detections as tuples: (class_id, confidence, x1, y1, x2, y2)
        """
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                verbose=False
            )

            detections = []

            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        # Get box coordinates
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

                        # Get confidence and class
                        confidence = boxes.conf[i].cpu().numpy()
                        class_id = int(boxes.cls[i].cpu().numpy())

                        detections.append((class_id, float(confidence), float(x1), float(y1), float(x2), float(y2)))

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def get_class_names(self) -> dict:
        """Get the class names dictionary."""
        return self.model.names

    def warm_up(self, input_shape: Tuple[int, int] = (640, 640)):
        """Warm up the model with a dummy input."""
        logger.info("Warming up the model...")
        dummy_input = np.random.randint(0, 255, (input_shape[1], input_shape[0], 3), dtype=np.uint8)
        self.detect(dummy_input)
        logger.info("Model warm-up completed")
