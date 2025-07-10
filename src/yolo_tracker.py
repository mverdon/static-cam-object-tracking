"""
Simplified tracker using YOLO's built-in tracking capabilities.
"""

import logging
from typing import List, Dict, Any
from .detector_with_tracking import YOLODetectorWithTracking
from . import config

logger = logging.getLogger(__name__)


class YOLOTracker:
    """Simplified tracker using YOLO's built-in ByteTrack/BotsSort."""

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL,
        confidence: float = config.CONFIDENCE_THRESHOLD,
        tracker_type: str = config.TRACKER_TYPE
    ):
        """
        Initialize YOLO tracker.

        Args:
            model_path: Path to YOLO model
            confidence: Confidence threshold
            tracker_type: Type of tracker ('bytetrack' or 'botsort')
        """
        self.detector = YOLODetectorWithTracking(
            model_path=model_path,
            confidence=confidence,
            tracker_type=tracker_type
        )

        self.tracker_type = tracker_type

        logger.info(f"YOLOTracker initialized with {tracker_type}")

    def update(self, frame, mask=None) -> List[Dict[str, Any]]:
        """
        Update tracker with new frame.

        Args:
            frame: Input frame
            mask: Optional mask

        Returns:
            List of tracked objects
        """
        return self.detector.detect_and_track(frame, mask)

    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """Get currently active tracks."""
        return self.detector.get_active_tracks()

    def get_class_names(self) -> Dict[int, str]:
        """Get class names dictionary."""
        return self.detector.get_class_names()

    def warm_up(self):
        """Warm up the detector."""
        self.detector.warm_up()



