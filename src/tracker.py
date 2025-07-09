"""
Object tracking implementation using YOLO detections.
"""

import logging
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, deque
import math

from . import config

logger = logging.getLogger(__name__)


class Track:
    """Individual track for an object."""

    def __init__(self, track_id: int, class_id: int, bbox: Tuple[float, float, float, float], confidence: float):
        self.track_id = track_id
        self.class_id = class_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.age = 0
        self.disappeared = 0
        self.positions = deque(maxlen=30)  # Store last 30 positions for trajectory
        self.center = self._get_center(bbox)
        self.positions.append(self.center)

    def _get_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Get the center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, bbox: Tuple[float, float, float, float], confidence: float):
        """Update track with new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.center = self._get_center(bbox)
        self.positions.append(self.center)
        self.age += 1
        self.disappeared = 0

    def mark_disappeared(self):
        """Mark track as disappeared for one frame."""
        self.disappeared += 1
        self.age += 1


class ObjectTracker:
    """Multi-object tracker using centroid tracking with YOLO detections."""

    def __init__(
        self,
        max_disappeared: int = config.MAX_DISAPPEARED,
        max_distance: float = config.MAX_DISTANCE
    ):
        """
        Initialize the object tracker.

        Args:
            max_disappeared: Maximum frames a track can be missing before deletion
            max_distance: Maximum distance for associating detections with tracks
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[Tuple[int, float, float, float, float, float]]) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections (class_id, confidence, x1, y1, x2, y2)

        Returns:
            List of active tracks
        """
        # If no detections, mark all tracks as disappeared
        if len(detections) == 0:
            for track in self.tracks.values():
                track.mark_disappeared()

            # Remove tracks that have been missing too long
            self._remove_old_tracks()
            return list(self.tracks.values())

        # Convert detections to input format
        input_objects = []
        for det in detections:
            class_id, confidence, x1, y1, x2, y2 = det
            input_objects.append((class_id, confidence, (x1, y1, x2, y2)))

        # If no existing tracks, create new ones
        if len(self.tracks) == 0:
            for class_id, confidence, bbox in input_objects:
                self._create_new_track(class_id, bbox, confidence)
        else:
            # Associate detections with existing tracks
            self._associate_detections(input_objects)

        # Remove tracks that have been missing too long
        self._remove_old_tracks()

        return list(self.tracks.values())

    def _associate_detections(self, detections: List[Tuple[int, float, Tuple[float, float, float, float]]]):
        """Associate detections with existing tracks."""
        if len(detections) == 0:
            return

        # Get centroids of existing tracks
        track_centroids = {}
        for track_id, track in self.tracks.items():
            track_centroids[track_id] = track.center

        # Get centroids of detections
        detection_centroids = []
        for class_id, confidence, bbox in detections:
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            detection_centroids.append(center)

        # Compute distance matrix
        distance_matrix = self._compute_distance_matrix(
            list(track_centroids.values()),
            detection_centroids
        )

        # Perform assignment using Hungarian algorithm (simplified greedy approach)
        used_track_indices = set()
        used_detection_indices = set()

        # Sort by distance and assign greedily
        assignments = []
        for i, track_id in enumerate(track_centroids.keys()):
            for j, _ in enumerate(detections):
                if i in used_track_indices or j in used_detection_indices:
                    continue

                distance = distance_matrix[i][j]
                if distance < self.max_distance:
                    # Check class consistency (allow some flexibility)
                    track_class = self.tracks[track_id].class_id
                    detection_class = detections[j][0]

                    # For now, allow any assignment (can be made stricter)
                    assignments.append((track_id, j, distance))

        # Sort by distance and apply assignments
        assignments.sort(key=lambda x: x[2])

        for track_id, detection_idx, _ in assignments:
            if track_id not in used_track_indices and detection_idx not in used_detection_indices:
                # Update existing track
                class_id, confidence, bbox = detections[detection_idx]
                self.tracks[track_id].update(bbox, confidence)
                used_track_indices.add(track_id)
                used_detection_indices.add(detection_idx)

        # Mark unmatched tracks as disappeared
        for track_id in track_centroids.keys():
            if track_id not in used_track_indices:
                self.tracks[track_id].mark_disappeared()

        # Create new tracks for unmatched detections
        for j, detection in enumerate(detections):
            if j not in used_detection_indices:
                class_id, confidence, bbox = detection
                self._create_new_track(class_id, bbox, confidence)

    def _compute_distance_matrix(self, track_centers: List[Tuple[float, float]],
                                detection_centers: List[Tuple[float, float]]) -> List[List[float]]:
        """Compute Euclidean distance matrix between tracks and detections."""
        distance_matrix = []

        for track_center in track_centers:
            row = []
            for detection_center in detection_centers:
                distance = math.sqrt(
                    (track_center[0] - detection_center[0]) ** 2 +
                    (track_center[1] - detection_center[1]) ** 2
                )
                row.append(distance)
            distance_matrix.append(row)

        return distance_matrix

    def _create_new_track(self, class_id: int, bbox: Tuple[float, float, float, float], confidence: float):
        """Create a new track."""
        track = Track(self.next_id, class_id, bbox, confidence)
        self.tracks[self.next_id] = track
        self.next_id += 1

    def _remove_old_tracks(self):
        """Remove tracks that have been missing for too long."""
        to_remove = []
        for track_id, track in self.tracks.items():
            if track.disappeared > self.max_disappeared:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]

    def get_active_tracks(self) -> List[Track]:
        """Get all currently active tracks."""
        return [track for track in self.tracks.values() if track.disappeared == 0]
