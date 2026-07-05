
"""Simple IoU-based tracker."""

from dataclasses import dataclass, field

from basketball_possession_analyzer.detection import (
    Detection,
    DetectionLabel,
    DetectionResult,
)
from basketball_possession_analyzer.tracking.result import TrackingResult
from basketball_possession_analyzer.tracking.track import Track


@dataclass(frozen=True)
class SimpleIoUTrackerConfig:
    """Configuration for SimpleIoUTracker."""

    iou_threshold: float = 0.3
    max_missed_frames: int = 2
    track_labels: tuple[DetectionLabel, ...] = field(
        default_factory=lambda: (DetectionLabel.PLAYER,)
    )

    def __post_init__(self) -> None:
        """Validate tracker configuration."""
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames must be >= 0")
        if not self.track_labels:
            raise ValueError("track_labels must not be empty")


class SimpleIoUTracker:
    """A deterministic IoU tracker for basic player tracking."""

    def __init__(self, config: SimpleIoUTrackerConfig | None = None) -> None:
        self.config = config or SimpleIoUTrackerConfig()
        self._next_track_id = 0
        self._tracks: dict[int, Track] = {}

    @property
    def tracks(self) -> list[Track]:
        """Return active tracks."""
        return list(self._tracks.values())

    def reset(self) -> None:
        """Reset tracker state."""
        self._next_track_id = 0
        self._tracks.clear()

    def update(self, detection_result: DetectionResult) -> TrackingResult:
        """Update tracker with detections from one frame."""
        frame_index = detection_result.frame_index
        detections = self._filter_trackable_detections(detection_result.detections)

        matches = self._match_detections_to_tracks(detections)
        matched_detection_indexes = set(matches)
        matched_track_ids = set(matches.values())

        for detection_index, track_id in matches.items():
            self._tracks[track_id].update(
                detection=detections[detection_index],
                frame_index=frame_index,
            )

        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_track_ids:
                track.mark_missed()
                if not track.is_active(self.config.max_missed_frames):
                    del self._tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detection_indexes:
                self._create_track(detection=detection, frame_index=frame_index)

        return TrackingResult(
            frame_index=frame_index,
            tracks=self.tracks,
        )

    def _filter_trackable_detections(
        self,
        detections: list[Detection],
    ) -> list[Detection]:
        """Return detections that should be tracked."""
        return [
            detection
            for detection in detections
            if detection.label in self.config.track_labels
        ]

    def _create_track(self, detection: Detection, frame_index: int) -> None:
        """Create a new track."""
        track = Track(
            track_id=self._next_track_id,
            detection=detection,
            first_frame_index=frame_index,
            last_frame_index=frame_index,
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1

    def _match_detections_to_tracks(
        self,
        detections: list[Detection],
    ) -> dict[int, int]:
        """Greedily match detections to existing tracks by IoU."""
        candidate_matches: list[tuple[float, int, int]] = []

        for detection_index, detection in enumerate(detections):
            for track_id, track in self._tracks.items():
                if detection.label != track.label:
                    continue

                iou = detection.bbox.iou(track.bbox)
                if iou >= self.config.iou_threshold:
                    candidate_matches.append((iou, detection_index, track_id))

        candidate_matches.sort(reverse=True)

        matches: dict[int, int] = {}
        used_detections: set[int] = set()
        used_tracks: set[int] = set()

        for _iou, detection_index, track_id in candidate_matches:
            if detection_index in used_detections or track_id in used_tracks:
                continue

            matches[detection_index] = track_id
            used_detections.add(detection_index)
            used_tracks.add(track_id)

        return matches
