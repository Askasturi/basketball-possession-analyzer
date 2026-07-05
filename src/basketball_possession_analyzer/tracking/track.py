
"""Track data model."""

from dataclasses import dataclass

from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
)


@dataclass
class Track:
    """A tracked object across frames."""

    track_id: int
    detection: Detection
    first_frame_index: int
    last_frame_index: int
    hits: int = 1
    missed_frames: int = 0

    def __post_init__(self) -> None:
        """Validate track fields."""
        if self.track_id < 0:
            raise ValueError("track_id must be >= 0")
        if self.first_frame_index < 0:
            raise ValueError("first_frame_index must be >= 0")
        if self.last_frame_index < self.first_frame_index:
            raise ValueError("last_frame_index must be >= first_frame_index")
        if self.hits <= 0:
            raise ValueError("hits must be > 0")
        if self.missed_frames < 0:
            raise ValueError("missed_frames must be >= 0")

    @property
    def bbox(self) -> BoundingBox:
        """Return the current track bounding box."""
        return self.detection.bbox

    @property
    def label(self) -> DetectionLabel:
        """Return the current track label."""
        return self.detection.label

    @property
    def confidence(self) -> float:
        """Return the current detection confidence."""
        return self.detection.confidence

    @property
    def age(self) -> int:
        """Return track age in frames."""
        return self.last_frame_index - self.first_frame_index + 1

    @property
    def is_confirmed(self) -> bool:
        """Return whether the track has been seen more than once."""
        return self.hits > 1

    def update(self, detection: Detection, frame_index: int) -> None:
        """Update track with a matched detection."""
        if frame_index < self.last_frame_index:
            raise ValueError("frame_index must be >= last_frame_index")

        self.detection = detection
        self.last_frame_index = frame_index
        self.hits += 1
        self.missed_frames = 0

    def mark_missed(self) -> None:
        """Mark track as missed for one frame."""
        self.missed_frames += 1

    def is_active(self, max_missed_frames: int) -> bool:
        """Return whether the track should remain active."""
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be >= 0")
        return self.missed_frames <= max_missed_frames
