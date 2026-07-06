
"""Shot event data models."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from basketball_possession_analyzer.court import CourtPoint, ImagePoint


class ShotResult(StrEnum):
    """Shot result labels."""

    MAKE = "make"
    MISS = "miss"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ShotEvent:
    """Detected basketball shot event."""

    frame_index: int
    shooter_track_id: int | None = None
    ball_image_point: ImagePoint | None = None
    ball_court_point: CourtPoint | None = None
    result: ShotResult = ShotResult.UNKNOWN
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shot event."""
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if self.shooter_track_id is not None and self.shooter_track_id < 0:
            raise ValueError("shooter_track_id must be >= 0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

    @property
    def is_made(self) -> bool:
        """Return whether this shot is made."""
        return self.result == ShotResult.MAKE

    @property
    def is_missed(self) -> bool:
        """Return whether this shot is missed."""
        return self.result == ShotResult.MISS
