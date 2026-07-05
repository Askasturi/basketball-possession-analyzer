
"""Track projection models."""

from dataclasses import dataclass, field
from typing import Any

from basketball_possession_analyzer.court.points import CourtPoint, ImagePoint


@dataclass(frozen=True)
class TrackCourtProjection:
    """A track projected from image coordinates to court coordinates."""

    track_id: int
    image_point: ImagePoint
    court_point: CourtPoint
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate projection."""
        if self.track_id < 0:
            raise ValueError("track_id must be >= 0")


@dataclass(frozen=True)
class CourtProjectionResult:
    """Court projections for one frame."""

    frame_index: int
    projections: list[TrackCourtProjection] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate court projection result."""
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")

    def __len__(self) -> int:
        """Return number of projections."""
        return len(self.projections)

    def get_projection(self, track_id: int) -> TrackCourtProjection | None:
        """Return projection by track ID."""
        for projection in self.projections:
            if projection.track_id == track_id:
                return projection
        return None

    @property
    def track_ids(self) -> list[int]:
        """Return projected track IDs."""
        return [projection.track_id for projection in self.projections]
