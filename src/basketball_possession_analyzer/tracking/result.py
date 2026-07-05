
"""Tracking result model."""

from dataclasses import dataclass, field

from basketball_possession_analyzer.tracking.track import Track


@dataclass(frozen=True)
class TrackingResult:
    """Tracking output for one frame."""

    frame_index: int
    tracks: list[Track] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate tracking result."""
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")

    def __len__(self) -> int:
        """Return number of tracks."""
        return len(self.tracks)

    def get_track(self, track_id: int) -> Track | None:
        """Return a track by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    @property
    def confirmed_tracks(self) -> list[Track]:
        """Return tracks seen more than once."""
        return [track for track in self.tracks if track.is_confirmed]

    @property
    def active_track_ids(self) -> list[int]:
        """Return active track IDs."""
        return [track.track_id for track in self.tracks]
