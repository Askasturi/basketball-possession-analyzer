
"""Video metadata model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata for a video clip."""

    path: str
    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        """Return video duration in seconds."""
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps

    @property
    def frame_size(self) -> tuple[int, int]:
        """Return frame size as width, height."""
        return self.width, self.height
