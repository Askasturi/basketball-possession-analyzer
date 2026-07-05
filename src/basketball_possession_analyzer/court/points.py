
"""Court and image point data models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImagePoint:
    """A point in image pixel coordinates."""

    x: float
    y: float

    def to_tuple(self) -> tuple[float, float]:
        """Return point as an x, y tuple."""
        return self.x, self.y


@dataclass(frozen=True)
class CourtPoint:
    """A point in court coordinates."""

    x: float
    y: float

    def to_tuple(self) -> tuple[float, float]:
        """Return point as an x, y tuple."""
        return self.x, self.y
