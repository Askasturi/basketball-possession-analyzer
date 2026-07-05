
"""Bounding box data model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box using xyxy coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        """Validate bounding box coordinates."""
        if self.x2 <= self.x1:
            raise ValueError("x2 must be greater than x1")
        if self.y2 <= self.y1:
            raise ValueError("y2 must be greater than y1")

    @property
    def width(self) -> float:
        """Return box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Return box height."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Return box area."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Return box center as x, y."""
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def foot_point(self) -> tuple[float, float]:
        """Return bottom-center point for court projection."""
        return (self.x1 + self.x2) / 2, self.y2

    def intersection_area(self, other: "BoundingBox") -> float:
        """Return intersection area with another box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        return (x2 - x1) * (y2 - y1)

    def union_area(self, other: "BoundingBox") -> float:
        """Return union area with another box."""
        return self.area + other.area - self.intersection_area(other)

    def iou(self, other: "BoundingBox") -> float:
        """Return intersection over union."""
        union = self.union_area(other)
        if union <= 0:
            return 0.0
        return self.intersection_area(other) / union

    def ios(self, other: "BoundingBox") -> float:
        """Return intersection over self area."""
        if self.area <= 0:
            return 0.0
        return self.intersection_area(other) / self.area

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return box as xyxy tuple."""
        return self.x1, self.y1, self.x2, self.y2

    @classmethod
    def from_xywh(
        cls,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> "BoundingBox":
        """Create a bounding box from xywh coordinates."""
        if width <= 0:
            raise ValueError("width must be greater than 0")
        if height <= 0:
            raise ValueError("height must be greater than 0")
        return cls(x1=x, y1=y, x2=x + width, y2=y + height)
