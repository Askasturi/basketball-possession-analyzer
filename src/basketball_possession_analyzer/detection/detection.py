
"""Detection data model."""

from dataclasses import dataclass, field
from typing import Any

from basketball_possession_analyzer.detection.bbox import BoundingBox
from basketball_possession_analyzer.detection.labels import DetectionLabel


@dataclass(frozen=True)
class Detection:
    """Single detected object."""

    label: DetectionLabel
    bbox: BoundingBox
    confidence: float
    class_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate detection fields."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        if self.class_id is not None and self.class_id < 0:
            raise ValueError("class_id must be >= 0")

    @property
    def is_player(self) -> bool:
        """Return whether this detection is a player."""
        return self.label == DetectionLabel.PLAYER

    @property
    def is_ball(self) -> bool:
        """Return whether this detection is a ball."""
        return self.label == DetectionLabel.BALL

    @property
    def is_jersey_number(self) -> bool:
        """Return whether this detection is a jersey number."""
        return self.label == DetectionLabel.JERSEY_NUMBER
