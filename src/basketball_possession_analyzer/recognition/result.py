
"""Jersey number recognition result model."""

from dataclasses import dataclass, field
from typing import Any

from basketball_possession_analyzer.detection import BoundingBox


@dataclass(frozen=True)
class NumberRecognitionResult:
    """Recognized jersey number for a detected number region."""

    frame_index: int
    bbox: BoundingBox
    number: str
    confidence: float
    track_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate recognition result."""
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if not self.number.strip():
            raise ValueError("number must not be empty")
        if not self.number.isdigit():
            raise ValueError("number must contain only digits")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.track_id is not None and self.track_id < 0:
            raise ValueError("track_id must be >= 0")

    @property
    def number_int(self) -> int:
        """Return number as an integer."""
        return int(self.number)
