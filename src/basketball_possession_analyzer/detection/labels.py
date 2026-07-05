
"""Detection class labels."""

from enum import StrEnum


class DetectionLabel(StrEnum):
    """Supported basketball object detection labels."""

    PLAYER = "player"
    BALL = "ball"
    RIM = "rim"
    JERSEY_NUMBER = "jersey_number"
    REFEREE = "referee"
    HOOP = "hoop"

    @classmethod
    def from_string(cls, value: str) -> "DetectionLabel":
        """Create a label from a string."""
        normalized = value.strip().lower()
        for label in cls:
            if label.value == normalized:
                return label
        raise ValueError(f"Unknown detection label: {value}")
