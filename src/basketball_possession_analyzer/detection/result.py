
"""Detection result data model."""

from dataclasses import dataclass, field

from basketball_possession_analyzer.detection.detection import Detection
from basketball_possession_analyzer.detection.labels import DetectionLabel


@dataclass(frozen=True)
class DetectionResult:
    """Detections for one video frame."""

    frame_index: int
    detections: list[Detection] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate detection result."""
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")

    def __len__(self) -> int:
        """Return number of detections."""
        return len(self.detections)

    def filter_by_label(self, label: DetectionLabel) -> list[Detection]:
        """Return detections matching a label."""
        return [detection for detection in self.detections if detection.label == label]

    @property
    def players(self) -> list[Detection]:
        """Return player detections."""
        return self.filter_by_label(DetectionLabel.PLAYER)

    @property
    def balls(self) -> list[Detection]:
        """Return ball detections."""
        return self.filter_by_label(DetectionLabel.BALL)

    @property
    def rims(self) -> list[Detection]:
        """Return rim detections."""
        return self.filter_by_label(DetectionLabel.RIM)

    @property
    def jersey_numbers(self) -> list[Detection]:
        """Return jersey number detections."""
        return self.filter_by_label(DetectionLabel.JERSEY_NUMBER)

    @property
    def referees(self) -> list[Detection]:
        """Return referee detections."""
        return self.filter_by_label(DetectionLabel.REFEREE)

    @property
    def hoops(self) -> list[Detection]:
        """Return hoop detections."""
        return self.filter_by_label(DetectionLabel.HOOP)
