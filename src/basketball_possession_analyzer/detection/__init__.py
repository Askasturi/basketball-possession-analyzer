
"""Detection data models."""

from basketball_possession_analyzer.detection.bbox import BoundingBox
from basketball_possession_analyzer.detection.detection import Detection
from basketball_possession_analyzer.detection.labels import DetectionLabel
from basketball_possession_analyzer.detection.result import DetectionResult

__all__ = [
    "BoundingBox",
    "Detection",
    "DetectionLabel",
    "DetectionResult",
]
