
"""Detection package."""

from basketball_possession_analyzer.detection.adapters import (
    ModelAdapterConfig,
    OptionalModelNotAvailableError,
    RFDETRDetectorAdapter,
    YOLODetectorAdapter,
)
from basketball_possession_analyzer.detection.base import BaseDetector, DetectorConfig
from basketball_possession_analyzer.detection.bbox import BoundingBox
from basketball_possession_analyzer.detection.detection import Detection
from basketball_possession_analyzer.detection.labels import DetectionLabel
from basketball_possession_analyzer.detection.mock import (
    MockDetector,
    MockDetectorConfig,
)
from basketball_possession_analyzer.detection.result import DetectionResult

__all__ = [
    "BaseDetector",
    "BoundingBox",
    "Detection",
    "DetectionLabel",
    "DetectionResult",
    "DetectorConfig",
    "MockDetector",
    "MockDetectorConfig",
    "ModelAdapterConfig",
    "OptionalModelNotAvailableError",
    "RFDETRDetectorAdapter",
    "YOLODetectorAdapter",
]
