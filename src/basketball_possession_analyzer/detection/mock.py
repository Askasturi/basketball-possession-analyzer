
"""Mock detector for tests and demos."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from basketball_possession_analyzer.detection.base import BaseDetector, DetectorConfig
from basketball_possession_analyzer.detection.detection import Detection
from basketball_possession_analyzer.detection.result import DetectionResult

DetectionFactory = Callable[[np.ndarray, int], list[Detection]]


@dataclass(frozen=True)
class MockDetectorConfig(DetectorConfig):
    """Configuration for MockDetector."""

    detections_by_frame: dict[int, list[Detection]] = field(default_factory=dict)


class MockDetector(BaseDetector):
    """Detector that returns predefined or generated detections."""

    def __init__(
        self,
        config: MockDetectorConfig | None = None,
        detection_factory: DetectionFactory | None = None,
    ) -> None:
        super().__init__(config=config or MockDetectorConfig())
        self.config: MockDetectorConfig
        self.detection_factory = detection_factory

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
    ) -> DetectionResult:
        """Return mock detections for a frame."""
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")

        if self.detection_factory is not None:
            detections = self.detection_factory(frame, frame_index)
        else:
            detections = self.config.detections_by_frame.get(frame_index, [])

        filtered_detections = [
            detection
            for detection in detections
            if detection.confidence >= self.config.confidence_threshold
        ]

        return DetectionResult(
            frame_index=frame_index,
            detections=filtered_detections,
        )
