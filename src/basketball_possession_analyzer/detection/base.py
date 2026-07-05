
"""Detector interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from basketball_possession_analyzer.detection.result import DetectionResult


@dataclass(frozen=True)
class DetectorConfig:
    """Base detector configuration."""

    confidence_threshold: float = 0.25
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate detector configuration."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        if not self.device.strip():
            raise ValueError("device must not be empty")


class BaseDetector(ABC):
    """Abstract interface for all detectors."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
    ) -> DetectionResult:
        """Detect objects in a single frame."""

    def detect_batch(
        self,
        frames: list[np.ndarray],
        start_frame_index: int = 0,
    ) -> list[DetectionResult]:
        """Detect objects in a batch of frames."""
        if start_frame_index < 0:
            raise ValueError("start_frame_index must be >= 0")

        return [
            self.detect(frame=frame, frame_index=start_frame_index + index)
            for index, frame in enumerate(frames)
        ]
