
"""Jersey number recognizer interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from basketball_possession_analyzer.detection import DetectionResult
from basketball_possession_analyzer.recognition.result import NumberRecognitionResult


@dataclass(frozen=True)
class NumberRecognizerConfig:
    """Base number recognizer configuration."""

    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        """Validate recognizer configuration."""
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")


class BaseNumberRecognizer(ABC):
    """Abstract interface for jersey number recognizers."""

    def __init__(self, config: NumberRecognizerConfig | None = None) -> None:
        self.config = config or NumberRecognizerConfig()

    @abstractmethod
    def recognize(
        self,
        frame: np.ndarray,
        detection_result: DetectionResult,
    ) -> list[NumberRecognitionResult]:
        """Recognize jersey numbers in one frame."""
