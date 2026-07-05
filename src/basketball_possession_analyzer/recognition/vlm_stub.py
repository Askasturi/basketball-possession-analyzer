
"""Future VLM-based jersey number recognizer stub."""

import numpy as np

from basketball_possession_analyzer.detection import DetectionResult
from basketball_possession_analyzer.recognition.base import (
    BaseNumberRecognizer,
    NumberRecognizerConfig,
)
from basketball_possession_analyzer.recognition.result import NumberRecognitionResult


class VLMRecognizerNotAvailableError(RuntimeError):
    """Raised when a VLM recognizer backend is unavailable."""


class VLMNumberRecognizer(BaseNumberRecognizer):
    """Future SmolVLM-style jersey number recognizer stub."""

    def __init__(self, config: NumberRecognizerConfig | None = None) -> None:
        super().__init__(config=config or NumberRecognizerConfig())

    def recognize(
        self,
        frame: np.ndarray,
        detection_result: DetectionResult,
    ) -> list[NumberRecognitionResult]:
        """Recognize jersey numbers using a future VLM backend."""
        raise VLMRecognizerNotAvailableError(
            "VLMNumberRecognizer is a stub. Implement a VLM backend before using "
            "this recognizer."
        )
