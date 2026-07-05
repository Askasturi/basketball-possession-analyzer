
"""Mock jersey number recognizer for tests and demos."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from basketball_possession_analyzer.detection import DetectionResult
from basketball_possession_analyzer.recognition.base import (
    BaseNumberRecognizer,
    NumberRecognizerConfig,
)
from basketball_possession_analyzer.recognition.result import NumberRecognitionResult

RecognitionFactory = Callable[
    [np.ndarray, DetectionResult],
    list[NumberRecognitionResult],
]


@dataclass(frozen=True)
class MockNumberRecognizerConfig(NumberRecognizerConfig):
    """Configuration for MockNumberRecognizer."""

    results_by_frame: dict[int, list[NumberRecognitionResult]] = field(
        default_factory=dict
    )


class MockNumberRecognizer(BaseNumberRecognizer):
    """Recognizer that returns predefined or generated number results."""

    def __init__(
        self,
        config: MockNumberRecognizerConfig | None = None,
        recognition_factory: RecognitionFactory | None = None,
    ) -> None:
        super().__init__(config=config or MockNumberRecognizerConfig())
        self.config: MockNumberRecognizerConfig
        self.recognition_factory = recognition_factory

    def recognize(
        self,
        frame: np.ndarray,
        detection_result: DetectionResult,
    ) -> list[NumberRecognitionResult]:
        """Return mock recognition results."""
        if self.recognition_factory is not None:
            results = self.recognition_factory(frame, detection_result)
        else:
            results = self.config.results_by_frame.get(
                detection_result.frame_index,
                [],
            )

        return [
            result
            for result in results
            if result.confidence >= self.config.min_confidence
        ]
