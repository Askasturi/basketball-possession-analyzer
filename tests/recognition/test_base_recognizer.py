
import numpy as np
import pytest

from basketball_possession_analyzer.detection import DetectionResult
from basketball_possession_analyzer.recognition import (
    BaseNumberRecognizer,
    NumberRecognitionResult,
    NumberRecognizerConfig,
)


class DummyNumberRecognizer(BaseNumberRecognizer):
    def recognize(
        self,
        frame: np.ndarray,
        detection_result: DetectionResult,
    ) -> list[NumberRecognitionResult]:
        return []


def test_number_recognizer_config_defaults() -> None:
    config = NumberRecognizerConfig()

    assert config.min_confidence == 0.0


def test_number_recognizer_config_rejects_invalid_min_confidence() -> None:
    with pytest.raises(ValueError):
        NumberRecognizerConfig(min_confidence=-0.1)

    with pytest.raises(ValueError):
        NumberRecognizerConfig(min_confidence=1.1)


def test_base_number_recognizer_uses_default_config() -> None:
    recognizer = DummyNumberRecognizer()

    assert recognizer.config.min_confidence == 0.0
