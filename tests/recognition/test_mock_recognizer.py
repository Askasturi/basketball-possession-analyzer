
import numpy as np

from basketball_possession_analyzer.detection import BoundingBox, DetectionResult
from basketball_possession_analyzer.recognition import (
    MockNumberRecognizer,
    MockNumberRecognizerConfig,
    NumberRecognitionResult,
)


def make_result(confidence: float = 0.9) -> NumberRecognitionResult:
    return NumberRecognitionResult(
        frame_index=3,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        number="23",
        confidence=confidence,
    )


def test_mock_number_recognizer_returns_predefined_results() -> None:
    result = make_result()
    recognizer = MockNumberRecognizer(
        config=MockNumberRecognizerConfig(results_by_frame={3: [result]})
    )

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    detection_result = DetectionResult(frame_index=3)

    results = recognizer.recognize(frame=frame, detection_result=detection_result)

    assert results == [result]


def test_mock_number_recognizer_returns_empty_for_unknown_frame() -> None:
    recognizer = MockNumberRecognizer()
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    results = recognizer.recognize(
        frame=frame,
        detection_result=DetectionResult(frame_index=99),
    )

    assert results == []


def test_mock_number_recognizer_filters_by_min_confidence() -> None:
    low = make_result(confidence=0.2)
    high = make_result(confidence=0.9)
    recognizer = MockNumberRecognizer(
        config=MockNumberRecognizerConfig(
            min_confidence=0.5,
            results_by_frame={3: [low, high]},
        )
    )

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    results = recognizer.recognize(
        frame=frame,
        detection_result=DetectionResult(frame_index=3),
    )

    assert results == [high]


def test_mock_number_recognizer_uses_factory() -> None:
    def recognition_factory(
        frame: np.ndarray,
        detection_result: DetectionResult,
    ) -> list[NumberRecognitionResult]:
        assert frame.shape == (20, 20, 3)
        assert detection_result.frame_index == 3
        return [make_result()]

    recognizer = MockNumberRecognizer(recognition_factory=recognition_factory)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    results = recognizer.recognize(
        frame=frame,
        detection_result=DetectionResult(frame_index=3),
    )

    assert len(results) == 1
