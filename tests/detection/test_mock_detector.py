
import numpy as np
import pytest

from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
    MockDetector,
    MockDetectorConfig,
)


def make_detection(confidence: float = 0.9) -> Detection:
    return Detection(
        label=DetectionLabel.PLAYER,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=20),
        confidence=confidence,
    )


def test_mock_detector_returns_predefined_detections() -> None:
    detection = make_detection()
    detector = MockDetector(
        config=MockDetectorConfig(
            detections_by_frame={
                3: [detection],
            }
        )
    )

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detector.detect(frame=frame, frame_index=3)

    assert result.frame_index == 3
    assert result.detections == [detection]


def test_mock_detector_returns_empty_result_for_unknown_frame() -> None:
    detector = MockDetector()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detector.detect(frame=frame, frame_index=99)

    assert result.frame_index == 99
    assert result.detections == []


def test_mock_detector_filters_by_confidence_threshold() -> None:
    low_confidence = make_detection(confidence=0.2)
    high_confidence = make_detection(confidence=0.9)

    detector = MockDetector(
        config=MockDetectorConfig(
            confidence_threshold=0.5,
            detections_by_frame={
                0: [low_confidence, high_confidence],
            },
        )
    )

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detector.detect(frame=frame)

    assert result.detections == [high_confidence]


def test_mock_detector_uses_detection_factory() -> None:
    def detection_factory(
        frame: np.ndarray,
        frame_index: int,
    ) -> list[Detection]:
        assert frame.shape == (100, 100, 3)
        assert frame_index == 7
        return [make_detection()]

    detector = MockDetector(detection_factory=detection_factory)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    result = detector.detect(frame=frame, frame_index=7)

    assert len(result.detections) == 1


def test_mock_detector_rejects_negative_frame_index() -> None:
    detector = MockDetector()

    with pytest.raises(ValueError):
        detector.detect(
            frame=np.zeros((100, 100, 3), dtype=np.uint8),
            frame_index=-1,
        )
