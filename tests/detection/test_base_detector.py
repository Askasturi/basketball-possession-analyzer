
import numpy as np
import pytest

from basketball_possession_analyzer.detection import (
    BaseDetector,
    DetectionResult,
    DetectorConfig,
)


class DummyDetector(BaseDetector):
    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
    ) -> DetectionResult:
        return DetectionResult(frame_index=frame_index)


def test_detector_config_defaults() -> None:
    config = DetectorConfig()

    assert config.confidence_threshold == 0.25
    assert config.device == "cpu"


def test_detector_config_rejects_invalid_confidence_threshold() -> None:
    with pytest.raises(ValueError):
        DetectorConfig(confidence_threshold=-0.1)

    with pytest.raises(ValueError):
        DetectorConfig(confidence_threshold=1.1)


def test_detector_config_rejects_empty_device() -> None:
    with pytest.raises(ValueError):
        DetectorConfig(device="")


def test_base_detector_detect_batch() -> None:
    detector = DummyDetector()
    frames = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.uint8),
    ]

    results = detector.detect_batch(frames=frames, start_frame_index=5)

    assert [result.frame_index for result in results] == [5, 6]


def test_base_detector_detect_batch_rejects_negative_start_index() -> None:
    detector = DummyDetector()

    with pytest.raises(ValueError):
        detector.detect_batch(
            frames=[np.zeros((10, 10, 3), dtype=np.uint8)],
            start_frame_index=-1,
        )
