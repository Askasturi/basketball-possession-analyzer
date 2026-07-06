
import pytest

from basketball_possession_analyzer.detection import DetectionResult
from basketball_possession_analyzer.shots import (
    ShotClassifierNotAvailableError,
    ShotEvent,
    ShotResult,
    SimpleShotResultClassifier,
    SimpleShotResultClassifierConfig,
    VisionShotResultClassifier,
)


def test_simple_shot_result_classifier_keeps_unknown_result() -> None:
    classifier = SimpleShotResultClassifier()
    shot_event = ShotEvent(frame_index=2, confidence=0.7)

    result = classifier.classify(
        shot_event=shot_event,
        detection_result=DetectionResult(frame_index=2),
    )

    assert result.result == ShotResult.UNKNOWN
    assert result.confidence == 0.7
    assert result.metadata["classifier"] == "simple_placeholder"


def test_simple_shot_result_classifier_config_validation() -> None:
    with pytest.raises(ValueError):
        SimpleShotResultClassifierConfig(make_confidence=-0.1)

    with pytest.raises(ValueError):
        SimpleShotResultClassifierConfig(make_confidence=1.1)


def test_vision_shot_result_classifier_stub_raises() -> None:
    classifier = VisionShotResultClassifier()

    with pytest.raises(ShotClassifierNotAvailableError):
        classifier.classify(
            shot_event=ShotEvent(frame_index=0),
            detection_result=DetectionResult(frame_index=0),
        )
