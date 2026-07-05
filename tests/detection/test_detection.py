
import pytest

from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
)


def test_detection_properties() -> None:
    detection = Detection(
        label=DetectionLabel.PLAYER,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=20),
        confidence=0.9,
        class_id=0,
        metadata={"track_hint": "starter"},
    )

    assert detection.is_player is True
    assert detection.is_ball is False
    assert detection.is_jersey_number is False
    assert detection.metadata["track_hint"] == "starter"


def test_ball_detection_property() -> None:
    detection = Detection(
        label=DetectionLabel.BALL,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        confidence=0.8,
    )

    assert detection.is_ball is True


def test_jersey_number_detection_property() -> None:
    detection = Detection(
        label=DetectionLabel.JERSEY_NUMBER,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        confidence=0.8,
    )

    assert detection.is_jersey_number is True


def test_detection_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        Detection(
            label=DetectionLabel.PLAYER,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=20),
            confidence=1.1,
        )

    with pytest.raises(ValueError):
        Detection(
            label=DetectionLabel.PLAYER,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=20),
            confidence=-0.1,
        )


def test_detection_rejects_negative_class_id() -> None:
    with pytest.raises(ValueError):
        Detection(
            label=DetectionLabel.PLAYER,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=20),
            confidence=0.9,
            class_id=-1,
        )
