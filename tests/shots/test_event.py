
import pytest

from basketball_possession_analyzer.court import CourtPoint, ImagePoint
from basketball_possession_analyzer.shots import ShotEvent, ShotResult


def test_shot_result_values() -> None:
    assert ShotResult.MAKE == "make"
    assert ShotResult.MISS == "miss"
    assert ShotResult.UNKNOWN == "unknown"


def test_shot_event_properties() -> None:
    event = ShotEvent(
        frame_index=10,
        shooter_track_id=3,
        ball_image_point=ImagePoint(x=20, y=30),
        ball_court_point=CourtPoint(x=5, y=6),
        result=ShotResult.MAKE,
        confidence=0.8,
        metadata={"source": "test"},
    )

    assert event.frame_index == 10
    assert event.shooter_track_id == 3
    assert event.is_made is True
    assert event.is_missed is False
    assert event.metadata["source"] == "test"


def test_shot_event_miss_property() -> None:
    event = ShotEvent(frame_index=0, result=ShotResult.MISS)

    assert event.is_made is False
    assert event.is_missed is True


def test_shot_event_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ShotEvent(frame_index=-1)

    with pytest.raises(ValueError):
        ShotEvent(frame_index=0, shooter_track_id=-1)

    with pytest.raises(ValueError):
        ShotEvent(frame_index=0, confidence=1.1)
