
import pytest

from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
)
from basketball_possession_analyzer.tracking import Track


def make_detection(label: DetectionLabel = DetectionLabel.PLAYER) -> Detection:
    return Detection(
        label=label,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=20),
        confidence=0.9,
    )


def test_track_properties() -> None:
    detection = make_detection()
    track = Track(
        track_id=1,
        detection=detection,
        first_frame_index=3,
        last_frame_index=3,
    )

    assert track.track_id == 1
    assert track.bbox == detection.bbox
    assert track.label == DetectionLabel.PLAYER
    assert track.confidence == 0.9
    assert track.age == 1
    assert track.is_confirmed is False


def test_track_update() -> None:
    track = Track(
        track_id=1,
        detection=make_detection(),
        first_frame_index=3,
        last_frame_index=3,
    )
    new_detection = Detection(
        label=DetectionLabel.PLAYER,
        bbox=BoundingBox(x1=1, y1=1, x2=11, y2=21),
        confidence=0.8,
    )

    track.update(detection=new_detection, frame_index=4)

    assert track.detection == new_detection
    assert track.last_frame_index == 4
    assert track.hits == 2
    assert track.missed_frames == 0
    assert track.is_confirmed is True
    assert track.age == 2


def test_track_mark_missed_and_active_state() -> None:
    track = Track(
        track_id=1,
        detection=make_detection(),
        first_frame_index=0,
        last_frame_index=0,
    )

    track.mark_missed()
    track.mark_missed()

    assert track.missed_frames == 2
    assert track.is_active(max_missed_frames=2) is True
    assert track.is_active(max_missed_frames=1) is False


def test_track_rejects_invalid_fields() -> None:
    detection = make_detection()

    with pytest.raises(ValueError):
        Track(
            track_id=-1,
            detection=detection,
            first_frame_index=0,
            last_frame_index=0,
        )

    with pytest.raises(ValueError):
        Track(
            track_id=1,
            detection=detection,
            first_frame_index=-1,
            last_frame_index=0,
        )

    with pytest.raises(ValueError):
        Track(
            track_id=1,
            detection=detection,
            first_frame_index=5,
            last_frame_index=4,
        )


def test_track_update_rejects_older_frame() -> None:
    track = Track(
        track_id=1,
        detection=make_detection(),
        first_frame_index=5,
        last_frame_index=5,
    )

    with pytest.raises(ValueError):
        track.update(detection=make_detection(), frame_index=4)


def test_track_is_active_rejects_invalid_max_missed_frames() -> None:
    track = Track(
        track_id=1,
        detection=make_detection(),
        first_frame_index=0,
        last_frame_index=0,
    )

    with pytest.raises(ValueError):
        track.is_active(max_missed_frames=-1)
