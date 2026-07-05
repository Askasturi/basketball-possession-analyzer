
import pytest

from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
)
from basketball_possession_analyzer.tracking import Track, TrackingResult


def make_track(track_id: int, hits: int = 1) -> Track:
    return Track(
        track_id=track_id,
        detection=Detection(
            label=DetectionLabel.PLAYER,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=20),
            confidence=0.9,
        ),
        first_frame_index=0,
        last_frame_index=0,
        hits=hits,
    )


def test_tracking_result_len_and_lookup() -> None:
    track_a = make_track(track_id=1)
    track_b = make_track(track_id=2)

    result = TrackingResult(frame_index=5, tracks=[track_a, track_b])

    assert len(result) == 2
    assert result.get_track(1) == track_a
    assert result.get_track(2) == track_b
    assert result.get_track(99) is None
    assert result.active_track_ids == [1, 2]


def test_tracking_result_confirmed_tracks() -> None:
    unconfirmed = make_track(track_id=1, hits=1)
    confirmed = make_track(track_id=2, hits=2)

    result = TrackingResult(frame_index=5, tracks=[unconfirmed, confirmed])

    assert result.confirmed_tracks == [confirmed]


def test_tracking_result_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError):
        TrackingResult(frame_index=-1)
