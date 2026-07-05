
import pytest

from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
)
from basketball_possession_analyzer.recognition import (
    NumberRecognitionResult,
    NumberTrackMatch,
    NumberTrackMatcher,
    NumberTrackMatcherConfig,
)
from basketball_possession_analyzer.tracking import Track


def make_track(
    track_id: int,
    bbox: BoundingBox,
    label: DetectionLabel = DetectionLabel.PLAYER,
) -> Track:
    return Track(
        track_id=track_id,
        detection=Detection(
            label=label,
            bbox=bbox,
            confidence=0.9,
        ),
        first_frame_index=0,
        last_frame_index=0,
    )


def make_result(
    number: str,
    bbox: BoundingBox,
    confidence: float = 0.9,
) -> NumberRecognitionResult:
    return NumberRecognitionResult(
        frame_index=0,
        bbox=bbox,
        number=number,
        confidence=confidence,
    )


def test_number_track_match_properties() -> None:
    match = NumberTrackMatch(
        track_id=1,
        number="23",
        confidence=0.8,
        overlap_score=0.7,
    )

    assert match.track_id == 1
    assert match.number == "23"
    assert match.confidence == 0.8
    assert match.overlap_score == 0.7


def test_number_track_match_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        NumberTrackMatch(
            track_id=-1,
            number="23",
            confidence=0.8,
            overlap_score=0.7,
        )

    with pytest.raises(ValueError):
        NumberTrackMatch(
            track_id=1,
            number="A",
            confidence=0.8,
            overlap_score=0.7,
        )

    with pytest.raises(ValueError):
        NumberTrackMatch(
            track_id=1,
            number="23",
            confidence=1.1,
            overlap_score=0.7,
        )

    with pytest.raises(ValueError):
        NumberTrackMatch(
            track_id=1,
            number="23",
            confidence=0.8,
            overlap_score=1.1,
        )


def test_number_track_matcher_matches_number_inside_player() -> None:
    matcher = NumberTrackMatcher(config=NumberTrackMatcherConfig(min_ios=0.5))
    track = make_track(
        track_id=5,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=200),
    )
    result = make_result(
        number="23",
        bbox=BoundingBox(x1=20, y1=20, x2=40, y2=60),
    )

    matches = matcher.match(recognition_results=[result], tracks=[track])

    assert len(matches) == 1
    assert matches[0].track_id == 5
    assert matches[0].number == "23"
    assert matches[0].overlap_score == 1.0


def test_number_track_matcher_ignores_low_overlap() -> None:
    matcher = NumberTrackMatcher(config=NumberTrackMatcherConfig(min_ios=0.5))
    track = make_track(
        track_id=5,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=200),
    )
    result = make_result(
        number="23",
        bbox=BoundingBox(x1=150, y1=150, x2=180, y2=180),
    )

    matches = matcher.match(recognition_results=[result], tracks=[track])

    assert matches == []


def test_number_track_matcher_uses_best_overlap() -> None:
    matcher = NumberTrackMatcher(config=NumberTrackMatcherConfig(min_ios=0.1))
    track_a = make_track(
        track_id=1,
        bbox=BoundingBox(x1=0, y1=0, x2=50, y2=100),
    )
    track_b = make_track(
        track_id=2,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=200),
    )
    result = make_result(
        number="23",
        bbox=BoundingBox(x1=40, y1=40, x2=80, y2=80),
    )

    matches = matcher.match(recognition_results=[result], tracks=[track_a, track_b])

    assert len(matches) == 1
    assert matches[0].track_id == 2


def test_number_track_matcher_assigns_each_track_once() -> None:
    matcher = NumberTrackMatcher(config=NumberTrackMatcherConfig(min_ios=0.1))
    track = make_track(
        track_id=1,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=200),
    )
    first = make_result(
        number="23",
        bbox=BoundingBox(x1=10, y1=10, x2=30, y2=30),
        confidence=0.8,
    )
    second = make_result(
        number="24",
        bbox=BoundingBox(x1=20, y1=20, x2=40, y2=40),
        confidence=0.9,
    )

    matches = matcher.match(recognition_results=[first, second], tracks=[track])

    assert len(matches) == 1
    assert matches[0].number in {"23", "24"}


def test_number_track_matcher_ignores_non_player_tracks() -> None:
    matcher = NumberTrackMatcher()
    ball_track = make_track(
        track_id=1,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        label=DetectionLabel.BALL,
    )
    result = make_result(
        number="23",
        bbox=BoundingBox(x1=10, y1=10, x2=30, y2=30),
    )

    matches = matcher.match(recognition_results=[result], tracks=[ball_track])

    assert matches == []


def test_number_track_matcher_config_validation() -> None:
    with pytest.raises(ValueError):
        NumberTrackMatcherConfig(min_ios=-0.1)

    with pytest.raises(ValueError):
        NumberTrackMatcherConfig(min_ios=1.1)
