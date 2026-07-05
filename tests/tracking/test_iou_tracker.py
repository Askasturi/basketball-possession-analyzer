
import pytest

from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
    DetectionResult,
)
from basketball_possession_analyzer.tracking import (
    SimpleIoUTracker,
    SimpleIoUTrackerConfig,
)


def make_detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: DetectionLabel = DetectionLabel.PLAYER,
) -> Detection:
    return Detection(
        label=label,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=0.9,
    )


def test_simple_iou_tracker_creates_new_track() -> None:
    tracker = SimpleIoUTracker()
    detection = make_detection(0, 0, 10, 20)

    result = tracker.update(
        DetectionResult(frame_index=0, detections=[detection]),
    )

    assert len(result.tracks) == 1
    assert result.tracks[0].track_id == 0
    assert result.tracks[0].detection == detection


def test_simple_iou_tracker_matches_same_object_across_frames() -> None:
    tracker = SimpleIoUTracker(
        config=SimpleIoUTrackerConfig(iou_threshold=0.2),
    )

    first = make_detection(0, 0, 10, 20)
    second = make_detection(1, 1, 11, 21)

    first_result = tracker.update(
        DetectionResult(frame_index=0, detections=[first]),
    )
    second_result = tracker.update(
        DetectionResult(frame_index=1, detections=[second]),
    )

    assert first_result.tracks[0].track_id == 0
    assert second_result.tracks[0].track_id == 0
    assert second_result.tracks[0].hits == 2
    assert second_result.tracks[0].detection == second


def test_simple_iou_tracker_creates_new_track_when_iou_is_low() -> None:
    tracker = SimpleIoUTracker(
        config=SimpleIoUTrackerConfig(iou_threshold=0.5),
    )

    first = make_detection(0, 0, 10, 20)
    second = make_detection(100, 100, 110, 120)

    tracker.update(DetectionResult(frame_index=0, detections=[first]))
    result = tracker.update(DetectionResult(frame_index=1, detections=[second]))

    assert sorted(track.track_id for track in result.tracks) == [0, 1]


def test_simple_iou_tracker_removes_track_after_too_many_misses() -> None:
    tracker = SimpleIoUTracker(
        config=SimpleIoUTrackerConfig(max_missed_frames=1),
    )

    tracker.update(
        DetectionResult(
            frame_index=0,
            detections=[make_detection(0, 0, 10, 20)],
        )
    )
    missed_once = tracker.update(DetectionResult(frame_index=1, detections=[]))
    missed_twice = tracker.update(DetectionResult(frame_index=2, detections=[]))

    assert len(missed_once.tracks) == 1
    assert len(missed_twice.tracks) == 0


def test_simple_iou_tracker_ignores_untracked_labels_by_default() -> None:
    tracker = SimpleIoUTracker()
    ball = make_detection(0, 0, 10, 10, label=DetectionLabel.BALL)

    result = tracker.update(DetectionResult(frame_index=0, detections=[ball]))

    assert result.tracks == []


def test_simple_iou_tracker_can_track_custom_labels() -> None:
    tracker = SimpleIoUTracker(
        config=SimpleIoUTrackerConfig(track_labels=(DetectionLabel.BALL,)),
    )
    ball = make_detection(0, 0, 10, 10, label=DetectionLabel.BALL)

    result = tracker.update(DetectionResult(frame_index=0, detections=[ball]))

    assert len(result.tracks) == 1
    assert result.tracks[0].label == DetectionLabel.BALL


def test_simple_iou_tracker_reset() -> None:
    tracker = SimpleIoUTracker()

    tracker.update(
        DetectionResult(
            frame_index=0,
            detections=[make_detection(0, 0, 10, 20)],
        )
    )
    tracker.reset()

    assert tracker.tracks == []

    result = tracker.update(
        DetectionResult(
            frame_index=1,
            detections=[make_detection(0, 0, 10, 20)],
        )
    )

    assert result.tracks[0].track_id == 0


def test_simple_iou_tracker_config_validation() -> None:
    with pytest.raises(ValueError):
        SimpleIoUTrackerConfig(iou_threshold=-0.1)

    with pytest.raises(ValueError):
        SimpleIoUTrackerConfig(iou_threshold=1.1)

    with pytest.raises(ValueError):
        SimpleIoUTrackerConfig(max_missed_frames=-1)

    with pytest.raises(ValueError):
        SimpleIoUTrackerConfig(track_labels=())
