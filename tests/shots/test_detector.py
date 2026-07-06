
import pytest

from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
    DetectionResult,
)
from basketball_possession_analyzer.shots import (
    ShotResult,
    SimpleShotDetector,
    SimpleShotDetectorConfig,
)
from basketball_possession_analyzer.tracking import Track, TrackingResult


def make_detection(
    label: DetectionLabel,
    bbox: BoundingBox,
    confidence: float = 0.9,
) -> Detection:
    return Detection(label=label, bbox=bbox, confidence=confidence)


def make_track(track_id: int, bbox: BoundingBox) -> Track:
    return Track(
        track_id=track_id,
        detection=make_detection(DetectionLabel.PLAYER, bbox),
        first_frame_index=0,
        last_frame_index=0,
    )


def test_simple_shot_detector_detects_ball_near_rim() -> None:
    detector = SimpleShotDetector(
        config=SimpleShotDetectorConfig(rim_distance_threshold=50),
    )
    detection_result = DetectionResult(
        frame_index=4,
        detections=[
            make_detection(
                DetectionLabel.BALL,
                BoundingBox(x1=90, y1=90, x2=100, y2=100),
            ),
            make_detection(
                DetectionLabel.RIM,
                BoundingBox(x1=100, y1=100, x2=120, y2=120),
            ),
        ],
    )

    event = detector.detect(detection_result=detection_result)

    assert event is not None
    assert event.frame_index == 4
    assert event.result == ShotResult.UNKNOWN
    assert event.ball_image_point.to_tuple() == (95.0, 95.0)
    assert event.metadata["rim_distance"] < 50


def test_simple_shot_detector_returns_none_without_ball_or_rim() -> None:
    detector = SimpleShotDetector()

    assert detector.detect(DetectionResult(frame_index=0)) is None

    only_ball = DetectionResult(
        frame_index=0,
        detections=[
            make_detection(
                DetectionLabel.BALL,
                BoundingBox(x1=0, y1=0, x2=10, y2=10),
            )
        ],
    )

    assert detector.detect(only_ball) is None


def test_simple_shot_detector_returns_none_when_ball_far_from_rim() -> None:
    detector = SimpleShotDetector(
        config=SimpleShotDetectorConfig(rim_distance_threshold=20),
    )
    detection_result = DetectionResult(
        frame_index=0,
        detections=[
            make_detection(
                DetectionLabel.BALL,
                BoundingBox(x1=0, y1=0, x2=10, y2=10),
            ),
            make_detection(
                DetectionLabel.RIM,
                BoundingBox(x1=100, y1=100, x2=120, y2=120),
            ),
        ],
    )

    assert detector.detect(detection_result) is None


def test_simple_shot_detector_assigns_nearest_shooter() -> None:
    detector = SimpleShotDetector(
        config=SimpleShotDetectorConfig(
            rim_distance_threshold=50,
            shooter_distance_threshold=150,
        ),
    )
    detection_result = DetectionResult(
        frame_index=4,
        detections=[
            make_detection(
                DetectionLabel.BALL,
                BoundingBox(x1=90, y1=90, x2=100, y2=100),
            ),
            make_detection(
                DetectionLabel.RIM,
                BoundingBox(x1=100, y1=100, x2=120, y2=120),
            ),
        ],
    )
    tracking_result = TrackingResult(
        frame_index=4,
        tracks=[
            make_track(1, BoundingBox(x1=80, y1=80, x2=100, y2=120)),
            make_track(2, BoundingBox(x1=300, y1=300, x2=330, y2=360)),
        ],
    )

    event = detector.detect(
        detection_result=detection_result,
        tracking_result=tracking_result,
    )

    assert event.shooter_track_id == 1
    assert event.metadata["shooter_distance"] is not None


def test_simple_shot_detector_config_validation() -> None:
    with pytest.raises(ValueError):
        SimpleShotDetectorConfig(rim_distance_threshold=0)

    with pytest.raises(ValueError):
        SimpleShotDetectorConfig(shooter_distance_threshold=0)

    with pytest.raises(ValueError):
        SimpleShotDetectorConfig(min_ball_confidence=-0.1)

    with pytest.raises(ValueError):
        SimpleShotDetectorConfig(min_rim_confidence=1.1)
