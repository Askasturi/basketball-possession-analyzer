
import numpy as np

from basketball_possession_analyzer.classification import (
    ColorTeamClassifierConfig,
    SimpleColorTeamClassifier,
    TeamSide,
)
from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
)
from basketball_possession_analyzer.tracking import Track


def make_track(
    track_id: int = 0,
    bbox: BoundingBox | None = None,
) -> Track:
    return Track(
        track_id=track_id,
        detection=Detection(
            label=DetectionLabel.PLAYER,
            bbox=bbox or BoundingBox(x1=0, y1=0, x2=10, y2=10),
            confidence=0.9,
        ),
        first_frame_index=0,
        last_frame_index=0,
    )


def test_simple_color_classifier_classifies_home_color() -> None:
    frame = np.full((20, 20, 3), (255, 255, 255), dtype=np.uint8)
    classifier = SimpleColorTeamClassifier(
        config=ColorTeamClassifierConfig(
            home_bgr=(255, 255, 255),
            away_bgr=(0, 0, 0),
            unknown_distance_margin=0.0,
        )
    )

    result = classifier.classify(frame=frame, track=make_track())

    assert result.team.side == TeamSide.HOME
    assert result.confidence > 0
    assert result.metadata["mean_bgr"] == (255.0, 255.0, 255.0)


def test_simple_color_classifier_classifies_away_color() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    classifier = SimpleColorTeamClassifier(
        config=ColorTeamClassifierConfig(
            home_bgr=(255, 255, 255),
            away_bgr=(0, 0, 0),
            unknown_distance_margin=0.0,
        )
    )

    result = classifier.classify(frame=frame, track=make_track())

    assert result.team.side == TeamSide.AWAY
    assert result.confidence > 0


def test_simple_color_classifier_returns_unknown_for_ambiguous_color() -> None:
    frame = np.full((20, 20, 3), (128, 128, 128), dtype=np.uint8)
    classifier = SimpleColorTeamClassifier(
        config=ColorTeamClassifierConfig(
            home_bgr=(255, 255, 255),
            away_bgr=(0, 0, 0),
            unknown_distance_margin=10.0,
        )
    )

    result = classifier.classify(frame=frame, track=make_track())

    assert result.team.side == TeamSide.UNKNOWN
    assert result.confidence == 0.0


def test_simple_color_classifier_respects_min_confidence() -> None:
    frame = np.full((20, 20, 3), (200, 200, 200), dtype=np.uint8)
    classifier = SimpleColorTeamClassifier(
        config=ColorTeamClassifierConfig(
            home_bgr=(255, 255, 255),
            away_bgr=(0, 0, 0),
            min_confidence=0.95,
            unknown_distance_margin=0.0,
        )
    )

    result = classifier.classify(frame=frame, track=make_track())

    assert result.team.side == TeamSide.UNKNOWN
    assert result.confidence == 0.0


def test_simple_color_classifier_clips_bbox_to_frame() -> None:
    frame = np.full((20, 20, 3), (255, 255, 255), dtype=np.uint8)
    classifier = SimpleColorTeamClassifier()
    track = make_track(bbox=BoundingBox(x1=-5, y1=-5, x2=10, y2=10))

    result = classifier.classify(frame=frame, track=track)

    assert result.team.side == TeamSide.HOME


def test_simple_color_classifier_handles_empty_crop() -> None:
    frame = np.full((20, 20, 3), (255, 255, 255), dtype=np.uint8)
    classifier = SimpleColorTeamClassifier()
    track = make_track(bbox=BoundingBox(x1=30, y1=30, x2=40, y2=40))

    result = classifier.classify(frame=frame, track=track)

    assert result.team.side == TeamSide.UNKNOWN
    assert result.confidence == 0.0
    assert result.metadata["reason"] == "empty_crop"
