
import numpy as np
import pytest

from basketball_possession_analyzer.classification import (
    BaseTeamClassifier,
    Team,
    TeamClassificationResult,
    TeamClassifierConfig,
    TeamSide,
)
from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
)
from basketball_possession_analyzer.tracking import Track


def make_track(track_id: int = 0) -> Track:
    return Track(
        track_id=track_id,
        detection=Detection(
            label=DetectionLabel.PLAYER,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            confidence=0.9,
        ),
        first_frame_index=0,
        last_frame_index=0,
    )


class DummyTeamClassifier(BaseTeamClassifier):
    def classify(
        self,
        frame: np.ndarray,
        track: Track,
    ) -> TeamClassificationResult:
        return TeamClassificationResult(
            track_id=track.track_id,
            team=Team(TeamSide.HOME),
            confidence=1.0,
        )


def test_team_classifier_config_defaults() -> None:
    config = TeamClassifierConfig()

    assert config.min_confidence == 0.0


def test_team_classifier_config_rejects_invalid_min_confidence() -> None:
    with pytest.raises(ValueError):
        TeamClassifierConfig(min_confidence=-0.1)

    with pytest.raises(ValueError):
        TeamClassifierConfig(min_confidence=1.1)


def test_base_team_classifier_classify_tracks() -> None:
    classifier = DummyTeamClassifier()
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    tracks = [make_track(track_id=1), make_track(track_id=2)]

    results = classifier.classify_tracks(frame=frame, tracks=tracks)

    assert [result.track_id for result in results] == [1, 2]
    assert all(result.team.side == TeamSide.HOME for result in results)
