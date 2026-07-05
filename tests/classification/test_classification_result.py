
import pytest

from basketball_possession_analyzer.classification import (
    Team,
    TeamClassificationResult,
    TeamSide,
)


def test_team_classification_result_properties() -> None:
    result = TeamClassificationResult(
        track_id=3,
        team=Team(TeamSide.HOME),
        confidence=0.8,
        metadata={"source": "color"},
    )

    assert result.track_id == 3
    assert result.team.side == TeamSide.HOME
    assert result.confidence == 0.8
    assert result.metadata["source"] == "color"
    assert result.is_unknown is False


def test_team_classification_result_unknown() -> None:
    result = TeamClassificationResult(
        track_id=3,
        team=Team(TeamSide.UNKNOWN),
        confidence=0.0,
    )

    assert result.is_unknown is True


def test_team_classification_result_rejects_invalid_track_id() -> None:
    with pytest.raises(ValueError):
        TeamClassificationResult(
            track_id=-1,
            team=Team(TeamSide.HOME),
            confidence=0.8,
        )


def test_team_classification_result_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        TeamClassificationResult(
            track_id=1,
            team=Team(TeamSide.HOME),
            confidence=1.1,
        )

    with pytest.raises(ValueError):
        TeamClassificationResult(
            track_id=1,
            team=Team(TeamSide.HOME),
            confidence=-0.1,
        )
