
import pytest

from basketball_possession_analyzer.classification import Team, TeamSide
from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
)
from basketball_possession_analyzer.identity import (
    IdentityResolutionResult,
    PlayerIdentity,
)
from basketball_possession_analyzer.tracking import Track


def make_track(track_id: int) -> Track:
    return Track(
        track_id=track_id,
        detection=Detection(
            label=DetectionLabel.PLAYER,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=20),
            confidence=0.9,
        ),
        first_frame_index=0,
        last_frame_index=0,
    )


def make_identity(
    track_id: int,
    with_team: bool = False,
    with_number: bool = False,
) -> PlayerIdentity:
    return PlayerIdentity(
        track_id=track_id,
        track=make_track(track_id),
        team=Team(TeamSide.HOME) if with_team else None,
        jersey_number="23" if with_number else None,
        team_confidence=0.8 if with_team else 0.0,
        number_confidence=0.9 if with_number else 0.0,
    )


def test_identity_resolution_result_len_and_lookup() -> None:
    identity_a = make_identity(track_id=1)
    identity_b = make_identity(track_id=2)

    result = IdentityResolutionResult(
        frame_index=5,
        identities=[identity_a, identity_b],
    )

    assert len(result) == 2
    assert result.get_identity(1) == identity_a
    assert result.get_identity(2) == identity_b
    assert result.get_identity(99) is None


def test_identity_resolution_result_filtered_properties() -> None:
    unknown = make_identity(track_id=1)
    team_only = make_identity(track_id=2, with_team=True)
    number_only = make_identity(track_id=3, with_number=True)
    full = make_identity(track_id=4, with_team=True, with_number=True)

    result = IdentityResolutionResult(
        frame_index=0,
        identities=[unknown, team_only, number_only, full],
    )

    assert result.team_players == [team_only, full]
    assert result.numbered_players == [number_only, full]
    assert result.identified_players == [full]


def test_identity_resolution_result_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError):
        IdentityResolutionResult(frame_index=-1)
