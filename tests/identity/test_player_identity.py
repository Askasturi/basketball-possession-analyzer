
import pytest

from basketball_possession_analyzer.classification import Team, TeamSide
from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
)
from basketball_possession_analyzer.identity import PlayerIdentity
from basketball_possession_analyzer.tracking import Track


def make_track(track_id: int = 1) -> Track:
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


def test_player_identity_properties() -> None:
    track = make_track(track_id=7)
    identity = PlayerIdentity(
        track_id=7,
        track=track,
        team=Team(TeamSide.HOME),
        jersey_number="23",
        team_confidence=0.8,
        number_confidence=0.9,
        metadata={"source": "resolver"},
    )

    assert identity.track_id == 7
    assert identity.has_team is True
    assert identity.has_number is True
    assert identity.display_label == "ID 7 | home | #23"
    assert identity.metadata["source"] == "resolver"


def test_player_identity_without_team_or_number() -> None:
    track = make_track(track_id=7)
    identity = PlayerIdentity(track_id=7, track=track)

    assert identity.has_team is False
    assert identity.has_number is False
    assert identity.display_label == "ID 7"


def test_player_identity_rejects_mismatched_track_id() -> None:
    with pytest.raises(ValueError):
        PlayerIdentity(track_id=1, track=make_track(track_id=2))


def test_player_identity_rejects_invalid_jersey_number() -> None:
    track = make_track(track_id=1)

    with pytest.raises(ValueError):
        PlayerIdentity(track_id=1, track=track, jersey_number="")

    with pytest.raises(ValueError):
        PlayerIdentity(track_id=1, track=track, jersey_number="2A")


def test_player_identity_rejects_invalid_confidence() -> None:
    track = make_track(track_id=1)

    with pytest.raises(ValueError):
        PlayerIdentity(track_id=1, track=track, team_confidence=1.1)

    with pytest.raises(ValueError):
        PlayerIdentity(track_id=1, track=track, number_confidence=-0.1)
