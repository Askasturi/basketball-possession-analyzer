
import pytest

from basketball_possession_analyzer.classification import (
    Team,
    TeamClassificationResult,
    TeamSide,
)
from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
)
from basketball_possession_analyzer.identity import (
    IdentityResolver,
    IdentityResolverConfig,
)
from basketball_possession_analyzer.recognition import NumberTrackMatch
from basketball_possession_analyzer.tracking import Track, TrackingResult


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


def test_identity_resolver_creates_identity_for_each_track() -> None:
    resolver = IdentityResolver()
    track_a = make_track(track_id=1)
    track_b = make_track(track_id=2)

    result = resolver.resolve(
        tracking_result=TrackingResult(
            frame_index=0,
            tracks=[track_a, track_b],
        ),
    )

    assert len(result.identities) == 2
    assert result.get_identity(1).track == track_a
    assert result.get_identity(2).track == track_b


def test_identity_resolver_adds_team_and_number() -> None:
    resolver = IdentityResolver()
    track = make_track(track_id=5)

    result = resolver.resolve(
        tracking_result=TrackingResult(frame_index=0, tracks=[track]),
        team_results=[
            TeamClassificationResult(
                track_id=5,
                team=Team(TeamSide.HOME),
                confidence=0.8,
            )
        ],
        number_matches=[
            NumberTrackMatch(
                track_id=5,
                number="23",
                confidence=0.9,
                overlap_score=1.0,
            )
        ],
    )

    identity = result.get_identity(5)

    assert identity.team.side == TeamSide.HOME
    assert identity.jersey_number == "23"
    assert identity.team_confidence == 0.8
    assert identity.number_confidence == 0.9


def test_identity_resolver_remembers_identity_across_frames() -> None:
    resolver = IdentityResolver()
    track = make_track(track_id=5)

    resolver.resolve(
        tracking_result=TrackingResult(frame_index=0, tracks=[track]),
        team_results=[
            TeamClassificationResult(
                track_id=5,
                team=Team(TeamSide.AWAY),
                confidence=0.8,
            )
        ],
        number_matches=[
            NumberTrackMatch(
                track_id=5,
                number="11",
                confidence=0.9,
                overlap_score=1.0,
            )
        ],
    )

    next_result = resolver.resolve(
        tracking_result=TrackingResult(frame_index=1, tracks=[track]),
    )

    identity = next_result.get_identity(5)

    assert identity.team.side == TeamSide.AWAY
    assert identity.jersey_number == "11"


def test_identity_resolver_can_disable_memory() -> None:
    resolver = IdentityResolver(
        config=IdentityResolverConfig(remember_identities=False),
    )
    track = make_track(track_id=5)

    result = resolver.resolve(
        tracking_result=TrackingResult(frame_index=0, tracks=[track]),
        team_results=[
            TeamClassificationResult(
                track_id=5,
                team=Team(TeamSide.HOME),
                confidence=0.8,
            )
        ],
        number_matches=[
            NumberTrackMatch(
                track_id=5,
                number="23",
                confidence=0.9,
                overlap_score=1.0,
            )
        ],
    )

    identity = result.get_identity(5)

    assert identity.team is None
    assert identity.jersey_number is None


def test_identity_resolver_respects_confidence_thresholds() -> None:
    resolver = IdentityResolver(
        config=IdentityResolverConfig(
            min_team_confidence=0.7,
            min_number_confidence=0.7,
        ),
    )
    track = make_track(track_id=5)

    result = resolver.resolve(
        tracking_result=TrackingResult(frame_index=0, tracks=[track]),
        team_results=[
            TeamClassificationResult(
                track_id=5,
                team=Team(TeamSide.HOME),
                confidence=0.6,
            )
        ],
        number_matches=[
            NumberTrackMatch(
                track_id=5,
                number="23",
                confidence=0.6,
                overlap_score=1.0,
            )
        ],
    )

    identity = result.get_identity(5)

    assert identity.team is None
    assert identity.jersey_number is None


def test_identity_resolver_prefers_higher_confidence_memory() -> None:
    resolver = IdentityResolver()
    track = make_track(track_id=5)

    resolver.resolve(
        tracking_result=TrackingResult(frame_index=0, tracks=[track]),
        number_matches=[
            NumberTrackMatch(
                track_id=5,
                number="11",
                confidence=0.9,
                overlap_score=1.0,
            )
        ],
    )

    result = resolver.resolve(
        tracking_result=TrackingResult(frame_index=1, tracks=[track]),
        number_matches=[
            NumberTrackMatch(
                track_id=5,
                number="12",
                confidence=0.5,
                overlap_score=1.0,
            )
        ],
    )

    identity = result.get_identity(5)

    assert identity.jersey_number == "11"


def test_identity_resolver_reset() -> None:
    resolver = IdentityResolver()
    track = make_track(track_id=5)

    resolver.resolve(
        tracking_result=TrackingResult(frame_index=0, tracks=[track]),
        number_matches=[
            NumberTrackMatch(
                track_id=5,
                number="23",
                confidence=0.9,
                overlap_score=1.0,
            )
        ],
    )
    resolver.reset()

    result = resolver.resolve(
        tracking_result=TrackingResult(frame_index=1, tracks=[track]),
    )

    assert result.get_identity(5).jersey_number is None


def test_identity_resolver_config_validation() -> None:
    with pytest.raises(ValueError):
        IdentityResolverConfig(min_team_confidence=-0.1)

    with pytest.raises(ValueError):
        IdentityResolverConfig(min_number_confidence=1.1)
