
"""Resolve stable player identities from tracks, teams, and jersey numbers."""

from dataclasses import dataclass

from basketball_possession_analyzer.classification import (
    Team,
    TeamClassificationResult,
)
from basketball_possession_analyzer.identity.player_identity import PlayerIdentity
from basketball_possession_analyzer.identity.result import IdentityResolutionResult
from basketball_possession_analyzer.recognition import NumberTrackMatch
from basketball_possession_analyzer.tracking import TrackingResult


@dataclass(frozen=True)
class IdentityResolverConfig:
    """Configuration for identity resolution."""

    min_team_confidence: float = 0.0
    min_number_confidence: float = 0.0
    remember_identities: bool = True

    def __post_init__(self) -> None:
        """Validate resolver configuration."""
        if not 0.0 <= self.min_team_confidence <= 1.0:
            raise ValueError("min_team_confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.min_number_confidence <= 1.0:
            raise ValueError("min_number_confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class RememberedTeam:
    """Remembered team assignment."""

    team: Team
    confidence: float


@dataclass(frozen=True)
class RememberedNumber:
    """Remembered jersey number assignment."""

    number: str
    confidence: float


class IdentityResolver:
    """Resolve and remember player identities across frames."""

    def __init__(self, config: IdentityResolverConfig | None = None) -> None:
        self.config = config or IdentityResolverConfig()
        self._teams_by_track_id: dict[int, RememberedTeam] = {}
        self._numbers_by_track_id: dict[int, RememberedNumber] = {}

    def reset(self) -> None:
        """Clear remembered identities."""
        self._teams_by_track_id.clear()
        self._numbers_by_track_id.clear()

    def resolve(
        self,
        tracking_result: TrackingResult,
        team_results: list[TeamClassificationResult] | None = None,
        number_matches: list[NumberTrackMatch] | None = None,
    ) -> IdentityResolutionResult:
        """Resolve identities for one frame."""
        team_results = team_results or []
        number_matches = number_matches or []

        self._update_team_memory(team_results)
        self._update_number_memory(number_matches)

        identities = [
            self._build_identity(track=track)
            for track in tracking_result.tracks
        ]

        return IdentityResolutionResult(
            frame_index=tracking_result.frame_index,
            identities=identities,
        )

    def _update_team_memory(
        self,
        team_results: list[TeamClassificationResult],
    ) -> None:
        """Update remembered team assignments."""
        for result in team_results:
            if result.confidence < self.config.min_team_confidence:
                continue

            previous = self._teams_by_track_id.get(result.track_id)
            if previous is None or result.confidence >= previous.confidence:
                self._teams_by_track_id[result.track_id] = RememberedTeam(
                    team=result.team,
                    confidence=result.confidence,
                )

    def _update_number_memory(
        self,
        number_matches: list[NumberTrackMatch],
    ) -> None:
        """Update remembered jersey number assignments."""
        for match in number_matches:
            if match.confidence < self.config.min_number_confidence:
                continue

            previous = self._numbers_by_track_id.get(match.track_id)
            if previous is None or match.confidence >= previous.confidence:
                self._numbers_by_track_id[match.track_id] = RememberedNumber(
                    number=match.number,
                    confidence=match.confidence,
                )

    def _build_identity(self, track: object) -> PlayerIdentity:
        """Build identity for one track."""
        track_id = track.track_id
        remembered_team = self._teams_by_track_id.get(track_id)
        remembered_number = self._numbers_by_track_id.get(track_id)

        if not self.config.remember_identities:
            remembered_team = None
            remembered_number = None

        return PlayerIdentity(
            track_id=track_id,
            track=track,
            team=remembered_team.team if remembered_team else None,
            jersey_number=remembered_number.number if remembered_number else None,
            team_confidence=remembered_team.confidence if remembered_team else 0.0,
            number_confidence=(
                remembered_number.confidence if remembered_number else 0.0
            ),
        )
