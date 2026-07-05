"""Identity resolution result model."""

from dataclasses import dataclass, field

from basketball_possession_analyzer.identity.player_identity import PlayerIdentity


@dataclass(frozen=True)
class IdentityResolutionResult:
    """Resolved identities for one frame."""

    frame_index: int
    identities: list[PlayerIdentity] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate identity result."""
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")

    def __len__(self) -> int:
        """Return number of identities."""
        return len(self.identities)

    def get_identity(self, track_id: int) -> PlayerIdentity | None:
        """Return identity by track ID."""
        for identity in self.identities:
            if identity.track_id == track_id:
                return identity
        return None

    @property
    def identified_players(self) -> list[PlayerIdentity]:
        """Return identities with both team and number."""
        return [
            identity
            for identity in self.identities
            if identity.has_team and identity.has_number
        ]

    @property
    def numbered_players(self) -> list[PlayerIdentity]:
        """Return identities with jersey numbers."""
        return [identity for identity in self.identities if identity.has_number]

    @property
    def team_players(self) -> list[PlayerIdentity]:
        """Return identities with teams."""
        return [identity for identity in self.identities if identity.has_team]
