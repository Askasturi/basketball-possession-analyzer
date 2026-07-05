
"""Player identity model."""

from dataclasses import dataclass, field
from typing import Any

from basketball_possession_analyzer.classification import Team
from basketball_possession_analyzer.tracking import Track


@dataclass(frozen=True)
class PlayerIdentity:
    """Resolved player identity for one tracked player."""

    track_id: int
    track: Track
    team: Team | None = None
    jersey_number: str | None = None
    team_confidence: float = 0.0
    number_confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate player identity."""
        if self.track_id < 0:
            raise ValueError("track_id must be >= 0")
        if self.track.track_id != self.track_id:
            raise ValueError("track.track_id must match track_id")
        if self.jersey_number is not None:
            if not self.jersey_number.strip():
                raise ValueError("jersey_number must not be empty")
            if not self.jersey_number.isdigit():
                raise ValueError("jersey_number must contain only digits")
        if not 0.0 <= self.team_confidence <= 1.0:
            raise ValueError("team_confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.number_confidence <= 1.0:
            raise ValueError("number_confidence must be between 0.0 and 1.0")

    @property
    def has_team(self) -> bool:
        """Return whether identity has a team."""
        return self.team is not None

    @property
    def has_number(self) -> bool:
        """Return whether identity has a jersey number."""
        return self.jersey_number is not None

    @property
    def display_label(self) -> str:
        """Return compact label for overlays."""
        parts: list[str] = [f"ID {self.track_id}"]

        if self.team is not None:
            parts.append(self.team.display_name)

        if self.jersey_number is not None:
            parts.append(f"#{self.jersey_number}")

        return " | ".join(parts)
