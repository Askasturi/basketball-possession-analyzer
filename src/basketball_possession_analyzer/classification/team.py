
"""Team data models."""

from dataclasses import dataclass
from enum import StrEnum


class TeamSide(StrEnum):
    """Simple team-side labels."""

    HOME = "home"
    AWAY = "away"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Team:
    """Basketball team identity."""

    side: TeamSide
    name: str | None = None

    @property
    def display_name(self) -> str:
        """Return display name for overlays."""
        if self.name:
            return self.name
        return self.side.value
