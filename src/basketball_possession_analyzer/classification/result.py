
"""Team classification result model."""

from dataclasses import dataclass, field
from typing import Any

from basketball_possession_analyzer.classification.team import Team, TeamSide


@dataclass(frozen=True)
class TeamClassificationResult:
    """Team classification result for one track."""

    track_id: int
    team: Team
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate classification result."""
        if self.track_id < 0:
            raise ValueError("track_id must be >= 0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

    @property
    def is_unknown(self) -> bool:
        """Return whether team is unknown."""
        return self.team.side == TeamSide.UNKNOWN
