
"""Team classifier interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from basketball_possession_analyzer.classification.result import (
    TeamClassificationResult,
)
from basketball_possession_analyzer.tracking import Track


@dataclass(frozen=True)
class TeamClassifierConfig:
    """Base team classifier configuration."""

    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        """Validate classifier configuration."""
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")


class BaseTeamClassifier(ABC):
    """Abstract interface for team classifiers."""

    def __init__(self, config: TeamClassifierConfig | None = None) -> None:
        self.config = config or TeamClassifierConfig()

    @abstractmethod
    def classify(
        self,
        frame: np.ndarray,
        track: Track,
    ) -> TeamClassificationResult:
        """Classify one track into a team."""

    def classify_tracks(
        self,
        frame: np.ndarray,
        tracks: list[Track],
    ) -> list[TeamClassificationResult]:
        """Classify many tracks."""
        return [self.classify(frame=frame, track=track) for track in tracks]
