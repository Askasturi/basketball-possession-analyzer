
"""Team classification package."""

from basketball_possession_analyzer.classification.base import (
    BaseTeamClassifier,
    TeamClassifierConfig,
)
from basketball_possession_analyzer.classification.color_classifier import (
    ColorTeamClassifierConfig,
    SimpleColorTeamClassifier,
)
from basketball_possession_analyzer.classification.embedding_classifier import (
    EmbeddingClassifierNotAvailableError,
    EmbeddingTeamClassifier,
)
from basketball_possession_analyzer.classification.result import (
    TeamClassificationResult,
)
from basketball_possession_analyzer.classification.team import Team, TeamSide

__all__ = [
    "BaseTeamClassifier",
    "ColorTeamClassifierConfig",
    "EmbeddingClassifierNotAvailableError",
    "EmbeddingTeamClassifier",
    "SimpleColorTeamClassifier",
    "Team",
    "TeamClassifierConfig",
    "TeamClassificationResult",
    "TeamSide",
]
