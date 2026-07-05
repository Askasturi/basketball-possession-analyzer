
"""Future embedding-based team classifier stub."""

import numpy as np

from basketball_possession_analyzer.classification.base import (
    BaseTeamClassifier,
    TeamClassifierConfig,
)
from basketball_possession_analyzer.classification.result import (
    TeamClassificationResult,
)
from basketball_possession_analyzer.tracking import Track


class EmbeddingClassifierNotAvailableError(RuntimeError):
    """Raised when embedding-based classifier backend is unavailable."""


class EmbeddingTeamClassifier(BaseTeamClassifier):
    """Future SigLIP/UMAP/K-means team classifier stub."""

    def __init__(self, config: TeamClassifierConfig | None = None) -> None:
        super().__init__(config=config or TeamClassifierConfig())

    def classify(
        self,
        frame: np.ndarray,
        track: Track,
    ) -> TeamClassificationResult:
        """Classify a track using a future embedding backend."""
        raise EmbeddingClassifierNotAvailableError(
            "EmbeddingTeamClassifier is a stub. Implement a SigLIP/UMAP/K-means "
            "backend before using this classifier."
        )
