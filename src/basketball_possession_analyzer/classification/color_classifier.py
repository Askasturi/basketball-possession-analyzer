
"""Simple color-based team classifier."""

from dataclasses import dataclass

import numpy as np

from basketball_possession_analyzer.classification.base import (
    BaseTeamClassifier,
    TeamClassifierConfig,
)
from basketball_possession_analyzer.classification.result import (
    TeamClassificationResult,
)
from basketball_possession_analyzer.classification.team import Team, TeamSide
from basketball_possession_analyzer.detection import BoundingBox
from basketball_possession_analyzer.tracking import Track


@dataclass(frozen=True)
class ColorTeamClassifierConfig(TeamClassifierConfig):
    """Configuration for the simple color classifier."""

    home_bgr: tuple[int, int, int] = (255, 255, 255)
    away_bgr: tuple[int, int, int] = (0, 0, 0)
    unknown_distance_margin: float = 10.0


class SimpleColorTeamClassifier(BaseTeamClassifier):
    """Classify teams by comparing jersey crop color to reference BGR colors."""

    def __init__(self, config: ColorTeamClassifierConfig | None = None) -> None:
        super().__init__(config=config or ColorTeamClassifierConfig())
        self.config: ColorTeamClassifierConfig

    def classify(
        self,
        frame: np.ndarray,
        track: Track,
    ) -> TeamClassificationResult:
        """Classify a track using mean crop color."""
        crop = self._crop_frame(frame=frame, bbox=track.bbox)

        if crop.size == 0:
            return TeamClassificationResult(
                track_id=track.track_id,
                team=Team(TeamSide.UNKNOWN),
                confidence=0.0,
                metadata={"reason": "empty_crop"},
            )

        mean_bgr = self._mean_bgr(crop)
        home_distance = self._color_distance(mean_bgr, self.config.home_bgr)
        away_distance = self._color_distance(mean_bgr, self.config.away_bgr)

        if abs(home_distance - away_distance) <= self.config.unknown_distance_margin:
            side = TeamSide.UNKNOWN
            confidence = 0.0
        elif home_distance < away_distance:
            side = TeamSide.HOME
            confidence = self._distance_confidence(home_distance, away_distance)
        else:
            side = TeamSide.AWAY
            confidence = self._distance_confidence(away_distance, home_distance)

        if confidence < self.config.min_confidence:
            side = TeamSide.UNKNOWN
            confidence = 0.0

        return TeamClassificationResult(
            track_id=track.track_id,
            team=Team(side),
            confidence=confidence,
            metadata={
                "mean_bgr": mean_bgr,
                "home_distance": home_distance,
                "away_distance": away_distance,
            },
        )

    def _crop_frame(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Crop a bbox from a frame, clipping to frame boundaries."""
        height, width = frame.shape[:2]

        x1 = max(0, int(round(bbox.x1)))
        y1 = max(0, int(round(bbox.y1)))
        x2 = min(width, int(round(bbox.x2)))
        y2 = min(height, int(round(bbox.y2)))

        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)

        return frame[y1:y2, x1:x2]

    def _mean_bgr(self, crop: np.ndarray) -> tuple[float, float, float]:
        """Return mean BGR color."""
        mean_values = crop.reshape(-1, crop.shape[-1]).mean(axis=0)
        return float(mean_values[0]), float(mean_values[1]), float(mean_values[2])

    def _color_distance(
        self,
        color_a: tuple[float, float, float],
        color_b: tuple[int, int, int],
    ) -> float:
        """Return Euclidean BGR distance."""
        diff = np.array(color_a, dtype=np.float32) - np.array(color_b, dtype=np.float32)
        return float(np.linalg.norm(diff))

    def _distance_confidence(
        self,
        winning_distance: float,
        losing_distance: float,
    ) -> float:
        """Convert distance gap into a 0..1 confidence."""
        total = winning_distance + losing_distance
        if total <= 0:
            return 1.0
        return max(0.0, min(1.0, (losing_distance - winning_distance) / total))
