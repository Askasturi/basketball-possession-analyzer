
"""Shot result classifier stubs."""

from dataclasses import dataclass

from basketball_possession_analyzer.detection import DetectionResult
from basketball_possession_analyzer.shots.event import ShotEvent, ShotResult


class ShotClassifierNotAvailableError(RuntimeError):
    """Raised when a heavy shot classifier backend is unavailable."""


@dataclass(frozen=True)
class SimpleShotResultClassifierConfig:
    """Configuration for simple shot result classification."""

    make_confidence: float = 0.5

    def __post_init__(self) -> None:
        """Validate config."""
        if not 0.0 <= self.make_confidence <= 1.0:
            raise ValueError("make_confidence must be between 0.0 and 1.0")


class SimpleShotResultClassifier:
    """Very simple placeholder result classifier."""

    def __init__(
        self,
        config: SimpleShotResultClassifierConfig | None = None,
    ) -> None:
        self.config = config or SimpleShotResultClassifierConfig()

    def classify(
        self,
        shot_event: ShotEvent,
        detection_result: DetectionResult,
    ) -> ShotEvent:
        """Classify a shot event as make/miss/unknown.

        This deterministic placeholder keeps the result unknown until a real
        rim/ball temporal classifier is implemented.
        """
        return ShotEvent(
            frame_index=shot_event.frame_index,
            shooter_track_id=shot_event.shooter_track_id,
            ball_image_point=shot_event.ball_image_point,
            ball_court_point=shot_event.ball_court_point,
            result=ShotResult.UNKNOWN,
            confidence=shot_event.confidence,
            metadata={
                **shot_event.metadata,
                "classifier": "simple_placeholder",
                "detections_seen": len(detection_result),
            },
        )


class VisionShotResultClassifier:
    """Future make/miss classifier stub."""

    def classify(
        self,
        shot_event: ShotEvent,
        detection_result: DetectionResult,
    ) -> ShotEvent:
        """Classify shot result using a future vision model."""
        raise ShotClassifierNotAvailableError(
            "VisionShotResultClassifier is a stub. Implement a shot result "
            "backend before using this classifier."
        )
