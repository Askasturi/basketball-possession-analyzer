
"""End-to-end frame analysis result."""

from dataclasses import dataclass, field

import numpy as np

from basketball_possession_analyzer.classification import TeamClassificationResult
from basketball_possession_analyzer.court import CourtProjectionResult
from basketball_possession_analyzer.detection import DetectionResult
from basketball_possession_analyzer.identity import IdentityResolutionResult
from basketball_possession_analyzer.recognition import (
    NumberRecognitionResult,
    NumberTrackMatch,
)
from basketball_possession_analyzer.shots import ShotEvent
from basketball_possession_analyzer.tracking import TrackingResult


@dataclass(frozen=True)
class FrameAnalysisResult:
    """All analysis outputs for one video frame."""

    frame_index: int
    detection_result: DetectionResult
    tracking_result: TrackingResult
    team_results: list[TeamClassificationResult] = field(default_factory=list)
    number_results: list[NumberRecognitionResult] = field(default_factory=list)
    number_matches: list[NumberTrackMatch] = field(default_factory=list)
    identity_result: IdentityResolutionResult | None = None
    court_projection_result: CourtProjectionResult | None = None
    shot_event: ShotEvent | None = None
    rendered_frame: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate frame analysis result."""
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if self.detection_result.frame_index != self.frame_index:
            raise ValueError("detection_result.frame_index must match frame_index")
        if self.tracking_result.frame_index != self.frame_index:
            raise ValueError("tracking_result.frame_index must match frame_index")
        if (
            self.identity_result is not None
            and self.identity_result.frame_index != self.frame_index
        ):
            raise ValueError("identity_result.frame_index must match frame_index")
        if (
            self.court_projection_result is not None
            and self.court_projection_result.frame_index != self.frame_index
        ):
            raise ValueError(
                "court_projection_result.frame_index must match frame_index"
            )

    @property
    def has_shot(self) -> bool:
        """Return whether a shot event was detected."""
        return self.shot_event is not None

    @property
    def detected_count(self) -> int:
        """Return number of detections."""
        return len(self.detection_result)

    @property
    def track_count(self) -> int:
        """Return number of tracks."""
        return len(self.tracking_result)
