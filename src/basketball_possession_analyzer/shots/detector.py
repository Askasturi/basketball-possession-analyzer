
"""Simple deterministic shot detector."""

from dataclasses import dataclass

from basketball_possession_analyzer.court import CourtPoint, ImagePoint
from basketball_possession_analyzer.detection import DetectionLabel, DetectionResult
from basketball_possession_analyzer.shots.event import ShotEvent, ShotResult
from basketball_possession_analyzer.tracking import TrackingResult


@dataclass(frozen=True)
class SimpleShotDetectorConfig:
    """Configuration for SimpleShotDetector."""

    rim_distance_threshold: float = 60.0
    shooter_distance_threshold: float = 120.0
    min_ball_confidence: float = 0.0
    min_rim_confidence: float = 0.0

    def __post_init__(self) -> None:
        """Validate config."""
        if self.rim_distance_threshold <= 0:
            raise ValueError("rim_distance_threshold must be > 0")
        if self.shooter_distance_threshold <= 0:
            raise ValueError("shooter_distance_threshold must be > 0")
        if not 0.0 <= self.min_ball_confidence <= 1.0:
            raise ValueError("min_ball_confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.min_rim_confidence <= 1.0:
            raise ValueError("min_rim_confidence must be between 0.0 and 1.0")


class SimpleShotDetector:
    """Detect likely shot events from ball/rim proximity."""

    def __init__(self, config: SimpleShotDetectorConfig | None = None) -> None:
        self.config = config or SimpleShotDetectorConfig()

    def detect(
        self,
        detection_result: DetectionResult,
        tracking_result: TrackingResult | None = None,
    ) -> ShotEvent | None:
        """Detect a shot event in one frame."""
        ball = self._best_detection(
            detection_result=detection_result,
            label=DetectionLabel.BALL,
            min_confidence=self.config.min_ball_confidence,
        )
        rim = self._best_detection(
            detection_result=detection_result,
            label=DetectionLabel.RIM,
            min_confidence=self.config.min_rim_confidence,
        )

        if ball is None or rim is None:
            return None

        ball_center = ImagePoint(*ball.bbox.center)
        rim_center = ImagePoint(*rim.bbox.center)
        rim_distance = self._distance(ball_center, rim_center)

        if rim_distance > self.config.rim_distance_threshold:
            return None

        shooter_track_id = None
        shooter_distance = None

        if tracking_result is not None:
            shooter_track_id, shooter_distance = self._nearest_player_track(
                point=ball_center,
                tracking_result=tracking_result,
            )

        return ShotEvent(
            frame_index=detection_result.frame_index,
            shooter_track_id=shooter_track_id,
            ball_image_point=ball_center,
            ball_court_point=CourtPoint(x=ball_center.x, y=ball_center.y),
            result=ShotResult.UNKNOWN,
            confidence=max(ball.confidence, rim.confidence),
            metadata={
                "rim_distance": rim_distance,
                "shooter_distance": shooter_distance,
            },
        )

    def _best_detection(
        self,
        detection_result: DetectionResult,
        label: DetectionLabel,
        min_confidence: float,
    ):
        """Return highest-confidence detection for a label."""
        candidates = [
            detection
            for detection in detection_result.detections
            if detection.label == label and detection.confidence >= min_confidence
        ]

        if not candidates:
            return None

        return max(candidates, key=lambda detection: detection.confidence)

    def _nearest_player_track(
        self,
        point: ImagePoint,
        tracking_result: TrackingResult,
    ) -> tuple[int | None, float | None]:
        """Return nearest player track to point."""
        best_track_id = None
        best_distance = None

        for track in tracking_result.tracks:
            if track.label != DetectionLabel.PLAYER:
                continue

            track_point = ImagePoint(*track.bbox.center)
            distance = self._distance(point, track_point)

            if distance > self.config.shooter_distance_threshold:
                continue

            if best_distance is None or distance < best_distance:
                best_track_id = track.track_id
                best_distance = distance

        return best_track_id, best_distance

    def _distance(self, point_a: ImagePoint, point_b: ImagePoint) -> float:
        """Return Euclidean distance."""
        return ((point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2) ** 0.5
