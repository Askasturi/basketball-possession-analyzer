
"""Frame renderer for basketball possession analysis."""

from dataclasses import dataclass

import cv2
import numpy as np

from basketball_possession_analyzer.court import CourtProjectionResult
from basketball_possession_analyzer.detection import Detection, DetectionResult
from basketball_possession_analyzer.identity import IdentityResolutionResult
from basketball_possession_analyzer.shots import ShotEvent
from basketball_possession_analyzer.tracking import Track, TrackingResult

Color = tuple[int, int, int]


@dataclass(frozen=True)
class RendererConfig:
    """Configuration for frame rendering."""

    box_thickness: int = 2
    point_radius: int = 4
    text_scale: float = 0.5
    text_thickness: int = 1
    detection_color: Color = (0, 255, 0)
    track_color: Color = (255, 0, 0)
    identity_color: Color = (0, 255, 255)
    court_projection_color: Color = (255, 255, 0)
    shot_color: Color = (0, 0, 255)
    text_color: Color = (255, 255, 255)

    def __post_init__(self) -> None:
        """Validate renderer configuration."""
        if self.box_thickness <= 0:
            raise ValueError("box_thickness must be > 0")
        if self.point_radius <= 0:
            raise ValueError("point_radius must be > 0")
        if self.text_scale <= 0:
            raise ValueError("text_scale must be > 0")
        if self.text_thickness <= 0:
            raise ValueError("text_thickness must be > 0")


class BasketballFrameRenderer:
    """Render analysis overlays on video frames."""

    def __init__(self, config: RendererConfig | None = None) -> None:
        self.config = config or RendererConfig()

    def render(
        self,
        frame: np.ndarray,
        detection_result: DetectionResult | None = None,
        tracking_result: TrackingResult | None = None,
        identity_result: IdentityResolutionResult | None = None,
        court_projection_result: CourtProjectionResult | None = None,
        shot_event: ShotEvent | None = None,
    ) -> np.ndarray:
        """Render all available annotations onto a copy of the frame."""
        output = frame.copy()

        if detection_result is not None:
            self._draw_detections(output, detection_result)

        if tracking_result is not None:
            self._draw_tracks(output, tracking_result)

        if identity_result is not None:
            self._draw_identities(output, identity_result)

        if court_projection_result is not None:
            self._draw_court_projections(output, court_projection_result)

        if shot_event is not None:
            self._draw_shot_event(output, shot_event)

        return output

    def _draw_detections(
        self,
        frame: np.ndarray,
        detection_result: DetectionResult,
    ) -> None:
        """Draw detection boxes."""
        for detection in detection_result.detections:
            self._draw_detection(
                frame=frame,
                detection=detection,
                label=f"{detection.label.value} {detection.confidence:.2f}",
                color=self.config.detection_color,
            )

    def _draw_tracks(
        self,
        frame: np.ndarray,
        tracking_result: TrackingResult,
    ) -> None:
        """Draw track boxes."""
        for track in tracking_result.tracks:
            self._draw_track(frame=frame, track=track)

    def _draw_identities(
        self,
        frame: np.ndarray,
        identity_result: IdentityResolutionResult,
    ) -> None:
        """Draw resolved player identity labels."""
        for identity in identity_result.identities:
            self._draw_bbox(
                frame=frame,
                bbox=identity.track.bbox.to_xyxy(),
                color=self.config.identity_color,
            )
            self._draw_text(
                frame=frame,
                text=identity.display_label,
                x=identity.track.bbox.x1,
                y=identity.track.bbox.y1 - 20,
                color=self.config.identity_color,
            )

    def _draw_court_projections(
        self,
        frame: np.ndarray,
        court_projection_result: CourtProjectionResult,
    ) -> None:
        """Draw projected track foot-points."""
        for projection in court_projection_result.projections:
            x = int(round(projection.image_point.x))
            y = int(round(projection.image_point.y))
            cv2.circle(
                frame,
                (x, y),
                self.config.point_radius,
                self.config.court_projection_color,
                -1,
            )
            self._draw_text(
                frame=frame,
                text=f"court {projection.court_point.x:.1f},"
                f"{projection.court_point.y:.1f}",
                x=x + 5,
                y=y - 5,
                color=self.config.court_projection_color,
            )

    def _draw_shot_event(self, frame: np.ndarray, shot_event: ShotEvent) -> None:
        """Draw shot event marker."""
        if shot_event.ball_image_point is None:
            return

        x = int(round(shot_event.ball_image_point.x))
        y = int(round(shot_event.ball_image_point.y))

        cv2.circle(
            frame,
            (x, y),
            self.config.point_radius + 3,
            self.config.shot_color,
            self.config.box_thickness,
        )
        self._draw_text(
            frame=frame,
            text=f"SHOT {shot_event.result.value}",
            x=x + 8,
            y=y - 8,
            color=self.config.shot_color,
        )

    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: Detection,
        label: str,
        color: Color,
    ) -> None:
        """Draw one detection."""
        self._draw_bbox(frame=frame, bbox=detection.bbox.to_xyxy(), color=color)
        self._draw_text(
            frame=frame,
            text=label,
            x=detection.bbox.x1,
            y=detection.bbox.y1 - 5,
            color=color,
        )

    def _draw_track(self, frame: np.ndarray, track: Track) -> None:
        """Draw one track."""
        self._draw_bbox(
            frame=frame,
            bbox=track.bbox.to_xyxy(),
            color=self.config.track_color,
        )
        self._draw_text(
            frame=frame,
            text=f"track {track.track_id}",
            x=track.bbox.x1,
            y=track.bbox.y2 + 15,
            color=self.config.track_color,
        )

    def _draw_bbox(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        color: Color,
    ) -> None:
        """Draw one bounding box."""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(
            frame,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            self.config.box_thickness,
        )

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        x: float,
        y: float,
        color: Color,
    ) -> None:
        """Draw text with safe image coordinates."""
        safe_x = max(0, int(round(x)))
        safe_y = max(15, int(round(y)))

        cv2.putText(
            frame,
            text,
            (safe_x, safe_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.text_scale,
            color,
            self.config.text_thickness,
            cv2.LINE_AA,
        )
