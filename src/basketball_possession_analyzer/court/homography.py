
"""Homography-based court coordinate transformation."""

from dataclasses import dataclass

import cv2
import numpy as np

from basketball_possession_analyzer.court.points import CourtPoint, ImagePoint
from basketball_possession_analyzer.court.projection import (
    CourtProjectionResult,
    TrackCourtProjection,
)
from basketball_possession_analyzer.tracking import TrackingResult


@dataclass(frozen=True)
class HomographyConfig:
    """Homography transformer configuration."""

    min_points: int = 4

    def __post_init__(self) -> None:
        """Validate config."""
        if self.min_points < 4:
            raise ValueError("min_points must be >= 4")


class HomographyTransformer:
    """Map image points to top-down court coordinates."""

    def __init__(
        self,
        image_points: list[ImagePoint],
        court_points: list[CourtPoint],
        config: HomographyConfig | None = None,
    ) -> None:
        self.config = config or HomographyConfig()
        self._validate_points(image_points=image_points, court_points=court_points)
        self.image_points = image_points
        self.court_points = court_points
        self.matrix = self._compute_matrix(
            image_points=image_points,
            court_points=court_points,
        )

    def transform_point(self, point: ImagePoint) -> CourtPoint:
        """Transform one image point into a court point."""
        source = np.array([[[point.x, point.y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(source, self.matrix)
        x, y = transformed[0][0]
        return CourtPoint(x=float(x), y=float(y))

    def project_tracking_result(
        self,
        tracking_result: TrackingResult,
    ) -> CourtProjectionResult:
        """Project track foot-points to court coordinates."""
        projections: list[TrackCourtProjection] = []

        for track in tracking_result.tracks:
            foot_x, foot_y = track.bbox.foot_point
            image_point = ImagePoint(x=foot_x, y=foot_y)
            court_point = self.transform_point(image_point)

            projections.append(
                TrackCourtProjection(
                    track_id=track.track_id,
                    image_point=image_point,
                    court_point=court_point,
                    metadata={
                        "source": "foot_point",
                    },
                )
            )

        return CourtProjectionResult(
            frame_index=tracking_result.frame_index,
            projections=projections,
        )

    def _validate_points(
        self,
        image_points: list[ImagePoint],
        court_points: list[CourtPoint],
    ) -> None:
        """Validate point correspondences."""
        if len(image_points) != len(court_points):
            raise ValueError("image_points and court_points must have the same length")
        if len(image_points) < self.config.min_points:
            raise ValueError(f"at least {self.config.min_points} points are required")

    def _compute_matrix(
        self,
        image_points: list[ImagePoint],
        court_points: list[CourtPoint],
    ) -> np.ndarray:
        """Compute homography matrix."""
        source = np.array(
            [point.to_tuple() for point in image_points],
            dtype=np.float32,
        )
        destination = np.array(
            [point.to_tuple() for point in court_points],
            dtype=np.float32,
        )

        matrix, _mask = cv2.findHomography(source, destination)

        if matrix is None:
            raise ValueError("could not compute homography matrix")

        return matrix
