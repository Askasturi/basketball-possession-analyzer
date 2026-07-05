
import pytest

from basketball_possession_analyzer.court import (
    CourtPoint,
    HomographyConfig,
    HomographyTransformer,
    ImagePoint,
)
from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
)
from basketball_possession_analyzer.tracking import Track, TrackingResult


def make_transformer() -> HomographyTransformer:
    return HomographyTransformer(
        image_points=[
            ImagePoint(x=0, y=0),
            ImagePoint(x=100, y=0),
            ImagePoint(x=100, y=100),
            ImagePoint(x=0, y=100),
        ],
        court_points=[
            CourtPoint(x=0, y=0),
            CourtPoint(x=50, y=0),
            CourtPoint(x=50, y=94),
            CourtPoint(x=0, y=94),
        ],
    )


def make_track(track_id: int = 1) -> Track:
    return Track(
        track_id=track_id,
        detection=Detection(
            label=DetectionLabel.PLAYER,
            bbox=BoundingBox(x1=40, y1=20, x2=60, y2=80),
            confidence=0.9,
        ),
        first_frame_index=0,
        last_frame_index=0,
    )


def test_homography_config_validation() -> None:
    with pytest.raises(ValueError):
        HomographyConfig(min_points=3)


def test_homography_transformer_rejects_mismatched_points() -> None:
    with pytest.raises(ValueError):
        HomographyTransformer(
            image_points=[
                ImagePoint(x=0, y=0),
                ImagePoint(x=100, y=0),
                ImagePoint(x=100, y=100),
                ImagePoint(x=0, y=100),
            ],
            court_points=[
                CourtPoint(x=0, y=0),
                CourtPoint(x=50, y=0),
                CourtPoint(x=50, y=94),
            ],
        )


def test_homography_transformer_rejects_too_few_points() -> None:
    with pytest.raises(ValueError):
        HomographyTransformer(
            image_points=[
                ImagePoint(x=0, y=0),
                ImagePoint(x=100, y=0),
                ImagePoint(x=100, y=100),
            ],
            court_points=[
                CourtPoint(x=0, y=0),
                CourtPoint(x=50, y=0),
                CourtPoint(x=50, y=94),
            ],
        )


def test_homography_transform_point() -> None:
    transformer = make_transformer()

    point = transformer.transform_point(ImagePoint(x=50, y=50))

    assert point.x == pytest.approx(25.0)
    assert point.y == pytest.approx(47.0)


def test_homography_project_tracking_result() -> None:
    transformer = make_transformer()
    track = make_track(track_id=7)

    result = transformer.project_tracking_result(
        TrackingResult(frame_index=3, tracks=[track]),
    )

    assert result.frame_index == 3
    assert len(result.projections) == 1

    projection = result.projections[0]

    assert projection.track_id == 7
    assert projection.image_point.to_tuple() == (50.0, 80)
    assert projection.court_point.x == pytest.approx(25.0)
    assert projection.court_point.y == pytest.approx(75.2)
    assert projection.metadata["source"] == "foot_point"
