
import numpy as np
import pytest

from basketball_possession_analyzer.classification import Team, TeamSide
from basketball_possession_analyzer.court import (
    CourtPoint,
    CourtProjectionResult,
    ImagePoint,
    TrackCourtProjection,
)
from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
    DetectionResult,
)
from basketball_possession_analyzer.identity import (
    IdentityResolutionResult,
    PlayerIdentity,
)
from basketball_possession_analyzer.shots import ShotEvent, ShotResult
from basketball_possession_analyzer.tracking import Track, TrackingResult
from basketball_possession_analyzer.visualization import (
    BasketballFrameRenderer,
    RendererConfig,
)


def make_frame() -> np.ndarray:
    return np.zeros((120, 160, 3), dtype=np.uint8)


def make_detection(label: DetectionLabel = DetectionLabel.PLAYER) -> Detection:
    return Detection(
        label=label,
        bbox=BoundingBox(x1=20, y1=20, x2=60, y2=90),
        confidence=0.9,
    )


def make_track(track_id: int = 1) -> Track:
    return Track(
        track_id=track_id,
        detection=make_detection(),
        first_frame_index=0,
        last_frame_index=0,
    )


def test_renderer_config_validation() -> None:
    with pytest.raises(ValueError):
        RendererConfig(box_thickness=0)

    with pytest.raises(ValueError):
        RendererConfig(point_radius=0)

    with pytest.raises(ValueError):
        RendererConfig(text_scale=0)

    with pytest.raises(ValueError):
        RendererConfig(text_thickness=0)


def test_renderer_returns_copy_without_mutating_input() -> None:
    frame = make_frame()
    original = frame.copy()
    renderer = BasketballFrameRenderer()

    rendered = renderer.render(frame)

    assert rendered is not frame
    assert np.array_equal(frame, original)


def test_renderer_draws_detections() -> None:
    frame = make_frame()
    renderer = BasketballFrameRenderer()
    detection_result = DetectionResult(
        frame_index=0,
        detections=[make_detection()],
    )

    rendered = renderer.render(frame, detection_result=detection_result)

    assert rendered.sum() > 0


def test_renderer_draws_tracks() -> None:
    frame = make_frame()
    renderer = BasketballFrameRenderer()
    tracking_result = TrackingResult(frame_index=0, tracks=[make_track()])

    rendered = renderer.render(frame, tracking_result=tracking_result)

    assert rendered.sum() > 0


def test_renderer_draws_identities() -> None:
    frame = make_frame()
    renderer = BasketballFrameRenderer()
    track = make_track(track_id=7)
    identity_result = IdentityResolutionResult(
        frame_index=0,
        identities=[
            PlayerIdentity(
                track_id=7,
                track=track,
                team=Team(TeamSide.HOME),
                jersey_number="23",
                team_confidence=0.8,
                number_confidence=0.9,
            )
        ],
    )

    rendered = renderer.render(frame, identity_result=identity_result)

    assert rendered.sum() > 0


def test_renderer_draws_court_projections() -> None:
    frame = make_frame()
    renderer = BasketballFrameRenderer()
    court_result = CourtProjectionResult(
        frame_index=0,
        projections=[
            TrackCourtProjection(
                track_id=1,
                image_point=ImagePoint(x=50, y=80),
                court_point=CourtPoint(x=25, y=47),
            )
        ],
    )

    rendered = renderer.render(frame, court_projection_result=court_result)

    assert rendered.sum() > 0


def test_renderer_draws_shot_event() -> None:
    frame = make_frame()
    renderer = BasketballFrameRenderer()
    shot_event = ShotEvent(
        frame_index=0,
        ball_image_point=ImagePoint(x=80, y=40),
        result=ShotResult.UNKNOWN,
        confidence=0.7,
    )

    rendered = renderer.render(frame, shot_event=shot_event)

    assert rendered.sum() > 0


def test_renderer_ignores_shot_without_ball_point() -> None:
    frame = make_frame()
    renderer = BasketballFrameRenderer()
    shot_event = ShotEvent(frame_index=0)

    rendered = renderer.render(frame, shot_event=shot_event)

    assert np.array_equal(rendered, frame)


def test_renderer_draws_combined_annotations() -> None:
    frame = make_frame()
    renderer = BasketballFrameRenderer()
    track = make_track(track_id=1)

    rendered = renderer.render(
        frame=frame,
        detection_result=DetectionResult(
            frame_index=0,
            detections=[make_detection(DetectionLabel.BALL)],
        ),
        tracking_result=TrackingResult(frame_index=0, tracks=[track]),
        identity_result=IdentityResolutionResult(
            frame_index=0,
            identities=[
                PlayerIdentity(
                    track_id=1,
                    track=track,
                    team=Team(TeamSide.AWAY),
                    jersey_number="11",
                    team_confidence=0.8,
                    number_confidence=0.9,
                )
            ],
        ),
        court_projection_result=CourtProjectionResult(
            frame_index=0,
            projections=[
                TrackCourtProjection(
                    track_id=1,
                    image_point=ImagePoint(x=40, y=90),
                    court_point=CourtPoint(x=20, y=70),
                )
            ],
        ),
        shot_event=ShotEvent(
            frame_index=0,
            ball_image_point=ImagePoint(x=80, y=40),
            result=ShotResult.UNKNOWN,
            confidence=0.7,
        ),
    )

    assert rendered.shape == frame.shape
    assert rendered.sum() > 0
