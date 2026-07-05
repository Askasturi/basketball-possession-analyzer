
import pytest

from basketball_possession_analyzer.court import (
    CourtPoint,
    CourtProjectionResult,
    ImagePoint,
    TrackCourtProjection,
)


def test_track_court_projection_properties() -> None:
    projection = TrackCourtProjection(
        track_id=3,
        image_point=ImagePoint(x=10, y=20),
        court_point=CourtPoint(x=5, y=10),
        metadata={"source": "foot_point"},
    )

    assert projection.track_id == 3
    assert projection.image_point.to_tuple() == (10, 20)
    assert projection.court_point.to_tuple() == (5, 10)
    assert projection.metadata["source"] == "foot_point"


def test_track_court_projection_rejects_invalid_track_id() -> None:
    with pytest.raises(ValueError):
        TrackCourtProjection(
            track_id=-1,
            image_point=ImagePoint(x=10, y=20),
            court_point=CourtPoint(x=5, y=10),
        )


def test_court_projection_result_len_lookup_and_track_ids() -> None:
    projection_a = TrackCourtProjection(
        track_id=1,
        image_point=ImagePoint(x=10, y=20),
        court_point=CourtPoint(x=5, y=10),
    )
    projection_b = TrackCourtProjection(
        track_id=2,
        image_point=ImagePoint(x=30, y=40),
        court_point=CourtPoint(x=15, y=20),
    )

    result = CourtProjectionResult(
        frame_index=7,
        projections=[projection_a, projection_b],
    )

    assert len(result) == 2
    assert result.get_projection(1) == projection_a
    assert result.get_projection(2) == projection_b
    assert result.get_projection(99) is None
    assert result.track_ids == [1, 2]


def test_court_projection_result_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError):
        CourtProjectionResult(frame_index=-1)
