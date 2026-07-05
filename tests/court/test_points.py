
from basketball_possession_analyzer.court import CourtPoint, ImagePoint


def test_image_point_to_tuple() -> None:
    point = ImagePoint(x=10.5, y=20.25)

    assert point.to_tuple() == (10.5, 20.25)


def test_court_point_to_tuple() -> None:
    point = CourtPoint(x=5.0, y=12.0)

    assert point.to_tuple() == (5.0, 12.0)
