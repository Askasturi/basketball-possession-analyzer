
import pytest

from basketball_possession_analyzer.detection import BoundingBox


def test_bounding_box_properties() -> None:
    bbox = BoundingBox(x1=10, y1=20, x2=50, y2=80)

    assert bbox.width == 40
    assert bbox.height == 60
    assert bbox.area == 2400
    assert bbox.center == (30, 50)
    assert bbox.foot_point == (30, 80)
    assert bbox.to_xyxy() == (10, 20, 50, 80)


def test_bounding_box_from_xywh() -> None:
    bbox = BoundingBox.from_xywh(x=10, y=20, width=40, height=60)

    assert bbox == BoundingBox(x1=10, y1=20, x2=50, y2=80)


def test_bounding_box_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError):
        BoundingBox(x1=10, y1=20, x2=10, y2=80)

    with pytest.raises(ValueError):
        BoundingBox(x1=10, y1=20, x2=50, y2=20)


def test_bounding_box_from_xywh_rejects_invalid_size() -> None:
    with pytest.raises(ValueError):
        BoundingBox.from_xywh(x=0, y=0, width=0, height=10)

    with pytest.raises(ValueError):
        BoundingBox.from_xywh(x=0, y=0, width=10, height=0)


def test_bounding_box_intersection_union_iou_and_ios() -> None:
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    b = BoundingBox(x1=5, y1=5, x2=15, y2=15)

    assert a.intersection_area(b) == 25
    assert a.union_area(b) == 175
    assert a.iou(b) == pytest.approx(25 / 175)
    assert a.ios(b) == pytest.approx(25 / 100)


def test_bounding_box_no_overlap() -> None:
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    b = BoundingBox(x1=20, y1=20, x2=30, y2=30)

    assert a.intersection_area(b) == 0
    assert a.iou(b) == 0
    assert a.ios(b) == 0
