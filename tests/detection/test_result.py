
import pytest

from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
    DetectionResult,
)


def make_detection(label: DetectionLabel) -> Detection:
    return Detection(
        label=label,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        confidence=0.9,
    )


def test_detection_result_len_and_filtering() -> None:
    player = make_detection(DetectionLabel.PLAYER)
    ball = make_detection(DetectionLabel.BALL)
    rim = make_detection(DetectionLabel.RIM)

    result = DetectionResult(
        frame_index=12,
        detections=[player, ball, rim],
    )

    assert len(result) == 3
    assert result.filter_by_label(DetectionLabel.PLAYER) == [player]
    assert result.filter_by_label(DetectionLabel.BALL) == [ball]
    assert result.filter_by_label(DetectionLabel.RIM) == [rim]


def test_detection_result_convenience_properties() -> None:
    player = make_detection(DetectionLabel.PLAYER)
    ball = make_detection(DetectionLabel.BALL)
    rim = make_detection(DetectionLabel.RIM)
    jersey_number = make_detection(DetectionLabel.JERSEY_NUMBER)
    referee = make_detection(DetectionLabel.REFEREE)
    hoop = make_detection(DetectionLabel.HOOP)

    result = DetectionResult(
        frame_index=0,
        detections=[player, ball, rim, jersey_number, referee, hoop],
    )

    assert result.players == [player]
    assert result.balls == [ball]
    assert result.rims == [rim]
    assert result.jersey_numbers == [jersey_number]
    assert result.referees == [referee]
    assert result.hoops == [hoop]


def test_detection_result_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError):
        DetectionResult(frame_index=-1)
