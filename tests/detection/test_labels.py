
import pytest

from basketball_possession_analyzer.detection import DetectionLabel


def test_detection_labels_have_expected_values() -> None:
    assert DetectionLabel.PLAYER == "player"
    assert DetectionLabel.BALL == "ball"
    assert DetectionLabel.RIM == "rim"
    assert DetectionLabel.JERSEY_NUMBER == "jersey_number"
    assert DetectionLabel.REFEREE == "referee"
    assert DetectionLabel.HOOP == "hoop"


def test_detection_label_from_string() -> None:
    assert DetectionLabel.from_string("player") == DetectionLabel.PLAYER
    assert DetectionLabel.from_string(" BALL ") == DetectionLabel.BALL


def test_detection_label_from_unknown_string_raises() -> None:
    with pytest.raises(ValueError):
        DetectionLabel.from_string("coach")
