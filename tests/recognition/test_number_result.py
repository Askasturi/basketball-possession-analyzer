
import pytest

from basketball_possession_analyzer.detection import BoundingBox
from basketball_possession_analyzer.recognition import NumberRecognitionResult


def test_number_recognition_result_properties() -> None:
    result = NumberRecognitionResult(
        frame_index=5,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        number="23",
        confidence=0.8,
        track_id=7,
        metadata={"source": "mock"},
    )

    assert result.frame_index == 5
    assert result.number == "23"
    assert result.number_int == 23
    assert result.confidence == 0.8
    assert result.track_id == 7
    assert result.metadata["source"] == "mock"


def test_number_recognition_result_rejects_invalid_frame_index() -> None:
    with pytest.raises(ValueError):
        NumberRecognitionResult(
            frame_index=-1,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            number="23",
            confidence=0.8,
        )


def test_number_recognition_result_rejects_empty_number() -> None:
    with pytest.raises(ValueError):
        NumberRecognitionResult(
            frame_index=0,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            number="",
            confidence=0.8,
        )


def test_number_recognition_result_rejects_non_digit_number() -> None:
    with pytest.raises(ValueError):
        NumberRecognitionResult(
            frame_index=0,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            number="2A",
            confidence=0.8,
        )


def test_number_recognition_result_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        NumberRecognitionResult(
            frame_index=0,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            number="23",
            confidence=1.1,
        )

    with pytest.raises(ValueError):
        NumberRecognitionResult(
            frame_index=0,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            number="23",
            confidence=-0.1,
        )


def test_number_recognition_result_rejects_invalid_track_id() -> None:
    with pytest.raises(ValueError):
        NumberRecognitionResult(
            frame_index=0,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            number="23",
            confidence=0.8,
            track_id=-1,
        )
