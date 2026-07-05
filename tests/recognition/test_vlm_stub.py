
import numpy as np
import pytest

from basketball_possession_analyzer.detection import DetectionResult
from basketball_possession_analyzer.recognition import (
    VLMNumberRecognizer,
    VLMRecognizerNotAvailableError,
)


def test_vlm_number_recognizer_stub_raises() -> None:
    recognizer = VLMNumberRecognizer()
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    with pytest.raises(VLMRecognizerNotAvailableError):
        recognizer.recognize(
            frame=frame,
            detection_result=DetectionResult(frame_index=0),
        )
