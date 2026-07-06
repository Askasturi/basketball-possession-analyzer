
import numpy as np
import pytest

from basketball_possession_analyzer.detection import DetectionResult
from basketball_possession_analyzer.pipeline import FrameAnalysisResult
from basketball_possession_analyzer.tracking import TrackingResult


def test_frame_analysis_result_properties() -> None:
    result = FrameAnalysisResult(
        frame_index=3,
        detection_result=DetectionResult(frame_index=3),
        tracking_result=TrackingResult(frame_index=3),
        rendered_frame=np.zeros((20, 20, 3), dtype=np.uint8),
    )

    assert result.frame_index == 3
    assert result.detected_count == 0
    assert result.track_count == 0
    assert result.has_shot is False
    assert result.rendered_frame is not None


def test_frame_analysis_result_rejects_invalid_frame_index() -> None:
    with pytest.raises(ValueError):
        FrameAnalysisResult(
            frame_index=-1,
            detection_result=DetectionResult(frame_index=0),
            tracking_result=TrackingResult(frame_index=0),
        )


def test_frame_analysis_result_rejects_mismatched_detection_frame() -> None:
    with pytest.raises(ValueError):
        FrameAnalysisResult(
            frame_index=1,
            detection_result=DetectionResult(frame_index=2),
            tracking_result=TrackingResult(frame_index=1),
        )


def test_frame_analysis_result_rejects_mismatched_tracking_frame() -> None:
    with pytest.raises(ValueError):
        FrameAnalysisResult(
            frame_index=1,
            detection_result=DetectionResult(frame_index=1),
            tracking_result=TrackingResult(frame_index=2),
        )
