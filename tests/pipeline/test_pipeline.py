
from pathlib import Path

import numpy as np

from basketball_possession_analyzer.classification import (
    ColorTeamClassifierConfig,
    SimpleColorTeamClassifier,
)
from basketball_possession_analyzer.court import (
    CourtPoint,
    HomographyTransformer,
    ImagePoint,
)
from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
    MockDetector,
    MockDetectorConfig,
)
from basketball_possession_analyzer.pipeline import (
    BasketballAnalysisPipeline,
    BasketballAnalysisPipelineConfig,
)
from basketball_possession_analyzer.recognition import (
    MockNumberRecognizer,
    MockNumberRecognizerConfig,
    NumberRecognitionResult,
)
from basketball_possession_analyzer.shots import (
    SimpleShotDetector,
    SimpleShotDetectorConfig,
    SimpleShotResultClassifier,
)
from basketball_possession_analyzer.tracking import SimpleIoUTracker
from basketball_possession_analyzer.video import VideoLoader, VideoWriter


def make_player_detection() -> Detection:
    return Detection(
        label=DetectionLabel.PLAYER,
        bbox=BoundingBox(x1=20, y1=20, x2=60, y2=90),
        confidence=0.95,
    )


def make_number_result(frame_index: int) -> NumberRecognitionResult:
    return NumberRecognitionResult(
        frame_index=frame_index,
        bbox=BoundingBox(x1=30, y1=35, x2=45, y2=50),
        number="23",
        confidence=0.9,
    )


def make_pipeline(render: bool = True) -> BasketballAnalysisPipeline:
    player = make_player_detection()
    number = Detection(
        label=DetectionLabel.JERSEY_NUMBER,
        bbox=BoundingBox(x1=30, y1=35, x2=45, y2=50),
        confidence=0.9,
    )
    ball = Detection(
        label=DetectionLabel.BALL,
        bbox=BoundingBox(x1=90, y1=90, x2=100, y2=100),
        confidence=0.9,
    )
    rim = Detection(
        label=DetectionLabel.RIM,
        bbox=BoundingBox(x1=100, y1=100, x2=120, y2=120),
        confidence=0.9,
    )

    homography = HomographyTransformer(
        image_points=[
            ImagePoint(x=0, y=0),
            ImagePoint(x=160, y=0),
            ImagePoint(x=160, y=120),
            ImagePoint(x=0, y=120),
        ],
        court_points=[
            CourtPoint(x=0, y=0),
            CourtPoint(x=50, y=0),
            CourtPoint(x=50, y=94),
            CourtPoint(x=0, y=94),
        ],
    )

    return BasketballAnalysisPipeline(
        detector=MockDetector(
            config=MockDetectorConfig(
                detections_by_frame={
                    0: [player, number, ball, rim],
                    1: [player, number, ball, rim],
                }
            )
        ),
        tracker=SimpleIoUTracker(),
        team_classifier=SimpleColorTeamClassifier(
            config=ColorTeamClassifierConfig(
                home_bgr=(255, 255, 255),
                away_bgr=(0, 0, 0),
                unknown_distance_margin=0.0,
            )
        ),
        number_recognizer=MockNumberRecognizer(
            config=MockNumberRecognizerConfig(
                results_by_frame={
                    0: [make_number_result(0)],
                    1: [make_number_result(1)],
                }
            )
        ),
        homography_transformer=homography,
        shot_detector=SimpleShotDetector(
            config=SimpleShotDetectorConfig(rim_distance_threshold=50)
        ),
        shot_result_classifier=SimpleShotResultClassifier(),
        config=BasketballAnalysisPipelineConfig(render=render),
    )


def test_pipeline_analyze_frame_runs_all_components() -> None:
    pipeline = make_pipeline()
    frame = np.full((120, 160, 3), 255, dtype=np.uint8)

    result = pipeline.analyze_frame(frame=frame, frame_index=0)

    assert result.frame_index == 0
    assert result.detected_count == 4
    assert result.track_count == 1
    assert len(result.team_results) == 1
    assert len(result.number_results) == 1
    assert len(result.number_matches) == 1
    assert result.identity_result is not None
    assert result.identity_result.get_identity(0).jersey_number == "23"
    assert result.court_projection_result is not None
    assert result.shot_event is not None
    assert result.rendered_frame is not None


def test_pipeline_can_disable_rendering() -> None:
    pipeline = make_pipeline(render=False)
    frame = np.full((120, 160, 3), 255, dtype=np.uint8)

    result = pipeline.analyze_frame(frame=frame, frame_index=0)

    assert result.rendered_frame is None


def test_pipeline_reset_clears_tracker_state() -> None:
    pipeline = make_pipeline(render=False)
    frame = np.full((120, 160, 3), 255, dtype=np.uint8)

    first = pipeline.analyze_frame(frame=frame, frame_index=0)
    pipeline.reset()
    second = pipeline.analyze_frame(frame=frame, frame_index=0)

    assert first.tracking_result.tracks[0].track_id == 0
    assert second.tracking_result.tracks[0].track_id == 0


def test_pipeline_analyze_video_writes_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"

    with VideoWriter(path=input_path, fps=10.0, frame_size=(160, 120)) as writer:
        for _ in range(2):
            writer.write(np.full((120, 160, 3), 255, dtype=np.uint8))

    pipeline = make_pipeline()
    results = pipeline.analyze_video(input_path=input_path, output_path=output_path)

    assert len(results) == 2
    assert output_path.exists()

    metadata = VideoLoader(output_path).metadata()
    assert metadata.frame_count == 2
