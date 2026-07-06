
"""Run a synthetic end-to-end basketball possession analyzer demo."""

from pathlib import Path

import cv2
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
from basketball_possession_analyzer.pipeline import BasketballAnalysisPipeline
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
from basketball_possession_analyzer.video import VideoWriter


def create_synthetic_video(path: Path) -> None:
    """Create a tiny synthetic basketball clip."""
    with VideoWriter(path=path, fps=10.0, frame_size=(320, 240)) as writer:
        for frame_index in range(5):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)

            player_x = 40 + frame_index * 12
            cv2.rectangle(
                frame,
                (player_x, 80),
                (player_x + 40, 180),
                (255, 255, 255),
                -1,
            )
            cv2.circle(frame, (220, 70), 18, (0, 140, 255), 2)
            cv2.circle(frame, (205, 85), 7, (0, 140, 255), -1)

            writer.write(frame)


def build_demo_pipeline() -> BasketballAnalysisPipeline:
    """Build a deterministic demo pipeline."""
    detections_by_frame: dict[int, list[Detection]] = {}

    for frame_index in range(5):
        player_x = 40 + frame_index * 12
        detections_by_frame[frame_index] = [
            Detection(
                label=DetectionLabel.PLAYER,
                bbox=BoundingBox(
                    x1=player_x,
                    y1=80,
                    x2=player_x + 40,
                    y2=180,
                ),
                confidence=0.95,
            ),
            Detection(
                label=DetectionLabel.JERSEY_NUMBER,
                bbox=BoundingBox(
                    x1=player_x + 12,
                    y1=105,
                    x2=player_x + 28,
                    y2=125,
                ),
                confidence=0.90,
            ),
            Detection(
                label=DetectionLabel.BALL,
                bbox=BoundingBox(x1=198, y1=78, x2=212, y2=92),
                confidence=0.90,
            ),
            Detection(
                label=DetectionLabel.RIM,
                bbox=BoundingBox(x1=202, y1=52, x2=238, y2=88),
                confidence=0.90,
            ),
        ]

    number_results = {
        frame_index: [
            NumberRecognitionResult(
                frame_index=frame_index,
                bbox=detections_by_frame[frame_index][1].bbox,
                number="23",
                confidence=0.92,
            )
        ]
        for frame_index in range(5)
    }

    homography = HomographyTransformer(
        image_points=[
            ImagePoint(x=0, y=0),
            ImagePoint(x=320, y=0),
            ImagePoint(x=320, y=240),
            ImagePoint(x=0, y=240),
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
            config=MockDetectorConfig(detections_by_frame=detections_by_frame)
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
            config=MockNumberRecognizerConfig(results_by_frame=number_results)
        ),
        homography_transformer=homography,
        shot_detector=SimpleShotDetector(
            config=SimpleShotDetectorConfig(rim_distance_threshold=60)
        ),
        shot_result_classifier=SimpleShotResultClassifier(),
    )


def main() -> None:
    """Run the demo."""
    input_path = Path("outputs/synthetic_input.mp4")
    output_path = Path("outputs/synthetic_annotated.mp4")

    create_synthetic_video(input_path)

    pipeline = build_demo_pipeline()
    results = pipeline.analyze_video(
        input_path=input_path,
        output_path=output_path,
    )

    print(f"Analyzed {len(results)} frames")
    print(f"Wrote annotated video to {output_path}")


if __name__ == "__main__":
    main()
