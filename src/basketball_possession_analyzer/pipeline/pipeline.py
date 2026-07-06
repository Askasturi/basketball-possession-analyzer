
"""End-to-end basketball possession analysis pipeline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from basketball_possession_analyzer.classification import BaseTeamClassifier
from basketball_possession_analyzer.court import HomographyTransformer
from basketball_possession_analyzer.detection import BaseDetector
from basketball_possession_analyzer.identity import IdentityResolver
from basketball_possession_analyzer.pipeline.result import FrameAnalysisResult
from basketball_possession_analyzer.recognition import (
    BaseNumberRecognizer,
    NumberTrackMatcher,
)
from basketball_possession_analyzer.shots import (
    SimpleShotDetector,
    SimpleShotResultClassifier,
)
from basketball_possession_analyzer.tracking import SimpleIoUTracker
from basketball_possession_analyzer.video import FrameIterator, VideoLoader, VideoWriter
from basketball_possession_analyzer.visualization import BasketballFrameRenderer


@dataclass(frozen=True)
class BasketballAnalysisPipelineConfig:
    """Configuration for the analysis pipeline."""

    render: bool = True

    def __post_init__(self) -> None:
        """Validate pipeline config."""


class BasketballAnalysisPipeline:
    """Run the full basketball possession analysis pipeline."""

    def __init__(
        self,
        detector: BaseDetector,
        tracker: SimpleIoUTracker,
        team_classifier: BaseTeamClassifier | None = None,
        number_recognizer: BaseNumberRecognizer | None = None,
        number_matcher: NumberTrackMatcher | None = None,
        identity_resolver: IdentityResolver | None = None,
        homography_transformer: HomographyTransformer | None = None,
        shot_detector: SimpleShotDetector | None = None,
        shot_result_classifier: SimpleShotResultClassifier | None = None,
        renderer: BasketballFrameRenderer | None = None,
        config: BasketballAnalysisPipelineConfig | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.team_classifier = team_classifier
        self.number_recognizer = number_recognizer
        self.number_matcher = number_matcher or NumberTrackMatcher()
        self.identity_resolver = identity_resolver or IdentityResolver()
        self.homography_transformer = homography_transformer
        self.shot_detector = shot_detector
        self.shot_result_classifier = shot_result_classifier
        self.renderer = renderer or BasketballFrameRenderer()
        self.config = config or BasketballAnalysisPipelineConfig()

    def reset(self) -> None:
        """Reset stateful pipeline components."""
        self.tracker.reset()
        self.identity_resolver.reset()

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
    ) -> FrameAnalysisResult:
        """Analyze one frame."""
        detection_result = self.detector.detect(
            frame=frame,
            frame_index=frame_index,
        )
        tracking_result = self.tracker.update(detection_result)

        team_results = []
        if self.team_classifier is not None:
            team_results = self.team_classifier.classify_tracks(
                frame=frame,
                tracks=tracking_result.tracks,
            )

        number_results = []
        if self.number_recognizer is not None:
            number_results = self.number_recognizer.recognize(
                frame=frame,
                detection_result=detection_result,
            )

        number_matches = self.number_matcher.match(
            recognition_results=number_results,
            tracks=tracking_result.tracks,
        )

        identity_result = self.identity_resolver.resolve(
            tracking_result=tracking_result,
            team_results=team_results,
            number_matches=number_matches,
        )

        court_projection_result = None
        if self.homography_transformer is not None:
            court_projection_result = (
                self.homography_transformer.project_tracking_result(
                    tracking_result=tracking_result,
                )
            )

        shot_event = None
        if self.shot_detector is not None:
            shot_event = self.shot_detector.detect(
                detection_result=detection_result,
                tracking_result=tracking_result,
            )

            if shot_event is not None and self.shot_result_classifier is not None:
                shot_event = self.shot_result_classifier.classify(
                    shot_event=shot_event,
                    detection_result=detection_result,
                )

        rendered_frame = None
        if self.config.render:
            rendered_frame = self.renderer.render(
                frame=frame,
                detection_result=detection_result,
                tracking_result=tracking_result,
                identity_result=identity_result,
                court_projection_result=court_projection_result,
                shot_event=shot_event,
            )

        return FrameAnalysisResult(
            frame_index=frame_index,
            detection_result=detection_result,
            tracking_result=tracking_result,
            team_results=team_results,
            number_results=number_results,
            number_matches=number_matches,
            identity_result=identity_result,
            court_projection_result=court_projection_result,
            shot_event=shot_event,
            rendered_frame=rendered_frame,
        )

    def analyze_video(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> list[FrameAnalysisResult]:
        """Analyze a video and optionally write rendered output."""
        metadata = VideoLoader(input_path).metadata()
        writer = None

        if output_path is not None:
            writer = VideoWriter(
                path=output_path,
                fps=metadata.fps,
                frame_size=metadata.frame_size,
            )

        results: list[FrameAnalysisResult] = []

        try:
            if writer is not None:
                writer.open()

            for frame_index, frame in FrameIterator(
                path=input_path,
                start_frame=start_frame,
                end_frame=end_frame,
            ):
                result = self.analyze_frame(frame=frame, frame_index=frame_index)
                results.append(result)

                if writer is not None and result.rendered_frame is not None:
                    writer.write(result.rendered_frame)
        finally:
            if writer is not None:
                writer.close()

        return results
