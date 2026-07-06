from pathlib import Path

from basketball_possession_analyzer.video import VideoLoader
from examples.run_demo import build_demo_pipeline, create_synthetic_video


def test_create_synthetic_video(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic.mp4"

    create_synthetic_video(video_path)

    metadata = VideoLoader(video_path).metadata()

    assert metadata.frame_count == 5
    assert metadata.width == 320
    assert metadata.height == 240


def test_build_demo_pipeline_analyzes_synthetic_video(tmp_path: Path) -> None:
    input_path = tmp_path / "synthetic_input.mp4"
    output_path = tmp_path / "synthetic_output.mp4"

    create_synthetic_video(input_path)
    pipeline = build_demo_pipeline()

    results = pipeline.analyze_video(input_path=input_path, output_path=output_path)

    assert len(results) == 5
    assert output_path.exists()
    assert any(result.shot_event is not None for result in results)
