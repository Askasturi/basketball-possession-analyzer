
from basketball_possession_analyzer.video import VideoMetadata


def test_video_metadata_duration_seconds() -> None:
    metadata = VideoMetadata(
        path="sample.mp4",
        frame_count=60,
        fps=30.0,
        width=1280,
        height=720,
    )

    assert metadata.duration_seconds == 2.0


def test_video_metadata_duration_zero_when_fps_invalid() -> None:
    metadata = VideoMetadata(
        path="sample.mp4",
        frame_count=60,
        fps=0.0,
        width=1280,
        height=720,
    )

    assert metadata.duration_seconds == 0.0


def test_video_metadata_frame_size() -> None:
    metadata = VideoMetadata(
        path="sample.mp4",
        frame_count=60,
        fps=30.0,
        width=1280,
        height=720,
    )

    assert metadata.frame_size == (1280, 720)
