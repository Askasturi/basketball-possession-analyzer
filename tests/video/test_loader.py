
from pathlib import Path

import numpy as np
import pytest

from basketball_possession_analyzer.video import (
    VideoLoader,
    VideoOpenError,
    VideoWriter,
)


def create_test_video(path: Path, frame_count: int = 5) -> None:
    with VideoWriter(path=path, fps=10.0, frame_size=(64, 48)) as writer:
        for _ in range(frame_count):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            writer.write(frame)


def test_video_loader_reads_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    create_test_video(video_path, frame_count=5)

    metadata = VideoLoader(video_path).metadata()

    assert metadata.path == str(video_path)
    assert metadata.frame_count == 5
    assert metadata.fps > 0
    assert metadata.width == 64
    assert metadata.height == 48


def test_video_loader_raises_for_missing_file(tmp_path: Path) -> None:
    video_path = tmp_path / "missing.mp4"

    with pytest.raises(VideoOpenError):
        VideoLoader(video_path).open()
