
from pathlib import Path

import numpy as np
import pytest

from basketball_possession_analyzer.video import (
    VideoLoader,
    VideoWriteError,
    VideoWriter,
)


def test_video_writer_creates_video_file(tmp_path: Path) -> None:
    video_path = tmp_path / "output.mp4"

    with VideoWriter(path=video_path, fps=10.0, frame_size=(64, 48)) as writer:
        for _ in range(3):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            writer.write(frame)

    assert video_path.exists()

    metadata = VideoLoader(video_path).metadata()
    assert metadata.frame_count == 3
    assert metadata.width == 64
    assert metadata.height == 48


def test_video_writer_rejects_invalid_fps(tmp_path: Path) -> None:
    video_path = tmp_path / "output.mp4"

    with pytest.raises(ValueError):
        VideoWriter(path=video_path, fps=0.0, frame_size=(64, 48))


def test_video_writer_rejects_invalid_frame_size(tmp_path: Path) -> None:
    video_path = tmp_path / "output.mp4"

    with pytest.raises(ValueError):
        VideoWriter(path=video_path, fps=10.0, frame_size=(0, 48))


def test_video_writer_rejects_mismatched_frame_size(tmp_path: Path) -> None:
    video_path = tmp_path / "output.mp4"
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    with pytest.raises(VideoWriteError):
        with VideoWriter(path=video_path, fps=10.0, frame_size=(64, 48)) as writer:
            writer.write(frame)
