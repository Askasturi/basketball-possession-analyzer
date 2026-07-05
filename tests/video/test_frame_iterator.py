
from pathlib import Path

import numpy as np
import pytest

from basketball_possession_analyzer.video import FrameIterator, VideoWriter


def create_test_video(path: Path, frame_count: int = 5) -> None:
    with VideoWriter(path=path, fps=10.0, frame_size=(64, 48)) as writer:
        for index in range(frame_count):
            frame = np.full((48, 64, 3), index, dtype=np.uint8)
            writer.write(frame)


def test_frame_iterator_yields_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    create_test_video(video_path, frame_count=5)

    frames = list(FrameIterator(video_path))

    assert len(frames) == 5
    assert frames[0][0] == 0
    assert frames[-1][0] == 4
    assert frames[0][1].shape == (48, 64, 3)


def test_frame_iterator_respects_start_and_end_frame(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    create_test_video(video_path, frame_count=5)

    frames = list(FrameIterator(video_path, start_frame=1, end_frame=4))

    assert [frame_index for frame_index, _ in frames] == [1, 2, 3]


def test_frame_iterator_rejects_negative_start_frame(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"

    with pytest.raises(ValueError):
        FrameIterator(video_path, start_frame=-1)


def test_frame_iterator_rejects_end_before_start(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"

    with pytest.raises(ValueError):
        FrameIterator(video_path, start_frame=5, end_frame=4)
