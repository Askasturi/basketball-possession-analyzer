
"""Video writing utilities."""

from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from basketball_possession_analyzer.video.exceptions import VideoWriteError


class VideoWriter:
    """Write frames to a video file."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")

        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("frame_size values must be > 0")

        self.path = Path(path)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self._writer: cv2.VideoWriter | None = None

    def open(self) -> None:
        """Open the video writer."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            str(self.path),
            fourcc,
            self.fps,
            self.frame_size,
        )

        if not self._writer.isOpened():
            self._writer.release()
            self._writer = None
            raise VideoWriteError(f"Could not open video writer: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame."""
        if self._writer is None:
            self.open()

        expected_width, expected_height = self.frame_size
        actual_height, actual_width = frame.shape[:2]

        if (actual_width, actual_height) != (expected_width, expected_height):
            raise VideoWriteError(
                "Frame size does not match writer frame_size: "
                f"expected {(expected_width, expected_height)}, "
                f"got {(actual_width, actual_height)}"
            )

        if self._writer is None:
            raise VideoWriteError("Video writer is not open")

        self._writer.write(frame)

    def close(self) -> None:
        """Close the video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
