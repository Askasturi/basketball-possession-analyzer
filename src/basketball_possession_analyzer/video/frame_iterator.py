
"""Frame iteration utilities."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from basketball_possession_analyzer.video.loader import VideoLoader


class FrameIterator:
    """Iterate over frames from a video file."""

    def __init__(
        self,
        path: str | Path,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> None:
        if start_frame < 0:
            raise ValueError("start_frame must be >= 0")
        if end_frame is not None and end_frame < start_frame:
            raise ValueError("end_frame must be >= start_frame")

        self.path = Path(path)
        self.start_frame = start_frame
        self.end_frame = end_frame

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield frame index and frame image."""
        capture = VideoLoader(self.path).open()

        try:
            if self.start_frame:
                capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            frame_index = self.start_frame

            while True:
                if self.end_frame is not None and frame_index >= self.end_frame:
                    break

                success, frame = capture.read()
                if not success:
                    break

                yield frame_index, frame
                frame_index += 1
        finally:
            capture.release()
