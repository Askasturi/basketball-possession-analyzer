
"""Video loading utilities."""

from pathlib import Path

import cv2

from basketball_possession_analyzer.video.exceptions import VideoOpenError
from basketball_possession_analyzer.video.metadata import VideoMetadata


class VideoLoader:
    """Open a video file and read its metadata."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def open(self) -> cv2.VideoCapture:
        """Open the video and return an OpenCV capture object."""
        if not self.path.exists():
            raise VideoOpenError(f"Video file does not exist: {self.path}")

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise VideoOpenError(f"Could not open video file: {self.path}")

        return capture

    def metadata(self) -> VideoMetadata:
        """Read metadata from the video file."""
        capture = self.open()
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            return VideoMetadata(
                path=str(self.path),
                frame_count=frame_count,
                fps=fps,
                width=width,
                height=height,
            )
        finally:
            capture.release()
