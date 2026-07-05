
"""Video infrastructure."""

from basketball_possession_analyzer.video.exceptions import (
    VideoError,
    VideoOpenError,
    VideoWriteError,
)
from basketball_possession_analyzer.video.frame_iterator import FrameIterator
from basketball_possession_analyzer.video.loader import VideoLoader
from basketball_possession_analyzer.video.metadata import VideoMetadata
from basketball_possession_analyzer.video.writer import VideoWriter

__all__ = [
    "FrameIterator",
    "VideoError",
    "VideoLoader",
    "VideoMetadata",
    "VideoOpenError",
    "VideoWriteError",
    "VideoWriter",
]
