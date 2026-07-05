"""Video-related exceptions."""


class VideoError(Exception):
    """Base exception for video infrastructure errors."""


class VideoOpenError(VideoError):
    """Raised when a video cannot be opened."""


class VideoWriteError(VideoError):
    """Raised when a video cannot be written."""
