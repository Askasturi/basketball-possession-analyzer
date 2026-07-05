
"""Optional heavy-model detector adapter stubs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from basketball_possession_analyzer.detection.base import BaseDetector, DetectorConfig
from basketball_possession_analyzer.detection.result import DetectionResult


class OptionalModelNotAvailableError(RuntimeError):
    """Raised when an optional heavy detector adapter is unavailable."""


@dataclass(frozen=True)
class ModelAdapterConfig(DetectorConfig):
    """Configuration for optional detector adapters."""

    model_path: str | None = None

    def __post_init__(self) -> None:
        """Validate adapter configuration."""
        super().__post_init__()

        if self.model_path is not None:
            path = Path(self.model_path)
            if not str(path).strip():
                raise ValueError("model_path must not be empty")


class YOLODetectorAdapter(BaseDetector):
    """Future YOLO detector adapter stub."""

    def __init__(self, config: ModelAdapterConfig | None = None) -> None:
        super().__init__(config=config or ModelAdapterConfig())

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
    ) -> DetectionResult:
        """Detect objects using a future YOLO adapter."""
        raise OptionalModelNotAvailableError(
            "YOLODetectorAdapter is a stub. Install and implement a YOLO backend "
            "before using this detector."
        )


class RFDETRDetectorAdapter(BaseDetector):
    """Future RF-DETR detector adapter stub."""

    def __init__(self, config: ModelAdapterConfig | None = None) -> None:
        super().__init__(config=config or ModelAdapterConfig())

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
    ) -> DetectionResult:
        """Detect objects using a future RF-DETR adapter."""
        raise OptionalModelNotAvailableError(
            "RFDETRDetectorAdapter is a stub. Install and implement an RF-DETR "
            "backend before using this detector."
        )
