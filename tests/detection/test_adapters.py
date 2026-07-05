import numpy as np
import pytest

from basketball_possession_analyzer.detection import (
    ModelAdapterConfig,
    OptionalModelNotAvailableError,
    RFDETRDetectorAdapter,
    YOLODetectorAdapter,
)


def test_model_adapter_config_defaults() -> None:
    config = ModelAdapterConfig()

    assert config.confidence_threshold == 0.25
    assert config.device == "cpu"
    assert config.model_path is None


def test_model_adapter_config_accepts_model_path() -> None:
    config = ModelAdapterConfig(model_path="models/example.pt")

    assert config.model_path == "models/example.pt"


def test_yolo_adapter_stub_raises() -> None:
    adapter = YOLODetectorAdapter()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(OptionalModelNotAvailableError):
        adapter.detect(frame)


def test_rfdetr_adapter_stub_raises() -> None:
    adapter = RFDETRDetectorAdapter()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(OptionalModelNotAvailableError):
        adapter.detect(frame)
