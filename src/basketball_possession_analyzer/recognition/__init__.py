
"""Jersey number recognition package."""

from basketball_possession_analyzer.recognition.base import (
    BaseNumberRecognizer,
    NumberRecognizerConfig,
)
from basketball_possession_analyzer.recognition.matcher import (
    NumberTrackMatch,
    NumberTrackMatcher,
    NumberTrackMatcherConfig,
)
from basketball_possession_analyzer.recognition.mock import (
    MockNumberRecognizer,
    MockNumberRecognizerConfig,
    RecognitionFactory,
)
from basketball_possession_analyzer.recognition.result import NumberRecognitionResult
from basketball_possession_analyzer.recognition.vlm_stub import (
    VLMNumberRecognizer,
    VLMRecognizerNotAvailableError,
)

__all__ = [
    "BaseNumberRecognizer",
    "MockNumberRecognizer",
    "MockNumberRecognizerConfig",
    "NumberRecognitionResult",
    "NumberRecognizerConfig",
    "NumberTrackMatch",
    "NumberTrackMatcher",
    "NumberTrackMatcherConfig",
    "RecognitionFactory",
    "VLMNumberRecognizer",
    "VLMRecognizerNotAvailableError",
]
