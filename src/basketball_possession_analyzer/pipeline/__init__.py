
"""End-to-end analysis pipeline package."""

from basketball_possession_analyzer.pipeline.pipeline import (
    BasketballAnalysisPipeline,
    BasketballAnalysisPipelineConfig,
)
from basketball_possession_analyzer.pipeline.result import FrameAnalysisResult

__all__ = [
    "BasketballAnalysisPipeline",
    "BasketballAnalysisPipelineConfig",
    "FrameAnalysisResult",
]
