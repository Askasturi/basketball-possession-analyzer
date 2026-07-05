
import numpy as np
import pytest

from basketball_possession_analyzer.classification import (
    EmbeddingClassifierNotAvailableError,
    EmbeddingTeamClassifier,
)
from basketball_possession_analyzer.detection import (
    BoundingBox,
    Detection,
    DetectionLabel,
)
from basketball_possession_analyzer.tracking import Track


def test_embedding_team_classifier_stub_raises() -> None:
    classifier = EmbeddingTeamClassifier()
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    track = Track(
        track_id=0,
        detection=Detection(
            label=DetectionLabel.PLAYER,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            confidence=0.9,
        ),
        first_frame_index=0,
        last_frame_index=0,
    )

    with pytest.raises(EmbeddingClassifierNotAvailableError):
        classifier.classify(frame=frame, track=track)
