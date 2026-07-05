
"""Match recognized jersey numbers to tracked players."""

from dataclasses import dataclass

from basketball_possession_analyzer.detection import DetectionLabel
from basketball_possession_analyzer.recognition.result import NumberRecognitionResult
from basketball_possession_analyzer.tracking import Track


@dataclass(frozen=True)
class NumberTrackMatch:
    """A jersey number assigned to a track."""

    track_id: int
    number: str
    confidence: float
    overlap_score: float

    def __post_init__(self) -> None:
        """Validate number-track match."""
        if self.track_id < 0:
            raise ValueError("track_id must be >= 0")
        if not self.number.strip():
            raise ValueError("number must not be empty")
        if not self.number.isdigit():
            raise ValueError("number must contain only digits")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.overlap_score <= 1.0:
            raise ValueError("overlap_score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class NumberTrackMatcherConfig:
    """Configuration for matching jersey numbers to player tracks."""

    min_ios: float = 0.3

    def __post_init__(self) -> None:
        """Validate matcher configuration."""
        if not 0.0 <= self.min_ios <= 1.0:
            raise ValueError("min_ios must be between 0.0 and 1.0")


class NumberTrackMatcher:
    """Assign recognized jersey numbers to player tracks using IoS overlap."""

    def __init__(self, config: NumberTrackMatcherConfig | None = None) -> None:
        self.config = config or NumberTrackMatcherConfig()

    def match(
        self,
        recognition_results: list[NumberRecognitionResult],
        tracks: list[Track],
    ) -> list[NumberTrackMatch]:
        """Match recognition results to tracks."""
        player_tracks = [
            track for track in tracks if track.label == DetectionLabel.PLAYER
        ]
        candidates: list[tuple[float, float, int, int]] = []

        for result_index, result in enumerate(recognition_results):
            for track_index, track in enumerate(player_tracks):
                overlap_score = result.bbox.ios(track.bbox)
                if overlap_score >= self.config.min_ios:
                    candidates.append(
                        (
                            overlap_score,
                            result.confidence,
                            result_index,
                            track_index,
                        )
                    )

        candidates.sort(reverse=True)

        matches: list[NumberTrackMatch] = []
        used_results: set[int] = set()
        used_tracks: set[int] = set()

        for overlap_score, _confidence, result_index, track_index in candidates:
            if result_index in used_results or track_index in used_tracks:
                continue

            result = recognition_results[result_index]
            track = player_tracks[track_index]

            matches.append(
                NumberTrackMatch(
                    track_id=track.track_id,
                    number=result.number,
                    confidence=result.confidence,
                    overlap_score=overlap_score,
                )
            )
            used_results.add(result_index)
            used_tracks.add(track_index)

        return matches
