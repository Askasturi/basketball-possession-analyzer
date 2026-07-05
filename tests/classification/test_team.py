
from basketball_possession_analyzer.classification import Team, TeamSide


def test_team_side_values() -> None:
    assert TeamSide.HOME == "home"
    assert TeamSide.AWAY == "away"
    assert TeamSide.UNKNOWN == "unknown"


def test_team_display_name_uses_name_when_present() -> None:
    team = Team(side=TeamSide.HOME, name="Lakers")

    assert team.display_name == "Lakers"


def test_team_display_name_falls_back_to_side() -> None:
    team = Team(side=TeamSide.AWAY)

    assert team.display_name == "away"
