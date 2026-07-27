from pathlib import Path

from ppp_simplified.models import Episode, Preference
from ppp_simplified.providers import ScriptedSmokeAgent
from ppp_simplified.runner import AgentRunner
from ppp_simplified.simulator import DeterministicUserSimulator


def episode() -> Episode:
    return Episode(
        instance_id="demo",
        repository="demo/repo",
        base_commit="abc",
        visible_issue="Something breaks.",
        full_issue="Widget.run raises an error.",
        hint="",
        patch="",
        expected_functions=("pkg/widget.py:Widget.run",),
        is_vague=True,
        preference=Preference(
            name="concise_question",
            description="Ask concise questions.",
            reward_rule="Penalize long questions.",
        ),
        source_path="fixture",
        row_index=0,
    )


def test_end_to_end_offline_run(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/widget.py").write_text(
        "class Widget:\n    def run(self):\n        pass\n"
    )
    fixture = episode()
    report = AgentRunner(
        provider=ScriptedSmokeAgent(fixture.expected_functions),
        simulator=DeterministicUserSimulator(),
        max_turns=4,
    ).run(fixture, tmp_path)
    assert report.predicted_functions == fixture.expected_functions
    assert report.reward.productivity == 1.0
    assert report.reward.questions_asked == 1
    assert [step.action.tool for step in report.trajectory] == [
        "list_files",
        "ask_user",
        "finish",
    ]
