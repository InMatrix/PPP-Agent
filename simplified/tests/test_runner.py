from pathlib import Path

from ppp_simplified.models import AgentAction, Episode, Preference
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
        "list_tree",
        "ask_user",
        "finish",
    ]
    assert report.termination == "natural_finish"
    assert report.duplicate_actions_suppressed == 0


class RepeatingAgent:
    name = "repeating-agent"

    def __init__(self) -> None:
        self.turn = 0

    def next_action(
        self,
        *,
        system_prompt,
        messages,
        allowed_tools=None,
    ) -> AgentAction:
        del system_prompt, messages, allowed_tools
        self.turn += 1
        if self.turn <= 2:
            return AgentAction(
                tool="search_code",
                arguments={"query": "Widget", "path": "", "glob": "*.py"},
            )
        return AgentAction(
            tool="finish",
            arguments={"functions": ["pkg/widget.py:Widget.run"]},
        )


def test_runner_suppresses_identical_actions(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/widget.py").write_text(
        "class Widget:\n"
        "    def run(self):\n"
        "        pass\n"
    )

    report = AgentRunner(
        provider=RepeatingAgent(),
        simulator=DeterministicUserSimulator(),
        max_turns=4,
    ).run(episode(), tmp_path)

    assert report.duplicate_actions_suppressed == 1
    assert report.trajectory[1].duplicate_suppressed is True
    assert report.trajectory[1].attempt == 1
    assert report.trajectory[1].executed is False
    assert report.trajectory[2].turn == 2
    assert report.trajectory[2].attempt == 2
    assert "Duplicate action suppressed" in report.trajectory[1].observation
    assert report.termination == "natural_finish"
    assert report.model_calls == 3


class DeadlineAgent:
    name = "deadline-agent"

    def __init__(self) -> None:
        self.allowed_tools = []
        self.prompts = []

    def next_action(
        self,
        *,
        system_prompt,
        messages,
        allowed_tools=None,
    ) -> AgentAction:
        del messages
        self.allowed_tools.append(allowed_tools)
        self.prompts.append(system_prompt)
        if allowed_tools == ("finish",):
            return AgentAction(
                tool="finish",
                arguments={"functions": ["pkg/widget.py:Widget.run"]},
            )
        return AgentAction(
            tool="list_files",
            arguments={"path": "", "max_entries": 10},
        )


def test_runner_reserves_last_turn_for_finish(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/widget.py").write_text(
        "class Widget:\n"
        "    def run(self):\n"
        "        pass\n"
    )
    provider = DeadlineAgent()

    report = AgentRunner(
        provider=provider,
        simulator=DeterministicUserSimulator(),
        max_turns=2,
    ).run(episode(), tmp_path)

    assert provider.allowed_tools == [None, ("finish",)]
    assert "turn 1 of 2" in provider.prompts[0]
    assert "FINALIZATION TURN" in provider.prompts[1]
    assert report.termination == "deadline_finish"
    assert report.predicted_functions == ("pkg/widget.py:Widget.run",)
    assert report.finish_validation_passed is True


class CorrectingFinishAgent:
    name = "correcting-finish-agent"

    def __init__(self, always_invalid: bool = False) -> None:
        self.calls = 0
        self.always_invalid = always_invalid

    def next_action(
        self,
        *,
        system_prompt,
        messages,
        allowed_tools=None,
    ) -> AgentAction:
        del system_prompt, messages, allowed_tools
        self.calls += 1
        function = (
            "pkg/widget.py:Widget.still_missing"
            if self.always_invalid and self.calls > 1
            else (
                "pkg/widget.py:Widget.run"
                if self.calls > 1
                else "missing.py:Widget.run"
            )
        )
        return AgentAction(
            tool="finish",
            arguments={"functions": [function]},
        )


def test_runner_corrects_invalid_finish_once(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/widget.py").write_text(
        "class Widget:\n"
        "    def run(self):\n"
        "        pass\n"
    )

    report = AgentRunner(
        provider=CorrectingFinishAgent(),
        simulator=DeterministicUserSimulator(),
        max_turns=1,
    ).run(episode(), tmp_path)

    assert report.model_calls == 2
    assert report.finish_correction_attempted is True
    assert report.finish_validation_passed is True
    assert report.invalid_predictions == ()
    assert report.predicted_functions == ("pkg/widget.py:Widget.run",)
    assert [step.executed for step in report.trajectory] == [False, True]
    assert [step.attempt for step in report.trajectory] == [1, 2]


def test_runner_preserves_unverified_correction_for_scoring(
    tmp_path: Path,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/widget.py").write_text("class Widget:\n    pass\n")

    report = AgentRunner(
        provider=CorrectingFinishAgent(always_invalid=True),
        simulator=DeterministicUserSimulator(),
        max_turns=1,
    ).run(episode(), tmp_path)

    assert report.finish_correction_attempted is True
    assert report.finish_validation_passed is False
    assert report.invalid_predictions == (
        "pkg/widget.py:Widget.still_missing",
    )
    assert report.predicted_functions == (
        "pkg/widget.py:Widget.still_missing",
    )


class PersistentlyRepeatingAgent:
    name = "persistently-repeating-agent"

    def __init__(self) -> None:
        self.allowed_tools = []

    def next_action(
        self,
        *,
        system_prompt,
        messages,
        allowed_tools=None,
    ) -> AgentAction:
        del system_prompt, messages
        self.allowed_tools.append(allowed_tools)
        if allowed_tools == ("finish",):
            return AgentAction(
                tool="finish",
                arguments={"functions": ["pkg/widget.py:Widget.run"]},
            )
        if allowed_tools is not None and "search_code" not in allowed_tools:
            return AgentAction(
                tool="list_tree",
                arguments={"path": "", "max_depth": 2, "max_entries": 20},
            )
        return AgentAction(
            tool="search_code",
            arguments={"query": "Widget", "path": "", "glob": "*.py"},
        )


def test_navigation_v3_retries_each_duplicate_signature_only_once(
    tmp_path: Path,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/widget.py").write_text(
        "class Widget:\n"
        "    def run(self):\n"
        "        pass\n"
    )

    provider = PersistentlyRepeatingAgent()
    report = AgentRunner(
        provider=provider,
        simulator=DeterministicUserSimulator(),
        max_turns=4,
        policy_version="navigation-v3",
    ).run(episode(), tmp_path)

    assert report.duplicate_actions_suppressed == 2
    assert report.model_calls == 5
    assert [
        (step.turn, step.attempt)
        for step in report.trajectory
        if step.duplicate_suppressed
    ] == [(2, 1), (2, 2)]
    assert provider.allowed_tools[3] is not None
    assert "search_code" not in provider.allowed_tools[3]
    assert report.predicted_functions == ("pkg/widget.py:Widget.run",)
