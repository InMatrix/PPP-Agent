from types import SimpleNamespace
from dataclasses import replace

from ppp_simplified.models import Episode, Preference
import pytest

from ppp_simplified.simulator import (
    CachedUserSimulator,
    GeminiUserSimulator,
    SimulatorCallBudgetExceeded,
    UserSimulator,
)


class FakeInteractions:
    def create(self, **kwargs):
        assert kwargs["model"] == "gemini-3.5-flash-lite"
        return SimpleNamespace(
            output_text=(
                '{"text":"pkg/a.py","cost_level":4,'
                '"preference_ok":true,"rationale":"file path"}'
            )
        )


def test_gemini_adapter_with_injected_client() -> None:
    fixture = Episode(
        instance_id="demo",
        repository="demo/repo",
        base_commit="abc",
        visible_issue="vague",
        full_issue="full",
        hint="",
        patch="",
        expected_functions=("pkg/a.py:run",),
        is_vague=True,
        preference=Preference("concise_question", "Be concise.", "rule"),
        source_path="fixture",
        row_index=0,
    )
    simulator = GeminiUserSimulator(
        client=SimpleNamespace(interactions=FakeInteractions())
    )
    reply = simulator.respond(
        episode=fixture,
        question="Which file?",
        question_number=1,
    )
    assert reply.cost_level == 4
    assert reply.preference_ok is True


def test_gemini_adapter_retries_malformed_response() -> None:
    fixture = Episode(
        instance_id="demo",
        repository="demo/repo",
        base_commit="abc",
        visible_issue="vague",
        full_issue="full",
        hint="",
        patch="",
        expected_functions=("pkg/a.py:run",),
        is_vague=True,
        preference=Preference("no_preference", "No preference.", "None"),
        source_path="fixture",
        row_index=0,
    )

    class FlakyInteractions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(output_text="not JSON")
            return SimpleNamespace(
                output_text=(
                    '{"text":"pkg/a.py","cost_level":4,'
                    '"preference_ok":null,"rationale":"file path"}'
                )
            )

    interactions = FlakyInteractions()
    simulator = GeminiUserSimulator(
        client=SimpleNamespace(interactions=interactions),
        retry_delay_seconds=0,
    )
    reply = simulator.respond(
        episode=fixture,
        question="Which file?",
        question_number=1,
    )
    assert reply.cost_level == 4
    assert interactions.calls == 2


class CountingSimulator(UserSimulator):
    name = "counting"

    def __init__(self):
        self.calls = 0

    def respond(self, *, episode, question, question_number):
        from ppp_simplified.models import UserReply

        self.calls += 1
        return UserReply("answer", 1, True, "fixture")


def test_cached_simulator_reuses_reply_and_enforces_budget(tmp_path) -> None:
    fixture = Episode(
        instance_id="demo",
        repository="demo/repo",
        base_commit="abc",
        visible_issue="vague",
        full_issue="full",
        hint="",
        patch="",
        expected_functions=("pkg/a.py:run",),
        is_vague=True,
        preference=Preference("concise_question", "Be concise.", "rule"),
        source_path="fixture",
        row_index=0,
    )
    inner = CountingSimulator()
    cached = CachedUserSimulator(
        inner,
        cache_path=tmp_path / "replies.sqlite3",
        max_live_calls=1,
    )
    first = cached.respond(episode=fixture, question="Which file?", question_number=1)
    second = cached.respond(episode=fixture, question="Which file?", question_number=1)
    assert first == second
    assert inner.calls == 1
    with pytest.raises(SimulatorCallBudgetExceeded):
        cached.respond(episode=fixture, question="Which method?", question_number=2)

    second_process_view = CachedUserSimulator(
        CountingSimulator(),
        cache_path=tmp_path / "replies.sqlite3",
        max_live_calls=1,
    )
    with pytest.raises(SimulatorCallBudgetExceeded):
        second_process_view.respond(
            episode=fixture,
            question="Which class?",
            question_number=3,
        )


def test_cached_gemini_budget_counts_every_physical_retry(tmp_path) -> None:
    fixture = Episode(
        instance_id="demo",
        repository="demo/repo",
        base_commit="abc",
        visible_issue="vague",
        full_issue="full",
        hint="",
        patch="",
        expected_functions=("pkg/a.py:run",),
        is_vague=True,
        preference=Preference("no_preference", "No preference.", "None"),
        source_path="fixture",
        row_index=0,
    )

    class AlwaysMalformed:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(output_text="not JSON")

    interactions = AlwaysMalformed()
    cached = CachedUserSimulator(
        GeminiUserSimulator(
            client=SimpleNamespace(interactions=interactions),
            retry_delay_seconds=0,
        ),
        cache_path=tmp_path / "gemini.sqlite3",
        max_live_calls=1,
    )
    with pytest.raises(SimulatorCallBudgetExceeded):
        cached.respond(
            episode=fixture,
            question="Which file?",
            question_number=1,
        )
    assert interactions.calls == 1
    assert cached.live_calls == 1


def test_cache_invalidates_when_hidden_episode_context_changes(tmp_path) -> None:
    fixture = Episode(
        instance_id="demo",
        repository="demo/repo",
        base_commit="abc",
        visible_issue="vague",
        full_issue="first hidden issue",
        hint="",
        patch="",
        expected_functions=("pkg/a.py:run",),
        is_vague=True,
        preference=Preference("no_preference", "No preference.", "None"),
        source_path="fixture",
        row_index=0,
    )
    inner = CountingSimulator()
    cached = CachedUserSimulator(
        inner,
        cache_path=tmp_path / "context.sqlite3",
        max_live_calls=2,
    )
    cached.respond(episode=fixture, question="Which file?", question_number=1)
    cached.respond(
        episode=replace(fixture, full_issue="different hidden issue"),
        question="Which file?",
        question_number=1,
    )
    assert inner.calls == 2
