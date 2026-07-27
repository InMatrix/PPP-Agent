from types import SimpleNamespace

from ppp_simplified.models import Episode, Preference
from ppp_simplified.simulator import GeminiUserSimulator


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
