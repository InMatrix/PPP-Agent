import pytest

from ppp_simplified.models import Episode, Preference, UserReply
from ppp_simplified.rewards import calculate_reward, set_f1


def episode() -> Episode:
    return Episode(
        instance_id="demo",
        repository="demo/repo",
        base_commit="abc",
        visible_issue="vague",
        full_issue="full",
        hint="",
        patch="",
        expected_functions=("pkg/mod.py:Widget.run",),
        is_vague=True,
        preference=Preference(
            name="concise_question",
            description="Be concise.",
            reward_rule="Long questions violate the preference.",
        ),
        source_path="fixture",
        row_index=0,
    )


def test_function_f1_normalizes_workspace_prefix() -> None:
    assert (
        set_f1(
            ["/testbed/pkg/mod.py:Widget.run"],
            ["pkg/mod.py:Widget.run"],
        )
        == 1.0
    )


def test_high_disclosure_can_offset_productivity() -> None:
    reward = calculate_reward(
        episode=episode(),
        predicted_functions=["pkg/mod.py:Widget.run"],
        replies=[
            UserReply(
                text="answer",
                cost_level=5,
                preference_ok=True,
            )
        ],
    )
    assert reward.productivity == 1.0
    assert reward.proactivity_adjustment == pytest.approx(-0.4)
    assert reward.personalization_adjustment == 0.05
    assert reward.total == pytest.approx(0.65)


def test_failed_vague_guess_is_penalized() -> None:
    reward = calculate_reward(
        episode=episode(),
        predicted_functions=[],
        replies=[],
    )
    assert reward.proactivity_adjustment == -0.1
    assert reward.total == 0.0
