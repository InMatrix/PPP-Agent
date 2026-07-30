import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ppp_simplified.data import episode_from_training_payload
from ppp_simplified.verl_loop import (
    ActionContractError,
    _one,
    _parse_action,
    _rollout_identity,
    run_simplified_rollout,
)
from ppp_simplified.workspace import RepositoryWorkspace


def test_one_unwraps_scalars_without_truncating_strings_or_dicts():
    assert _one("FuncLocEnv@payload") == "FuncLocEnv@payload"
    assert _one({"index": "sample-1"}) == {"index": "sample-1"}
    assert _one(["first"]) == "first"


def test_rollout_identity_preserves_foldgrpo_group_and_sample_ids():
    assert _rollout_identity(
        {"uid": ["question-1"], "gen_uid": ["trajectory-7"]}
    ) == ("question-1", "trajectory-7")
    with pytest.raises(ValueError, match="gen_uid"):
        _rollout_identity({"uid": "question-1"})


def test_verl_loop_class_is_worker_importable_from_hydra_config():
    config = (
        Path(__file__).parents[1] / "config" / "verl_agent_loops.yaml"
    ).read_text()
    assert "ppp_simplified.verl_agent_loop_impl.SimplifiedPPPAgentLoop" in config
    source = (
        Path(__file__).parents[1]
        / "src"
        / "ppp_simplified"
        / "verl_agent_loop_impl.py"
    ).read_text()
    assert "\nclass SimplifiedPPPAgentLoop(" in source
    rollout_source = (
        Path(__file__).parents[1]
        / "src"
        / "ppp_simplified"
        / "verl_loop.py"
    ).read_text()
    assert '"arguments": action.arguments,' not in rollout_source
    assert rollout_source.count("sanitize_action_arguments(") >= 4


def test_vendored_rollout_client_requests_logprobs_and_honors_turn_cap():
    source = (Path(__file__).parents[2] / "agents" / "utils.py").read_text()
    assert "max_new_tokens = min(max_new_tokens, self.config.plugin.turn_max_new_tokens)" in source
    assert "'logprobs': True" in source
    assert "requested_sampling_params.get('structured_outputs')" in source
    assert "sampling_params['structured_outputs'] = structured_outputs" in source
    assert "vLLM did not return the requested sampled-token log" in source
    assert "len(response_log_probs) != len(output.token_ids)" in source
    assert "sampling_params=sampling_params" in source


def test_rollout_returns_verls_output_and_metrics_models():
    source = (
        Path(__file__).parents[1]
        / "src"
        / "ppp_simplified"
        / "verl_loop.py"
    ).read_text()
    assert "from agents.utils import Agent, CallLLM" in source
    assert "from verl.experimental.agent_loop.agent_loop import (" in source
    assert "AgentLoopMetrics," in source
    assert "AgentLoopOutput," in source
    assert '"mask_rollout": termination == "turn_limit"' in source
    assert '"process_reward_mask": trace["process_reward_mask"]' in source


def test_parse_action_keeps_navigation_v2_tool_contract():
    action = _parse_action(
        json.dumps(
            {
                "tool": "find_symbol",
                "arguments": {"name": "run"},
                "reasoning": "Locate the entry point.",
            }
        ),
        ("find_symbol", "finish"),
    )
    assert action.tool == "find_symbol"
    assert action.arguments == {"name": "run"}
    with pytest.raises(ValueError, match="not allowed"):
        _parse_action(
            '{"tool":"inspect_symbol","arguments":{},"reasoning":"x"}',
            ("find_symbol", "finish"),
        )


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("not json", "invalid_json"),
        ('{"arguments":{},"reasoning":"x"}', "missing_tool"),
        (
            '{"tool":"ask_user","arguments":{},"reasoning":"x"}',
            "disallowed_tool",
        ),
        (
            '{"tool":"finish","arguments":[],"reasoning":"x"}',
            "arguments_not_object",
        ),
        (
            '{"tool":"finish","arguments":{},"reasoning":3}',
            "reasoning_not_string",
        ),
    ],
)
def test_parse_action_emits_sanitized_error_categories(
    text: str,
    category: str,
) -> None:
    with pytest.raises(ActionContractError) as raised:
        _parse_action(text, ("finish",))

    assert raised.value.category == category
    assert text not in str(raised.value)


def _training_item() -> SimpleNamespace:
    ability = "FuncLocEnv@" + json.dumps(
        {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": "abc",
            "problem_statement": "Precise issue",
            "hints_text": "",
            "patch": "diff",
            "edited_functions": ["pkg/widget.py:Widget.run"],
        }
    )
    return SimpleNamespace(
        non_tensor_batch={
            "ability": ability,
            "uid": "question-1",
            "gen_uid": "trajectory-1",
            "extra_info": {
                "index": "sample-1",
                "source_path": "data/train.parquet",
                "row_index": 7,
                "is_vague": True,
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            "--- BEGIN ISSUE ---\n"
                            "Vague issue\n"
                            "--- END ISSUE ---"
                        ),
                    }
                ],
                "preference": {
                    "preference_name": "no_preference",
                    "no_preference": {
                        "preference": "No preference.",
                        "reward": "None",
                    },
                },
            },
        }
    )


def _install_fake_agent_modules(
    monkeypatch: pytest.MonkeyPatch,
    responses: Sequence[str],
) -> type:
    class FakeCallLLM:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class FakeAgent:
        remaining = list(responses)
        sampling_params: list[dict] = []

        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def step(self, *, max_new_tokens, sampling_params):
            del max_new_tokens
            self.sampling_params.append(sampling_params)
            return self.remaining.pop(0)

        def append(self, message):
            del message

        async def get_data(self):
            return {
                "prompt_ids": [1],
                "response_ids": [2],
                "response_mask": [1],
                "response_logprobs": [-0.1],
                "process_reward_mask": [0],
                "num_turns": 3,
            }

    class FakeAgentLoopMetrics(SimpleNamespace):
        pass

    class FakeAgentLoopOutput(SimpleNamespace):
        pass

    agents_utils = ModuleType("agents.utils")
    agents_utils.Agent = FakeAgent
    agents_utils.CallLLM = FakeCallLLM
    agent_loop = ModuleType("verl.experimental.agent_loop.agent_loop")
    agent_loop.AgentLoopMetrics = FakeAgentLoopMetrics
    agent_loop.AgentLoopOutput = FakeAgentLoopOutput
    monkeypatch.setitem(sys.modules, "agents.utils", agents_utils)
    monkeypatch.setitem(
        sys.modules,
        "verl.experimental.agent_loop.agent_loop",
        agent_loop,
    )
    return FakeAgent


def _run_fake_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    responses: Sequence[str],
):
    (tmp_path / "pkg").mkdir(exist_ok=True)
    (tmp_path / "pkg/widget.py").write_text(
        "class Widget:\n"
        "    def run(self):\n"
        "        pass\n"
    )
    monkeypatch.setattr(
        RepositoryWorkspace,
        "prepare",
        lambda self, episode: tmp_path,
    )
    fake_agent = _install_fake_agent_modules(monkeypatch, responses)
    rollout_config = SimpleNamespace(
        prompt_length=6144,
        response_length=4096,
        plugin=SimpleNamespace(
            workspace_root=str(tmp_path / "unused"),
            simulator="deterministic",
            sanitized_trajectory_dir=None,
        ),
    )
    context = SimpleNamespace(
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(rollout=rollout_config)
        ),
        tokenizer=object(),
        llm_client=SimpleNamespace(loop=object()),
        global_step=1,
    )
    result = asyncio.run(
        run_simplified_rollout(
            item=_training_item(),
            context=context,
            server_manager=object(),
        )
    )
    return result, fake_agent


def test_verl_finish_correction_is_schema_restricted_and_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, fake_agent = _run_fake_rollout(
        tmp_path,
        monkeypatch,
        (
            json.dumps(
                {
                    "tool": "finish",
                    "arguments": {"functions": ["missing.py:Widget.run"]},
                    "reasoning": "first try",
                }
            ),
            json.dumps(
                {
                    "tool": "finish",
                    "arguments": {
                        "functions": ["pkg/widget.py:Widget.run"]
                    },
                    "reasoning": "corrected",
                }
            ),
        ),
    )

    schemas = [
        item["structured_outputs"]["json"]["properties"]["tool"]["enum"]
        for item in fake_agent.sampling_params
    ]
    assert schemas[0] == [
        "list_tree",
        "find_symbol",
        "search_code",
        "read_file",
        "ask_user",
        "finish",
    ]
    assert schemas[1] == ["finish"]
    assert result.extra_fields["predicted_functions"] == [
        "pkg/widget.py:Widget.run"
    ]
    assert result.extra_fields["finish_validation_passed"] is True
    assert result.extra_fields["finish_correction_attempted"] is True
    assert [
        item["attempt"]
        for item in result.extra_fields["sanitized_trajectory"]
    ] == [1, 2]


def test_verl_records_invalid_action_category_without_model_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = "I should inspect the repository before emitting JSON."
    result, _ = _run_fake_rollout(
        tmp_path,
        monkeypatch,
        (
            malformed,
            json.dumps(
                {
                    "tool": "finish",
                    "arguments": {
                        "functions": ["pkg/widget.py:Widget.run"]
                    },
                    "reasoning": "done",
                }
            ),
        ),
    )

    assert result.extra_fields["model_calls"] == 2
    assert result.extra_fields["parsed_actions"] == 1
    assert result.extra_fields["action_parse_rate"] == 0.5
    assert result.extra_fields["invalid_action_categories"] == {
        "invalid_json": 1
    }
    encoded = json.dumps(result.extra_fields["sanitized_trajectory"])
    assert malformed not in encoded
    assert "invalid_json" in encoded


def test_verl_spends_only_one_finish_correction_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejected = "missing.py:Widget.run"
    result, _ = _run_fake_rollout(
        tmp_path,
        monkeypatch,
        (
            json.dumps(
                {
                    "tool": "finish",
                    "arguments": {"functions": [rejected]},
                    "reasoning": "first try",
                }
            ),
            json.dumps(
                {
                    "tool": "ask_user",
                    "arguments": {"question": "Where is it?"},
                    "reasoning": "disallowed correction",
                }
            ),
        ),
    )

    assert result.extra_fields["model_calls"] == 2
    assert result.extra_fields["predicted_functions"] == [rejected]
    assert result.extra_fields["finish_validation_passed"] is False
    assert result.extra_fields["finish_correction_parse_failed"] is True
    assert result.extra_fields["invalid_action_categories"] == {
        "disallowed_tool": 1
    }


def test_episode_from_training_payload_matches_simplified_contract():
    ability = "FuncLocEnv@" + json.dumps(
        {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": "abc",
            "problem_statement": "Precise issue",
            "hints_text": "A hint",
            "patch": "diff",
            "edited_functions": ["pkg/a.py:Thing.run"],
        }
    )
    extra_info = {
        "is_vague": True,
        "prompt": [
            {
                "role": "user",
                "content": "--- BEGIN ISSUE ---\nVague issue\n--- END ISSUE ---",
            }
        ],
        "preference": {
            "preference_name": "no_preference",
            "no_preference": {"preference": "No preference.", "reward": "None"},
        },
    }
    episode = episode_from_training_payload(
        ability=ability,
        extra_info=extra_info,
        source_path="data/train.parquet",
        row_index=7,
    )
    assert episode.visible_issue == "Vague issue"
    assert episode.full_issue == "Precise issue"
    assert episode.expected_functions == ("pkg/a.py:Thing.run",)
    assert episode.row_index == 7
