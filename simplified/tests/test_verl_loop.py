import json
from pathlib import Path

import pytest

from ppp_simplified.data import episode_from_training_payload
from ppp_simplified.verl_loop import _one, _parse_action, _rollout_identity


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
