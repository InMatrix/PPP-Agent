from pathlib import Path

import pytest

from ppp_simplified.evaluation import (
    BatchEvaluator,
    build_manifest,
    compare_summaries,
    summarize_records,
    validate_comparable_manifests,
)
from ppp_simplified.models import Episode, Preference


def test_summarize_records_aggregates_interaction_metrics() -> None:
    records = [
        {
            "status": "completed",
            "duration_seconds": 2.0,
            "episode": {
                "instance_id": "one",
                "preference": "concise_question",
                "inference_seed": 11,
            },
            "report": {
                "predicted_functions": ["pkg/mod.py:run"],
                "termination": "natural_finish",
                "duplicate_actions_suppressed": 0,
                "finish_validation_passed": True,
                "finish_correction_attempted": False,
                "invalid_predictions": [],
                "trajectory": [
                    {"action": {"tool": "search_code"}},
                    {"action": {"tool": "finish"}},
                ],
                "reward": {
                    "productivity": 1.0,
                    "proactivity_adjustment": -0.2,
                    "personalization_adjustment": 0.05,
                    "total": 0.85,
                    "questions_asked": 1,
                    "disclosure_levels": [3],
                    "preference_ok": True,
                },
            },
        },
        {
            "status": "completed",
            "duration_seconds": 4.0,
            "episode": {
                "instance_id": "two",
                "preference": "no_ask",
                "inference_seed": 22,
            },
            "report": {
                "predicted_functions": [],
                "termination": "turn_limit",
                "duplicate_actions_suppressed": 2,
                "finish_validation_passed": False,
                "finish_correction_attempted": True,
                "invalid_predictions": ["missing.py:run"],
                "trajectory": [{"action": {"tool": "search_code"}}],
                "reward": {
                    "productivity": 0.0,
                    "proactivity_adjustment": -0.1,
                    "personalization_adjustment": 0.0,
                    "total": 0.0,
                    "questions_asked": 0,
                    "disclosure_levels": [],
                    "preference_ok": None,
                },
            },
        },
    ]

    summary = summarize_records(records)

    assert summary["episodes_completed"] == 2
    assert summary["exact_localization_rate"] == 0.5
    assert summary["mean_productivity_f1"] == 0.5
    assert summary["agent_finish_rate"] == 0.5
    assert summary["natural_finish_rate"] == 0.5
    assert summary["deadline_finish_rate"] == 0.0
    assert summary["turn_limit_rate"] == 0.5
    assert summary["empty_prediction_rate"] == 0.5
    assert summary["mean_duplicate_actions_suppressed"] == 1.0
    assert summary["duplicate_action_episode_rate"] == 0.5
    assert summary["question_rate"] == 0.5
    assert summary["mean_disclosure_level_per_question"] == 3.0
    assert summary["preference_compliance_rate_when_judged"] == 1.0
    assert summary["mean_turns"] == 1.5
    assert summary["mean_duration_seconds"] == 3.0
    assert summary["finish_validation_pass_rate"] == 0.5
    assert summary["finish_correction_rate"] == 0.5
    assert summary["invalid_prediction_episode_rate"] == 0.5
    assert summary["by_inference_seed"]["11"]["episodes_completed"] == 1
    assert summary["by_inference_seed"]["22"]["episodes_completed"] == 1


def test_manifest_expands_episodes_across_inference_seeds() -> None:
    episodes = tuple(
        Episode(
            instance_id=f"task-{index}",
            repository=f"org/repo-{index}",
            base_commit=f"abc{index}",
            visible_issue="Issue",
            full_issue="Issue",
            hint="",
            patch="",
            expected_functions=(f"pkg/mod.py:run_{index}",),
            is_vague=True,
            preference=Preference(
                name="concise_question",
                description="Be concise.",
                reward_rule="Comply.",
            ),
            source_path="fixture",
            row_index=index,
        )
        for index in range(2)
    )

    manifest = build_manifest(
        episodes=episodes,
        mode="live",
        model="qwen",
        simulator="gemini",
        max_turns=8,
        sample_seed=7,
        inference_seeds=(11, 22),
        policy_label="termination-v1",
        policy_version="navigation-v2",
        tool_schema_version="v1",
        code_revision="abc123",
    )

    assert manifest["sample_seed"] == 7
    assert manifest["schema_version"] == 3
    assert manifest["inference_seeds"] == [11, 22]
    assert manifest["policy_version"] == "navigation-v2"
    assert len(manifest["cases"]) == 4
    assert [case["inference_seed"] for case in manifest["cases"]] == [
        11,
        11,
        22,
        22,
    ]


def test_resume_rejects_a_different_manifest(tmp_path: Path) -> None:
    evaluator = BatchEvaluator(
        workspace_root=tmp_path / "workspaces",
        output_dir=tmp_path / "results",
        provider_factory=lambda episode, seed: None,
        simulator_factory=lambda episode: None,
        max_turns=8,
    )
    manifest = {
        "schema_version": 2,
        "mode": "offline",
        "model": "scripted",
        "simulator": "deterministic",
        "inference_seeds": [11],
        "policy_label": "termination-v1",
    }
    evaluator.run((), (11,), manifest)

    with pytest.raises(ValueError, match="different evaluation manifest"):
        evaluator.run(
            (),
            (11,),
            {
                "schema_version": 2,
                "mode": "offline",
                "model": "scripted",
                "simulator": "deterministic",
                "inference_seeds": [11],
                "policy_label": "navigation-v2",
            },
        )


def test_comparison_applies_the_4b_decision_rule() -> None:
    control = {
        "episodes_requested": 12,
        "episodes_completed": 12,
        "episodes_failed": 0,
        "mean_productivity_f1": 0.20,
        "mean_total_reward": 0.18,
        "preference_compliance_rate_when_judged": 0.75,
        "empty_prediction_rate": 0.0,
    }
    candidate = {
        "episodes_requested": 12,
        "episodes_completed": 12,
        "episodes_failed": 0,
        "mean_productivity_f1": 0.35,
        "mean_total_reward": 0.30,
        "preference_compliance_rate_when_judged": 1.0,
        "empty_prediction_rate": 0.0,
    }

    comparison = compare_summaries(control, candidate)

    assert comparison["metrics"]["mean_productivity_f1"]["delta"] == (
        pytest.approx(0.15)
    )
    assert comparison["decision"]["completion_ok"] is True
    assert comparison["decision"]["recommendation"] == "continue_with_4b"


def test_comparison_recommends_larger_model_below_f1_threshold() -> None:
    control = {
        "episodes_requested": 12,
        "episodes_completed": 12,
        "episodes_failed": 0,
        "mean_productivity_f1": 0.25,
        "mean_total_reward": 0.25,
        "preference_compliance_rate_when_judged": 1.0,
        "empty_prediction_rate": 0.0,
    }
    candidate = dict(control, mean_productivity_f1=0.30)

    comparison = compare_summaries(control, candidate)

    assert comparison["decision"]["f1_gain_met"] is False
    assert comparison["decision"]["recommendation"] == "compare_larger_model"


def test_comparison_requires_identical_frozen_suite_cases() -> None:
    suite = {
        "name": "heldout-v1",
        "role": "heldout",
        "source_path": "data/test_ood.parquet",
        "source_sha256": "abc123",
    }
    control = {
        "evaluation_suite": suite,
        "cases": [
            {
                "row_index": 10,
                "instance_id": "task-1",
                "preference": "json",
                "inference_seed": 11,
            }
        ],
    }
    candidate = {
        "evaluation_suite": dict(suite),
        "cases": [dict(control["cases"][0])],
    }

    assert validate_comparable_manifests(control, candidate) == suite

    candidate["cases"][0]["inference_seed"] = 22
    with pytest.raises(ValueError, match="different evaluation cases"):
        validate_comparable_manifests(control, candidate)


def test_comparison_rejects_cross_suite_results() -> None:
    control = {
        "evaluation_suite": {
            "name": "dev-v1",
            "role": "development",
            "source_path": "data/test_id.parquet",
            "source_sha256": "one",
        },
        "cases": [],
    }
    candidate = {
        "evaluation_suite": {
            "name": "heldout-v1",
            "role": "heldout",
            "source_path": "data/test_ood.parquet",
            "source_sha256": "two",
        },
        "cases": [],
    }

    with pytest.raises(ValueError, match="different evaluation suites"):
        validate_comparable_manifests(control, candidate)
