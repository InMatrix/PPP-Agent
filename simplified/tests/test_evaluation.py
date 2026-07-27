from ppp_simplified.evaluation import build_manifest, summarize_records
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
        tool_schema_version="v1",
        code_revision="abc123",
    )

    assert manifest["sample_seed"] == 7
    assert manifest["inference_seeds"] == [11, 22]
    assert len(manifest["cases"]) == 4
    assert [case["inference_seed"] for case in manifest["cases"]] == [
        11,
        11,
        22,
        22,
    ]
