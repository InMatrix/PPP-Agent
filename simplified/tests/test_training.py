import json
from dataclasses import replace
from pathlib import Path

import pytest

from ppp_simplified.models import (
    AgentAction,
    Episode,
    Preference,
    RewardBreakdown,
    RunReport,
    TrajectoryStep,
)
from ppp_simplified.training import (
    PPPTrainingConfig,
    TrainingExampleRef,
    TrainingSubsetManifest,
    build_training_artifact,
    group_advantages,
    materialize_training_subset,
    model_token_mask,
    sanitize_observation,
    sanitize_report,
    terminal_reward_vector,
    write_training_artifact,
)


def example_report() -> RunReport:
    episode = Episode(
        instance_id="owner__repo-1",
        repository="owner/repo",
        base_commit="secret-commit",
        visible_issue="Something fails.",
        full_issue="Hidden precise issue.",
        hint="Hidden hint.",
        patch="hidden patch",
        expected_functions=("pkg/mod.py:Thing.run",),
        is_vague=True,
        preference=Preference("no_preference", "No preference.", "None"),
        source_path="data/train.parquet",
        row_index=3,
    )
    return RunReport(
        episode=episode,
        model="scripted",
        simulator="deterministic",
        predicted_functions=("pkg/mod.py:Thing.run",),
        reward=RewardBreakdown(1.0, 0.05, 0.0, 1.0, 1, (1,), None),
        trajectory=(
            TrajectoryStep(
                turn=1,
                action=AgentAction(
                    "ask_user",
                    {"question": "Which path? The package path."},
                    "The package path.",
                ),
                observation="The package path.",
            ),
        ),
        workspace="/private/workspace",
        model_calls=1,
    )


def subset() -> TrainingSubsetManifest:
    return TrainingSubsetManifest(
        schema_version=1,
        dataset="data/train.parquet",
        dataset_sha256="abc",
        selection_seed=42,
        issue_groups=("owner__repo-1",),
        examples=(
            TrainingExampleRef(
                "owner__repo-1", "no_preference", True, 3, "data/train.parquet"
            ),
        ),
    )


def test_training_config_locks_method_invariants():
    config = PPPTrainingConfig()
    assert config.group_size == 8
    assert config.clip_ratio_high > config.clip_ratio_low
    assert len(config.identity) == 64
    with pytest.raises(ValueError, match="group size"):
        replace(config, group_size=4)


def test_group_advantages_match_sample_normalization():
    advantages = group_advantages((0.0, 0.0, 1.0, 1.0))
    assert sum(advantages) == pytest.approx(0.0)
    assert advantages[0] < 0 < advantages[-1]
    assert group_advantages((0.5,) * 8) == (0.0,) * 8


def test_model_mask_and_terminal_reward_ignore_environment_tokens():
    mask = model_token_mask((("model", 3), ("environment", 4), ("model", 2)))
    assert mask == (1, 1, 1, 0, 0, 0, 0, 1, 1)
    assert terminal_reward_vector(response_mask=mask, reward=0.75) == (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.75,
    )


def test_sanitized_report_excludes_oracle_and_workspace_fields():
    payload = sanitize_report(example_report())
    encoded = json.dumps(payload)
    assert "Hidden precise issue" not in encoded
    assert "Hidden hint" not in encoded
    assert "hidden patch" not in encoded
    assert "secret-commit" not in encoded
    assert "/private/workspace" not in encoded
    assert "The package path." not in encoded
    assert "Which path?" not in encoded
    assert "simulator reply" in payload["trajectory"][0]["observation"]
    assert "agent question" in payload["trajectory"][0]["arguments"]["question"]
    assert "model reasoning" in payload["trajectory"][0]["reasoning"]
    assert payload["trajectory"][0]["tool"] == "ask_user"


def test_observation_sanitizer_never_exports_source_or_hidden_reply():
    source = "def secret_implementation():\n    return hidden_value"
    hidden_reply = "The exact edited function is package.py:Class.method"
    assert source not in sanitize_observation("read_file", source)
    assert hidden_reply not in sanitize_observation("ask_user", hidden_reply)
    assert "repository observation" in sanitize_observation("read_file", source)
    assert "simulator reply" in sanitize_observation("ask_user", hidden_reply)


def test_report_sanitizer_redacts_model_text_that_echoes_source():
    source = "def secret_implementation(): return hidden_value"
    report = replace(
        example_report(),
        trajectory=(
            TrajectoryStep(
                turn=1,
                action=AgentAction(
                    "search_code",
                    {"query": source, "path": "pkg", "glob": "*.py"},
                    source,
                ),
                observation=f"pkg/a.py:1:{source}",
            ),
        ),
    )
    payload = sanitize_report(report)
    encoded = json.dumps(payload)
    assert source not in encoded
    assert payload["trajectory"][0]["arguments"]["path"] == "pkg"
    assert "search query" in payload["trajectory"][0]["arguments"]["query"]


def test_artifact_identities_and_secret_guard(tmp_path: Path):
    payload = build_training_artifact(
        config=PPPTrainingConfig(),
        subset=subset(),
        stage="deterministic_smoke",
        reports=[example_report()],
        metrics={"group_advantages": [0.0] * 8},
    )
    output = tmp_path / "artifact.json"
    write_training_artifact(output, payload)
    assert json.loads(output.read_text())["stage"] == "deterministic_smoke"
    with pytest.raises(ValueError, match="secret"):
        write_training_artifact(
            tmp_path / "bad.json",
            {**payload, "GEMINI_API_KEY": "do-not-write"},
        )


def test_materialize_subset_uses_only_manifest_rows(tmp_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"value": [10, 20, 30]}), source)
    import hashlib

    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = TrainingSubsetManifest(
        schema_version=1,
        dataset=str(source),
        dataset_sha256=checksum,
        selection_seed=42,
        issue_groups=("a",),
        examples=(
            TrainingExampleRef("a", "no_preference", True, 2, str(source)),
            TrainingExampleRef("a", "concise_question", True, 0, str(source)),
        ),
    )
    output = tmp_path / "selected.parquet"
    materialize_training_subset(source, manifest, output)
    assert pq.read_table(output).column("value").to_pylist() == [10, 30]
