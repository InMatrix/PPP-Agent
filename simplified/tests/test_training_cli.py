import hashlib
import json
from pathlib import Path

import pytest

from ppp_simplified.training import PPPTrainingConfig, TrainingExampleRef, TrainingSubsetManifest, build_training_artifact
from ppp_simplified.training_cli import (
    DEFAULT_LIVE_INFERENCE_SEEDS,
    _model_id,
    _require_group_seeds,
    _markdown_summary,
    _validate_artifact,
    build_parser,
    command_contract_gate,
    command_schema_gate,
    command_summarize_run,
    command_check_extension,
    command_verify_continuation,
    command_train,
    command_evaluate,
    command_export,
)


def subset() -> TrainingSubsetManifest:
    return TrainingSubsetManifest(
        schema_version=1,
        dataset="data/train_12n1.parquet",
        dataset_sha256="abc",
        selection_seed=42,
        issue_groups=("owner__repo-1",),
        examples=(TrainingExampleRef("owner__repo-1", "no_preference", True, 0, "data/train_12n1.parquet"),),
    )


def artifact() -> dict:
    return build_training_artifact(
        config=PPPTrainingConfig(),
        subset=subset(),
        stage="deterministic_smoke",
        reports=[],
        metrics={"group_size": 8, "group_advantages": [0.0] * 8},
    )


def test_parser_exposes_local_and_guarded_gpu_commands():
    parser = build_parser()
    assert {"doctor", "prepare", "contract-gate", "schema-gate", "smoke", "live-group", "export", "summarize-run", "check-extension", "verify-continuation", "train", "evaluate"} <= set(parser._subparsers._group_actions[0].choices)
    train = parser.parse_args(["train", "--steps", "1", "--simulator", "deterministic"])
    assert train.steps == 1
    assert train.execute is False
    assert train.model == "qwen35"
    assert parser.parse_args(["train", "--steps", "2"]).steps == 2
    live = parser.parse_args(["live-group"])
    assert live.inference_seeds == DEFAULT_LIVE_INFERENCE_SEEDS
    assert live.model == "qwen35"
    schema = parser.parse_args(["schema-gate"])
    assert schema.model == "qwen35"
    assert schema.max_model_len == 2048
    assert schema.max_tokens == 256
    assert schema.execute is False


def test_contract_gate_writes_sanitized_summary(monkeypatch, tmp_path: Path):
    payload = {
        "schema_version": 1,
        "stage": "adapter_to_foldgrpo_contract",
        "status": "passed",
        "group_size": 8,
    }
    monkeypatch.setattr(
        "ppp_simplified.verl_contract.run_verl_contract_gate",
        lambda: payload,
    )
    output = tmp_path / "contract.json"
    assert command_contract_gate(
        type("Args", (), {"output": output})()
    ) == 0
    assert json.loads(output.read_text()) == payload


def test_schema_gate_routes_model_and_bounded_limits(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    calls = []
    payload = {
        "status": "passed",
        "model_id": "Qwen/Qwen3.5-4B",
        "cases": [],
    }

    def fake_gate(**kwargs):
        calls.append(kwargs)
        return payload

    monkeypatch.setattr(
        "ppp_simplified.vllm_schema_gate.run_vllm_schema_gate",
        fake_gate,
    )
    monkeypatch.setenv(
        "CONFIRM_PAID_TRAINING",
        "I_UNDERSTAND_LAMBDA_IS_BILLING",
    )
    output = tmp_path / "schema.json"
    assert command_schema_gate(
        type(
            "Args",
            (),
            {
                "model": "qwen35",
                "output": output,
                "max_model_len": 2048,
                "max_tokens": 256,
                "gpu_memory_utilization": 0.2,
                "execute": True,
            },
        )()
    ) == 0
    assert calls == [
        {
            "model_id": "Qwen/Qwen3.5-4B",
            "output_path": output,
            "max_model_len": 2048,
            "max_tokens": 256,
            "gpu_memory_utilization": 0.2,
        }
    ]
    assert "schema gate artifact" in capsys.readouterr().out


def test_schema_gate_defaults_to_nonexecuting_paid_run(capsys):
    assert command_schema_gate(
        type("Args", (), {"execute": False})()
    ) == 0
    assert "Dry run only" in capsys.readouterr().out


def test_schema_gate_requires_explicit_paid_confirmation(monkeypatch):
    monkeypatch.delenv("CONFIRM_PAID_TRAINING", raising=False)
    with pytest.raises(ValueError, match="CONFIRM_PAID_TRAINING"):
        command_schema_gate(
            type("Args", (), {"execute": True})()
        )


def test_live_group_requires_eight_distinct_seeds_and_model_routes():
    _require_group_seeds(DEFAULT_LIVE_INFERENCE_SEEDS, 8)
    with pytest.raises(ValueError, match="exactly 8"):
        _require_group_seeds((11, 22), 8)
    with pytest.raises(ValueError, match="distinct"):
        _require_group_seeds((11,) * 8, 8)
    assert _model_id("qwen35") == "Qwen/Qwen3.5-4B"
    assert _model_id("qwen3") == "Qwen/Qwen3-4B"


def test_export_validates_sanitized_artifact_and_writes_summary(tmp_path: Path):
    source = tmp_path / "source.json"
    output = tmp_path / "exported.json"
    summary = tmp_path / "summary.md"
    source.write_text(json.dumps(artifact()))
    command_export(type("Args", (), {"artifact": source, "output": output, "summary": summary})())
    assert json.loads(output.read_text())["stage"] == "deterministic_smoke"
    assert "Group size: `8`" in summary.read_text()


def test_export_rejects_secret_like_keys():
    payload = artifact()
    payload["nested"] = {"api_key": "not-a-real-value"}
    with pytest.raises(ValueError, match="secret-like"):
        _validate_artifact(payload)
    payload = artifact()
    payload["unlabeled"] = "AIza0123456789abcdefghijklmnopqrstuv"
    with pytest.raises(ValueError, match="secret-like"):
        _validate_artifact(payload)


def test_markdown_summary_has_no_repository_contents():
    summary = _markdown_summary(artifact())
    assert "PPP training artifact" in summary
    assert "repo-1" not in summary


def test_evaluate_prints_adapter_serve_and_seeded_suite(tmp_path: Path, capsys):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    result = command_evaluate(
        type(
            "Args",
            (),
            {
                "adapter": adapter,
                "qwen_base_url": "http://localhost:8000/v1",
                "output_dir": tmp_path / "evaluation",
                "model": "qwen35",
            },
        )()
    )
    output = capsys.readouterr().out
    assert result == 0
    assert "--lora-modules trained=" in output
    assert "--language-model-only" in output
    assert "--inference-seeds 11,22,33" in output


def test_train_uses_argument_list_not_shell(monkeypatch, tmp_path: Path):
    calls = []

    class Result:
        returncode = 0

    def fake_run(command, *, check, env):
        calls.append((command, check, env["PPP_PYTHON"]))
        return Result()

    monkeypatch.setattr("ppp_simplified.training_cli.subprocess.run", fake_run)
    assert command_train(
        type("Args", (), {"execute": False, "steps": 1, "simulator": "deterministic", "model": "qwen35", "projected_compute_usd": None})()
    ) == 0
    assert calls == [
        (
            [
                "bash",
                "simplified/scripts/run_ppp_rl_4b.sh",
                "--print-command",
                "--steps",
                "1",
                "--simulator",
                "deterministic",
                "--model",
                "qwen35",
            ],
            False,
            __import__("sys").executable,
        )
    ]


def test_summarize_run_calculates_group_advantages(tmp_path: Path):
    run_dir = tmp_path / "run"
    trajectories = run_dir / "sanitized-trajectories"
    trajectories.mkdir(parents=True)
    PPPTrainingConfig().write_json(run_dir / "training-config.json")
    (run_dir / "training-subset.json").write_text(
        json.dumps(
            {
                "dataset_sha256": "abc",
                "selection_seed": 42,
                "issue_groups": ["owner__repo-1"],
            }
        )
    )
    for index, reward in enumerate((0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)):
        (trajectories / f"{index}.json").write_text(
            json.dumps(
                {
                    "group_id": "prompt-1",
                    "instance_id": "owner__repo-1",
                    "predicted_functions": [],
                    "reward_breakdown": {"total": reward},
                    "sanitized_trajectory": [],
                    "termination": "natural_finish",
                    "model_calls": 2,
                    "parsed_actions": 1 if index == 0 else 2,
                    "action_parse_rate": 0.5 if index == 0 else 1.0,
                    "invalid_action_count": 1 if index == 0 else 0,
                    "invalid_action_categories": (
                        {"invalid_json": 1} if index == 0 else {}
                    ),
                    "schema_constrained_model_calls": 2,
                    "finish_validation_passed": True,
                    "finish_correction_attempted": False,
                    "finish_correction_parse_failed": False,
                    "invalid_predictions": [],
                    "model_generated_tokens": 10,
                    "environment_tokens": 4,
                    "latency_seconds": 1.0,
                    "simulator_live_calls": 0,
                }
            )
        )
    assert command_summarize_run(
        type(
            "Args",
            (),
            {"run_dir": run_dir, "compute_seconds": 3600, "hourly_usd": 1.09},
        )()
    ) == 0
    report = json.loads((run_dir / "training-report.json").read_text())
    group = report["metrics"]["groups"][0]
    assert len(group["advantages"]) == 8
    assert sum(group["advantages"]) == pytest.approx(0.0)
    assert report["metrics"]["groups_with_nonzero_reward_variance"] == 1
    assert report["metrics"]["estimated_compute_usd"] == pytest.approx(1.09)
    assert report["metrics"]["action_parse_rate"] == pytest.approx(15 / 16)
    assert report["metrics"]["invalid_action_count"] == 1
    assert report["metrics"]["invalid_action_categories"] == {
        "invalid_json": 1
    }
    assert report["metrics"]["valid_finish_count"] == 8
    assert report["metrics"]["schema_constrained_model_calls"] == 16

    # A crashed/resumed step may leave a partial or duplicate group. Never
    # count it as a completed FoldGRPO update.
    (trajectories / "duplicate.json").write_text(
        (trajectories / "0.json").read_text()
    )
    with pytest.raises(ValueError, match="expected exactly 8"):
        command_summarize_run(
            type(
                "Args",
                (),
                {
                    "run_dir": run_dir,
                    "compute_seconds": 3600,
                    "hourly_usd": 1.09,
                },
            )()
        )


def test_extension_gate_checks_variance_resume_metrics_adapter_and_budget(tmp_path: Path):
    run_dir = tmp_path / "run"
    adapter = run_dir / "global_step_20" / "actor" / "lora_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    (run_dir / "resume-verified").touch()
    groups = [
        {"group_id": str(index), "rewards": [0, 1] if index < 5 else [0, 0]}
        for index in range(20)
    ]
    (run_dir / "training-report.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "groups": groups,
                    "groups_with_nonzero_reward_variance": 5,
                }
            }
        )
    )
    (run_dir / "training-metrics.jsonl").write_text(
        json.dumps({"data": {"actor/pg_loss": 0.1, "actor/ppo_kl": 0.01}})
        + "\n"
    )
    assert command_check_extension(
        type(
            "Args",
            (),
            {"run_dir": run_dir, "projected_compute_usd": 44.0},
        )()
    ) == 0
    with pytest.raises(ValueError, match=r"between \$0 and \$45"):
        command_check_extension(
            type(
                "Args",
                (),
                {"run_dir": run_dir, "projected_compute_usd": 46.0},
            )()
        )


def test_continuation_gate_requires_adapter_delta_and_positive_gradient(tmp_path: Path):
    run_dir = tmp_path / "run"
    before = run_dir / "global_step_1/actor/lora_adapter/adapter_model.safetensors"
    after = run_dir / "global_step_2/actor/lora_adapter/adapter_model.safetensors"
    before.parent.mkdir(parents=True)
    after.parent.mkdir(parents=True)
    before.write_bytes(b"before")
    after.write_bytes(b"after")
    trajectories = run_dir / "sanitized-trajectories"
    trajectories.mkdir()
    for index in range(16):
        (trajectories / f"{index}.json").write_text("{}")
    metrics_path = run_dir / "training-metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "step": 2,
                "data": {
                    "actor/pg_loss": 0.01,
                    "actor/grad_norm": 0.2,
                    "rollout_corr/kl": 0.001,
                },
            }
        )
        + "\n"
    )
    expected_before = hashlib.sha256(b"before").hexdigest()
    output = tmp_path / "continuation.json"
    args = type(
        "Args",
        (),
        {
            "run_dir": run_dir,
            "from_step": 1,
            "to_step": 2,
            "baseline_trajectories": 8,
            "expected_before_sha256": expected_before,
            "output": output,
        },
    )()

    assert command_verify_continuation(args) == 0
    assert json.loads(output.read_text())["adapter_changed"] is True

    args.baseline_trajectories = 7
    with pytest.raises(ValueError, match="exactly eight baseline"):
        command_verify_continuation(args)
    args.baseline_trajectories = 8

    metrics_path.write_text(
        json.dumps(
            {
                "step": 2,
                "data": {
                    "actor/pg_loss": 0.0,
                    "actor/grad_norm": 0.0,
                    "rollout_corr/kl": 0.0,
                },
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="gradient norm must be positive"):
        command_verify_continuation(args)
