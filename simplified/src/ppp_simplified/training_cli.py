"""Prepare, run, evaluate, and inspect the teaching-scale PPP RL workflow.

Local commands stop before an optimizer. The guarded ``train --execute`` path
is reserved for the paid Lambda compatibility and training gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .cli import load_local_environment
from .data import load_episode
from .providers import OpenAICompatibleAgent, ScriptedSmokeAgent
from .runner import AgentRunner
from .simulator import (
    CachedUserSimulator,
    DeterministicUserSimulator,
    GeminiUserSimulator,
)
from .training import (
    PPPTrainingConfig,
    TrainingSubsetManifest,
    build_training_artifact,
    group_advantages,
    materialize_training_subset,
    select_training_subset,
    write_training_artifact,
)
from .workspace import RepositoryWorkspace


DEFAULT_DATA = Path("data/train_12n1.parquet")
DEFAULT_OUTPUT_DIR = Path("simplified/results/training")
SECRET_KEY_SUFFIXES = ("api_key", "apikey", "_secret", "_password")
DEFAULT_LIVE_INFERENCE_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)


def _print_check(name: str, ready: bool, detail: str) -> None:
    print(f"{'PASS' if ready else 'WAIT'}  {name:<16} {detail}")


def _optional_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _model_id(route: str) -> str:
    config = PPPTrainingConfig()
    if route == "qwen35":
        return config.model_id
    if route == "qwen3":
        return config.fallback_model_id
    raise ValueError(f"Unsupported model route: {route}")


def _parse_inference_seeds(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--inference-seeds must be comma-separated integers"
        ) from error


def _require_group_seeds(seeds: tuple[int, ...], group_size: int) -> None:
    if len(seeds) != group_size:
        raise ValueError(
            f"A live PPP group requires exactly {group_size} inference seeds."
        )
    if len(set(seeds)) != group_size:
        raise ValueError("A live PPP group requires eight distinct inference seeds.")


def _workspace_for(args: argparse.Namespace, episode, *, label: str) -> Path:
    if args.workspace is None:
        return RepositoryWorkspace(args.workspace_root).prepare(episode)
    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"{label} workspace does not exist: {workspace}")
    return workspace


def command_doctor(args: argparse.Namespace) -> int:
    """Report readiness without importing CUDA-heavy dependencies or secrets."""

    checks = [
        ("dataset", args.data.is_file(), str(args.data)),
        ("git", shutil.which("git") is not None, shutil.which("git") or "missing"),
        ("Gemini", bool(os.getenv("GEMINI_API_KEY")), "configured" if os.getenv("GEMINI_API_KEY") else "GEMINI_API_KEY missing"),
        ("Verl", _optional_package("verl"), "installed" if _optional_package("verl") else "optional: install on Lambda"),
    ]
    torch_present = _optional_package("torch")
    cuda_ready = False
    cuda_detail = "optional: PyTorch not installed"
    if torch_present:
        try:
            import torch  # type: ignore[import-not-found]

            cuda_ready = bool(torch.cuda.is_available())
            cuda_detail = (
                torch.cuda.get_device_name(0) if cuda_ready else "PyTorch installed; CUDA unavailable"
            )
        except Exception as error:  # pragma: no cover - environment dependent
            cuda_detail = f"PyTorch probe failed: {error}"
    checks.append(("CUDA", cuda_ready, cuda_detail))
    for check in checks:
        _print_check(*check)
    # Only local preparation prerequisites determine this command's status.
    return 0 if all(ready for _, ready, _ in checks[:2]) else 1


def command_prepare(args: argparse.Namespace) -> int:
    config = PPPTrainingConfig()
    subset, _ = select_training_subset(
        args.data,
        issue_groups=config.training_issue_groups,
        seed=args.selection_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.write_json(args.output_dir / "training-config.json")
    subset.write_json(args.output_dir / "training-subset.json")
    materialized = args.output_dir / "train-16x13.parquet"
    checksum = materialize_training_subset(args.data, subset, materialized)
    print(f"configuration: {args.output_dir / 'training-config.json'}")
    print(f"subset: {args.output_dir / 'training-subset.json'}")
    print(f"training parquet: {materialized} ({checksum[:12]})")
    print(f"issue groups: {len(subset.issue_groups)}; examples: {len(subset.examples)}")
    return 0


def command_contract_gate(args: argparse.Namespace) -> int:
    """Run the synthetic Verl/FoldGRPO contract without loading a model."""

    from .verl_contract import run_verl_contract_gate

    payload = run_verl_contract_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"contract artifact: {args.output}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_schema_gate(args: argparse.Namespace) -> int:
    """Run two real vLLM schema-constrained completions without Verl."""

    if not args.execute:
        print(
            "Dry run only. This command loads a real model. After explicit "
            "paid-compute confirmation, set "
            "CONFIRM_PAID_TRAINING=I_UNDERSTAND_LAMBDA_IS_BILLING and add "
            "--execute."
        )
        return 0
    if (
        os.getenv("CONFIRM_PAID_TRAINING")
        != "I_UNDERSTAND_LAMBDA_IS_BILLING"
    ):
        raise ValueError(
            "Set CONFIRM_PAID_TRAINING=I_UNDERSTAND_LAMBDA_IS_BILLING "
            "before executing the real vLLM schema gate."
        )
    from .vllm_schema_gate import run_vllm_schema_gate

    payload = run_vllm_schema_gate(
        model_id=_model_id(args.model),
        output_path=args.output,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(f"schema gate artifact: {args.output}")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "model_id": payload["model_id"],
                "cases": payload["cases"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_subset_for_group(
    args: argparse.Namespace,
    episode,
) -> TrainingSubsetManifest:
    manifest_path = args.subset_manifest
    if manifest_path is not None:
        payload = json.loads(manifest_path.read_text())
        # Recreate the dataclass only to retain its stable identity contract.
        from .training import TrainingExampleRef

        return TrainingSubsetManifest(
            schema_version=int(payload["schema_version"]),
            dataset=str(payload["dataset"]),
            dataset_sha256=str(payload["dataset_sha256"]),
            selection_seed=int(payload["selection_seed"]),
            issue_groups=tuple(str(value) for value in payload["issue_groups"]),
            examples=tuple(TrainingExampleRef(**item) for item in payload["examples"]),
        )
    from .training import TrainingExampleRef

    hasher = hashlib.sha256()
    with args.data.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    return TrainingSubsetManifest(
        schema_version=1,
        dataset=str(args.data),
        dataset_sha256=digest,
        selection_seed=args.selection_seed,
        issue_groups=(episode.instance_id,),
        examples=(
            TrainingExampleRef(
                episode.instance_id,
                episode.preference.name,
                episode.is_vague,
                episode.row_index,
                episode.source_path,
            ),
        ),
    )


def _episode_from_args(args: argparse.Namespace):
    return load_episode(
        args.data,
        row_index=args.row_index,
        instance_id=None if args.row_index is not None else args.instance_id,
        preference_name=None if args.row_index is not None else args.preference,
        vague=None if args.row_index is not None else not args.precise,
    )


def command_smoke(args: argparse.Namespace) -> int:
    config = PPPTrainingConfig()
    episode = _episode_from_args(args)
    subset = _load_subset_for_group(args, episode)
    workspace = _workspace_for(args, episode, label="Smoke")

    reports = []
    for _ in range(config.group_size):
        reports.append(
            AgentRunner(
                provider=ScriptedSmokeAgent(episode.expected_functions),
                simulator=DeterministicUserSimulator(),
                max_turns=config.max_turns,
                policy_version=config.policy_version,
            ).run(episode, workspace)
        )
    rewards = [report.reward.total for report in reports]
    payload = build_training_artifact(
        config=config,
        subset=subset,
        stage="deterministic_smoke",
        reports=reports,
        metrics={
            "group_size": config.group_size,
            "group_rewards": rewards,
            "group_advantages": list(group_advantages(rewards)),
            "simulator_calls": sum(report.reward.questions_asked for report in reports),
            "deterministic": True,
        },
    )
    write_training_artifact(args.output, payload)
    print(f"artifact: {args.output}")
    print(json.dumps(payload["metrics"], indent=2))
    return 0


def command_live_group(args: argparse.Namespace) -> int:
    """Run eight real policy trajectories without constructing an optimizer."""

    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is required for a live rollout group.")
    config = PPPTrainingConfig()
    episode = _episode_from_args(args)
    subset = _load_subset_for_group(args, episode)
    _require_group_seeds(args.inference_seeds, config.group_size)
    workspace = _workspace_for(args, episode, label="Live-group")
    simulator = CachedUserSimulator(
        GeminiUserSimulator(model=config.simulator_model),
        cache_path=args.simulator_cache,
        max_live_calls=args.simulator_call_budget,
    )
    reports = []
    for seed in args.inference_seeds:
        reports.append(
            AgentRunner(
                provider=OpenAICompatibleAgent(
                    model=_model_id(args.model),
                    base_url=args.qwen_base_url,
                    seed=seed,
                    tool_schema_version=config.tool_schema_version,
                ),
                simulator=simulator,
                max_turns=config.max_turns,
                policy_version=config.policy_version,
            ).run(episode, workspace)
        )
    rewards = [report.reward.total for report in reports]
    payload = build_training_artifact(
        config=config,
        subset=subset,
        stage="live_group",
        reports=reports,
        metrics={
            "group_size": config.group_size,
            "group_rewards": rewards,
            "group_advantages": list(group_advantages(rewards)),
            "model_calls": sum(report.model_calls for report in reports),
            "simulator_live_calls": simulator.live_calls,
            "deterministic": False,
        },
    )
    write_training_artifact(args.output, payload)
    print(f"artifact: {args.output}")
    print(json.dumps(payload["metrics"], indent=2))
    return 0


def _has_secret_key(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).lower().replace("-", "_")
            # ``response_tokens`` is expected training metadata, so reject
            # credential-shaped keys rather than every key containing "token".
            if (
                normalized in {"token", "secret", "password"}
                or normalized.endswith(SECRET_KEY_SUFFIXES)
            ):
                return True
            if _has_secret_key(value):
                return True
    elif isinstance(payload, list):
        return any(_has_secret_key(value) for value in payload)
    return False


def _has_secret_value(payload: Any) -> bool:
    """Catch common credential values even when a caller chose a bland key."""

    if isinstance(payload, dict):
        return any(_has_secret_value(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_has_secret_value(value) for value in payload)
    if isinstance(payload, str):
        return bool(
            re.search(r"AIza[0-9A-Za-z_-]{20,}", payload)
            or re.search(r"\bsk-[0-9A-Za-z_-]{16,}\b", payload)
        )
    return False


def _validate_artifact(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Artifact must be a JSON object.")
    required = {"schema_version", "stage", "configuration", "dataset", "reports", "metrics"}
    missing = required - set(payload)
    if missing:
        raise ValueError("Artifact is missing required fields: " + ", ".join(sorted(missing)))
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported artifact schema version.")
    if not isinstance(payload["reports"], list) or not isinstance(payload["metrics"], dict):
        raise ValueError("Artifact reports and metrics must be a list and object.")
    if _has_secret_key(payload) or _has_secret_value(payload):
        raise ValueError("Refusing to export an artifact containing a secret-like key or value.")
    return payload


def _markdown_summary(payload: dict[str, Any]) -> str:
    reports = payload["reports"]
    rewards = [float(item.get("reward", {}).get("total", 0.0)) for item in reports]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    metrics = payload["metrics"]
    lines = [
        "# PPP training artifact",
        "",
        f"- Stage: `{payload['stage']}`",
        f"- Reports: `{len(reports)}`",
        f"- Mean terminal reward: `{mean_reward:.3f}`",
        f"- Group size: `{metrics.get('group_size', 'n/a')}`",
    ]
    if "group_advantages" in metrics:
        lines.append(f"- Group advantages: `{metrics['group_advantages']}`")
    return "\n".join(lines) + "\n"


def command_export(args: argparse.Namespace) -> int:
    payload = _validate_artifact(json.loads(args.artifact.read_text()))
    if args.output is not None:
        write_training_artifact(args.output, payload)
        print(f"artifact: {args.output}")
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(_markdown_summary(payload))
        print(f"summary: {args.summary}")
    if args.output is None and args.summary is None:
        print(_markdown_summary(payload), end="")
    return 0


def command_summarize_run(args: argparse.Namespace) -> int:
    """Aggregate sanitized worker records after a Verl run completes."""

    config_path = args.run_dir / "training-config.json"
    subset_path = args.run_dir / "training-subset.json"
    trajectory_dir = args.run_dir / "sanitized-trajectories"
    for required in (config_path, subset_path, trajectory_dir):
        if not required.exists():
            raise FileNotFoundError(f"Run input does not exist: {required}")
    configuration = json.loads(config_path.read_text())
    subset = json.loads(subset_path.read_text())
    records = [
        json.loads(path.read_text())
        for path in sorted(trajectory_dir.glob("*.json"))
    ]
    if not records:
        raise ValueError("No sanitized trajectory records were produced.")

    grouped: dict[str, list[dict[str, Any]]] = {}
    reports = []
    invalid_action_categories: Counter[str] = Counter()
    for record in records:
        grouped.setdefault(str(record["group_id"]), []).append(record)
        invalid_action_categories.update(
            record.get("invalid_action_categories", {})
        )
        reports.append(
            {
                "instance_id": record["instance_id"],
                "predicted_functions": record["predicted_functions"],
                "reward": record["reward_breakdown"],
                "trajectory": record["sanitized_trajectory"],
                "termination": record["termination"],
                "model_calls": record["model_calls"],
                "parsed_actions": record.get("parsed_actions"),
                "action_parse_rate": record.get("action_parse_rate"),
                "invalid_action_count": record.get(
                    "invalid_action_count",
                    0,
                ),
                "invalid_action_categories": record.get(
                    "invalid_action_categories",
                    {},
                ),
                "schema_constrained_model_calls": record.get(
                    "schema_constrained_model_calls",
                    0,
                ),
                "finish_validation_passed": record[
                    "finish_validation_passed"
                ],
                "finish_correction_attempted": record[
                    "finish_correction_attempted"
                ],
                "finish_correction_parse_failed": record.get(
                    "finish_correction_parse_failed",
                    False,
                ),
                "invalid_predictions": record["invalid_predictions"],
                "model_generated_tokens": record["model_generated_tokens"],
                "environment_tokens": record["environment_tokens"],
                "latency_seconds": record["latency_seconds"],
                "simulator_live_calls": record["simulator_live_calls"],
            }
        )

    group_metrics = []
    nonzero_variance = 0
    for group_id, group_records in sorted(grouped.items()):
        if len(group_records) != PPPTrainingConfig().group_size:
            raise ValueError(
                f"Rollout group {group_id!r} contains {len(group_records)} records; "
                f"expected exactly {PPPTrainingConfig().group_size}."
            )
        rewards = [
            float(record["reward_breakdown"]["total"])
            for record in group_records
        ]
        advantages = list(group_advantages(rewards))
        if len(set(rewards)) > 1:
            nonzero_variance += 1
        group_metrics.append(
            {
                "group_id": group_id,
                "rollouts": len(group_records),
                "rewards": rewards,
                "advantages": advantages,
            }
        )

    total_model_calls = sum(record["model_calls"] for record in records)
    total_parsed_actions = sum(
        int(record.get("parsed_actions", 0)) for record in records
    )
    payload = {
        "schema_version": 1,
        "stage": "training",
        "configuration": configuration,
        "dataset": {
            "sha256": subset["dataset_sha256"],
            "selection_seed": subset["selection_seed"],
            "issue_groups": subset["issue_groups"],
        },
        "reports": reports,
        "metrics": {
            "groups": group_metrics,
            "group_size": PPPTrainingConfig().group_size,
            "groups_with_nonzero_reward_variance": nonzero_variance,
            "model_calls": total_model_calls,
            "parsed_actions": total_parsed_actions,
            "invalid_action_count": sum(
                int(record.get("invalid_action_count", 0))
                for record in records
            ),
            "invalid_action_categories": dict(
                sorted(invalid_action_categories.items())
            ),
            "schema_constrained_model_calls": sum(
                int(record.get("schema_constrained_model_calls", 0))
                for record in records
            ),
            "action_parse_rate": (
                total_parsed_actions / total_model_calls
                if total_model_calls
                else 0.0
            ),
            "valid_finish_count": sum(
                record["finish_validation_passed"] is True
                for record in records
            ),
            "simulator_live_calls": sum(
                record["simulator_live_calls"] for record in records
            ),
            "mean_latency_seconds": sum(
                float(record["latency_seconds"]) for record in records
            )
            / len(records),
            "compute_seconds": args.compute_seconds,
            "estimated_compute_usd": (
                args.compute_seconds / 3600 * args.hourly_usd
            ),
            "hourly_usd": args.hourly_usd,
        },
    }
    _validate_artifact(payload)
    report_path = args.run_dir / "training-report.json"
    summary_path = args.run_dir / "summary.md"
    write_training_artifact(report_path, payload)
    summary_path.write_text(_markdown_summary(payload))
    print(f"training report: {report_path}")
    print(f"summary: {summary_path}")
    return 0


def command_check_extension(args: argparse.Namespace) -> int:
    """Enforce the teaching plan's 20-to-40-step extension gate."""

    if not (0 <= args.projected_compute_usd <= 45):
        raise ValueError("Projected phase compute must remain between $0 and $45.")
    report = json.loads((args.run_dir / "training-report.json").read_text())
    groups = report["metrics"].get("groups", [])
    if len(groups) < PPPTrainingConfig().initial_steps:
        raise ValueError("At least 20 completed prompt groups are required.")
    nonzero = int(
        report["metrics"].get("groups_with_nonzero_reward_variance", 0)
    )
    if nonzero / len(groups) < 0.25:
        raise ValueError(
            "At least 25% of prompt groups must have nonzero reward variance."
        )
    if not (args.run_dir / "resume-verified").is_file():
        raise ValueError("Checkpoint reload/resume has not been verified.")
    adapters = list(
        args.run_dir.glob(
            "global_step_*/actor/lora_adapter/adapter_model.safetensors"
        )
    )
    if not adapters:
        raise ValueError("No saved LoRA adapter checkpoint was found.")

    metrics_path = args.run_dir / "training-metrics.jsonl"
    saw_loss = False
    saw_kl = False
    for line in metrics_path.read_text().splitlines():
        data = json.loads(line).get("data", {})
        for key, value in data.items():
            lowered = key.lower()
            if "loss" not in lowered and "kl" not in lowered:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                raise ValueError(f"Non-finite training metric: {key}={value}")
            saw_loss = saw_loss or "loss" in lowered
            saw_kl = saw_kl or "kl" in lowered
    if not saw_loss or not saw_kl:
        raise ValueError("Finite loss and KL metrics are required for extension.")
    print("PASS: 40-step extension criteria are satisfied.")
    return 0


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def command_verify_continuation(args: argparse.Namespace) -> int:
    """Prove that one resumed group produced an effective LoRA update."""

    if (args.from_step, args.to_step) != (1, 2):
        raise ValueError("The bounded continuation verifier only accepts step 1 to 2.")
    if args.baseline_trajectories != PPPTrainingConfig().group_size:
        raise ValueError("Step-2 continuation requires exactly eight baseline trajectories.")

    from_adapter = (
        args.run_dir
        / f"global_step_{args.from_step}"
        / "actor"
        / "lora_adapter"
        / "adapter_model.safetensors"
    )
    to_adapter = (
        args.run_dir
        / f"global_step_{args.to_step}"
        / "actor"
        / "lora_adapter"
        / "adapter_model.safetensors"
    )
    for required in (from_adapter, to_adapter):
        if not required.is_file():
            raise FileNotFoundError(f"Continuation adapter does not exist: {required}")

    before_sha256 = _sha256_file(from_adapter)
    after_sha256 = _sha256_file(to_adapter)
    if before_sha256 != args.expected_before_sha256:
        raise ValueError("Step-1 adapter changed after the continuation started.")
    if before_sha256 == after_sha256:
        raise ValueError("Continuation did not change the LoRA adapter.")

    trajectory_dir = args.run_dir / "sanitized-trajectories"
    trajectory_count = len(list(trajectory_dir.glob("*.json")))
    expected_trajectories = args.baseline_trajectories + PPPTrainingConfig().group_size
    if trajectory_count != expected_trajectories:
        raise ValueError(
            "Continuation must add exactly eight trajectories: "
            f"expected {expected_trajectories}, found {trajectory_count}."
        )

    step_metrics = None
    metrics_path = args.run_dir / "training-metrics.jsonl"
    for line in metrics_path.read_text().splitlines():
        record = json.loads(line)
        if int(record.get("step", -1)) == args.to_step:
            step_metrics = record.get("data", {})
    if step_metrics is None:
        raise ValueError(f"No scalar metrics found for continuation step {args.to_step}.")

    required_metrics = {
        "policy_loss": "actor/pg_loss",
        "gradient_norm": "actor/grad_norm",
        "rollout_kl": "rollout_corr/kl",
    }
    verified_metrics: dict[str, float] = {}
    for output_name, metric_name in required_metrics.items():
        try:
            value = float(step_metrics[metric_name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Continuation metric is missing or invalid: {metric_name}."
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"Continuation metric is non-finite: {metric_name}.")
        verified_metrics[output_name] = value
    if verified_metrics["gradient_norm"] <= 0:
        raise ValueError("Continuation gradient norm must be positive.")

    payload = {
        "schema_version": 1,
        "stage": "step_2_continuation",
        "status": "passed",
        "from_step": args.from_step,
        "to_step": args.to_step,
        "group_size": PPPTrainingConfig().group_size,
        "baseline_trajectories": args.baseline_trajectories,
        "final_trajectories": trajectory_count,
        "new_trajectories": trajectory_count - args.baseline_trajectories,
        "before_adapter_sha256": before_sha256,
        "after_adapter_sha256": after_sha256,
        "adapter_changed": True,
        **verified_metrics,
    }
    output = args.output or args.run_dir / "continuation-step-2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"continuation evidence: {output}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_train(args: argparse.Namespace) -> int:
    launcher = Path("simplified/scripts/run_ppp_rl_4b.sh")
    if not launcher.is_file():
        raise FileNotFoundError(f"GPU launcher does not exist: {launcher}")
    command = [
        "bash",
        str(launcher),
        "--execute" if args.execute else "--print-command",
        "--steps",
        str(args.steps),
        "--simulator",
        args.simulator,
        "--model",
        args.model,
    ]
    if args.projected_compute_usd is not None:
        command.extend(
            [
                "--projected-compute-usd",
                str(args.projected_compute_usd),
            ]
        )
    environment = {**os.environ, "PPP_PYTHON": sys.executable}
    return subprocess.run(
        command,
        check=False,
        env=environment,
    ).returncode


def command_evaluate(args: argparse.Namespace) -> int:
    if not args.adapter.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory does not exist: {args.adapter}")
    model_id = _model_id(args.model)
    serve = (
        f"vllm serve {model_id} "
        f"--enable-lora --lora-modules trained={args.adapter} "
        "--max-model-len 10240 --gpu-memory-utilization 0.35"
        + (" --language-model-only" if args.model == "qwen35" else "")
    )
    evaluate = (
        "ppp-simple evaluate --suite dev-v1 --mode live "
        f"--qwen-base-url {args.qwen_base_url} --qwen-model trained "
        "--inference-seeds 11,22,33 --policy-version navigation-v2 "
        "--tool-schema-version v2 "
        f"--output-dir {args.output_dir}"
    )
    print("# Terminal 1: serve the saved adapter")
    print(serve)
    print("\n# Terminal 2: run the frozen development suite")
    print(evaluate)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Report local and optional GPU readiness.")
    doctor.add_argument("--data", type=Path, default=DEFAULT_DATA)
    doctor.set_defaults(handler=command_doctor)

    prepare = subparsers.add_parser("prepare", help="Write the frozen 16-group subset and config.")
    prepare.add_argument("--data", type=Path, default=DEFAULT_DATA)
    prepare.add_argument("--selection-seed", type=int, default=42)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "prepared")
    prepare.set_defaults(handler=command_prepare)

    contract_gate = subparsers.add_parser(
        "contract-gate",
        help="Validate the synthetic Verl-to-FoldGRPO batch contract.",
    )
    contract_gate.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "contract-gate.json",
    )
    contract_gate.set_defaults(handler=command_contract_gate)

    schema_gate = subparsers.add_parser(
        "schema-gate",
        help=(
            "Run two real vLLM completions to verify JSON-schema enforcement "
            "and chosen-token log probabilities."
        ),
    )
    schema_gate.add_argument(
        "--model",
        choices=("qwen35", "qwen3"),
        default="qwen35",
    )
    schema_gate.add_argument("--max-model-len", type=int, default=2048)
    schema_gate.add_argument("--max-tokens", type=int, default=256)
    schema_gate.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.20,
    )
    schema_gate.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "vllm-schema-gate.json",
    )
    schema_gate.add_argument(
        "--execute",
        action="store_true",
        help="Load the real model after setting the paid-compute confirmation.",
    )
    schema_gate.set_defaults(handler=command_schema_gate)

    def add_group_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--data", type=Path, default=DEFAULT_DATA)
        command.add_argument("--selection-seed", type=int, default=42)
        command.add_argument("--subset-manifest", type=Path)
        command.add_argument("--row-index", type=int)
        command.add_argument("--instance-id", default="Project-MONAI__MONAI-3205")
        command.add_argument("--preference", default="concise_question")
        command.add_argument("--precise", action="store_true")
        command.add_argument(
            "--workspace",
            type=Path,
            help="Existing pinned workspace; avoids a network fetch.",
        )
        command.add_argument(
            "--workspace-root",
            type=Path,
            default=Path("simplified/workspaces"),
        )

    smoke = subparsers.add_parser(
        "smoke",
        help="Run eight scripted deterministic rollouts for one episode.",
    )
    add_group_arguments(smoke)
    smoke.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "deterministic-smoke.json",
    )
    smoke.set_defaults(handler=command_smoke)

    live_group = subparsers.add_parser(
        "live-group",
        help="Run eight real Qwen/Gemini trajectories without an optimizer.",
    )
    add_group_arguments(live_group)
    live_group.add_argument(
        "--model",
        choices=("qwen35", "qwen3"),
        default="qwen35",
        help="Use Qwen3.5 by default; select Qwen3 only after a recorded gate failure.",
    )
    live_group.add_argument(
        "--qwen-base-url",
        default=os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1"),
    )
    live_group.add_argument(
        "--inference-seeds",
        type=_parse_inference_seeds,
        default=DEFAULT_LIVE_INFERENCE_SEEDS,
    )
    live_group.add_argument(
        "--simulator-cache",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "simulator-cache.sqlite3",
    )
    live_group.add_argument("--simulator-call-budget", type=int, default=64)
    live_group.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "live-group.json",
    )
    live_group.set_defaults(handler=command_live_group)

    export = subparsers.add_parser("export", help="Validate and export a sanitized artifact.")
    export.add_argument("--artifact", type=Path, required=True)
    export.add_argument("--output", type=Path)
    export.add_argument("--summary", type=Path)
    export.set_defaults(handler=command_export)

    summarize = subparsers.add_parser(
        "summarize-run",
        help="Aggregate sanitized Verl worker records and group advantages.",
    )
    summarize.add_argument("--run-dir", type=Path, required=True)
    summarize.add_argument("--compute-seconds", type=float, default=0.0)
    summarize.add_argument("--hourly-usd", type=float, default=1.09)
    summarize.set_defaults(handler=command_summarize_run)

    extension = subparsers.add_parser(
        "check-extension",
        help="Validate reward variance, resume, metrics, checkpoint, and budget.",
    )
    extension.add_argument("--run-dir", type=Path, required=True)
    extension.add_argument("--projected-compute-usd", type=float, required=True)
    extension.set_defaults(handler=command_check_extension)

    continuation = subparsers.add_parser(
        "verify-continuation",
        help="Verify eight new trajectories, nonzero gradient, and adapter delta.",
    )
    continuation.add_argument("--run-dir", type=Path, required=True)
    continuation.add_argument("--from-step", type=int, default=1)
    continuation.add_argument("--to-step", type=int, default=2)
    continuation.add_argument("--baseline-trajectories", type=int, required=True)
    continuation.add_argument("--expected-before-sha256", required=True)
    continuation.add_argument("--output", type=Path)
    continuation.set_defaults(handler=command_verify_continuation)

    train = subparsers.add_parser(
        "train",
        help="Print the guarded Verl command, or execute it after paid-run confirmation.",
    )
    train.add_argument("--steps", type=int, choices=(1, 2, 20, 40), default=20)
    train.add_argument(
        "--simulator",
        choices=("deterministic", "gemini"),
        default="gemini",
    )
    train.add_argument(
        "--execute",
        action="store_true",
        help="Execute only when CONFIRM_PAID_TRAINING is set.",
    )
    train.add_argument(
        "--model",
        choices=("qwen35", "qwen3"),
        default="qwen35",
        help="Try Qwen3.5 first; use qwen3 only after recording gate failure evidence.",
    )
    train.add_argument(
        "--projected-compute-usd",
        type=float,
        help="Required with --steps 40 --execute; must keep phase compute at or below $45.",
    )
    train.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Print paired commands for serving and evaluating a saved adapter.",
    )
    evaluate.add_argument("--adapter", type=Path, required=True)
    evaluate.add_argument(
        "--qwen-base-url",
        default="http://localhost:8000/v1",
    )
    evaluate.add_argument(
        "--model",
        choices=("qwen35", "qwen3"),
        default="qwen35",
    )
    evaluate.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "adapter-dev-v1",
    )
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main() -> None:
    load_local_environment()
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
