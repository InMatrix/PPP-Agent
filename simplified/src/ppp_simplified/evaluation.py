"""Resumable batch evaluation and aggregate baseline metrics."""

from __future__ import annotations

import json
import re
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Episode
from .providers import AgentProvider
from .runner import AgentRunner
from .simulator import UserSimulator
from .workspace import RepositoryWorkspace


ProgressCallback = Callable[[str], None]
ProviderFactory = Callable[[Episode, int | None], AgentProvider]
SimulatorFactory = Callable[[Episode], UserSimulator]
COMPARISON_METRICS = (
    "exact_localization_rate",
    "mean_productivity_f1",
    "agent_finish_rate",
    "natural_finish_rate",
    "deadline_finish_rate",
    "empty_prediction_rate",
    "mean_total_reward",
    "question_rate",
    "preference_compliance_rate_when_judged",
    "mean_turns",
    "mean_model_calls",
    "mean_duplicate_actions_suppressed",
    "finish_validation_pass_rate",
    "finish_correction_rate",
    "invalid_prediction_episode_rate",
    "mean_duration_seconds",
)


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _episode_filename(
    position: int,
    episode: Episode,
    inference_seed: int | None,
) -> str:
    safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode.instance_id)
    safe_preference = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        episode.preference.name,
    )
    return (
        f"{position:02d}_seed-{inference_seed}_row-{episode.row_index}_"
        f"{safe_instance}_{safe_preference}.json"
    )


def build_manifest(
    *,
    episodes: Sequence[Episode],
    mode: str,
    model: str,
    simulator: str,
    max_turns: int,
    sample_seed: int,
    inference_seeds: Sequence[int | None],
    policy_label: str,
    tool_schema_version: str,
    code_revision: str,
    evaluation_suite: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cases = [
        {
            "position": position,
            "row_index": episode.row_index,
            "instance_id": episode.instance_id,
            "preference": episode.preference.name,
            "inference_seed": inference_seed,
        }
        for position, (inference_seed, episode) in enumerate(
            (
                (inference_seed, episode)
                for inference_seed in inference_seeds
                for episode in episodes
            ),
            start=1,
        )
    ]
    manifest = {
        "schema_version": 3,
        "mode": mode,
        "model": model,
        "simulator": simulator,
        "max_turns": max_turns,
        "sample_seed": sample_seed,
        "inference_seeds": list(inference_seeds),
        "policy_label": policy_label,
        "tool_schema_version": tool_schema_version,
        "code_revision": code_revision,
        "episodes": [
            {
                "position": position,
                "row_index": episode.row_index,
                "instance_id": episode.instance_id,
                "repository": episode.repository,
                "preference": episode.preference.name,
                "is_vague": episode.is_vague,
            }
            for position, episode in enumerate(episodes, start=1)
        ],
        "cases": cases,
    }
    if evaluation_suite is not None:
        manifest["evaluation_suite"] = evaluation_suite
    return manifest


def summarize_records(
    records: Sequence[dict[str, Any]],
    *,
    include_seed_groups: bool = True,
) -> dict[str, Any]:
    completed = [item for item in records if item.get("status") == "completed"]
    failed = [item for item in records if item.get("status") == "failed"]
    rewards = [item["report"]["reward"] for item in completed]
    durations = [float(item["duration_seconds"]) for item in completed]
    turns = [
        max(
            (
                int(step.get("turn", position))
                for position, step in enumerate(
                    item["report"]["trajectory"],
                    start=1,
                )
            ),
            default=0,
        )
        for item in completed
    ]
    questions = [int(reward["questions_asked"]) for reward in rewards]
    disclosure_levels = [
        int(level)
        for reward in rewards
        for level in reward["disclosure_levels"]
    ]
    predictions = [
        item["report"].get("predicted_functions", []) for item in completed
    ]
    finished = [
        bool(item["report"]["trajectory"])
        and item["report"]["trajectory"][-1].get("action", {}).get("tool")
        == "finish"
        for item in completed
    ]
    terminations = [
        item["report"].get("termination")
        or ("natural_finish" if did_finish else "turn_limit")
        for item, did_finish in zip(completed, finished)
    ]
    duplicate_counts = [
        int(
            item["report"].get(
                "duplicate_actions_suppressed",
                sum(
                    bool(step.get("duplicate_suppressed"))
                    for step in item["report"]["trajectory"]
                ),
            )
        )
        for item in completed
    ]
    judged_preferences = [
        bool(reward["preference_ok"])
        for reward in rewards
        if reward["preference_ok"] is not None
    ]
    productivities = [float(reward["productivity"]) for reward in rewards]
    model_calls = [
        int(item["report"].get("model_calls", len(item["report"]["trajectory"])))
        for item in completed
    ]
    validation_results = [
        item["report"]["finish_validation_passed"]
        for item in completed
        if item["report"].get("finish_validation_passed") is not None
    ]
    corrections = [
        bool(item["report"]["finish_correction_attempted"])
        for item in completed
        if "finish_correction_attempted" in item["report"]
    ]
    invalid_predictions = [
        item["report"]["invalid_predictions"]
        for item in completed
        if "invalid_predictions" in item["report"]
    ]

    summary = {
        "episodes_requested": len(records),
        "episodes_completed": len(completed),
        "episodes_failed": len(failed),
        "exact_localization_rate": (
            sum(value == 1.0 for value in productivities) / len(productivities)
            if productivities
            else None
        ),
        "mean_productivity_f1": _mean(productivities),
        "agent_finish_rate": (
            sum(finished) / len(finished) if finished else None
        ),
        "natural_finish_rate": (
            sum(value == "natural_finish" for value in terminations)
            / len(terminations)
            if terminations
            else None
        ),
        "deadline_finish_rate": (
            sum(value == "deadline_finish" for value in terminations)
            / len(terminations)
            if terminations
            else None
        ),
        "turn_limit_rate": (
            sum(value == "turn_limit" for value in terminations)
            / len(terminations)
            if terminations
            else None
        ),
        "empty_prediction_rate": (
            sum(not values for values in predictions) / len(predictions)
            if predictions
            else None
        ),
        "mean_total_reward": _mean(
            [float(reward["total"]) for reward in rewards]
        ),
        "mean_proactivity_adjustment": _mean(
            [float(reward["proactivity_adjustment"]) for reward in rewards]
        ),
        "mean_personalization_adjustment": _mean(
            [
                float(reward["personalization_adjustment"])
                for reward in rewards
            ]
        ),
        "question_rate": (
            sum(value > 0 for value in questions) / len(questions)
            if questions
            else None
        ),
        "mean_questions_per_episode": _mean(
            [float(value) for value in questions]
        ),
        "mean_duplicate_actions_suppressed": _mean(
            [float(value) for value in duplicate_counts]
        ),
        "duplicate_action_episode_rate": (
            sum(value > 0 for value in duplicate_counts)
            / len(duplicate_counts)
            if duplicate_counts
            else None
        ),
        "mean_disclosure_level_per_question": _mean(
            [float(value) for value in disclosure_levels]
        ),
        "preference_compliance_rate_when_judged": (
            sum(judged_preferences) / len(judged_preferences)
            if judged_preferences
            else None
        ),
        "mean_turns": _mean([float(value) for value in turns]),
        "mean_model_calls": _mean([float(value) for value in model_calls]),
        "finish_validation_pass_rate": (
            sum(validation_results)
            / len(validation_results)
            if validation_results
            else None
        ),
        "finish_correction_rate": (
            sum(corrections) / len(corrections) if corrections else None
        ),
        "invalid_prediction_episode_rate": (
            sum(bool(values) for values in invalid_predictions)
            / len(invalid_predictions)
            if invalid_predictions
            else None
        ),
        "mean_duration_seconds": _mean(durations),
        "failures": [
            {
                "instance_id": item["episode"]["instance_id"],
                "preference": item["episode"]["preference"],
                "error_type": item["error_type"],
                "error": item["error"],
            }
            for item in failed
        ],
    }
    if include_seed_groups:
        seeds = {
            item["episode"].get("inference_seed") for item in records
        }
        summary["by_inference_seed"] = {
            str(seed): summarize_records(
                [
                    item
                    for item in records
                    if item["episode"].get("inference_seed") == seed
                ],
                include_seed_groups=False,
            )
            for seed in sorted(seeds, key=lambda value: (value is None, value))
        }
        instances = {
            item["episode"]["instance_id"] for item in records
        }
        summary["by_instance"] = {
            instance_id: summarize_records(
                [
                    item
                    for item in records
                    if item["episode"]["instance_id"] == instance_id
                ],
                include_seed_groups=False,
            )
            for instance_id in sorted(instances)
        }
    return summary


def render_markdown_summary(
    manifest: dict[str, Any],
    records: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    def metric(name: str) -> str:
        value = summary[name]
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        "# Simplified PPP-Agent baseline",
        "",
        f"- Mode: `{manifest['mode']}`",
        f"- Agent: `{manifest['model']}`",
        f"- Simulator: `{manifest['simulator']}`",
        f"- Policy: `{manifest['policy_label']}`",
        f"- Inference seeds: `{manifest['inference_seeds']}`",
        f"- Completed: {summary['episodes_completed']}/"
        f"{summary['episodes_requested']}",
        f"- Exact localization rate: {metric('exact_localization_rate')}",
        f"- Mean function F1: {metric('mean_productivity_f1')}",
        f"- Agent finish rate: {metric('agent_finish_rate')}",
        f"- Natural finish rate: {metric('natural_finish_rate')}",
        f"- Deadline finish rate: {metric('deadline_finish_rate')}",
        f"- Empty prediction rate: {metric('empty_prediction_rate')}",
        f"- Episodes with suppressed duplicates: "
        f"{metric('duplicate_action_episode_rate')}",
        f"- Finish validation pass rate: "
        f"{metric('finish_validation_pass_rate')}",
        f"- Finish correction rate: {metric('finish_correction_rate')}",
        f"- Invalid prediction episode rate: "
        f"{metric('invalid_prediction_episode_rate')}",
        f"- Mean total reward: {metric('mean_total_reward')}",
        f"- Question rate: {metric('question_rate')}",
        f"- Mean disclosure level: "
        f"{metric('mean_disclosure_level_per_question')}",
        f"- Preference compliance when judged: "
        f"{metric('preference_compliance_rate_when_judged')}",
        "",
    ]
    suite = manifest.get("evaluation_suite")
    if suite:
        lines.extend(
            [
                "## Evaluation suite",
                "",
                f"- Name: `{suite['name']}`",
                f"- Role: `{suite['role']}`",
                f"- Sealed: `{suite['sealed']}`",
                f"- Trajectory policy: {suite['trajectory_policy']}",
                "",
            ]
        )
        if suite["role"] == "heldout":
            lines.extend(
                [
                    "> Held-out result: use aggregate metrics for model "
                    "selection. Inspecting individual trajectories turns those "
                    "tasks into development data.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Episodes",
            "",
            "| # | Seed | Instance | Preference | Turns | Calls | Dupes | "
            "Questions | F1 | Reward | Termination |",
            "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for position, item in enumerate(records, start=1):
        episode = item["episode"]
        if item["status"] == "completed":
            report = item["report"]
            reward = report["reward"]
            termination = report.get("termination") or (
                "natural_finish"
                if report["trajectory"]
                and report["trajectory"][-1].get("action", {}).get("tool")
                == "finish"
                else "turn_limit"
            )
            duplicates = int(
                report.get(
                    "duplicate_actions_suppressed",
                    sum(
                        bool(step.get("duplicate_suppressed"))
                        for step in report["trajectory"]
                    ),
                )
            )
            lines.append(
                f"| {position} | {episode.get('inference_seed', '—')} | "
                f"`{episode['instance_id']}` | `{episode['preference']}` | "
                f"{max((step.get('turn', index) for index, step in enumerate(report['trajectory'], start=1)), default=0)} | "
                f"{report.get('model_calls', len(report['trajectory']))} | "
                f"{duplicates} | {reward['questions_asked']} | "
                f"{float(reward['productivity']):.3f} | "
                f"{float(reward['total']):.3f} | `{termination}` |"
            )
        else:
            lines.append(
                f"| {position} | {episode.get('inference_seed', '—')} | "
                f"`{episode['instance_id']}` | `{episode['preference']}` | "
                f"— | — | — | — | — | — | failed: {item['error_type']} |"
            )
    lines.append("")
    return "\n".join(lines)


def compare_summaries(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in COMPARISON_METRICS:
        control_value = control.get(name)
        candidate_value = candidate.get(name)
        metrics[name] = {
            "control": control_value,
            "candidate": candidate_value,
            "delta": (
                candidate_value - control_value
                if control_value is not None and candidate_value is not None
                else None
            ),
        }
    f1_gain = metrics["mean_productivity_f1"]["delta"]
    reward_delta = metrics["mean_total_reward"]["delta"]
    preference_delta = metrics[
        "preference_compliance_rate_when_judged"
    ]["delta"]
    completion_ok = (
        candidate.get("episodes_failed") == 0
        and candidate.get("episodes_completed")
        == candidate.get("episodes_requested")
        and candidate.get("empty_prediction_rate") == 0
    )
    reward_non_regression = reward_delta is None or reward_delta >= 0
    preference_non_regression = (
        preference_delta is None or preference_delta >= 0
    )
    continue_with_4b = (
        completion_ok
        and f1_gain is not None
        and f1_gain >= 0.10
        and reward_non_regression
        and preference_non_regression
    )
    return {
        "metrics": metrics,
        "decision": {
            "completion_ok": completion_ok,
            "f1_gain_threshold": 0.10,
            "f1_gain_met": f1_gain is not None and f1_gain >= 0.10,
            "reward_non_regression": reward_non_regression,
            "preference_non_regression": preference_non_regression,
            "recommendation": (
                "continue_with_4b"
                if continue_with_4b
                else "compare_larger_model"
            ),
        },
    }


def validate_comparable_manifests(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    """Reject comparisons that cross suite or episode boundaries."""

    control_suite = control.get("evaluation_suite")
    candidate_suite = candidate.get("evaluation_suite")
    if (control_suite is None) != (candidate_suite is None):
        raise ValueError(
            "Cannot compare a named evaluation suite with an ad-hoc sample."
        )
    if control_suite is not None:
        identity_fields = ("name", "role", "source_path", "source_sha256")
        control_identity = {
            field: control_suite.get(field) for field in identity_fields
        }
        candidate_identity = {
            field: candidate_suite.get(field) for field in identity_fields
        }
        if control_identity != candidate_identity:
            raise ValueError(
                "Control and candidate use different evaluation suites."
            )

    def case_identity(manifest: dict[str, Any]) -> list[tuple[Any, ...]]:
        return [
            (
                item.get("row_index"),
                item.get("instance_id"),
                item.get("preference"),
                item.get("inference_seed"),
            )
            for item in manifest.get("cases", ())
        ]

    if case_identity(control) != case_identity(candidate):
        raise ValueError(
            "Control and candidate manifests contain different evaluation "
            "cases or inference seeds."
        )
    return control_suite


def render_markdown_comparison(comparison: dict[str, Any]) -> str:
    lines = ["# Control vs candidate", ""]
    suite = comparison.get("evaluation_suite")
    if suite:
        lines.extend(
            [
                f"- Evaluation suite: `{suite['name']}`",
                f"- Evaluation role: `{suite['role']}`",
                "",
            ]
        )
        if suite["role"] == "heldout":
            lines.extend(
                [
                    "> Frozen held-out comparison. Use the aggregate decision; "
                    "do not tune against individual episode trajectories.",
                    "",
                ]
            )
    lines.extend(
        [
            "| Metric | Control | Candidate | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, values in comparison["metrics"].items():
        rendered = [
            "n/a" if values[key] is None else f"{values[key]:.3f}"
            for key in ("control", "candidate", "delta")
        ]
        lines.append(
            f"| `{name}` | {rendered[0]} | {rendered[1]} | {rendered[2]} |"
        )
    decision = comparison["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Completion criterion: `{decision['completion_ok']}`",
            f"- F1 gain ≥ {decision['f1_gain_threshold']:.2f}: "
            f"`{decision['f1_gain_met']}`",
            f"- Reward non-regression: "
            f"`{decision['reward_non_regression']}`",
            f"- Preference non-regression: "
            f"`{decision['preference_non_regression']}`",
            f"- Recommendation: `{decision['recommendation']}`",
            "",
        ]
    )
    return "\n".join(lines)


class BatchEvaluator:
    """Run independent episodes and persist each result before continuing."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        output_dir: Path,
        provider_factory: ProviderFactory,
        simulator_factory: SimulatorFactory,
        max_turns: int,
        resume: bool = True,
        progress: ProgressCallback = print,
    ) -> None:
        self.workspace = RepositoryWorkspace(workspace_root)
        self.output_dir = output_dir
        self.provider_factory = provider_factory
        self.simulator_factory = simulator_factory
        self.max_turns = max_turns
        self.resume = resume
        self.progress = progress

    def run(
        self,
        episodes: Sequence[Episode],
        inference_seeds: Sequence[int | None],
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists() and self.resume:
            existing = json.loads(manifest_path.read_text())
            if existing != manifest:
                raise ValueError(
                    "The output directory contains a different evaluation "
                    "manifest. Choose another --output-dir or use --no-resume."
                )
        _write_json(manifest_path, manifest)

        records: list[dict[str, Any]] = []
        episodes_dir = self.output_dir / "episodes"
        cases = [
            (episode, inference_seed)
            for inference_seed in inference_seeds
            for episode in episodes
        ]
        for position, (episode, inference_seed) in enumerate(cases, start=1):
            path = episodes_dir / _episode_filename(
                position,
                episode,
                inference_seed,
            )
            if path.exists() and self.resume:
                previous = json.loads(path.read_text())
                if previous.get("status") == "completed":
                    self.progress(
                        f"[{position}/{len(cases)}] resume "
                        f"{episode.instance_id} seed={inference_seed}"
                    )
                    records.append(previous)
                    continue

            self.progress(
                f"[{position}/{len(cases)}] run {episode.instance_id} "
                f"({episode.preference.name}) seed={inference_seed}"
            )
            started = time.perf_counter()
            episode_metadata = {
                "row_index": episode.row_index,
                "instance_id": episode.instance_id,
                "repository": episode.repository,
                "preference": episode.preference.name,
                "inference_seed": inference_seed,
            }
            try:
                workspace = self.workspace.prepare(episode)
                report = AgentRunner(
                    provider=self.provider_factory(episode, inference_seed),
                    simulator=self.simulator_factory(episode),
                    max_turns=self.max_turns,
                ).run(episode, workspace)
                record = {
                    "status": "completed",
                    "duration_seconds": time.perf_counter() - started,
                    "episode": episode_metadata,
                    "report": asdict(report),
                }
            except Exception as error:
                record = {
                    "status": "failed",
                    "duration_seconds": time.perf_counter() - started,
                    "episode": episode_metadata,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            _write_json(path, record)
            records.append(record)

        summary = summarize_records(records)
        summary["generated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(self.output_dir / "summary.json", summary)
        (self.output_dir / "summary.md").write_text(
            render_markdown_summary(manifest, records, summary)
        )
        return summary
