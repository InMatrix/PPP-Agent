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
ProviderFactory = Callable[[Episode], AgentProvider]
SimulatorFactory = Callable[[Episode], UserSimulator]


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _episode_filename(position: int, episode: Episode) -> str:
    safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode.instance_id)
    safe_preference = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        episode.preference.name,
    )
    return (
        f"{position:02d}_row-{episode.row_index}_"
        f"{safe_instance}_{safe_preference}.json"
    )


def build_manifest(
    *,
    episodes: Sequence[Episode],
    mode: str,
    model: str,
    simulator: str,
    max_turns: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": mode,
        "model": model,
        "simulator": simulator,
        "max_turns": max_turns,
        "seed": seed,
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
    }


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in records if item.get("status") == "completed"]
    failed = [item for item in records if item.get("status") == "failed"]
    rewards = [item["report"]["reward"] for item in completed]
    durations = [float(item["duration_seconds"]) for item in completed]
    turns = [len(item["report"]["trajectory"]) for item in completed]
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

    return {
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
        f"- Mean total reward: {metric('mean_total_reward')}",
        f"- Question rate: {metric('question_rate')}",
        f"- Mean disclosure level: "
        f"{metric('mean_disclosure_level_per_question')}",
        f"- Preference compliance when judged: "
        f"{metric('preference_compliance_rate_when_judged')}",
        "",
        "## Episodes",
        "",
        "| # | Instance | Preference | Turns | Dupes | Questions | F1 | "
        "Reward | Termination |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
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
                f"| {position} | `{episode['instance_id']}` | "
                f"`{episode['preference']}` | {len(report['trajectory'])} | "
                f"{duplicates} | {reward['questions_asked']} | "
                f"{float(reward['productivity']):.3f} | "
                f"{float(reward['total']):.3f} | `{termination}` |"
            )
        else:
            lines.append(
                f"| {position} | `{episode['instance_id']}` | "
                f"`{episode['preference']}` | — | — | — | — | — | "
                f"failed: {item['error_type']} |"
            )
    lines.append("")
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
        for position, episode in enumerate(episodes, start=1):
            path = episodes_dir / _episode_filename(position, episode)
            if path.exists() and self.resume:
                previous = json.loads(path.read_text())
                if previous.get("status") == "completed":
                    self.progress(
                        f"[{position}/{len(episodes)}] resume "
                        f"{episode.instance_id}"
                    )
                    records.append(previous)
                    continue

            self.progress(
                f"[{position}/{len(episodes)}] run {episode.instance_id} "
                f"({episode.preference.name})"
            )
            started = time.perf_counter()
            episode_metadata = {
                "row_index": episode.row_index,
                "instance_id": episode.instance_id,
                "repository": episode.repository,
                "preference": episode.preference.name,
            }
            try:
                workspace = self.workspace.prepare(episode)
                report = AgentRunner(
                    provider=self.provider_factory(episode),
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
