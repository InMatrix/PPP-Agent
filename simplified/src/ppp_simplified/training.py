"""Pure-Python contracts for the teaching-scale PPP reinforcement-learning run.

This module deliberately has no Torch, Ray, or Verl imports.  Local tests and
the analysis notebook can therefore validate the data, reward, masking, and
artifact contracts before a GPU instance exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .data import iter_episodes
from .models import Episode, RunReport


TRAINING_PREFERENCES = (
    "no_preference",
    "concise_question",
    "detail_question",
    "answer_more",
    "only_begin",
    "no_ask",
    "do_selection",
    "professional",
    "amateur",
    "ask_many",
    "one_question",
    "first_try",
)


@dataclass(frozen=True)
class PPPTrainingConfig:
    """The reduced configuration, plus the invariants we refuse to shrink."""

    schema_version: int = 1
    model_id: str = "Qwen/Qwen3.5-4B"
    fallback_model_id: str = "Qwen/Qwen3-4B"
    simulator_model: str = "gemini-3.5-flash-lite"
    policy_version: str = "navigation-v2"
    tool_schema_version: str = "v2"
    learning_rate: float = 1e-6
    group_size: int = 8
    prompt_groups_per_update: int = 1
    initial_steps: int = 20
    maximum_steps: int = 40
    max_turns: int = 8
    prompt_tokens: int = 6144
    response_tokens: int = 4096
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: str = "all-linear"
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    loss_aggregation: str = "token-mean"
    advantage_estimator: str = "foldgrpo"
    selection_seed: int = 42
    inference_seeds: tuple[int, ...] = (11, 22, 33)
    training_issue_groups: int = 16
    lambda_gpu: str = "NVIDIA A6000 48GB"
    lambda_hourly_usd: float = 1.09
    a6000_gate_hours: float = 2.0
    a6000_gate_usd: float = 3.0
    phase_compute_guard_usd: float = 45.0
    phase_budget_usd: float = 50.0

    def __post_init__(self) -> None:
        if self.group_size != 8:
            raise ValueError("PPP teaching runs keep the paper's group size of 8.")
        if self.prompt_groups_per_update != 1:
            raise ValueError("The teaching configuration uses one prompt group per update.")
        if self.max_turns != 8:
            raise ValueError("Navigation-v2 training is frozen at eight logical turns.")
        if self.policy_version != "navigation-v2":
            raise ValueError("Tool and navigation changes are out of scope for this phase.")
        if self.tool_schema_version != "v2":
            raise ValueError("Navigation-v2 requires the v2 tool schema.")
        if self.clip_ratio_high <= self.clip_ratio_low:
            raise ValueError("DAPO Clip-Higher requires a larger upper clip ratio.")
        if not (0 < self.initial_steps <= self.maximum_steps):
            raise ValueError("Initial steps must be positive and no larger than maximum steps.")
        if self.phase_compute_guard_usd >= self.phase_budget_usd:
            raise ValueError("The compute guard must leave a phase-budget reserve.")

    @property
    def maximum_total_tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens

    @property
    def identity(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


@dataclass(frozen=True)
class TrainingExampleRef:
    instance_id: str
    preference: str
    is_vague: bool
    row_index: int
    source_path: str


@dataclass(frozen=True)
class TrainingSubsetManifest:
    schema_version: int
    dataset: str
    dataset_sha256: str
    selection_seed: int
    issue_groups: tuple[str, ...]
    examples: tuple[TrainingExampleRef, ...]

    @property
    def identity(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def select_training_subset(
    data_path: Path,
    *,
    issue_groups: int = 16,
    seed: int = 42,
) -> tuple[TrainingSubsetManifest, tuple[Episode, ...]]:
    """Select whole 13-row issue groups without inspecting evaluation data."""

    grouped: dict[str, list[Episode]] = {}
    for episode in iter_episodes(data_path):
        grouped.setdefault(episode.instance_id, []).append(episode)
    eligible = [
        instance_id
        for instance_id, episodes in grouped.items()
        if len(episodes) == 13
        and {item.preference.name for item in episodes} == set(TRAINING_PREFERENCES)
        and sum(not item.is_vague for item in episodes) == 1
    ]
    if len(eligible) < issue_groups:
        raise ValueError(
            f"Requested {issue_groups} complete issue groups, found {len(eligible)}."
        )
    selected_ids = tuple(sorted(random.Random(seed).sample(sorted(eligible), issue_groups)))
    selected = tuple(
        sorted(
            (episode for instance_id in selected_ids for episode in grouped[instance_id]),
            key=lambda item: (
                item.instance_id,
                item.is_vague,
                item.preference.name,
                item.row_index,
            ),
        )
    )
    refs = tuple(
        TrainingExampleRef(
            instance_id=item.instance_id,
            preference=item.preference.name,
            is_vague=item.is_vague,
            row_index=item.row_index,
            source_path=item.source_path,
        )
        for item in selected
    )
    manifest = TrainingSubsetManifest(
        schema_version=1,
        dataset=str(data_path),
        dataset_sha256=_sha256(data_path),
        selection_seed=seed,
        issue_groups=selected_ids,
        examples=refs,
    )
    return manifest, selected


def materialize_training_subset(
    source: Path,
    manifest: TrainingSubsetManifest,
    output: Path,
) -> str:
    """Write only manifest-selected rows for the GPU dataloader."""

    if _sha256(source) != manifest.dataset_sha256:
        raise ValueError("Training dataset checksum does not match the subset manifest.")
    import pyarrow.parquet as pq

    indices = [reference.row_index for reference in manifest.examples]
    if len(indices) != len(set(indices)):
        raise ValueError("Training subset contains duplicate parquet row indices.")
    parquet = pq.ParquetFile(source)
    if any(index < 0 or index >= parquet.metadata.num_rows for index in indices):
        raise ValueError("Training subset contains an out-of-range parquet row index.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stream small batches instead of taking rows from the full table. The
    # training parquet contains large embedded issue payloads; concatenating
    # all source string chunks can overflow Arrow's 32-bit string offsets even
    # though the selected 208 rows are modest. Source-row order is deterministic
    # and immaterial because Verl shuffles prompt groups during training.
    selected_indices = set(indices)
    temporary = output.with_suffix(output.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, parquet.schema_arrow, compression="zstd")
    source_offset = 0
    written = 0
    try:
        for batch in parquet.iter_batches(batch_size=32):
            local_indices = [
                index - source_offset
                for index in sorted(selected_indices)
                if source_offset <= index < source_offset + batch.num_rows
            ]
            if local_indices:
                writer.write_batch(batch.take(local_indices))
                written += len(local_indices)
            source_offset += batch.num_rows
    finally:
        writer.close()
    if written != len(indices):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Materialized {written} rows but expected {len(indices)}."
        )
    temporary.replace(output)
    return _sha256(output)


def group_advantages(rewards: Sequence[float]) -> tuple[float, ...]:
    """Match Verl's sample-standard-deviation normalization for one group."""

    if not rewards:
        raise ValueError("At least one reward is required.")
    if any(not math.isfinite(value) for value in rewards):
        raise ValueError("Rewards must be finite.")
    if len(rewards) == 1:
        return (0.0,)
    mean = statistics.fmean(rewards)
    deviation = statistics.stdev(rewards)
    if deviation == 0:
        return tuple(0.0 for _ in rewards)
    return tuple((value - mean) / (deviation + 1e-6) for value in rewards)


def terminal_reward_vector(
    *,
    response_mask: Sequence[int],
    reward: float,
) -> tuple[float, ...]:
    """Place outcome reward on the last generated token, as Verl expects."""

    vector = [0.0] * len(response_mask)
    generated = [index for index, value in enumerate(response_mask) if value]
    if generated:
        vector[generated[-1]] = float(reward)
    return tuple(vector)


def model_token_mask(
    turns: Iterable[tuple[str, int]],
    *,
    limit: int | None = None,
) -> tuple[int, ...]:
    """Build a response mask from ``(source, token_count)`` turn metadata."""

    values: list[int] = []
    for source, count in turns:
        if source not in {"model", "environment"}:
            raise ValueError(f"Unsupported token source: {source}")
        if count < 0:
            raise ValueError("Token counts cannot be negative.")
        values.extend([1 if source == "model" else 0] * count)
    return tuple(values if limit is None else values[:limit])


def sanitize_observation(tool: str, observation: str) -> str:
    """Describe an observation without exporting repository or hidden-user text."""

    category = (
        "simulator reply"
        if tool == "ask_user"
        else "finish validation"
        if tool == "finish"
        else "repository observation"
    )
    return f"<{category}: {len(observation)} characters redacted>"


def sanitize_action_arguments(
    tool: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Retain structural navigation metadata, never arbitrary model text."""

    safe: dict[str, Any] = {}

    def label(name: str) -> None:
        value = arguments.get(name)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_./*?@:-]{0,240}", value):
            safe[name] = value
        elif value is not None:
            safe[name] = f"<{name}: redacted>"

    def integer(name: str) -> None:
        value = arguments.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            safe[name] = value

    if tool == "list_tree":
        label("path")
        integer("max_depth")
        integer("max_entries")
    elif tool == "find_symbol":
        label("name")
        label("path")
        label("kind")
        integer("max_results")
    elif tool == "search_code":
        query = arguments.get("query")
        safe["query"] = f"<search query: {len(query) if isinstance(query, str) else 0} characters redacted>"
        label("path")
        label("glob")
    elif tool == "read_file":
        label("path")
        integer("start_line")
        integer("end_line")
    elif tool == "ask_user":
        question = arguments.get("question")
        safe["question"] = (
            f"<agent question: {len(question) if isinstance(question, str) else 0} "
            "characters redacted>"
        )
    elif tool == "finish":
        functions = arguments.get("functions")
        if isinstance(functions, list) and all(
            isinstance(value, str)
            and re.fullmatch(r"[A-Za-z0-9_./-]+:[A-Za-z0-9_.<>-]+", value)
            for value in functions
        ):
            safe["functions"] = functions
        else:
            safe["functions"] = "<invalid finish arguments redacted>"
    return safe


def sanitize_report(report: RunReport) -> dict[str, Any]:
    """Export learner-visible behavior without hidden labels or repository data."""

    return {
        "instance_id": report.episode.instance_id,
        "repository": report.episode.repository,
        "visible_issue": report.episode.visible_issue,
        "is_vague": report.episode.is_vague,
        "preference": {
            "name": report.episode.preference.name,
            "description": report.episode.preference.description,
        },
        "model": report.model,
        "simulator": report.simulator,
        "predicted_functions": list(report.predicted_functions),
        "reward": asdict(report.reward),
        "trajectory": [
            {
                "turn": step.turn,
                "attempt": step.attempt,
                "tool": step.action.tool,
                "arguments": sanitize_action_arguments(
                    step.action.tool,
                    step.action.arguments,
                ),
                "reasoning": (
                    f"<model reasoning: {len(step.action.reasoning)} "
                    "characters redacted>"
                ),
                "observation": sanitize_observation(
                    step.action.tool,
                    step.observation,
                ),
                "executed": step.executed,
                "duplicate_suppressed": step.duplicate_suppressed,
            }
            for step in report.trajectory
        ],
        "termination": report.termination,
        "model_calls": report.model_calls,
        "finish_validation_passed": report.finish_validation_passed,
        "finish_correction_attempted": report.finish_correction_attempted,
        "invalid_predictions": list(report.invalid_predictions),
        "inference_seed": report.inference_seed,
    }


def build_training_artifact(
    *,
    config: PPPTrainingConfig,
    subset: TrainingSubsetManifest,
    stage: str,
    reports: Sequence[RunReport],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in {
        "deterministic_smoke",
        "compatibility_gate",
        "live_group",
        "training",
        "evaluation",
    }:
        raise ValueError(f"Unsupported artifact stage: {stage}")
    return {
        "schema_version": 1,
        "stage": stage,
        "config_identity": config.identity,
        "subset_identity": subset.identity,
        "configuration": asdict(config),
        "dataset": {
            "path": subset.dataset,
            "sha256": subset.dataset_sha256,
            "selection_seed": subset.selection_seed,
            "issue_groups": list(subset.issue_groups),
        },
        "reports": [sanitize_report(report) for report in reports],
        "metrics": metrics or {},
    }


def write_training_artifact(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    forbidden = ("GEMINI_API_KEY", "LAMBDA_API_KEY", "OPENAI_API_KEY")
    if any(name in text for name in forbidden):
        raise ValueError("Refusing to export an artifact containing a secret key name.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
