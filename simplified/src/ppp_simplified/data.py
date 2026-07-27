"""Load one real PPP parquet row without pulling in pandas or VERL."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow.parquet as pq

from .models import Episode, Preference


@dataclass(frozen=True)
class EpisodeReference:
    """Lightweight row metadata used to choose a batch without retaining patches."""

    row_index: int
    instance_id: str
    repository: str
    preference_name: str


def _ability_payload(ability: str) -> dict[str, Any]:
    if "@" not in ability:
        raise ValueError("Expected an environment payload after '@' in ability.")
    _, payload = ability.split("@", 1)
    return json.loads(payload)


def _visible_issue(extra_info: dict[str, Any], fallback: str) -> str:
    prompt = extra_info.get("prompt") or []
    user_messages = [
        item.get("content", "")
        for item in prompt
        if isinstance(item, dict) and item.get("role") == "user"
    ]
    if not user_messages:
        return fallback
    text = user_messages[-1]
    matches = re.findall(
        r"--- BEGIN ISSUE ---\s*(.*?)\s*--- END ISSUE ---",
        text,
        flags=re.DOTALL,
    )
    return matches[-1].strip() if matches else text.strip()


def _preference(extra_info: dict[str, Any]) -> Preference:
    data = extra_info.get("preference") or {}
    name = data.get("preference_name") or "no_preference"
    details = data.get(name) or {}
    return Preference(
        name=str(name),
        description=str(
            details.get("preference")
            or "The user does not have any specific preferences."
        ),
        reward_rule=str(details.get("reward") or "None"),
    )


def _matches(
    payload: dict[str, Any],
    extra_info: dict[str, Any],
    *,
    instance_id: str | None,
    preference_name: str | None,
    vague: bool | None,
) -> bool:
    if instance_id and payload.get("instance_id") != instance_id:
        return False
    if preference_name:
        actual = (extra_info.get("preference") or {}).get("preference_name")
        if actual != preference_name:
            return False
    if vague is not None and bool(extra_info.get("is_vague", True)) != vague:
        return False
    return True


def _episode_from_row(
    row: dict[str, Any],
    *,
    path: Path,
    row_index: int,
) -> Episode:
    extra_info = row.get("extra_info") or {}
    payload = _ability_payload(str(row.get("ability") or ""))
    full_issue = str(payload.get("problem_statement") or "")
    return Episode(
        instance_id=str(payload["instance_id"]),
        repository=str(payload["repo"]),
        base_commit=str(payload["base_commit"]),
        visible_issue=_visible_issue(extra_info, full_issue),
        full_issue=full_issue,
        hint=str(payload.get("hints_text") or payload.get("hint") or ""),
        patch=str(payload.get("patch") or ""),
        expected_functions=tuple(payload.get("edited_functions") or ()),
        is_vague=bool(extra_info.get("is_vague", True)),
        preference=_preference(extra_info),
        source_path=str(path),
        row_index=row_index,
    )


def iter_episodes(path: Path) -> Iterator[Episode]:
    """Stream fully parsed episodes in parquet row order."""

    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches(columns=["ability", "extra_info"]):
        for row in batch.to_pylist():
            yield _episode_from_row(row, path=path, row_index=row_index)
            row_index += 1


def load_episode(
    path: Path,
    *,
    row_index: int | None = None,
    instance_id: str | None = None,
    preference_name: str | None = None,
    vague: bool | None = True,
) -> Episode:
    """Load a single episode selected by row or semantic identifiers."""

    for episode in iter_episodes(path):
        selected = (
            episode.row_index == row_index
            if row_index is not None
            else (
                (not instance_id or episode.instance_id == instance_id)
                and (
                    not preference_name
                    or episode.preference.name == preference_name
                )
                and (vague is None or episode.is_vague == vague)
            )
        )
        if selected:
            return episode
    selector = (
        f"row {row_index}"
        if row_index is not None
        else f"instance={instance_id!r}, preference={preference_name!r}"
    )
    raise LookupError(f"No episode matched {selector} in {path}.")


def select_evaluation_sample(
    path: Path,
    *,
    sample_size: int,
    preferences: Sequence[str],
    seed: int = 7,
    anchor_instance: str | None = None,
    anchor_preference: str | None = None,
) -> tuple[Episode, ...]:
    """Choose distinct tasks and repositories across preference conditions."""

    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")
    selected_preferences = tuple(dict.fromkeys(preferences))
    if not selected_preferences:
        raise ValueError("At least one preference is required.")

    by_preference: dict[str, list[EpisodeReference]] = {
        name: [] for name in selected_preferences
    }
    anchor: EpisodeReference | None = None
    for episode in iter_episodes(path):
        name = episode.preference.name
        if name not in by_preference:
            continue
        reference = EpisodeReference(
            row_index=episode.row_index,
            instance_id=episode.instance_id,
            repository=episode.repository,
            preference_name=name,
        )
        by_preference[name].append(reference)
        if (
            anchor_instance
            and episode.instance_id == anchor_instance
            and (anchor_preference is None or name == anchor_preference)
        ):
            anchor = reference

    missing = [name for name, values in by_preference.items() if not values]
    if missing:
        raise LookupError(
            "No dataset rows matched preferences: " + ", ".join(missing)
        )

    rng = random.Random(seed)
    for values in by_preference.values():
        rng.shuffle(values)

    chosen: list[EpisodeReference] = []
    if anchor_instance:
        if anchor is None:
            raise LookupError(
                f"No anchor row matched instance={anchor_instance!r}, "
                f"preference={anchor_preference!r}."
            )
        chosen.append(anchor)

    used_rows = {item.row_index for item in chosen}
    used_instances = {item.instance_id for item in chosen}
    used_repositories = {item.repository for item in chosen}
    while len(chosen) < sample_size:
        preference = selected_preferences[
            len(chosen) % len(selected_preferences)
        ]
        candidates = [
            item
            for item in by_preference[preference]
            if item.row_index not in used_rows
        ]
        if not candidates:
            raise LookupError(
                f"Not enough rows to select {sample_size} evaluation episodes."
            )
        reference = next(
            (
                item
                for item in candidates
                if item.instance_id not in used_instances
                and item.repository not in used_repositories
            ),
            next(
                (
                    item
                    for item in candidates
                    if item.instance_id not in used_instances
                ),
                candidates[0],
            ),
        )
        chosen.append(reference)
        used_rows.add(reference.row_index)
        used_instances.add(reference.instance_id)
        used_repositories.add(reference.repository)

    return tuple(load_episode(path, row_index=item.row_index) for item in chosen)
