"""Load one real PPP parquet row without pulling in pandas or VERL."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .models import Episode, Preference


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


def load_episode(
    path: Path,
    *,
    row_index: int | None = None,
    instance_id: str | None = None,
    preference_name: str | None = None,
    vague: bool | None = True,
) -> Episode:
    """Load a single episode selected by row or semantic identifiers."""

    parquet = pq.ParquetFile(path)
    absolute_index = 0
    for batch in parquet.iter_batches(columns=["ability", "extra_info"]):
        for row in batch.to_pylist():
            extra_info = row.get("extra_info") or {}
            payload = _ability_payload(str(row.get("ability") or ""))
            selected = (
                absolute_index == row_index
                if row_index is not None
                else _matches(
                    payload,
                    extra_info,
                    instance_id=instance_id,
                    preference_name=preference_name,
                    vague=vague,
                )
            )
            if selected:
                full_issue = str(payload.get("problem_statement") or "")
                expected = tuple(payload.get("edited_functions") or ())
                return Episode(
                    instance_id=str(payload["instance_id"]),
                    repository=str(payload["repo"]),
                    base_commit=str(payload["base_commit"]),
                    visible_issue=_visible_issue(extra_info, full_issue),
                    full_issue=full_issue,
                    hint=str(
                        payload.get("hints_text")
                        or payload.get("hint")
                        or ""
                    ),
                    patch=str(payload.get("patch") or ""),
                    expected_functions=expected,
                    is_vague=bool(extra_info.get("is_vague", True)),
                    preference=_preference(extra_info),
                    source_path=str(path),
                    row_index=absolute_index,
                )
            absolute_index += 1
    selector = (
        f"row {row_index}"
        if row_index is not None
        else f"instance={instance_id!r}, preference={preference_name!r}"
    )
    raise LookupError(f"No episode matched {selector} in {path}.")
