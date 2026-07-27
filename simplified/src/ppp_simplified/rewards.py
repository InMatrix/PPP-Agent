"""PPP reward decomposition aligned with the original UserVille formulas."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

from .models import Episode, RewardBreakdown, UserReply


def _clean_function(value: str) -> str:
    normalized = value.strip().lstrip("/")
    normalized = re.sub(
        r"^(?:\.?/)?(?:testbed/|workspace/)",
        "",
        normalized,
    )
    if ":" not in normalized:
        return os.path.normpath(normalized)
    path, symbol = normalized.split(":", 1)
    return f"{os.path.normpath(path)}:{symbol.strip()}"


def set_f1(predicted: Iterable[str], expected: Iterable[str]) -> float:
    pred = {_clean_function(item) for item in predicted if item.strip()}
    gold = {_clean_function(item) for item in expected if item.strip()}
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = len(pred & gold)
    return 0.0 if not overlap else 2 * overlap / (len(pred) + len(gold))


def _violation_penalty(preference_name: str) -> float:
    if preference_name in {
        "concise_question",
        "detail_question",
        "amateur",
        "first_try",
    }:
        return 0.1
    if preference_name in {"only_begin", "no_ask", "ask_many"}:
        return 1.0
    return 0.5


def calculate_reward(
    *,
    episode: Episode,
    predicted_functions: Iterable[str],
    replies: Iterable[UserReply],
) -> RewardBreakdown:
    reply_list = list(replies)
    productivity = set_f1(
        predicted_functions,
        episode.expected_functions,
    )
    levels = tuple(reply.cost_level for reply in reply_list)
    asked = bool(reply_list)
    if episode.is_vague:
        base_cost = (sum(levels) - len(levels)) * 0.1
        used_only_basic = asked and all(level == 1 for level in levels)
    else:
        base_cost = sum(levels) * 0.2
        used_only_basic = False
    proactivity = -base_cost
    if episode.is_vague and used_only_basic:
        proactivity += 0.05
    if episode.is_vague and not asked and productivity < 1.0:
        proactivity -= 0.1

    judgments = [
        reply.preference_ok
        for reply in reply_list
        if reply.preference_ok is not None
    ]
    personalization = 0.0
    preference_result: bool | None = None
    if judgments:
        preference_result = all(judgments)
        if preference_result and episode.is_vague and asked:
            personalization += 0.05
        elif not preference_result:
            personalization -= _violation_penalty(episode.preference.name)

    total = min(
        max(productivity + proactivity + personalization, 0.0),
        1.0,
    )
    return RewardBreakdown(
        productivity=productivity,
        proactivity_adjustment=proactivity,
        personalization_adjustment=personalization,
        total=total,
        questions_asked=len(reply_list),
        disclosure_levels=levels,
        preference_ok=preference_result,
    )
