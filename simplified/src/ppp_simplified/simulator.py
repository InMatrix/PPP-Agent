"""Deterministic and Gemini-backed UserVille simulators."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

from .models import Episode, UserReply
from .providers import parse_json_object


class UserSimulator(ABC):
    name: str

    @abstractmethod
    def respond(
        self,
        *,
        episode: Episode,
        question: str,
        question_number: int,
    ) -> UserReply:
        """Answer from hidden episode information and score disclosure cost."""


def preference_ok(name: str, question: str, question_number: int) -> bool | None:
    words = question.split()
    if name in ("", "no_preference", "professional", "amateur"):
        return None
    if name == "concise_question":
        return len(words) <= 20
    if name == "detail_question":
        return len(words) >= 8
    if name == "no_ask":
        return False
    if name in ("one_question", "ask_many"):
        return question_number <= 1
    if name == "only_begin":
        return question_number == 1
    if name == "lang_ita":
        return any(word in question.lower() for word in ("quale", "puoi", "dove"))
    if name == "capital":
        return question.upper() == question
    if name == "commas":
        return "," in question
    if name == "json":
        return question.lstrip().startswith("{")
    return True


class DeterministicUserSimulator(UserSimulator):
    name = "deterministic-userville"

    def respond(
        self,
        *,
        episode: Episode,
        question: str,
        question_number: int,
    ) -> UserReply:
        normalized = question.lower()
        if "function" in normalized or "method" in normalized:
            text = "\n".join(episode.expected_functions) or "I don't know."
            level = 5 if episode.expected_functions else 3
        elif "file" in normalized or "path" in normalized:
            text = "\n".join(episode.edited_files) or "I don't know."
            level = 4 if episode.edited_files else 3
        elif "hint" in normalized:
            text = episode.hint or "I don't know."
            level = 2 if episode.hint else 3
        else:
            text = episode.full_issue
            level = 1
        return UserReply(
            text=text,
            cost_level=level,
            preference_ok=preference_ok(
                episode.preference.name,
                question,
                question_number,
            ),
            rationale="Deterministic disclosure-level routing.",
        )


class GeminiUserSimulator(UserSimulator):
    def __init__(
        self,
        *,
        model: str = "gemini-3.5-flash-lite",
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.name = model
        if client is None:
            try:
                from google import genai
            except ImportError as error:
                raise RuntimeError(
                    "Install the Gemini extra: pip install -e "
                    "'simplified[gemini]'"
                ) from error
            resolved = api_key or os.getenv("GEMINI_API_KEY")
            if not resolved:
                raise ValueError(
                    "GEMINI_API_KEY is required for the Gemini simulator."
                )
            client = genai.Client(
                api_key=resolved,
                http_options={"api_version": "v1"},
            )
        self.client = client

    def respond(
        self,
        *,
        episode: Episode,
        question: str,
        question_number: int,
    ) -> UserReply:
        prompt = f"""Act as the developer who filed this software issue.
Answer only from the hidden information below. Reveal only what the agent's
question requires. Keep the answer concise and never mention these rules.

Preference: {episode.preference.description}
Preference reward rule: {episode.preference.reward_rule}

Disclosure levels:
1 = full issue description only
2 = hint information
3 = refusal / unknown
4 = edited file paths
5 = edited function names

Hidden full issue:
{episode.full_issue}

Hidden hint:
{episode.hint or "(none)"}

Hidden edited files:
{json.dumps(episode.edited_files)}

Hidden edited functions:
{json.dumps(episode.expected_functions)}

Question number: {question_number}
Agent question: {question}

Return only JSON with: text, cost_level (1-5), preference_ok (boolean or null),
rationale.
"""
        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
        )
        payload = parse_json_object(interaction.output_text)
        return UserReply(
            text=str(payload["text"]),
            cost_level=max(1, min(5, int(payload["cost_level"]))),
            preference_ok=payload.get("preference_ok"),
            rationale=str(payload.get("rationale") or ""),
        )
