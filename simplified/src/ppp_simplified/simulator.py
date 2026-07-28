"""Deterministic and Gemini-backed UserVille simulators."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

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


class SimulatorCallBudgetExceeded(RuntimeError):
    """Raised before a live simulator request would exceed its hard budget."""


class CachedUserSimulator(UserSimulator):
    """Persist exact simulator replies and bound new paid requests.

    SQLite gives concurrent rollout workers an atomic cache without placing
    hidden UserVille information in the exported training artifacts.
    """

    def __init__(
        self,
        simulator: UserSimulator,
        *,
        cache_path: Path,
        max_live_calls: int,
    ) -> None:
        if max_live_calls < 0:
            raise ValueError("max_live_calls cannot be negative.")
        self.simulator = simulator
        self.name = f"{simulator.name}+cache"
        self.cache_path = cache_path
        self.max_live_calls = max_live_calls
        self.live_calls = 0
        self._budget_each_attempt = isinstance(simulator, GeminiUserSimulator)
        if self._budget_each_attempt:
            if simulator.before_request is not None:
                raise ValueError(
                    "A budgeted Gemini simulator cannot already have a request hook."
                )
            simulator.before_request = self._reserve_live_call
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS replies (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS state (
                    name TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO state(name, value) VALUES ('live_calls', 0)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.cache_path, timeout=30)

    def _key(
        self,
        *,
        episode: Episode,
        question: str,
        question_number: int,
    ) -> str:
        import hashlib

        payload = {
            "simulator": getattr(
                self.simulator,
                "cache_identity",
                self.simulator.name,
            ),
            # Hash all context that can affect a simulator reply. This keeps
            # hidden data out of the SQLite key while invalidating stale
            # replies when the dataset or preference contract changes.
            "episode": asdict(episode),
            "question": question,
            "question_number": question_number,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def _reserve_live_call(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            used = int(
                connection.execute(
                    "SELECT value FROM state WHERE name = 'live_calls'"
                ).fetchone()[0]
            )
            if used >= self.max_live_calls:
                raise SimulatorCallBudgetExceeded(
                    "Simulator cache-wide live-call budget exhausted "
                    f"({self.max_live_calls})."
                )
            connection.execute(
                "UPDATE state SET value = value + 1 WHERE name = 'live_calls'"
            )
        self.live_calls += 1

    def respond(
        self,
        *,
        episode: Episode,
        question: str,
        question_number: int,
    ) -> UserReply:
        key = self._key(
            episode=episode,
            question=question,
            question_number=question_number,
        )
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM replies WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                return UserReply(**json.loads(row[0]))
        # Gemini invokes the guard before every physical API request, including
        # retries. Other simulators have one request per logical response.
        if not self._budget_each_attempt:
            self._reserve_live_call()
        reply = self.simulator.respond(
            episode=episode,
            question=question,
            question_number=question_number,
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO replies(cache_key, payload) VALUES (?, ?)",
                (key, json.dumps(asdict(reply), sort_keys=True)),
            )
        return reply


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
        max_attempts: int = 3,
        retry_delay_seconds: float = 1.0,
        before_request: Callable[[], None] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one.")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative.")
        self.model = model
        self.name = model
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.before_request = before_request
        self.cache_identity = f"{model}:userville-prompt-v1"
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
        for attempt in range(self.max_attempts):
            try:
                if self.before_request is not None:
                    self.before_request()
                interaction = self.client.interactions.create(
                    model=self.model,
                    input=prompt,
                )
                payload = parse_json_object(interaction.output_text)
                break
            except Exception:
                if attempt + 1 == self.max_attempts:
                    raise
                time.sleep(self.retry_delay_seconds * (2**attempt))
        return UserReply(
            text=str(payload["text"]),
            cost_level=max(1, min(5, int(payload["cost_level"]))),
            preference_ok=payload.get("preference_ok"),
            rationale=str(payload.get("rationale") or ""),
        )
