"""Agent model providers with one shared action contract."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import httpx

from .models import AgentAction


ACTION_SCHEMA_V2 = """Return exactly one JSON object:
{
  "tool": "list_tree|find_symbol|search_code|read_file|ask_user|finish",
  "arguments": { ... },
  "reasoning": "one concise sentence"
}

Tool arguments:
- list_tree: {"path": "", "max_depth": 2, "max_entries": 120}
- find_symbol: {"name": "exact_name", "path": "", "kind": "any|class|function", "max_results": 50}
- search_code: {"query": "literal or regex", "path": "", "glob": "*.py"}
- read_file: {"path": "relative/file.py", "start_line": 1, "end_line": 220}
- ask_user: {"question": "one targeted, easy-to-answer question"}
- finish: {"functions": ["path/file.py:QualifiedName"]}
"""

ACTION_SCHEMA_V3 = """Return exactly one JSON object:
{
  "tool": "list_tree|find_symbol|inspect_symbol|search_code|read_file|ask_user|finish",
  "arguments": { ... },
  "reasoning": "one concise sentence"
}

Tool arguments:
- list_tree: {"path": "", "max_depth": 2, "max_entries": 120}
- find_symbol: {"name": "exact_name", "path": "", "kind": "any|class|function", "max_results": 50}
- inspect_symbol: {"path": "relative/file.py", "name": "QualifiedName", "context_lines": 12}
- search_code: {"query": "literal or regex", "path": "", "glob": "*.py"}
- read_file: {"path": "relative/file.py", "start_line": 1, "end_line": 220}
- ask_user: {"question": "one targeted, easy-to-answer question"}
- finish: {"functions": ["path/file.py:QualifiedName"]}
"""

TOOL_NAMES = (
    "list_tree",
    "find_symbol",
    "inspect_symbol",
    "search_code",
    "read_file",
    "ask_user",
    "finish",
)
TOOL_NAMES_V2 = (
    "list_tree",
    "find_symbol",
    "search_code",
    "read_file",
    "ask_user",
    "finish",
)


def action_json_schema(allowed_tools: Sequence[str]) -> dict[str, Any]:
    """Return the shared LM Studio/vLLM action schema."""

    selected_tools = tuple(allowed_tools)
    if not selected_tools:
        raise ValueError("At least one action tool must be allowed.")
    return {
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "enum": list(selected_tools),
            },
            "arguments": {"type": "object"},
            "reasoning": {"type": "string"},
        },
        "required": ["tool", "arguments", "reasoning"],
        "additionalProperties": False,
    }


def action_response_format(allowed_tools: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ppp_agent_action",
            "strict": True,
            "schema": action_json_schema(allowed_tools),
        },
    }


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if candidate.startswith("```"):
        candidate = (
            candidate.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Model did not return a JSON action: {text[:240]}")
        payload = json.loads(candidate[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model action must be a JSON object.")
    return payload


class AgentProvider(ABC):
    name: str

    @abstractmethod
    def next_action(
        self,
        *,
        system_prompt: str,
        messages: Sequence[dict[str, str]],
        allowed_tools: Sequence[str] | None = None,
    ) -> AgentAction:
        """Select the next repository or user action."""


class OpenAICompatibleAgent(AgentProvider):
    """Talk to Qwen through a local OpenAI-compatible serving endpoint."""

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3.5-4B",
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        seed: int | None = None,
        tool_schema_version: str = "v2",
        timeout_seconds: float = 300,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.name = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("QWEN_API_KEY") or "EMPTY"
        self.seed = seed
        if tool_schema_version not in {"v2", "v3"}:
            raise ValueError(
                f"Unsupported tool schema version: {tool_schema_version}"
            )
        self.tool_schema_version = tool_schema_version
        self.default_tools = (
            TOOL_NAMES if tool_schema_version == "v3" else TOOL_NAMES_V2
        )
        self.action_schema = (
            ACTION_SCHEMA_V3
            if tool_schema_version == "v3"
            else ACTION_SCHEMA_V2
        )
        self.timeout_seconds = timeout_seconds
        # LM Studio is a local service. Ignoring environment/system proxies keeps
        # localhost requests from being intercepted on proxy-configured Macs.
        self.client = client or httpx.Client(
            timeout=timeout_seconds,
            trust_env=False,
        )

    def next_action(
        self,
        *,
        system_prompt: str,
        messages: Sequence[dict[str, str]],
        allowed_tools: Sequence[str] | None = None,
    ) -> AgentAction:
        selected_tools = tuple(allowed_tools or self.default_tools)
        unknown = set(selected_tools) - set(self.default_tools)
        if not selected_tools or unknown:
            raise ValueError(f"Invalid allowed tools: {selected_tools}")
        restriction = (
            ""
            if selected_tools == self.default_tools
            else "\n\nOnly these tools are permitted now: "
            + ", ".join(selected_tools)
            + "."
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        request_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        system_prompt
                        + restriction
                        + "\n\n"
                        + self.action_schema
                    ),
                },
                *messages,
            ],
            "temperature": 0.2,
            "max_tokens": 900,
            "response_format": action_response_format(selected_tools),
        }
        if self.seed is not None:
            request_payload["seed"] = self.seed
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=request_payload,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        # LM Studio currently returns Qwen 3.5's schema-constrained payload in
        # reasoning_content even when the normal content field is empty.
        text = message.get("content") or message.get("reasoning_content") or ""
        payload = parse_json_object(text)
        if payload.get("tool") not in selected_tools:
            raise ValueError(
                f"Model selected disallowed tool: {payload.get('tool')}"
            )
        return AgentAction(
            tool=str(payload["tool"]),
            arguments=dict(payload.get("arguments") or {}),
            reasoning=str(payload.get("reasoning") or ""),
        )


class ScriptedSmokeAgent(AgentProvider):
    """Deterministic control path that tests plumbing, not model quality."""

    name = "scripted-smoke-agent"

    def __init__(self, expected_functions: Sequence[str]) -> None:
        self.expected_functions = tuple(expected_functions)
        self.turn = 0

    def next_action(
        self,
        *,
        system_prompt: str,
        messages: Sequence[dict[str, str]],
        allowed_tools: Sequence[str] | None = None,
    ) -> AgentAction:
        del system_prompt, messages
        if allowed_tools is not None and tuple(allowed_tools) == ("finish",):
            return AgentAction(
                tool="finish",
                arguments={"functions": list(self.expected_functions)},
                reasoning="Return the fixture answer on the finalization turn.",
            )
        self.turn += 1
        if self.turn == 1:
            return AgentAction(
                tool="list_tree",
                arguments={"path": "", "max_depth": 2, "max_entries": 30},
                reasoning="Verify that the pinned repository is accessible.",
            )
        if self.turn == 2:
            return AgentAction(
                tool="ask_user",
                arguments={
                    "question": (
                        "Which existing function names and file paths need "
                        "changing?"
                    )
                },
                reasoning="Request the highest-sensitivity answer to test UserVille.",
            )
        return AgentAction(
            tool="finish",
            arguments={"functions": list(self.expected_functions)},
            reasoning="Finish with the fixture answer to verify scoring.",
        )


def check_openai_compatible_endpoint(
    base_url: str,
    api_key: str | None = None,
) -> tuple[bool, str]:
    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
            timeout=10,
            trust_env=False,
        )
        response.raise_for_status()
        models = response.json().get("data", [])
        names = [str(item.get("id")) for item in models]
        return True, ", ".join(names) or "endpoint responded"
    except Exception as error:
        return False, str(error)
