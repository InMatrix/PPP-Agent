"""A deliberately read-only tool surface for function localization."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

from .models import AgentAction, UserReply


class ToolError(ValueError):
    pass


class ReadOnlyRepositoryTools:
    def __init__(
        self,
        root: Path,
        ask_user: Callable[[str], UserReply],
    ) -> None:
        self.root = root.resolve()
        self.ask_user = ask_user
        self.user_replies: list[UserReply] = []
        self.final_answer: tuple[str, ...] | None = None

    def execute(self, action: AgentAction) -> str:
        arguments = action.arguments
        if action.tool == "list_files":
            return self.list_files(
                str(arguments.get("path", "")),
                int(arguments.get("max_entries", 120)),
            )
        if action.tool == "search_code":
            return self.search_code(
                str(arguments.get("query", "")),
                str(arguments.get("path", "")),
                str(arguments.get("glob", "*.py")),
            )
        if action.tool == "read_file":
            return self.read_file(
                str(arguments.get("path", "")),
                int(arguments.get("start_line", 1)),
                int(arguments.get("end_line", 220)),
            )
        if action.tool == "ask_user":
            question = str(arguments.get("question", "")).strip()
            if not question:
                raise ToolError("ask_user requires a non-empty question.")
            reply = self.ask_user(question)
            self.user_replies.append(reply)
            return (
                f"{reply.text}\n[Cost {reply.cost_level}]"
                + (
                    f"\n[Preference {'OK' if reply.preference_ok else 'VIOLATION'}]"
                    if reply.preference_ok is not None
                    else ""
                )
            )
        if action.tool == "finish":
            raw = arguments.get("functions", arguments.get("answer", []))
            if isinstance(raw, str):
                functions = tuple(
                    line.strip() for line in raw.splitlines() if line.strip()
                )
            elif isinstance(raw, list):
                functions = tuple(str(item).strip() for item in raw if str(item).strip())
            else:
                raise ToolError("finish expects a string or list of functions.")
            self.final_answer = functions
            return "Task finished."
        raise ToolError(f"Unsupported tool: {action.tool}")

    def _safe_path(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ToolError("Path escapes the repository workspace.")
        if ".git" in candidate.relative_to(self.root).parts:
            raise ToolError("Git metadata is outside the agent tool surface.")
        return candidate

    def list_files(self, relative: str, max_entries: int) -> str:
        target = self._safe_path(relative)
        if not target.is_dir():
            raise ToolError(f"Not a directory: {relative}")
        entries: list[str] = []
        for path in sorted(target.rglob("*")):
            if ".git" in path.parts or not path.is_file():
                continue
            entries.append(path.relative_to(self.root).as_posix())
            if len(entries) >= max(1, min(max_entries, 500)):
                break
        return "\n".join(entries) or "(no files)"

    def search_code(self, query: str, relative: str, glob: str) -> str:
        if not query:
            raise ToolError("search_code requires a query.")
        target = self._safe_path(relative)
        result = subprocess.run(
            [
                "rg",
                "-n",
                "--no-heading",
                "--color",
                "never",
                "-g",
                glob,
                "--",
                query,
                os.fspath(target),
            ],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise ToolError(result.stderr.strip() or "rg failed")
        output = result.stdout
        output = output.replace(f"{self.root}{os.sep}", "")
        lines = output.splitlines()
        return "\n".join(lines[:200]) if lines else "(no matches)"

    def read_file(self, relative: str, start_line: int, end_line: int) -> str:
        target = self._safe_path(relative)
        if not target.is_file():
            raise ToolError(f"Not a file: {relative}")
        if target.stat().st_size > 2_000_000:
            raise ToolError("Refusing to read a file larger than 2 MB.")
        start = max(1, start_line)
        end = max(start, min(end_line, start + 400))
        lines = target.read_text(errors="replace").splitlines()
        return "\n".join(
            f"{number:>5}  {lines[number - 1]}"
            for number in range(start, min(end, len(lines)) + 1)
        )
