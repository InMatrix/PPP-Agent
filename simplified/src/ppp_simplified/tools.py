"""A deliberately read-only tool surface for function localization."""

from __future__ import annotations

import ast
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models import AgentAction, UserReply


class ToolError(ValueError):
    pass


IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


@dataclass(frozen=True)
class SymbolLocation:
    path: str
    line: int
    end_line: int
    qualified_name: str
    kind: str


@dataclass(frozen=True)
class FunctionValidationError:
    value: str
    message: str


class PythonSymbolIndex:
    """Lazily index Python definitions with qualified names."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._locations: tuple[SymbolLocation, ...] | None = None

    @staticmethod
    def _ignored(path: Path) -> bool:
        return any(
            part.startswith(".") or part in IGNORED_DIRECTORY_NAMES
            for part in path.parts
        )

    def _build(self) -> tuple[SymbolLocation, ...]:
        locations: list[SymbolLocation] = []
        for path in sorted(self.root.rglob("*.py")):
            relative = path.relative_to(self.root)
            if self._ignored(relative) or path.stat().st_size > 2_000_000:
                continue
            try:
                tree = ast.parse(path.read_text(errors="replace"))
            except (OSError, SyntaxError, UnicodeError):
                continue

            class DefinitionVisitor(ast.NodeVisitor):
                def __init__(self) -> None:
                    self.parents: list[str] = []

                def _record(self, node: ast.AST, name: str, kind: str) -> None:
                    qualified = ".".join((*self.parents, name))
                    locations.append(
                        SymbolLocation(
                            path=relative.as_posix(),
                            line=int(getattr(node, "lineno", 1)),
                            end_line=int(
                                getattr(
                                    node,
                                    "end_lineno",
                                    getattr(node, "lineno", 1),
                                )
                            ),
                            qualified_name=qualified,
                            kind=kind,
                        )
                    )

                def visit_ClassDef(self, node: ast.ClassDef) -> None:
                    self._record(node, node.name, "class")
                    self.parents.append(node.name)
                    self.generic_visit(node)
                    self.parents.pop()

                def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                    self._record(node, node.name, "function")
                    self.parents.append(node.name)
                    self.generic_visit(node)
                    self.parents.pop()

                def visit_AsyncFunctionDef(
                    self,
                    node: ast.AsyncFunctionDef,
                ) -> None:
                    self._record(node, node.name, "function")
                    self.parents.append(node.name)
                    self.generic_visit(node)
                    self.parents.pop()

            DefinitionVisitor().visit(tree)
        return tuple(locations)

    @property
    def locations(self) -> tuple[SymbolLocation, ...]:
        if self._locations is None:
            self._locations = self._build()
        return self._locations

    def find(
        self,
        *,
        name: str,
        path_prefix: str = "",
        kind: str = "any",
        max_results: int = 50,
    ) -> tuple[SymbolLocation, ...]:
        matches = [
            location
            for location in self.locations
            if (
                location.qualified_name == name
                or location.qualified_name.rsplit(".", 1)[-1] == name
            )
            and (kind == "any" or location.kind == kind)
            and (
                not path_prefix
                or location.path == path_prefix
                or location.path.startswith(path_prefix.rstrip("/") + "/")
            )
        ]
        return tuple(matches[: max(1, min(max_results, 100))])

    def contains(self, path: str, qualified_name: str) -> bool:
        return any(
            location.path == path
            and location.qualified_name == qualified_name
            for location in self.locations
        )

    def resolve(
        self,
        path: str,
        qualified_name: str,
    ) -> SymbolLocation | None:
        return next(
            (
                location
                for location in self.locations
                if location.path == path
                and location.qualified_name == qualified_name
            ),
            None,
        )

    def siblings(
        self,
        location: SymbolLocation,
        *,
        max_results: int = 30,
    ) -> tuple[SymbolLocation, ...]:
        parent = location.qualified_name.rpartition(".")[0]
        matches = [
            candidate
            for candidate in self.locations
            if candidate.path == location.path
            and candidate != location
            and candidate.qualified_name.rpartition(".")[0] == parent
        ]
        matches.sort(key=lambda item: (abs(item.line - location.line), item.line))
        return tuple(matches[: max(1, min(max_results, 50))])


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
        self.symbols = PythonSymbolIndex(self.root)

    def execute(self, action: AgentAction) -> str:
        arguments = action.arguments
        if action.tool == "list_files":
            return self.list_files(
                str(arguments.get("path", "")),
                int(arguments.get("max_entries", 120)),
            )
        if action.tool == "list_tree":
            return self.list_tree(
                str(arguments.get("path", "")),
                int(arguments.get("max_depth", 2)),
                int(arguments.get("max_entries", 120)),
            )
        if action.tool == "find_symbol":
            return self.find_symbol(
                str(arguments.get("name", "")),
                str(arguments.get("path", "")),
                str(arguments.get("kind", "any")),
                int(arguments.get("max_results", 50)),
            )
        if action.tool == "inspect_symbol":
            return self.inspect_symbol(
                str(arguments.get("path", "")),
                str(arguments.get("name", "")),
                int(arguments.get("context_lines", 12)),
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
            functions = self.finish_functions(arguments)
            self.accept_finish(functions)
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
        """Deprecated recursive listing retained for offline compatibility."""

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

    def list_tree(
        self,
        relative: str,
        max_depth: int,
        max_entries: int,
    ) -> str:
        target = self._safe_path(relative)
        if not target.is_dir():
            raise ToolError(f"Not a directory: {relative}")
        depth_limit = max(1, min(max_depth, 4))
        entry_limit = max(1, min(max_entries, 500))
        entries: list[str] = []

        def visible(path: Path) -> bool:
            return (
                not path.name.startswith(".")
                and path.name not in IGNORED_DIRECTORY_NAMES
                and not path.is_symlink()
            )

        def walk(directory: Path, depth: int) -> None:
            if len(entries) >= entry_limit:
                return
            children = [path for path in directory.iterdir() if visible(path)]
            directories = sorted(path for path in children if path.is_dir())
            files = sorted(path for path in children if path.is_file())
            for path in (*directories, *files):
                display = path.relative_to(self.root).as_posix()
                entries.append(display + ("/" if path.is_dir() else ""))
                if len(entries) >= entry_limit:
                    return
            if depth >= depth_limit:
                return
            for path in directories:
                walk(path, depth + 1)
                if len(entries) >= entry_limit:
                    return

        walk(target, 1)
        return "\n".join(entries) or "(no entries)"

    def find_symbol(
        self,
        name: str,
        relative: str,
        kind: str,
        max_results: int,
    ) -> str:
        if not name.strip():
            raise ToolError("find_symbol requires a symbol name.")
        if kind not in {"any", "class", "function"}:
            raise ToolError("find_symbol kind must be any, class, or function.")
        target = self._safe_path(relative)
        if not target.exists():
            raise ToolError(f"Path does not exist: {relative}")
        prefix = target.relative_to(self.root).as_posix()
        if prefix == ".":
            prefix = ""
        matches = self.symbols.find(
            name=name.strip(),
            path_prefix=prefix,
            kind=kind,
            max_results=max_results,
        )
        if not matches:
            return "(no symbol definitions)"
        return "\n".join(
            f"{item.path}:{item.line}:{item.qualified_name} [{item.kind}]"
            for item in matches
        )

    def inspect_symbol(
        self,
        relative: str,
        name: str,
        context_lines: int,
    ) -> str:
        if not relative or not name.strip():
            raise ToolError("inspect_symbol requires path and name.")
        target = self._safe_path(relative)
        if not target.is_file() or target.suffix != ".py":
            raise ToolError(f"Not a Python file: {relative}")
        normalized = target.relative_to(self.root).as_posix()
        requested = name.strip()
        location = self.symbols.resolve(normalized, requested)
        if location is None:
            matches = self.symbols.find(
                name=requested,
                path_prefix=normalized,
                max_results=20,
            )
            if len(matches) == 1:
                location = matches[0]
            elif matches:
                choices = ", ".join(item.qualified_name for item in matches)
                raise ToolError(
                    f"Ambiguous symbol {requested!r}; use one of: {choices}"
                )
            else:
                raise ToolError(
                    f"Symbol {requested!r} is not defined in {normalized}."
                )

        context = max(0, min(context_lines, 40))
        start = max(1, location.line - context)
        end = location.end_line + context
        siblings = self.symbols.siblings(location)
        sibling_lines = (
            "\n".join(
                f"- {item.path}:{item.line}:{item.qualified_name} [{item.kind}]"
                for item in siblings
            )
            if siblings
            else "(none)"
        )
        return (
            f"SYMBOL\n{location.path}:{location.line}-"
            f"{location.end_line}:{location.qualified_name} "
            f"[{location.kind}]\n\n"
            f"NEARBY SIBLINGS\n{sibling_lines}\n\n"
            f"SOURCE\n{self.read_file(normalized, start, end)}"
        )

    @staticmethod
    def finish_functions(arguments: dict) -> tuple[str, ...]:
        raw = arguments.get("functions", arguments.get("answer", []))
        if isinstance(raw, str):
            return tuple(
                line.strip() for line in raw.splitlines() if line.strip()
            )
        if isinstance(raw, list):
            return tuple(
                str(item).strip() for item in raw if str(item).strip()
            )
        raise ToolError("finish expects a string or list of functions.")

    def accept_finish(self, functions: tuple[str, ...]) -> None:
        self.final_answer = functions

    def validate_functions(
        self,
        functions: tuple[str, ...],
    ) -> tuple[FunctionValidationError, ...]:
        errors: list[FunctionValidationError] = []
        if not functions:
            return (
                FunctionValidationError(
                    value="<empty>",
                    message="finish requires at least one function",
                ),
            )
        for value in functions:
            if ":" not in value:
                errors.append(
                    FunctionValidationError(
                        value=value,
                        message="expected path.py:QualifiedName",
                    )
                )
                continue
            raw_path, qualified_name = value.split(":", 1)
            raw_path = os.path.normpath(
                raw_path.strip().lstrip("/")
            ).replace(os.sep, "/")
            qualified_name = qualified_name.strip()
            try:
                target = self._safe_path(raw_path)
            except ToolError as error:
                errors.append(
                    FunctionValidationError(value=value, message=str(error))
                )
                continue
            if not target.is_file():
                errors.append(
                    FunctionValidationError(
                        value=value,
                        message="file does not exist",
                    )
                )
                continue
            if target.suffix != ".py":
                errors.append(
                    FunctionValidationError(
                        value=value,
                        message="only Python symbols are supported",
                    )
                )
                continue
            location = self.symbols.resolve(raw_path, qualified_name)
            if location is None:
                errors.append(
                    FunctionValidationError(
                        value=value,
                        message="symbol is not defined in that file",
                    )
                )
            elif location.kind != "function":
                errors.append(
                    FunctionValidationError(
                        value=value,
                        message=(
                            "expected an existing function or method, found "
                            f"{location.kind}"
                        ),
                    )
                )
        return tuple(errors)

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
