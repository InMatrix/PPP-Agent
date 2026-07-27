from pathlib import Path

import pytest

from ppp_simplified.models import UserReply
from ppp_simplified.tools import ReadOnlyRepositoryTools, ToolError


def tools(root: Path) -> ReadOnlyRepositoryTools:
    return ReadOnlyRepositoryTools(
        root,
        lambda question: UserReply(
            text=question,
            cost_level=1,
            preference_ok=True,
        ),
    )


def test_list_tree_is_shallow_directory_first_and_ignores_noise(
    tmp_path: Path,
) -> None:
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github/workflow.yml").write_text("ignored")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/mod.pyc").write_bytes(b"ignored")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/guide.rst").write_text("Guide")
    (tmp_path / "src/pkg").mkdir(parents=True)
    (tmp_path / "src/pkg/mod.py").write_text("def run():\n    pass\n")
    (tmp_path / "README.md").write_text("Read me")

    result = tools(tmp_path).list_tree("", max_depth=2, max_entries=20)

    assert result.splitlines() == [
        "docs/",
        "src/",
        "README.md",
        "docs/guide.rst",
        "src/pkg/",
    ]


def test_list_tree_caps_entries_and_rejects_path_escape(
    tmp_path: Path,
) -> None:
    for index in range(5):
        (tmp_path / f"file-{index}.py").write_text("")
    repository_tools = tools(tmp_path)

    assert len(
        repository_tools.list_tree("", max_depth=2, max_entries=2).splitlines()
    ) == 2
    with pytest.raises(ToolError, match="escapes"):
        repository_tools.list_tree("..", max_depth=2, max_entries=20)


def test_find_symbol_returns_qualified_python_definitions(
    tmp_path: Path,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/mod.py").write_text(
        "class Widget:\n"
        "    def run(self):\n"
        "        pass\n\n"
        "async def fetch():\n"
        "    pass\n"
    )
    (tmp_path / "pkg/broken.py").write_text("def broken(:\n")
    repository_tools = tools(tmp_path)

    assert repository_tools.find_symbol("Widget", "", "class", 50) == (
        "pkg/mod.py:1:Widget [class]"
    )
    assert repository_tools.find_symbol("run", "", "function", 50) == (
        "pkg/mod.py:2:Widget.run [function]"
    )
    assert repository_tools.find_symbol("fetch", "pkg", "any", 50) == (
        "pkg/mod.py:5:fetch [function]"
    )
    assert (
        repository_tools.find_symbol("missing", "", "any", 50)
        == "(no symbol definitions)"
    )


def test_find_symbol_respects_result_cap(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"mod_{index}.py").write_text(
            "def repeated():\n    pass\n"
        )

    result = tools(tmp_path).find_symbol(
        "repeated",
        "",
        "function",
        max_results=2,
    )

    assert len(result.splitlines()) == 2


def test_inspect_symbol_returns_canonical_name_source_and_siblings(
    tmp_path: Path,
) -> None:
    (tmp_path / "pkg.py").write_text(
        "class Widget:\n"
        "    def before(self):\n"
        "        return 1\n\n"
        "    def run(self, value):\n"
        "        changed = value + 1\n"
        "        return changed\n\n"
        "    def after(self):\n"
        "        return 2\n"
    )
    repository_tools = tools(tmp_path)

    result = repository_tools.inspect_symbol(
        "pkg.py",
        "run",
        context_lines=0,
    )

    assert "pkg.py:5-7:Widget.run [function]" in result
    assert "pkg.py:2:Widget.before [function]" in result
    assert "pkg.py:9:Widget.after [function]" in result
    assert "changed = value + 1" in result


def test_inspect_symbol_rejects_ambiguous_bare_names(tmp_path: Path) -> None:
    (tmp_path / "pkg.py").write_text(
        "class One:\n"
        "    def run(self):\n"
        "        pass\n\n"
        "class Two:\n"
        "    def run(self):\n"
        "        pass\n"
    )

    with pytest.raises(ToolError, match="Ambiguous symbol"):
        tools(tmp_path).inspect_symbol("pkg.py", "run", context_lines=0)


def test_finish_validation_checks_file_and_qualified_symbol(
    tmp_path: Path,
) -> None:
    (tmp_path / "mod.py").write_text(
        "class Widget:\n"
        "    def run(self):\n"
        "        pass\n"
    )
    repository_tools = tools(tmp_path)

    assert repository_tools.validate_functions(
        ("mod.py:Widget.run",)
    ) == ()
    errors = repository_tools.validate_functions(
        (
            "missing.py:run",
            "mod.py:Widget.missing",
            "malformed",
        )
    )

    assert [error.value for error in errors] == [
        "missing.py:run",
        "mod.py:Widget.missing",
        "malformed",
    ]
