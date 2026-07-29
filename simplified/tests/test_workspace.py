import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ppp_simplified.workspace import RepositoryWorkspace, WorkspaceError


def _episode():
    return SimpleNamespace(
        instance_id="owner__repo-1",
        repository="owner/repo",
        base_commit="abc123",
    )


def test_concurrent_prepare_fetches_one_snapshot(tmp_path, monkeypatch):
    workspace = RepositoryWorkspace(tmp_path / "workspaces")
    git_calls = []

    def fake_git(target, *arguments):
        git_calls.append(arguments)
        time.sleep(0.02)

    monkeypatch.setattr(RepositoryWorkspace, "_git", staticmethod(fake_git))

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: workspace.prepare(_episode()), range(8)))

    assert len(set(paths)) == 1
    assert len(git_calls) == 4
    marker = paths[0] / ".ppp-snapshot.json"
    assert json.loads(marker.read_text()) == {
        "repository": "owner/repo",
        "base_commit": "abc123",
    }


def test_prepare_still_rejects_an_unmarked_existing_directory(tmp_path):
    workspace = RepositoryWorkspace(tmp_path / "workspaces")
    target = workspace.path_for(_episode())
    target.mkdir(parents=True)

    with pytest.raises(WorkspaceError, match="without a valid"):
        workspace.prepare(_episode())
