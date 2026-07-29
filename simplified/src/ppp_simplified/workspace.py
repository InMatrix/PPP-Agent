"""Prepare one pinned repository snapshot with ordinary Git."""

from __future__ import annotations

import fcntl
import json
import shutil
import subprocess
from pathlib import Path

from .models import Episode


class WorkspaceError(RuntimeError):
    pass


class RepositoryWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def path_for(self, episode: Episode) -> Path:
        safe_id = episode.instance_id.replace("/", "__")
        return self.root / safe_id

    def prepare(self, episode: Episode) -> Path:
        target = self.path_for(episode)
        # Verl expands one prompt into eight concurrent rollouts. They share
        # the same pinned repository snapshot, so serialize its first fetch
        # across threads and Ray worker processes. The marker remains the
        # source of truth after the lock is acquired.
        lock_dir = self.root / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{target.name}.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return self._prepare_locked(episode, target)

    def _prepare_locked(self, episode: Episode, target: Path) -> Path:
        marker = target / ".ppp-snapshot.json"
        expected = {
            "repository": episode.repository,
            "base_commit": episode.base_commit,
        }
        if marker.exists():
            if json.loads(marker.read_text()) == expected:
                return target
            raise WorkspaceError(
                f"{target} exists but points at a different repository snapshot."
            )
        if target.exists():
            raise WorkspaceError(
                f"{target} exists without a valid .ppp-snapshot.json marker."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        if not shutil.which("git"):
            raise WorkspaceError("Git is required to prepare a repository.")
        target.mkdir()
        try:
            self._git(target, "init")
            self._git(
                target,
                "remote",
                "add",
                "origin",
                f"https://github.com/{episode.repository}.git",
            )
            self._git(
                target,
                "fetch",
                "--depth",
                "1",
                "origin",
                episode.base_commit,
            )
            self._git(target, "checkout", "--detach", "FETCH_HEAD")
            marker.write_text(json.dumps(expected, indent=2) + "\n")
        except Exception:
            shutil.rmtree(target)
            raise
        return target

    @staticmethod
    def _git(target: Path, *arguments: str) -> None:
        result = subprocess.run(
            ["git", *arguments],
            cwd=target,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise WorkspaceError(
                f"git {' '.join(arguments)} failed: {detail}"
            )
