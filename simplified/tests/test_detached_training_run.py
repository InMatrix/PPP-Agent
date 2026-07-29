import os
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).parents[2]
MANAGER = ROOT / "simplified/scripts/manage_ppp_rl_run.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _environment(
    tmp_path: Path,
    *,
    exit_code: int = 0,
    sleep_seconds: float = 0.1,
    write_cleanup: bool = True,
):
    runner = tmp_path / "fake-training-runner"
    _write_executable(
        runner,
        """#!/usr/bin/env bash
printf 'runner %s\\n' "$*"
sleep "$PPP_TEST_SLEEP_SECONDS"
if [[ "$PPP_TEST_WRITE_CLEANUP" == "true" ]]; then
  printf 'passed\\n' > "$PPP_LIFECYCLE_DIR/cleanup-status"
  printf '0\\n' > "$PPP_LIFECYCLE_DIR/gpu-processes-after-cleanup"
fi
exit "$PPP_TEST_EXIT_CODE"
""",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CONFIRM_PAID_TRAINING": "I_UNDERSTAND_LAMBDA_IS_BILLING",
            "PPP_LAUNCH_ROOT": str(tmp_path / "launches"),
            "PPP_TRAIN_RUNNER": str(runner),
            "PPP_TEST_EXIT_CODE": str(exit_code),
            "PPP_TEST_SLEEP_SECONDS": str(sleep_seconds),
            "PPP_TEST_WRITE_CLEANUP": str(write_cleanup).lower(),
        }
    )
    environment.pop("GEMINI_API_KEY", None)
    return environment


def _run_manager(environment: dict[str, str], *arguments: str):
    return subprocess.run(
        ["bash", str(MANAGER), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )


def _wait_for_terminal_status(
    environment: dict[str, str], run_id: str
) -> subprocess.CompletedProcess[str]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = _run_manager(environment, "status", run_id)
        if "status=succeeded" in result.stdout or "status=failed" in result.stdout:
            return result
        time.sleep(0.05)
    raise AssertionError(f"Detached run did not finish:\n{result.stdout}\n{result.stderr}")


def test_detached_start_returns_before_worker_and_records_success(tmp_path):
    environment = _environment(tmp_path, sleep_seconds=0.3)

    started_at = time.monotonic()
    start = _run_manager(
        environment,
        "start",
        "offline-success",
        "--steps",
        "20",
        "--simulator",
        "deterministic",
        "--model",
        "qwen3",
    )
    elapsed = time.monotonic() - started_at

    assert start.returncode == 0
    assert elapsed < 0.25
    assert "run_id=offline-success" in start.stdout
    state_dir = tmp_path / "launches/offline-success"
    worker_pid = int((state_dir / "pid").read_text().strip())
    assert "GEMINI" not in (state_dir / "command.txt").read_text()

    # Model the controlling SSH shell disappearing after `start` returns.
    os.kill(worker_pid, signal.SIGHUP)
    status = _wait_for_terminal_status(environment, "offline-success")
    assert status.returncode == 0
    assert "status=succeeded" in status.stdout
    assert "exit_code=0" in status.stdout
    assert "cleanup_status=passed" in status.stdout
    assert "gpu_processes_after_cleanup=0" in status.stdout
    assert (state_dir / "started-at-utc").is_file()
    assert (state_dir / "finished-at-utc").is_file()
    assert "runner --execute --steps 20" in (state_dir / "run.log").read_text()


def test_detached_failure_is_reconnectable_and_preserves_exit_code(tmp_path):
    environment = _environment(tmp_path, exit_code=7)

    start = _run_manager(
        environment,
        "start",
        "offline-failure",
        "--steps",
        "20",
        "--simulator",
        "deterministic",
        "--model",
        "qwen3",
    )
    assert start.returncode == 0

    status = _wait_for_terminal_status(environment, "offline-failure")
    assert status.returncode == 1
    assert "status=failed" in status.stdout
    assert "exit_code=7" in status.stdout
    assert "cleanup_status=passed" in status.stdout

    logs = _run_manager(
        environment,
        "logs",
        "offline-failure",
        "--lines",
        "5",
    )
    assert logs.returncode == 0
    assert "runner --execute --steps 20" in logs.stdout


def test_detached_worker_marks_unobserved_cleanup_when_runner_exits_early(tmp_path):
    environment = _environment(tmp_path, exit_code=2, write_cleanup=False)

    start = _run_manager(
        environment,
        "start",
        "offline-early-failure",
        "--steps",
        "20",
        "--simulator",
        "deterministic",
        "--model",
        "qwen3",
    )
    assert start.returncode == 0

    status = _wait_for_terminal_status(environment, "offline-early-failure")
    assert status.returncode == 1
    assert "status=failed" in status.stdout
    assert "exit_code=2" in status.stdout
    assert "cleanup_status=not-observed" in status.stdout


def test_detached_start_refuses_duplicate_or_unsafe_run_ids(tmp_path):
    environment = _environment(tmp_path)
    arguments = (
        "--steps",
        "20",
        "--simulator",
        "deterministic",
        "--model",
        "qwen3",
    )

    first = _run_manager(environment, "start", "unique-run", *arguments)
    duplicate = _run_manager(environment, "start", "unique-run", *arguments)
    unsafe = _run_manager(environment, "start", "../escape", *arguments)

    assert first.returncode == 0
    assert duplicate.returncode == 2
    assert "choose a new RUN_ID" in duplicate.stderr
    assert unsafe.returncode == 2
    assert "RUN_ID must contain" in unsafe.stderr
    _wait_for_terminal_status(environment, "unique-run")


def test_detached_gemini_start_checks_key_without_persisting_it(tmp_path):
    environment = _environment(tmp_path)
    missing = _run_manager(
        environment,
        "start",
        "gemini-missing-key",
        "--steps",
        "20",
        "--simulator",
        "gemini",
        "--model",
        "qwen3",
    )
    assert missing.returncode == 2
    assert "GEMINI_API_KEY is required" in missing.stderr
    assert not (tmp_path / "launches/gemini-missing-key").exists()

    secret = "test-only-do-not-persist"
    environment["GEMINI_API_KEY"] = secret
    started = _run_manager(
        environment,
        "start",
        "gemini-with-key",
        "--steps",
        "20",
        "--simulator",
        "gemini",
        "--model",
        "qwen3",
    )
    assert started.returncode == 0
    _wait_for_terminal_status(environment, "gemini-with-key")
    for path in (tmp_path / "launches/gemini-with-key").iterdir():
        if path.is_file():
            assert secret not in path.read_text()
