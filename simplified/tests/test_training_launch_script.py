import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "simplified/scripts/run_ppp_rl_4b.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _run_guarded_launcher(
    tmp_path: Path,
    *,
    training_status: int = 0,
    busy_gpu: bool = False,
    resolve_runtime_from_path: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for name in ("training-config.json", "training-subset.json"):
        (prepared / name).write_text("{}")

    log = tmp_path / "calls.log"
    fake_python = tmp_path / ("python3" if resolve_runtime_from_path else "python")
    _write_executable(
        fake_python,
        """#!/usr/bin/env bash
printf 'python %s\\n' "$*" >> "$PPP_TEST_LOG"
if [[ "${2:-}" == "scripts.train_ppp_simplified" ]]; then
  exit "${PPP_TEST_TRAIN_STATUS:-0}"
fi
""",
    )
    fake_ray = tmp_path / "ray"
    _write_executable(
        fake_ray,
        """#!/usr/bin/env bash
printf 'ray %s\\n' "$*" >> "$PPP_TEST_LOG"
""",
    )
    fake_nvidia_smi = tmp_path / "nvidia-smi"
    _write_executable(
        fake_nvidia_smi,
        "#!/usr/bin/env bash\n"
        + ("printf '4242\\n'\n" if busy_gpu else ""),
    )

    environment = os.environ.copy()
    environment.pop("PPP_PYTHON", None)
    environment.pop("PPP_RAY_CLI", None)
    environment.update({
        "CONFIRM_PAID_TRAINING": "I_UNDERSTAND_LAMBDA_IS_BILLING",
        "PPP_NVIDIA_SMI": str(fake_nvidia_smi),
        "PPP_PREPARED_DIR": str(prepared),
        "PPP_RUN_DIR": str(tmp_path / "run"),
        "PPP_GPU_IDLE_ATTEMPTS": "1",
        "PPP_GPU_IDLE_INTERVAL_SECONDS": "0",
        "PPP_TEST_LOG": str(log),
        "PPP_TEST_TRAIN_STATUS": str(training_status),
        "PPP_HOURLY_USD": "2.29",
    })
    if resolve_runtime_from_path:
        environment["PATH"] = f"{tmp_path}{os.pathsep}{environment['PATH']}"
    else:
        environment["PPP_PYTHON"] = str(fake_python)
        environment["PPP_RAY_CLI"] = str(fake_ray)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--execute",
            "--steps",
            "1",
            "--simulator",
            "deterministic",
            "--model",
            "qwen3",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


def test_training_launcher_is_print_only_by_default():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--steps", "1", "--simulator", "deterministic"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "scripts.train_ppp_simplified" in result.stdout
    assert "total_training_steps=1" in result.stdout
    assert "model.path=Qwen/Qwen3.5-4B" in result.stdout
    assert "model.override_config.attn_implementation=sdpa" in result.stdout
    assert "actor.ppo_mini_batch_size=1" in result.stdout
    assert "data.train_batch_size=1" in result.stdout
    assert "rollout.n=8" in result.stdout
    assert "rollout.enforce_eager=True" in result.stdout
    assert "rollout.gpu_memory_utilization=0.20" in result.stdout
    assert "rollout.plugin.workflow=search" in result.stdout
    assert "data.val_files=simplified/results/training/prepared/train-16x13.parquet" in result.stdout
    assert "data/test_id.parquet" not in result.stdout
    assert "VERL_FILE_LOGGER_PATH=" in result.stdout
    assert "VLLM_USE_V1=1" in result.stdout
    assert "agent_loop_config_path=simplified/config/verl_agent_loops.yaml" in result.stdout
    assert "language_model_only=True" in result.stdout


def test_training_launcher_uses_explicit_python_interpreter():
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--steps",
            "1",
            "--simulator",
            "deterministic",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={"PPP_PYTHON": "/opt/ppp/bin/python"},
    )
    assert result.returncode == 0
    assert "/opt/ppp/bin/python" in result.stdout


def test_training_launcher_has_explicit_qwen3_fallback():
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--steps",
            "1",
            "--simulator",
            "deterministic",
            "--model",
            "qwen3",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "model.path=Qwen/Qwen3-4B" in result.stdout
    assert "experiment_name=qwen3-4b" in result.stdout
    assert "language_model_only" not in result.stdout


def test_training_launcher_requires_explicit_paid_confirmation():
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--execute",
            "--steps",
            "1",
            "--simulator",
            "deterministic",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={},
    )
    assert result.returncode == 2
    assert "Paid execution guard" in result.stderr


def test_40_step_extension_requires_projected_compute_guard():
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--execute",
            "--steps",
            "40",
            "--simulator",
            "deterministic",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={"CONFIRM_PAID_TRAINING": "I_UNDERSTAND_LAMBDA_IS_BILLING"},
    )
    assert result.returncode == 2
    assert "--projected-compute-usd is required" in result.stderr


def test_paid_launcher_stops_ray_before_and_after_success(tmp_path):
    result, calls = _run_guarded_launcher(tmp_path)

    assert result.returncode == 0
    assert calls.count("ray stop --force") == 2
    assert any("scripts.train_ppp_simplified" in call for call in calls)
    assert any("summarize-run" in call for call in calls)
    assert any("--hourly-usd 2.29" in call for call in calls)


def test_paid_launcher_resolves_ray_from_activated_path(tmp_path):
    result, calls = _run_guarded_launcher(
        tmp_path,
        resolve_runtime_from_path=True,
    )

    assert result.returncode == 0
    assert calls.count("ray stop --force") == 2
    assert any("scripts.train_ppp_simplified" in call for call in calls)


def test_paid_launcher_stops_ray_after_training_failure(tmp_path):
    result, calls = _run_guarded_launcher(tmp_path, training_status=7)

    assert result.returncode == 7
    assert calls.count("ray stop --force") == 2
    assert any("scripts.train_ppp_simplified" in call for call in calls)
    assert not any("summarize-run" in call for call in calls)


def test_paid_launcher_refuses_busy_gpu_after_cleanup(tmp_path):
    result, calls = _run_guarded_launcher(tmp_path, busy_gpu=True)

    assert result.returncode == 2
    assert calls.count("ray stop --force") == 2
    assert not any("scripts.train_ppp_simplified" in call for call in calls)
    assert "GPU compute processes remain after Ray cleanup: 4242" in result.stderr
