import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "simplified/scripts/run_ppp_rl_4b.sh"


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
    assert "actor.ppo_mini_batch_size=1" in result.stdout
    assert "data.train_batch_size=1" in result.stdout
    assert "rollout.n=8" in result.stdout
    assert "data.val_files=simplified/results/training/prepared/train-16x13.parquet" in result.stdout
    assert "data/test_id.parquet" not in result.stdout
    assert "VERL_FILE_LOGGER_PATH=" in result.stdout
    assert "agent_loop_config_path=simplified/config/verl_agent_loops.yaml" in result.stdout
    assert "language_model_only=True" in result.stdout


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
