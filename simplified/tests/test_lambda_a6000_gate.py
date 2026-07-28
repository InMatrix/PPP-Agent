import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "simplified/scripts/lambda_a6000_gate.sh"


def test_lambda_gate_help_is_local_and_describes_both_model_modes():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--model qwen35|qwen3" in result.stdout
    assert "does not launch, stop" in result.stdout


def test_lambda_gate_rejects_unknown_model_before_host_probe():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--doctor", "--model", "unknown"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--model must be qwen35 or qwen3" in result.stderr


def test_lambda_gate_keeps_distinct_rollout_stacks():
    source = SCRIPT.read_text()
    assert 'vllm_version="0.21.0"' in source
    assert 'transformers_requirement="transformers>=5,<6"' in source
    assert 'vllm_version="0.12.0"' in source
    assert 'transformers_requirement="transformers>=4.51,<5"' in source
    assert 'maximum 2 hours / \\$3 compute' in source


def test_lambda_requirements_support_transformers_five():
    requirements = (
        ROOT / "simplified" / "requirements.lambda-a6000.txt"
    ).read_text()
    assert "safetensors>=0.8.0" in requirements
    assert "safetensors==0.5.3" not in requirements


def test_vendored_verl_has_transformers_five_vision_alias():
    compatibility = (
        ROOT / "verl" / "utils" / "transformers_compat.py"
    ).read_text()
    assert "except ImportError:" in compatibility
    assert "AutoModelForVision2Seq = AutoModelForImageTextToText" in compatibility
    for relative in (
        "verl/utils/model.py",
        "verl/workers/fsdp_workers.py",
        "verl/model_merger/base_model_merger.py",
    ):
        source = (ROOT / relative).read_text()
        assert "verl.utils.transformers_compat import AutoModelForVision2Seq" in source
