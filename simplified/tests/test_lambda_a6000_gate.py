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
    assert "--accelerator a6000|h100|gh200" in result.stdout
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


def test_lambda_gate_rejects_unknown_accelerator_before_host_probe():
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--doctor",
            "--accelerator",
            "unknown",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--accelerator must be a6000, h100, or gh200" in result.stderr


def test_gh200_gate_is_arm64_and_binary_wheel_only():
    source = SCRIPT.read_text()

    assert 'if [[ "$accelerator" == "gh200" && "$host_arch" != "aarch64" ]]' in source
    assert 'minimum_gpu_memory_mib=90000' in source
    assert 'expected_gpu_name="GH200"' in source
    assert source.count("--only-binary=:all:") == 5
    assert "--index-url https://download.pytorch.org/whl/cu128" in source
    assert "torch==2.9.0+cu128" in source
    assert (
        "--no-deps --no-binary=:all: antlr4-python3-runtime==4.9.3"
        in source
    )
    assert (
        "nvidia-cusparselt-cu12 0.7.1 is not supported on this platform"
        in source
    )
    assert (
        "nvidia-cusparselt-cu13 0.8.0 is not supported on this platform"
        in source
    )
    assert "libcusparseLt.so.0" in source
    assert source.index('command -v nvidia-smi') < source.index(
        'if [[ "$mode" == "bootstrap" ]]'
    )


def test_lambda_requirements_support_transformers_five():
    requirements = (
        ROOT / "simplified" / "requirements.lambda-a6000.txt"
    ).read_text()
    assert "safetensors>=0.8.0" in requirements
    assert "safetensors==0.5.3" not in requirements
    assert "ray[default]>=2.48.0,<3" in requirements
    assert "unidiff==0.7.5" in requirements


def test_lambda_doctor_checks_the_exact_async_rollout_contract():
    source = SCRIPT.read_text()

    assert "vllm_async_server import vLLMReplica" in source
    assert "vllm_rollout import WorkerWrapperBase" in source
    assert "create_worker_wrapper(" in source
    assert "isinstance(worker_wrapper, WorkerWrapperBase)" in source
    assert 'for name in ("init_device", "execute_model")' in source
    assert 'worker_dispatch_mode = "legacy" if' in source
    assert 'if "enable_log_requests" not in async_llm_parameters:' in source
    assert "inspect.signature(init_app_state).parameters" in source
    assert '("engine_client", "vllm_config", "state", "args")' in source
    assert '"supported_tasks"' in source
    assert 'default is not inspect.Parameter.empty' in source
    assert "if not app_state_supported:" in source
    assert "StructuredOutputsParams(json=finish_schema)" in source
    assert "action_json_schema((\"finish\",))" in source
    assert "structured_outputs=structured" in source
    assert "logprobs=0" in source
    assert 'pathlib.Path(sysconfig.get_path("include")) / "Python.h"' in source


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
