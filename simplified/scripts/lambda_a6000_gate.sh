#!/usr/bin/env bash
# Prepare or inspect a single-GPU Lambda host.  This script never calls
# the Lambda API, starts an instance, reads a .env file, or sends an API key.
set -euo pipefail

mode="doctor"
repo_root=""
venv_dir=""
model="qwen35"
accelerator="a6000"

usage() {
  cat <<'EOF'
Usage: simplified/scripts/lambda_a6000_gate.sh [--doctor|--bootstrap] [--model qwen35|qwen3] [--accelerator a6000|h100|gh200] [--repo-root PATH] [--venv PATH]

  --doctor     Verify a prepared host only (default; no network writes).
  --bootstrap  Create the Python environment and install pinned dependencies.
  --model      qwen35 is the primary compatibility attempt; qwen3 is the
               documented fallback after recording qwen35 failure evidence.
  --accelerator
               Validate the selected GPU and its minimum memory. gh200 also
               requires an ARM64 host and binary-only Python dependencies.

This script does not launch, stop, or otherwise control a Lambda instance.
EOF
}

while (($#)); do
  case "$1" in
    --doctor) mode="doctor" ;;
    --bootstrap) mode="bootstrap" ;;
    --model) model="${2:?missing value for --model}"; shift ;;
    --accelerator) accelerator="${2:?missing value for --accelerator}"; shift ;;
    --repo-root) repo_root="${2:?missing value for --repo-root}"; shift ;;
    --venv) venv_dir="${2:?missing value for --venv}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

case "$model" in
  qwen35)
    model_id="Qwen/Qwen3.5-4B"
    vllm_version="0.21.0"
    transformers_requirement="transformers>=5,<6"
    ;;
  qwen3)
    model_id="Qwen/Qwen3-4B"
    vllm_version="0.12.0"
    transformers_requirement="transformers>=4.51,<5"
    ;;
  *)
    echo "--model must be qwen35 or qwen3" >&2
    exit 2
    ;;
esac

case "$accelerator" in
  a6000)
    expected_gpu_name="A6000"
    minimum_gpu_memory_mib=46000
    gate_budget="maximum 2 hours / \$3 compute"
    ;;
  h100)
    expected_gpu_name="H100"
    minimum_gpu_memory_mib=78000
    gate_budget="one-step compatibility gate only"
    ;;
  gh200)
    expected_gpu_name="GH200"
    minimum_gpu_memory_mib=90000
    gate_budget="maximum 1 hour / \$3 compute"
    ;;
  *)
    echo "--accelerator must be a6000, h100, or gh200" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "$repo_root" ]]; then
  repo_root="$(cd "$script_dir/../.." && pwd)"
fi
if [[ -z "$venv_dir" ]]; then
  venv_dir="$repo_root/simplified/.venv-lambda"
fi

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

require_file "$repo_root/simplified/pyproject.toml"
require_file "$repo_root/simplified/requirements.lambda-a6000.txt"
require_file "$repo_root/verl/__init__.py"

# Probe the billed hardware before downloading packages or constructing an
# environment. A wrong instance should fail in seconds, not after a large
# PyTorch/vLLM transfer.
host_arch="$(uname -m)"
if [[ "$host_arch" == "arm64" ]]; then
  host_arch="aarch64"
fi
if [[ "$accelerator" == "gh200" && "$host_arch" != "aarch64" ]]; then
  echo "FAIL: GH200 requires an ARM64 host; uname -m reported: $host_arch" >&2
  exit 1
fi

command -v nvidia-smi >/dev/null || {
  echo "FAIL: nvidia-smi is unavailable; this is not a usable CUDA host." >&2
  exit 1
}
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
gpu_memory_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
if [[ "$gpu_name" != *"$expected_gpu_name"* ]]; then
  echo "FAIL: expected $expected_gpu_name for --accelerator $accelerator, found: $gpu_name" >&2
  exit 1
fi
if [[ "$gpu_memory_mib" -lt "$minimum_gpu_memory_mib" ]]; then
  echo "FAIL: expected at least ${minimum_gpu_memory_mib} MiB GPU memory, found ${gpu_memory_mib} MiB." >&2
  exit 1
fi

if [[ "$mode" == "bootstrap" ]]; then
  command -v python3.11 >/dev/null || {
    echo "Python 3.11 is required. Install it before bootstrapping." >&2
    exit 1
  }
  python3.11 - <<'PY' || {
import pathlib
import sysconfig

raise SystemExit(not (pathlib.Path(sysconfig.get_path("include")) / "Python.h").is_file())
PY
    echo "Python 3.11 development headers are required. Install python3.11-dev before bootstrapping." >&2
    exit 1
  }
  # A model mode gets its own environment. Refusing to reuse a directory keeps
  # the vLLM/Transformers 5 primary stack from being mixed with the
  # Transformers 4 fallback after a failed compatibility attempt.
  if [[ -e "$venv_dir" ]]; then
    echo "FAIL: environment already exists: $venv_dir" >&2
    echo "Choose a new --venv path for a fresh compatibility attempt." >&2
    exit 2
  fi
  python3.11 -m venv "$venv_dir"
  "$venv_dir/bin/python" -m pip install --only-binary=:all: --upgrade pip==25.0.1 wheel==0.45.1 setuptools==75.8.0
  # vLLM's compiled wheel is the binding dependency, so let its selected
  # release install the matching PyTorch build. Requiring wheels prevents an
  # ARM64 host from silently spending paid time compiling a missing package.
  "$venv_dir/bin/python" -m pip install --only-binary=:all: "vllm==$vllm_version" "$transformers_requirement"
  if [[ "$accelerator" == "gh200" && "$model" == "qwen3" ]]; then
    # PyPI's ARM64 torch 2.9 wheel is CPU-only. The official PyTorch CUDA index
    # publishes the ABI-compatible GH200 build plus its ARM64 CUDA libraries.
    "$venv_dir/bin/python" -m pip install --only-binary=:all: \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.9.0+cu128 torchvision==0.24.0 torchaudio==2.9.0
  fi
  # Hydra 1.3.2 pins antlr4-python3-runtime 4.9.*, for which PyPI publishes no
  # wheel. This audited package is pure Python; install it without dependencies
  # as the sole source-archive exception before restoring the wheel-only rule.
  "$venv_dir/bin/python" -m pip install --no-deps --no-binary=:all: antlr4-python3-runtime==4.9.3
  "$venv_dir/bin/python" -m pip install --only-binary=:all: -r "$repo_root/simplified/requirements.lambda-a6000.txt"
  "$venv_dir/bin/python" -m pip install --only-binary=:all: -e "$repo_root/simplified[test,gemini]"
  pip_check_status=0
  pip_check_output="$("$venv_dir/bin/python" -m pip check 2>&1)" || pip_check_status=$?
  if [[ "$pip_check_status" -ne 0 ]]; then
    # NVIDIA's 0.7.1 ARM64 wheel contains an SBSA tag internally even though
    # the CUDA index serves it as aarch64. pip check rejects that metadata.
    # Accept only this exact warning and only when the shared library exists.
    known_cusparselt_cu12_warning="nvidia-cusparselt-cu12 0.7.1 is not supported on this platform"
    known_cusparselt_cu13_warning="nvidia-cusparselt-cu13 0.8.0 is not supported on this platform"
    unexpected_check_output="$(
      printf '%s\n' "$pip_check_output" |
        grep -Fvx \
          -e "$known_cusparselt_cu12_warning" \
          -e "$known_cusparselt_cu13_warning" || true
    )"
    cusparselt_library="$(
      find "$venv_dir/lib" -path '*/nvidia/cusparselt/lib/libcusparseLt.so.0' -print -quit
    )"
    if [[ "$accelerator" == "gh200" &&
          -n "$cusparselt_library" &&
          -z "$unexpected_check_output" ]]; then
      echo "WARN: accepted NVIDIA ARM64 cuSPARSELt wheel-tag mismatch: $cusparselt_library" >&2
    else
      printf '%s\n' "$pip_check_output" >&2
      exit "$pip_check_status"
    fi
  fi
fi

echo "Repository: $repo_root"
echo "Environment: $venv_dir"
echo "Host architecture: $host_arch"
echo "Accelerator gate: $accelerator ($gpu_name; ${gpu_memory_mib} MiB)"
echo "Model gate: $model_id"
echo "Rollout stack: vllm==$vllm_version; $transformers_requirement"
echo "Gate budget: $gate_budget"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ ! -x "$venv_dir/bin/python" ]]; then
  echo "FAIL: environment absent. Re-run with --bootstrap on the launched host." >&2
  exit 1
fi

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
"$venv_dir/bin/python" - <<'PY'
import importlib
import inspect
import pathlib
import platform
import sys
import sysconfig

print(f"python.architecture={platform.machine()}; platform={sysconfig.get_platform()}")
required = ("torch", "transformers", "vllm", "peft", "ray", "hydra", "omegaconf", "ppp_simplified", "verl")
missing = []
for name in required:
    try:
        module = importlib.import_module(name)
        print(f"{name}={getattr(module, '__version__', 'vendored')}")
    except Exception as exc:  # report every import problem before failing
        missing.append(f"{name}: {exc}")

if missing:
    print("FAIL: required imports:\n" + "\n".join(missing), file=sys.stderr)
    raise SystemExit(1)

python_header = pathlib.Path(sysconfig.get_path("include")) / "Python.h"
if not python_header.is_file():
    raise SystemExit(
        f"FAIL: missing {python_header}; install the Python development headers."
    )
print(f"python.headers={python_header}")

import torch
if not torch.cuda.is_available():
    raise SystemExit("FAIL: PyTorch cannot see CUDA.")
print(f"torch.cuda={torch.version.cuda}; device={torch.cuda.get_device_name(0)}")

# Import the exact async rollout path and verify the constructor contract used
# by the vendored server. Generic package imports did not catch these API
# drifts before paid GPU initialization.
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.entrypoints.openai.api_server import init_app_state
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from ppp_simplified.providers import action_json_schema
from verl.utils.vllm_compat import create_worker_wrapper
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
from verl.workers.rollout.vllm_rollout.vllm_rollout import WorkerWrapperBase

async_llm_parameters = inspect.signature(AsyncLLM.from_vllm_config).parameters
if "enable_log_requests" not in async_llm_parameters:
    raise SystemExit(
        "FAIL: AsyncLLM.from_vllm_config lacks enable_log_requests; "
        "the vendored rollout adapter and installed vLLM are incompatible."
    )
app_state_parameters = inspect.signature(init_app_state).parameters
app_state_names = tuple(app_state_parameters)
app_state_supported = (
    app_state_names == ("engine_client", "state", "args")
    or app_state_names == ("engine_client", "vllm_config", "state", "args")
    or (
        app_state_names == ("engine_client", "state", "args", "supported_tasks")
        and app_state_parameters["supported_tasks"].default is not inspect.Parameter.empty
    )
)
if not app_state_supported:
    raise SystemExit(
        "FAIL: init_app_state has an unsupported signature: "
        f"{inspect.signature(init_app_state)}"
    )
worker_wrapper = create_worker_wrapper(
    WorkerWrapperBase,
    vllm_config=object(),
)
if not isinstance(worker_wrapper, WorkerWrapperBase):
    raise SystemExit("FAIL: vLLM worker-wrapper compatibility returned the wrong type.")
legacy_worker_dispatch = callable(
    getattr(WorkerWrapperBase, "execute_method", None)
)
direct_worker_dispatch = all(
    callable(getattr(WorkerWrapperBase, name, None))
    for name in ("init_device", "execute_model")
)
if not (legacy_worker_dispatch or direct_worker_dispatch):
    raise SystemExit("FAIL: vLLM worker wrapper has no supported dispatch API.")
worker_dispatch_mode = "legacy" if legacy_worker_dispatch else "direct"
finish_schema = action_json_schema(("finish",))
structured = StructuredOutputsParams(json=finish_schema)
sampling = SamplingParams(
    max_tokens=8,
    logprobs=0,
    structured_outputs=structured,
)
if sampling.structured_outputs.json != finish_schema:
    raise SystemExit(
        "FAIL: vLLM did not retain the finish-only structured-output schema."
    )
print(
    "verl.async_rollout=imported; "
    "AsyncLLM.enable_log_requests=supported; "
    f"init_app_state.args={len(app_state_parameters)}; "
    f"worker_wrapper.args={len(inspect.signature(WorkerWrapperBase).parameters)}; "
    f"worker_dispatch={worker_dispatch_mode}; "
    "structured_outputs.json=supported; logprobs=enabled"
)
PY

echo "PASS: host is ready for the next compatibility command."
echo "NEXT:"
echo "  $venv_dir/bin/ppp-train doctor"
echo "  $venv_dir/bin/ppp-train train --steps 1 --simulator deterministic --model $model"
echo "Do not start a paid training run from this script. Terminate the Lambda instance manually when this gate is complete."
