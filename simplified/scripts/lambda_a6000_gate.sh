#!/usr/bin/env bash
# Prepare or inspect a single-GPU Lambda A6000 host.  This script never calls
# the Lambda API, starts an instance, reads a .env file, or sends an API key.
set -euo pipefail

mode="doctor"
repo_root=""
venv_dir=""
model="qwen35"

usage() {
  cat <<'EOF'
Usage: simplified/scripts/lambda_a6000_gate.sh [--doctor|--bootstrap] [--model qwen35|qwen3] [--repo-root PATH] [--venv PATH]

  --doctor     Verify a prepared host only (default; no network writes).
  --bootstrap  Create the Python environment and install pinned dependencies.
  --model      qwen35 is the primary compatibility attempt; qwen3 is the
               documented fallback after recording qwen35 failure evidence.

This script does not launch, stop, or otherwise control a Lambda instance.
EOF
}

while (($#)); do
  case "$1" in
    --doctor) mode="doctor" ;;
    --bootstrap) mode="bootstrap" ;;
    --model) model="${2:?missing value for --model}"; shift ;;
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
  "$venv_dir/bin/python" -m pip install --upgrade pip==25.0.1 wheel==0.45.1 setuptools==75.8.0
  # vLLM's compiled wheel is the binding dependency, so let its selected
  # release install the matching PyTorch build before adding the remaining
  # training layer.
  "$venv_dir/bin/python" -m pip install "vllm==$vllm_version" "$transformers_requirement"
  "$venv_dir/bin/python" -m pip install -r "$repo_root/simplified/requirements.lambda-a6000.txt"
  "$venv_dir/bin/python" -m pip install -e "$repo_root/simplified[test,gemini]"
  "$venv_dir/bin/python" -m pip check
fi

echo "Repository: $repo_root"
echo "Environment: $venv_dir"
echo "Model gate: $model_id"
echo "Rollout stack: vllm==$vllm_version; $transformers_requirement"
echo "Gate: A6000-first, maximum 2 hours / \$3 compute"

command -v nvidia-smi >/dev/null || {
  echo "FAIL: nvidia-smi is unavailable; this is not a usable CUDA host." >&2
  exit 1
}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
gpu_memory_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
if [[ "$gpu_name" != *"A6000"* ]]; then
  echo "WARN: expected an A6000 for this gate, found: $gpu_name" >&2
fi
if [[ "$gpu_memory_mib" -lt 46000 ]]; then
  echo "FAIL: expected at least 46 GiB usable GPU memory, found ${gpu_memory_mib} MiB." >&2
  exit 1
fi

if [[ ! -x "$venv_dir/bin/python" ]]; then
  echo "FAIL: environment absent. Re-run with --bootstrap on the launched host." >&2
  exit 1
fi

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
"$venv_dir/bin/python" - <<'PY'
import importlib
import inspect
import pathlib
import sys
import sysconfig

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
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

async_llm_parameters = inspect.signature(AsyncLLM.from_vllm_config).parameters
if "enable_log_requests" not in async_llm_parameters:
    raise SystemExit(
        "FAIL: AsyncLLM.from_vllm_config lacks enable_log_requests; "
        "the vendored rollout adapter and installed vLLM are incompatible."
    )
app_state_parameters = inspect.signature(init_app_state).parameters
app_state_supported = len(app_state_parameters) == 3 or (
    len(app_state_parameters) == 4 and "vllm_config" in app_state_parameters
)
if not app_state_supported:
    raise SystemExit(
        "FAIL: init_app_state has an unsupported signature: "
        f"{inspect.signature(init_app_state)}"
    )
print(
    "verl.async_rollout=imported; "
    "AsyncLLM.enable_log_requests=supported; "
    f"init_app_state.args={len(app_state_parameters)}"
)
PY

echo "PASS: host is ready for the next compatibility command."
echo "NEXT:"
echo "  $venv_dir/bin/ppp-train doctor"
echo "  $venv_dir/bin/ppp-train train --steps 1 --simulator deterministic --model $model"
echo "Do not start a paid training run from this script. Terminate the Lambda instance manually when this gate is complete."
