# Lambda single-GPU compatibility gate

This runbook prepares a GPU host for the first PPP reinforcement-learning
compatibility check. It does **not** launch Lambda infrastructure, run a model,
contact Gemini, or consume a secret.

The script retains its original `lambda_a6000_gate.sh` filename for command
compatibility, but now validates A6000, H100, and GH200 hosts explicitly.

## Current GH200 gate

The next attempt uses one GH200 96 GB. Its extra HBM gives the colocated actor
and rollout stack more headroom than the 80 GB H100 probe, while its Grace CPU
changes the host architecture from x86-64 to ARM64. Treat this as a fresh
package-compatibility result; do not copy the previous H100 virtual
environment.

Keep the repository on `codex/ppp-rl-4b` and ensure it is pushed or otherwise
available to the GPU host. Set a one-hour timer. Lambda bills a running
instance even when no command is active, so terminate it manually as soon as
the gate passes or fails.

## Host setup

From the GH200 host, first inspect the base image:

```bash
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3.11 --version
```

Expected results are `aarch64`, a GPU name containing `GH200`, at least 90,000
MiB of GPU memory, and Python 3.11. Then clone and bootstrap the Qwen3 fallback
stack that passed the earlier API-compatibility work:

```bash
git clone <YOUR_PPP_AGENT_REMOTE> PPP-Agent
cd PPP-Agent
git switch codex/ppp-rl-4b
sudo apt-get install -y python3.11-dev
bash simplified/scripts/lambda_a6000_gate.sh \
  --bootstrap \
  --accelerator gh200 \
  --model qwen3 \
  --venv simplified/.venv-lambda-qwen3
```

Hardware validation runs before any package download. Native and compiled
registry dependencies are installed with `--only-binary=:all:`. The sole
source-archive exception is `antlr4-python3-runtime==4.9.3`: Hydra pins that
version, PyPI provides no wheel, and the package is pure Python. If ARM64 lacks
any other required wheel, pip must fail rather than begin a paid source
compilation. Save that error, terminate the instance, and decide separately
whether a documented container or source-build path is worthwhile.

The script creates `simplified/.venv-lambda-qwen3`. The vendored `verl/` source
has no independent package manifest, so the script exports the repository root
through `PYTHONPATH` instead of pretending it is a published dependency.

This gate uses `vllm==0.12.0` and Transformers 4.x with
`Qwen/Qwen3-4B`. vLLM's compiled wheel selects its matching PyTorch package.
This is intentional: the compiled vLLM/PyTorch pair is the binding dependency.
A failed gate is a compatibility result—not permission to patch the algorithm
or reduce the rollout group.

To rerun only read-only host checks:

```bash
bash simplified/scripts/lambda_a6000_gate.sh \
  --doctor \
  --accelerator gh200 \
  --model qwen3 \
  --venv simplified/.venv-lambda-qwen3
```

The doctor prints host architecture, GPU model, memory, and driver
information; checks for at least 90,000 MiB; confirms CUDA is visible to
PyTorch; and imports the exact asynchronous vLLM/Verl path used during
training. Bootstrap also runs `pip check`. Neither command reads `.env` or
`GEMINI_API_KEY`.

## Run the gates

Prepare the frozen data and inspect the exact training command:

```bash
simplified/.venv-lambda-qwen3/bin/ppp-train doctor
simplified/.venv-lambda-qwen3/bin/ppp-train prepare

# Print only; this does not construct an optimizer.
simplified/.venv-lambda-qwen3/bin/ppp-train train \
  --steps 1 --simulator deterministic --model qwen3
```

Only after explicit paid-run confirmation, execute the one-step compatibility
gate:

```bash
export CONFIRM_PAID_TRAINING=I_UNDERSTAND_LAMBDA_IS_BILLING
simplified/.venv-lambda-qwen3/bin/ppp-train train \
  --steps 1 --simulator deterministic --model qwen3 --execute
```

That command must complete a model forward/generation pass, eight sequential
rollouts, old-policy log probabilities, FoldGRPO advantage calculation, one
DAPO backward/optimizer step, and checkpoint save. Inspect the rollout
artifact, peak memory, checkpoint, and console metrics before proceeding.
The command also writes scalar metrics, sanitized trajectories, a grouped
training report, and a Markdown summary under the compatibility run directory.

The paid launcher assumes a dedicated Lambda instance. Before training it
stops the current user's existing Ray runtime, waits up to 15 seconds for all
GPU compute PIDs to disappear, and refuses to continue on a contaminated GPU.
An exit trap stops Ray after success, failure, or interruption so orphaned
vLLM engine processes cannot consume memory in the next attempt. Do not use
this launcher on a shared host where another job intentionally owns Ray.

Run the same guarded one-step command a second time and confirm the log resumes
from `global_step_1`; the launcher records that successful reload. Do this
before the live simulator gate.

Do not enter a Gemini key until the deterministic gate passes. Store
`GEMINI_API_KEY` only in the instance-local environment. Run a text-only vLLM
server in one terminal, then use the no-optimizer live-group command in a
second terminal:

```bash
vllm serve Qwen/Qwen3-4B \
  --max-model-len 10240 --gpu-memory-utilization 0.20
ppp-train live-group --model qwen3
```

`live-group` produces exactly eight trajectories with distinct inference
seeds, uses the cached Gemini simulator, and exports only redacted repository
observations.

After the compatibility and live-group gates, start the 20-step run with
`ppp-train train --steps 20 --simulator gemini --model qwen3 --execute`.
Rerun that same command once to verify checkpoint resume. A 40-step extension
also requires `--projected-compute-usd AMOUNT`; the launcher refuses it unless
the saved report shows at least 25% non-flat groups, loss and KL are finite, a
LoRA adapter exists, resume was verified, and projected phase compute is at
most $45.

## Qwen3.5 primary-stack retry

Qwen3 is the current bounded fallback after Qwen3.5 compatibility work. Do not
retry the newer hybrid architecture during the GH200 memory gate. If a later
Qwen3.5 retry is approved, create a separate environment:

```bash
bash simplified/scripts/lambda_a6000_gate.sh \
  --bootstrap \
  --accelerator gh200 \
  --model qwen35 \
  --venv simplified/.venv-lambda-qwen35
```

Do not mix primary and fallback packages in one virtual environment. The
bootstrap script refuses to reuse an existing environment.

## Historical A6000 and H100 evidence

The original A6000 gate defined these escalation conditions:

- actor plus rollout components cannot load safely;
- peak allocated memory exceeds 46 GiB;
- a real optimizer step still fails after microbatch size 1, gradient
  checkpointing, conservative rollout memory, sequential rollouts, and the
  documented CPU offload; or
- fitting requires reducing the eight-rollout group or changing FoldGRPO.

The historical plan selected an H100 after A6000 availability and memory
constraints. The current GH200 gate supersedes that hardware choice without
weakening the eight-rollout learning objective.

The first H100 fallback probe showed that a colocated Qwen3-4B actor left about
20 GiB free. vLLM interprets `gpu_memory_utilization` against total device
memory, so the launcher reserves 20% (about 15.8 GiB on an 80 GiB H100) for
rollout rather than the original 35%. Group size, context limits, and training
semantics are unchanged.

The scripts encode BF16 base weights, rank-16 LoRA, 4K generated tokens, and
eight turns, but local preparation does not prove their combined GPU memory
footprint. The held-out evaluation suite remains sealed throughout this work.
