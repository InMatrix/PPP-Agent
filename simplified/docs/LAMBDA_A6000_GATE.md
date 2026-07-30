# Lambda single-GPU compatibility gate

This runbook prepares a GPU host for the first PPP reinforcement-learning
compatibility check. It does **not** launch Lambda infrastructure, run a model,
contact Gemini, or consume a secret.

The script retains its original `lambda_a6000_gate.sh` filename for command
compatibility, but now validates A6000, H100, and GH200 hosts explicitly.

Retrospective evidence from the nine GH200 compatibility attempts is indexed
in [`run-reports/README.md`](run-reports/README.md). Use
[`run-reports/TEMPLATE.md`](run-reports/TEMPLATE.md) for future attempts so a
sanitized report is durable before the instance is terminated.
The live status, next gate, and phase-level acceptance criteria are maintained
in [`PHASE_PLAN.md`](PHASE_PLAN.md).

## Next GH200 gate

The next attempt uses `Qwen/Qwen3.5-4B` on one GH200 96 GB. Its extra HBM gives
the colocated actor and rollout stack more headroom than the 80 GB H100 probe,
while its Grace CPU changes the host architecture from x86-64 to ARM64. Treat
this as a fresh model and package-compatibility result. Create a separate
Transformers 5/vLLM environment and leave the proven Qwen3 fallback
environment unchanged.

Keep the repository on `codex/ppp-rl-4b` and ensure it is pushed or otherwise
available to the GPU host. This branch passes the shared action JSON Schema
through the raw Verl/vLLM rollout path and directly records sanitized
invalid-action categories. Set a one-hour timer. Lambda bills a running
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
MiB of GPU memory, and Python 3.11. Then clone and bootstrap the Qwen3.5 primary
stack:

```bash
git clone <YOUR_PPP_AGENT_REMOTE> PPP-Agent
cd PPP-Agent
git switch codex/ppp-rl-4b
sudo apt-get install -y python3.11-dev
bash simplified/scripts/lambda_a6000_gate.sh \
  --bootstrap \
  --accelerator gh200 \
  --model qwen35 \
  --venv simplified/.venv-lambda-qwen35
```

Hardware validation runs before any package download. Native and compiled
registry dependencies are installed with `--only-binary=:all:`. The sole
source-archive exception is `antlr4-python3-runtime==4.9.3`: Hydra pins that
version, PyPI provides no wheel, and the package is pure Python. If ARM64 lacks
any other required wheel, pip must fail rather than begin a paid source
compilation. Save that error, terminate the instance, and decide separately
whether a documented container or source-build path is worthwhile.

The script creates `simplified/.venv-lambda-qwen35`. The vendored `verl/` source
has no independent package manifest, so the script exports the repository root
through `PYTHONPATH` instead of pretending it is a published dependency.

This gate uses `vllm==0.21.0` and Transformers 5.x with
`Qwen/Qwen3.5-4B`. It intentionally does not reuse the vLLM 0.12/Transformers
4 environment that established the Qwen3 fallback.

NVIDIA's `nvidia-cusparselt-cu12==0.7.1` and
`nvidia-cusparselt-cu13==0.8.0` ARM64 wheels can carry platform metadata that
`pip check` reports as unsupported even though the required AArch64 shared
library is installed. The bootstrap accepts only those exact warnings, only on
GH200, and only after finding `libcusparseLt.so.0`. Every other dependency
warning remains fatal.

A failed gate is a compatibility result—not permission to patch the algorithm
or reduce the rollout group.

To rerun only read-only host checks:

```bash
bash simplified/scripts/lambda_a6000_gate.sh \
  --doctor \
  --accelerator gh200 \
  --model qwen35 \
  --venv simplified/.venv-lambda-qwen35
```

The doctor prints host architecture, GPU model, memory, and driver
information; checks for at least 90,000 MiB; confirms CUDA is visible to
PyTorch; and imports the exact asynchronous vLLM/Verl path used during
training. It also constructs a finish-only structured-output request with
chosen-token log probabilities enabled. Bootstrap runs `pip check`. Neither
command reads `.env` or `GEMINI_API_KEY`.

## Run the gates

Prepare the frozen data and inspect the exact training command:

```bash
simplified/.venv-lambda-qwen35/bin/ppp-train doctor
simplified/.venv-lambda-qwen35/bin/ppp-train prepare

# Print only; this does not construct an optimizer.
simplified/.venv-lambda-qwen35/bin/ppp-train train \
  --steps 1 --simulator deterministic --model qwen35
```

Before Verl or an optimizer, run the targeted vLLM schema gate. The first
command is a dry run and does not load the model:

```bash
simplified/.venv-lambda-qwen35/bin/ppp-train schema-gate --model qwen35
```

After explicit paid-run confirmation, execute exactly two real completions
from one model load:

```bash
export CONFIRM_PAID_TRAINING=I_UNDERSTAND_LAMBDA_IS_BILLING
simplified/.venv-lambda-qwen35/bin/ppp-train schema-gate \
  --model qwen35 \
  --execute \
  --output simplified/results/training/attempt-15-schema-gate.json
```

The first completion permits all navigation-v2 tools. The second permits only
`finish`. The gate fails unless each output is exactly one JSON object with
the required keys, the selected tool belongs to that call's enum, and every
sampled token has a finite chosen-token log probability. Its artifact includes
schema hashes, tool names, counts, latency, versions, Torch-reported peak
allocation, and a sanitized failure stage. It never includes prompts, output
text, reasoning, or arguments. It does not construct Verl, Gemini, a LoRA
adapter, an optimizer, or an eight-rollout group.

If this gate fails, preserve its artifact and logs, analyze both completions,
write the next numbered run report, and stop before training. If it passes,
the same explicit confirmation permits the one-step compatibility gate:

```bash
simplified/.venv-lambda-qwen35/bin/ppp-train train \
  --steps 1 --simulator deterministic --model qwen35 --execute
```

That command must complete a schema-constrained model forward/generation pass,
eight sequential rollouts, old-policy log probabilities, FoldGRPO advantage
calculation, one DAPO backward/optimizer step, and checkpoint save. Inspect the rollout
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
vllm serve Qwen/Qwen3.5-4B \
  --max-model-len 10240 --gpu-memory-utilization 0.20
ppp-train live-group --model qwen35
```

`live-group` produces exactly eight trajectories with distinct inference
seeds, uses the cached Gemini simulator, and exports only redacted repository
observations.

When the first optimizer group is flat, do not infer the cause from aggregate
loss. Inspect all eight sanitized trajectories, including parse categories,
tool use, finish validation, termination, reward components, and masking. Only
then decide whether to verify one effective update by continuing the ordinary
checkpointed dataloader to total step 2:

```bash
source ~/.config/ppp-agent/gemini.env
export CONFIRM_PAID_TRAINING=I_UNDERSTAND_LAMBDA_IS_BILLING
export PPP_HOURLY_USD=2.29  # Replace if Lambda's displayed rate has changed.
simplified/.venv-lambda-qwen35/bin/ppp-train train \
  --steps 2 --simulator gemini --model qwen35 --execute
```

This bounded mode only accepts the verified `global_step_1` compatibility
checkpoint and its original prepared inputs. It resumes the saved dataloader
state, runs exactly the next eight-trajectory group, and rejects success unless
step 2 has a positive finite gradient norm and a different LoRA adapter hash.
It does not allow an alternate training parquet or prepared directory.

After the compatibility and live-group gates, start long paid work through the
detached controller. Do not own a 20-step job with a foreground SSH shell:

```bash
source ~/.config/ppp-agent/gemini.env
export CONFIRM_PAID_TRAINING=I_UNDERSTAND_LAMBDA_IS_BILLING
export PPP_PYTHON=/home/ubuntu/PPP-Agent/simplified/.venv-lambda-qwen35/bin/python
export PPP_HOURLY_USD=2.29  # Replace if Lambda's displayed rate has changed.

bash simplified/scripts/manage_ppp_rl_run.sh start attempt-15 \
  --steps 20 --simulator gemini --model qwen35
```

`start` prints the worker PID and returns after detaching stdin, stdout, and
stderr from the SSH session. It is then safe to disconnect. Reconnect and
inspect the same run with:

```bash
bash simplified/scripts/manage_ppp_rl_run.sh status attempt-15
bash simplified/scripts/manage_ppp_rl_run.sh logs attempt-15 --lines 100
bash simplified/scripts/manage_ppp_rl_run.sh logs attempt-15 --follow
```

Each run ID is immutable and path-safe. Its ignored lifecycle directory under
`simplified/results/training/launches/` contains:

- the detached worker PID and sanitized command;
- start, worker-start, and finish timestamps;
- `starting`, `running`, `succeeded`, or `failed` status;
- the process exit code and combined run log; and
- Ray cleanup status plus the observed post-cleanup GPU-process count.

The controller inherits the instance-local Gemini key but never writes it to
the command or lifecycle files. If `status` reports `orphaned`, `failed`
cleanup, or a nonzero GPU-process count, inspect the log and terminate the
instance rather than starting another run.

On checkpoint steps, Verl now fsyncs a `pre_checkpoint` scalar record before
serializing the checkpoint. A normally completed step writes a second, fuller
record afterward. Consumers use the last record for a step, while an
interrupted run retains the policy loss, gradient norm, KL, global step, and
epoch needed for diagnosis.

Rerun the same 20-step training configuration with a new lifecycle run ID to
verify checkpoint resume. A 40-step extension also requires
`--projected-compute-usd AMOUNT`; the launcher refuses it unless the saved
report shows at least 25% non-flat groups, loss and KL are finite, a LoRA
adapter exists, resume was verified, and projected phase compute is at most
$45.

## Qwen3 fallback preservation

Qwen3 remains the proven bounded fallback. Use it only after the Qwen3.5 gate
records a failure that would require invasive Verl changes. Create a separate
environment rather than modifying or reusing the primary stack:

```bash
bash simplified/scripts/lambda_a6000_gate.sh \
  --bootstrap \
  --accelerator gh200 \
  --model qwen3 \
  --venv simplified/.venv-lambda-qwen3
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
