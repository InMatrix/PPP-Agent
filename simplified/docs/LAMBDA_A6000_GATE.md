# Lambda A6000 compatibility gate

This runbook prepares a GPU host for the first PPP reinforcement-learning
compatibility check. It does **not** launch Lambda infrastructure, run a model,
contact Gemini, or consume a secret.

## Before launching

Keep the repository on `codex/ppp-rl-4b` and ensure it is pushed or otherwise
available to the GPU host. Add an SSH public key in Lambda, then manually
launch one A6000 48 GB instance. Do not launch an H100 yet.

The gate is capped at two hours and $3 of compute. Set a local timer before
connecting. Lambda bills a running instance even when no command is active, so
terminate it manually in the Lambda console as soon as the gate passes or
fails.

## Host setup

From the A6000 host, clone the desired branch and run the primary bootstrap:

```bash
git clone <YOUR_PPP_AGENT_REMOTE> PPP-Agent
cd PPP-Agent
git switch codex/ppp-rl-4b
bash simplified/scripts/lambda_a6000_gate.sh --bootstrap --model qwen35
```

The script creates `simplified/.venv-lambda` and installs the simplified agent
plus a pinned Qwen3.5 compatibility stack. The vendored `verl/` source has no
independent package manifest, so the script exports the repository root
through `PYTHONPATH` instead of pretending it is a published dependency.

The primary stack uses `vllm==0.21.0` and Transformers 5. vLLM's compiled wheel
selects its matching PyTorch package. This is intentional: the compiled
vLLM/PyTorch pair is the binding dependency. A failed primary gate is a
compatibility result—not permission to patch the algorithm or reduce the
rollout group.

To rerun only read-only host checks:

```bash
bash simplified/scripts/lambda_a6000_gate.sh --doctor --model qwen35
```

The doctor prints GPU model, memory, and driver information; checks for at
least 46 GiB; confirms CUDA is visible to PyTorch; and imports vLLM and the
training layer. Bootstrap also runs `pip check`. Neither command reads `.env`
or `GEMINI_API_KEY`.

## Run the gates

Prepare the frozen data and inspect the exact training command:

```bash
simplified/.venv-lambda/bin/ppp-train doctor
simplified/.venv-lambda/bin/ppp-train prepare

# Print only; this does not construct an optimizer.
simplified/.venv-lambda/bin/ppp-train train \
  --steps 1 --simulator deterministic --model qwen35
```

Only after explicit paid-run confirmation, execute the one-step compatibility
gate:

```bash
export CONFIRM_PAID_TRAINING=I_UNDERSTAND_LAMBDA_IS_BILLING
simplified/.venv-lambda/bin/ppp-train train \
  --steps 1 --simulator deterministic --model qwen35 --execute
```

That command must complete a model forward/generation pass, eight sequential
rollouts, old-policy log probabilities, FoldGRPO advantage calculation, one
DAPO backward/optimizer step, and checkpoint save. Inspect the rollout
artifact, peak memory, checkpoint, and console metrics before proceeding.
The command also writes scalar metrics, sanitized trajectories, a grouped
training report, and a Markdown summary under the compatibility run directory.
Run the same guarded one-step command a second time and confirm the log resumes
from `global_step_1`; the launcher records that successful reload. Do this
before the live simulator gate.

Do not enter a Gemini key until the deterministic gate passes. Store
`GEMINI_API_KEY` only in the instance-local environment. Run a text-only vLLM
server in one terminal, then use the no-optimizer live-group command in a
second terminal:

```bash
vllm serve Qwen/Qwen3.5-4B \
  --language-model-only --max-model-len 10240 --gpu-memory-utilization 0.35
ppp-train live-group --model qwen35
```

`live-group` produces exactly eight trajectories with distinct inference
seeds, uses the cached Gemini simulator, and exports only redacted repository
observations.

After the compatibility and live-group gates, start the 20-step run with
`ppp-train train --steps 20 --simulator gemini --model qwen35 --execute`.
Rerun that same command once to verify checkpoint resume. A 40-step extension
also requires `--projected-compute-usd AMOUNT`; the launcher refuses it unless
the saved report shows at least 25% non-flat groups, loss and KL are finite, a
LoRA adapter exists, resume was verified, and projected phase compute is at
most $45.

## Qwen3 fallback

If Qwen3.5 fails because the vendored Verl path cannot use its new hybrid
architecture, first save the traceback and package versions. Then create a
fresh fallback environment and repeat the same A6000 gate:

```bash
bash simplified/scripts/lambda_a6000_gate.sh \
  --bootstrap --model qwen3 --venv simplified/.venv-lambda-qwen3
export CONFIRM_PAID_TRAINING=I_UNDERSTAND_LAMBDA_IS_BILLING
simplified/.venv-lambda-qwen3/bin/ppp-train train \
  --steps 1 --simulator deterministic --model qwen3 --execute
```

The fallback pins vLLM 0.12.0, matching Verl 0.7.0, and Transformers 4.x. Do not mix primary and
fallback packages in one virtual environment. The bootstrap script refuses to
reuse an existing virtual environment; select a new `--venv` path for each
fresh compatibility attempt.

## H100 fallback

Stop the A6000 attempt and record its doctor and training output if:

- actor plus rollout components cannot load safely;
- peak allocated memory exceeds 46 GiB;
- a real optimizer step still fails after microbatch size 1, gradient
  checkpointing, conservative rollout memory, sequential rollouts, and the
  documented CPU offload; or
- fitting requires reducing the eight-rollout group or changing FoldGRPO.

Only then launch one H100 80 GB and repeat the same gate. Do not solve a memory
failure by weakening the eight-rollout learning objective.

The scripts encode BF16 base weights, rank-16 LoRA, 4K generated tokens, and
eight turns, but local preparation does not prove their combined GPU memory
footprint. The held-out evaluation suite remains sealed throughout this work.
