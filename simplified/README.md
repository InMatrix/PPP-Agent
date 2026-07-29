# Simplified PPP-Agent vertical slice

This directory proves the PPP-Agent stack with one real function-localization
episode before we build more course modules. It deliberately avoids VERL, Ray,
distributed training, and a repository HTTP service.

The later teaching-scale RL phase is tracked in
[`docs/PHASE_PLAN.md`](docs/PHASE_PLAN.md), with per-run evidence under
[`docs/run-reports/`](docs/run-reports/README.md).

The vertical slice still exercises the important boundaries:

1. Load a real vague/preference-conditioned row from the downloaded parquet.
2. Fetch the exact repository commit into a small, isolated workspace.
3. Let Qwen inspect the repository through read-only tools.
4. Let Qwen ask a simulated user a targeted question.
5. Use Gemini 3.5 Flash-Lite to answer from hidden task information.
6. Score function-localization F1, disclosure cost, and preference compliance.
7. Write the complete trajectory and decomposed reward to JSON.

## Why this stack

- **Qwen:** an OpenAI-compatible client works with vLLM, SGLang, LM Studio, or
  another compatible server. The default is `Qwen/Qwen3.5-4B`; the 9B model can
  be tested by changing one flag.
- **Gemini:** the current Google GenAI SDK uses an AI Studio key directly. No
  Vertex setup is required.
- **Repository access:** ordinary shallow Git fetches replace the full
  benchmark repository server for this proof.
- **Data:** PyArrow reads one matching row without loading the entire parquet
  into memory.

## Install

Use Python 3.11 or later from the repository root:

```bash
python -m venv simplified/.venv
simplified/.venv/bin/pip install -e 'simplified[test,gemini]'
```

## Prove the offline path first

The default episode is the small `pallets/flask` case in `test_id.parquet`.

```bash
simplified/.venv/bin/ppp-simple doctor
simplified/.venv/bin/ppp-simple inspect
simplified/.venv/bin/ppp-simple prepare
simplified/.venv/bin/ppp-simple run --mode offline
```

Offline mode uses the real parquet, real pinned repository, real tools, and real
reward implementation. Its scripted agent and deterministic user intentionally
use the oracle answer, so it validates plumbing—not model quality.

## Prove the live providers

### 1. Serve Qwen

The agent only requires an OpenAI-compatible endpoint. On a CUDA runtime,
serve Qwen with a current vLLM build:

```bash
vllm serve Qwen/Qwen3.5-4B \
  --port 8000 \
  --max-model-len 16384 \
  --reasoning-parser qwen3
```

For a local Mac, a compatible quantized Qwen3.5-4B model in LM Studio is also
suitable; set `QWEN_BASE_URL` to its `/v1` endpoint and `QWEN_MODEL` to the
identifier reported by LM Studio. On a 16 GB Apple Silicon Mac, start with the
4-bit MLX build so the repository tools, model cache, and KV cache have
comfortable headroom. No paid Qwen API is required.

For the recommended LM Studio Community 4-bit build, a typical local setup is:

```bash
export QWEN_BASE_URL=http://localhost:1234/v1
export QWEN_MODEL=qwen3.5-4b-mlx
```

### 2. Add the AI Studio key

Create a key in Google AI Studio and export it in the terminal:

```bash
export GEMINI_API_KEY='...'
```

Alternatively, copy `simplified/.env.example` to `simplified/.env` and place
the key there. That file is ignored by Git. Do not put the key into source
files or commit it.

### 3. Run one live episode

```bash
simplified/.venv/bin/ppp-simple doctor
simplified/.venv/bin/ppp-simple run --mode live --max-turns 12
```

Results are written below `simplified/results/`, which is ignored by Git.

## Evaluation protocol

The tracked `evaluation_suites.json` separates model development from final
measurement:

- `dev-v1` contains the four previously inspected `test_id` tasks. Use it for
  prompt, tool, and policy iteration; its trajectories are development data.
- `heldout-v1` freezes twelve unseen task instances from `test_ood`, covering
  eleven repositories and all eight available OOD preference conditions. Use
  it only for milestone comparisons after an implementation is frozen.

Each suite records the parquet SHA-256, row metadata, selection seed, role, and
trajectory policy. Loading verifies the dataset checksum and every frozen
row's instance, repository, and preference. The held-out suite explicitly
excludes all four development task IDs.

If you inspect an individual held-out issue, expected answer, or trajectory to
guide a change, that task is contaminated: add it to a future development
suite and replace it before the next held-out comparison. Aggregate held-out
metrics may be used for the predeclared model-selection decision.

Preview either suite without making model calls:

```bash
simplified/.venv/bin/ppp-simple evaluate \
  --suite dev-v1 \
  --mode live \
  --inference-seeds 11 \
  --dry-run

simplified/.venv/bin/ppp-simple evaluate \
  --suite heldout-v1 \
  --mode live \
  --inference-seeds 11 \
  --dry-run
```

Run the inexpensive development suite while iterating:

```bash
simplified/.venv/bin/ppp-simple evaluate \
  --suite dev-v1 \
  --mode live \
  --max-turns 8 \
  --inference-seeds 11 \
  --policy-label navigation-v2 \
  --policy-version navigation-v2 \
  --tool-schema-version v2 \
  --output-dir simplified/results/dev-navigation-v2
```

Freeze and commit a candidate before using the held-out suite. Begin with one
inference seed to control local-Qwen cost. Execution is blocked unless the
milestone intent is acknowledged explicitly, and sealed runs reject a dirty
Git worktree. Development manifests append `+dirty` to the revision when
appropriate so provisional results cannot be mistaken for a frozen build:

```bash
simplified/.venv/bin/ppp-simple evaluate \
  --suite heldout-v1 \
  --confirm-heldout \
  --mode live \
  --max-turns 8 \
  --inference-seeds 11 \
  --policy-label navigation-v3 \
  --policy-version navigation-v3 \
  --tool-schema-version v3 \
  --output-dir simplified/results/heldout-navigation-v3-seed-11
```

Ad-hoc sampling remains available by omitting `--suite` and using `--data`,
`--sample-size`, `--sample-seed`, and `--preferences`. Ad-hoc results must not
be labeled as held-out evidence.

Each episode is saved immediately. Repeating the command resumes completed
episodes, so an interrupted repository download or provider error does not
discard earlier results. Use `--no-resume` to rerun every selected episode.

The live runner manages the turn budget explicitly:

- Every request tells the agent its current turn and remaining budget.
- Exact repeated tool calls are suppressed and returned with a corrective
  observation.
- The penultimate turn is labeled as the last chance to gather evidence.
- The final turn uses a finish-only structured-output schema so the agent
  records its best-supported localization instead of silently timing out.

Reports distinguish `natural_finish`, `deadline_finish`, and `turn_limit`, and
record how many duplicate actions were suppressed.

Navigation v2 keeps the surface read-only and exposes six focused actions:
`list_tree`, `find_symbol`, `search_code`, `read_file`, `ask_user`, and
`finish`. `list_tree` shows shallow directory-first structure without hidden
or cache noise. `find_symbol` uses a cached Python AST index to return exact
definition paths, lines, and qualified names.

The first exact duplicate proposal receives one retry without consuming another
logical turn. Finish entries are checked against repository files and AST
function or method definitions; classes are rejected. One invalid finish
receives a correction attempt, and a still-invalid correction is retained for
scoring but marked unverified in the report.

Navigation v3 is an executable policy selected with
`--policy-version navigation-v3`; `--policy-label` remains the human-readable
experiment label. V3 adds `inspect_symbol`, which resolves a definition to its
canonical qualified name and returns its complete source plus nearby sibling
definitions. Its prompt asks the agent to trace supporting call/data paths
before finalizing. An exact action signature receives at most one duplicate
retry across the entire episode, preventing a stuck proposal from multiplying
model calls on every logical turn. If the immediate retry is also the same
duplicate, that tool is removed from the next turn's schema to force a
different navigation strategy. Navigation v2 remains selectable for matched
control runs.

`--sample-seed` controls which dataset rows are selected, while
`--inference-seeds` controls LM Studio generation. Multiple inference seeds
expand the fixed episode sample into independently persisted evaluation cases.
The manifest also records the policy label, tool-schema version, and Git
revision so incompatible runs cannot be mixed during resume.

The output directory contains:

- `manifest.json`: model, sampling, and episode configuration.
- `episodes/*.json`: complete trajectory and reward, or an isolated error.
- `summary.json`: aggregate metrics for analysis.
- `summary.md`: a compact, human-readable baseline report.

Compare two completed evaluation directories and apply the configured F1,
reward, preference, and completion decision rule with:

```bash
simplified/.venv/bin/ppp-simple compare \
  --control-dir simplified/results/control-termination-v1-seeded \
  --candidate-dir simplified/results/candidate-navigation-v2-seeded \
  --output-dir simplified/results/control-vs-navigation-v2
```

Comparison checks the manifests before reading the metrics. Named suites,
dataset hashes, frozen cases, preferences, and inference seeds must match, so a
development run cannot accidentally be compared with a held-out run.

The primary metrics are exact localization rate, mean function F1, natural and
deadline finish rates, empty prediction rate, suppressed duplicate actions,
question rate, disclosure level, preference compliance, turns, latency, and
total reward. Inspect individual trajectories only for development suites.

## Expected cost

The repository and offline run have no API cost. A live episode invokes Gemini
only when the agent calls `ask_user`; the runner is capped by `--max-turns`.
Qwen is local by default. Start with one episode and inspect its trajectory
before increasing the count or switching to Qwen3.5-9B.

## Current limitations

- The live proof evaluates inference and environment integration, not RL
  training.
- Preference scoring covers the common UserVille rules; full behavioral
  functions remain in `envs/userville.py`.
- Repository tools are read-only because this proof targets function
  localization, not patch generation.
- The 16K serving context is a low-cost smoke-test setting, not the model's
  maximum context.
