# Simplified PPP-Agent vertical slice

This directory proves the PPP-Agent stack with one real function-localization
episode before we build more course modules. It deliberately avoids VERL, Ray,
distributed training, and a repository HTTP service.

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

## Evaluate a small live baseline

Preview the deterministic, stratified sample before making any model calls:

```bash
simplified/.venv/bin/ppp-simple evaluate \
  --mode live \
  --sample-size 4 \
  --max-turns 8 \
  --dry-run
```

The default sample uses distinct tasks and repositories across
`concise_question`, `detail_question`, `no_ask`, and `one_question`. It keeps
the known Flask task as the first sanity check. Run the batch with:

```bash
simplified/.venv/bin/ppp-simple evaluate \
  --mode live \
  --sample-size 4 \
  --max-turns 8 \
  --inference-seeds 11,22,33 \
  --policy-label termination-v1 \
  --tool-schema-version v1 \
  --output-dir simplified/results/baseline-live-4
```

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
symbols; one invalid finish receives a correction attempt, and a still-invalid
correction is retained for scoring but marked unverified in the report.

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

The primary metrics are exact localization rate, mean function F1, natural and
deadline finish rates, empty prediction rate, suppressed duplicate actions,
question rate, disclosure level, preference compliance, turns, latency, and
total reward. Increase the sample only after inspecting the four-episode
report.

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
