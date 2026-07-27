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
