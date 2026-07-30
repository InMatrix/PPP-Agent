# Attempt 15 — Qwen3.5 constrained actions pass after bounded GH200 startup fixes

## Result

The targeted real-vLLM gate loaded `Qwen/Qwen3.5-4B` once and completed both
planned schema-constrained generations. The navigation completion selected
`ask_user`; the finish-only completion selected `finish`. Both outputs were
exact JSON objects with the required keys, selected a tool allowed by that
call's enum, parsed through the agent action contract, and returned one finite
chosen-token log probability per sampled token.

This was deliberately not an agent trajectory or training run. It did not
construct Verl, Gemini, a LoRA adapter, rewards, FoldGRPO advantages, a policy
loss, an optimizer, or a checkpoint. Therefore the eight-rollout, old-policy,
backward, checkpoint-save, and checkpoint-resume gates remain pending for
Qwen3.5.

## Configuration and evidence

- Date and observed targeted-gate interval: 2026-07-30,
  `04:38:42Z`–`04:48:12Z`, including diagnosis and bounded retries.
- Fix commit: `cfce77a` (`Fix Qwen3.5 GH200 schema gate startup`).
- Hardware and assumed hourly rate: one Lambda GH200 96 GB at `$2.29/hour`.
- Model and runtime: `Qwen/Qwen3.5-4B`, BF16, vLLM 0.21.0,
  Transformers 5.14.1, Torch 2.11.0+cu130, CUDA 13.0.
- Limits: 2,048-token context, 256 maximum generated tokens, one sequence,
  20% vLLM GPU-memory utilization, temperature 0.2.
- Simulator: none.
- Successful artifact runtime: 29.57 seconds, including an 18.50-second model
  load; generation latencies were 4.73 and 1.16 seconds.
- Model-load allocation reported by vLLM: 7.99 GiB. The artifact's
  `torch_peak_allocated_bytes` is zero because vLLM owns CUDA in a spawned
  process; it is not a valid peak-memory measurement.
- Approximate active command time across four executions: 329 seconds,
  `$0.21` before tax. The full observed diagnosis interval was 570 seconds,
  `$0.36`; bootstrap and other idle-instance time are not included.
- Durable sanitized evidence:
  [`artifacts/attempt-15-schema-gate.json`](artifacts/attempt-15-schema-gate.json).
- Raw completions, reasoning, arguments, prompts, repository contents, and
  secrets were not exported.

## Completion-level investigation

There were no agent trajectories, repository observations, tool executions,
rewards, termination reasons, or loss-eligible masks. Per the diagnostic-gate
protocol, both generated completions and each available validation stage were
inspected:

| Completion | Allowed tools | Selected tool | Tokens/logprobs | Exact JSON | Contract valid |
|---|---|---|---:|---|---|
| Navigation | Six navigation-v2 tools | `ask_user` | 63/63 | Yes | Yes |
| Finish correction | `finish` only | `finish` | 83/83 | Yes | Yes |

For both completions, all chosen-token log probabilities were finite and the
sampled-token and log-probability counts matched exactly. The two distinct
schema hashes prove that the finish-only request used a narrower schema than
the navigation request.

`action_valid` here means the JSON passed the model-facing action interface.
The gate intentionally did not execute the selected action or validate the
sanitized finish argument against a repository AST. It therefore proves
mechanical schema enforcement and training-logprob availability, not code
localization quality, tool choice quality, or valid final predictions. The
navigation prompt was synthetic, so selecting `ask_user` is not scored as a
behavioral success or failure.

## Startup failures and diagnosis

The successful result followed four bounded initialization executions:

1. The first stopped in model initialization because FlashInfer could not find
   `ninja`. The executable was installed beside the virtual-environment Python,
   but that directory was absent from `PATH`.
2. Adding that directory manually let JIT compilation finish. vLLM then
   rejected its default `max_num_seqs=1024`, because 20% memory provided 406
   Mamba cache blocks.
3. The first automatic `PATH` fix incorrectly resolved the virtual-environment
   Python symlink to `/usr/bin/python3.11`, again hiding the sibling `ninja`
   executable.
4. Preserving the literal `sys.executable` parent and setting
   `max_num_seqs=1` allowed initialization and both completions to pass.

The host bootstrap also exposed two setup-only compatibility checks before the
first model load. `pip check` rejected the installed CUDA 13 cuSPARSELt ARM64
wheel despite the required shared library being present, and vLLM 0.21 added
an optional `supported_tasks` parameter to `init_app_state`. PyTorch saw the
GH200 and CUDA 13.0, so the first was the same known wheel-tag metadata issue
as CUDA 12. The vendored adapter's three-argument call remains valid because
`supported_tasks` defaults to `None`.

## Corrective change

Commit `cfce77a`:

- accepts only the exact CUDA 12 or CUDA 13 cuSPARSELt ARM64 metadata warning,
  and only when `libcusparseLt.so.0` exists;
- recognizes vLLM's optional `supported_tasks` initializer signature without
  broadening the accepted API shapes;
- exposes native helpers installed beside the invoked Python without resolving
  away the virtual environment; and
- binds this two-completion gate to one concurrent sequence, matching its
  actual workload and avoiding the unrelated 1,024-sequence default.

The full local simplified suite passes: `133 passed`. On the GH200, the focused
pre-run suite passed `43` tests, the final schema-gate tests passed `7` tests,
the host doctor passed, and the real two-completion artifact reports
`status: passed`.

## Process signal

- Detectable offline: the virtual-environment path behavior and one-sequence
  configuration were unit-testable; the exact CUDA 13 metadata and vLLM 0.21
  signatures required inspecting the installed ARM64 stack.
- Cheapest live gate that caught the issues: this two-completion gate, before
  eight rollouts, Gemini, or optimizer construction.
- Paid time before a successful result: about 329 active command seconds,
  approximately `$0.21` before tax.
- New invariants: a launcher must expose its environment's native helpers;
  diagnostic concurrency must be explicit; spawned-engine memory must be
  measured in the child rather than with parent-process Torch counters.
- Retrospective question: should bootstrap execute a zero-token engine-start
  probe so JIT-helper and cache-sizing failures are separated from the first
  model-output contract test?
