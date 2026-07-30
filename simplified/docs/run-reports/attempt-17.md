# Attempt 17 — Qwen3.5 reaches direct vLLM worker dispatch

## Result

The constructor fix from Attempt 16 worked: vLLM 0.21 constructed and
initialized its lazy worker wrapper. The one-step gate then failed at the next
engine boundary, when Verl requested `init_device` through the legacy generic
dispatcher:

```text
'Worker' object has no attribute 'execute_method'
```

No trajectory began. This attempt did not complete an eight-rollout group,
old-policy log probabilities, rewards, FoldGRPO advantages, backward, an
optimizer call, checkpoint save, or checkpoint reload. Detached cleanup passed
and left zero GPU processes.

## Configuration and evidence

- Date and UTC interval: 2026-07-30,
  `05:21:19Z`–`05:22:14Z`.
- Tested commit: `eaf7136`
  (`Record Qwen3.5 worker-wrapper failure`).
- Corrective commit: `bc1a1e9`
  (`Adapt Verl worker dispatch to vLLM 0.21`).
- Hardware and hourly rate: one Lambda GH200 96 GB at `$2.29/hour`.
- Model and intended update: `Qwen/Qwen3.5-4B`, BF16 rank-16 LoRA, learning
  rate `1e-6`.
- Intended group and limits: one prompt group, eight trajectories, eight turns,
  4,096 generated response tokens per trajectory.
- Simulator: deterministic; Gemini was not used.
- Runtime and estimated compute: 55 seconds, approximately `$0.035` before
  tax.
- Peak GPU memory: unavailable before device initialization completed.
- Durable tracked evidence:
  [`artifacts/attempt-17-worker-dispatch.json`](artifacts/attempt-17-worker-dispatch.json).
- Durable ignored remote evidence:
  `simplified/results/training/launches/attempt-17-qwen35-one-step/`.
- Zero trajectories and checkpoints were written; raw model output, repository
  contents, hidden context, and secrets were not exported.

## Trajectory-level error analysis

Not applicable because all eight trajectories remained unstarted. No action
generation, parsing, repository navigation, finish validation, termination,
reward, advantage, or loss-mask evidence exists.

Relative to Attempt 16, the run progressed from wrapper construction through
`init_worker` and failed on the following `init_device` dispatch. That
progression directly validates the Attempt 16 constructor fix.

## Failure and diagnosis

Older vLLM worker wrappers exposed:

```text
wrapper.execute_method("init_device", ...)
```

vLLM 0.21 instead exposes lifecycle methods such as `init_device` and
`execute_model` directly on `WorkerWrapperBase`, forwarding other named methods
to its concrete worker. Its concrete `Worker` no longer supplies the generic
`execute_method` dispatcher. Vendored Verl unconditionally called that removed
dispatcher for every method other than its locally special-cased
`init_worker` and `load_model`.

This remains a bounded API bridge: both versions expose the same lifecycle
operation, but ownership moved from a generic dispatcher to direct wrapper
methods. It is not an algorithm, agent, prompt, reward, or memory change.

## Corrective change

Commit `bc1a1e9` centralizes method dispatch alongside constructor adaptation:

- legacy wrappers continue receiving method names through `execute_method`;
- vLLM 0.21 wrappers resolve and invoke the named method directly; and
- the async rollout no longer calls the removed API itself.

Tests cover legacy dispatch, direct dispatch with keyword arguments, and source
wiring that forbids the old direct call. The focused compatibility suite passes
`21` tests. The current complete workspace suite passes `148` tests; unrelated
uncommitted Lambda-cloud helper files were not staged into the corrective
commit.

Commit `38347c6` extends the Lambda doctor to identify and require either the
legacy dispatcher or vLLM 0.21's direct `init_device`/`execute_model` API before
another model launch.

## Process signal

- Detectable offline: yes, by checking both construction and lifecycle method
  dispatch against the installed wrapper.
- Cheapest gate that caught it: one-step engine initialization, before
  trajectories or optimizer work.
- Paid time before detection: 55 seconds, approximately `$0.035`.
- New invariant: vLLM compatibility checks must cover a lifecycle sequence,
  not isolated constructors.
- Retrospective question: should the next offline contract use a stateful fake
  worker to execute `construct → init_worker → init_device → load_model`
  before another paid engine start?
