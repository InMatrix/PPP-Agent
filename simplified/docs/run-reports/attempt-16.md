# Attempt 16 — Qwen3.5 one-step gate reaches the vLLM worker boundary

## Result

The first detached Qwen3.5/Verl one-step gate launched the actor workers and
vLLM HTTP server, then failed while vLLM initialized its engine worker:

```text
WorkerWrapperBase.__init__() got an unexpected keyword argument 'vllm_config'
```

No trajectory began. Consequently, this attempt did not complete an
eight-rollout group, old-policy log probabilities, reward or FoldGRPO
advantages, backward, an optimizer call, a LoRA checkpoint, or checkpoint
reload/resume. The detached controller recorded the failure, cleaned Ray, and
verified that no GPU process remained.

## Configuration and evidence

- Date and UTC interval: 2026-07-30,
  `05:04:27Z`–`05:05:23Z`.
- Tested commit: `4704447` (`Record Qwen3.5 schema gate run`).
- Corrective commit: `771fbb0`
  (`Adapt Verl worker wrapper to vLLM 0.21`).
- Hardware and hourly rate: one Lambda GH200 96 GB at `$2.29/hour`.
- Model and update: `Qwen/Qwen3.5-4B`, BF16 rank-16 LoRA, learning rate
  `1e-6`.
- Intended group and limits: one prompt group, eight trajectories, eight
  logical turns, 4,096 generated response tokens per trajectory.
- Simulator: deterministic; Gemini was not used.
- Runtime and estimated compute: 56 seconds, approximately `$0.0356` before
  tax.
- Peak GPU memory: unavailable before the engine worker completed
  initialization.
- Durable tracked evidence:
  [`artifacts/attempt-16-worker-wrapper.json`](artifacts/attempt-16-worker-wrapper.json).
- Durable ignored remote evidence:
  `simplified/results/training/launches/attempt-16-qwen35-one-step/`.
- The run wrote zero trajectories and no checkpoint. Raw model output,
  repository contents, hidden context, and secrets were not exported.

## Trajectory-level error analysis

Not applicable: all eight planned trajectories remained unstarted. There were
no model action calls, parsed or invalid actions, repository tools, finish
proposals, termination reasons, rewards, advantages, or loss masks to analyze.

The failure occurred after the async rollout server was launched but before its
colocated engine worker was ready. Therefore Attempt 15's two standalone
schema-constrained completions were not contradicted: this run reached a
different integration boundary, the external ZeroMQ worker used by Verl.

## Failure and diagnosis

vLLM 0.21 changed `WorkerWrapperBase` from eager configuration:

```text
WorkerWrapperBase(vllm_config=...)
```

to a lazy constructor:

```text
WorkerWrapperBase(rpc_rank=0, global_rank=None)
wrapper.init_worker(all_kwargs)
```

The vendored Verl code still passed `vllm_config` to the constructor even
though it also included that configuration in `all_kwargs` for
`init_worker`. The installed vLLM source confirms that the new
`init_worker` extracts and stores `vllm_config` itself. This is an observed API
contract mismatch, not a model, memory, dataset, reward, or agent-prompt
failure.

## Corrective change

Commit `771fbb0` adds a small signature-aware constructor helper:

- older wrappers receive `vllm_config` in their constructor;
- vLLM 0.21 wrappers are constructed without it and continue receiving the
  unchanged `all_kwargs` in `init_worker`; and
- the async rollout uses this helper instead of directly owning either vLLM
  API shape.

Tests cover both constructor contracts and assert that the real rollout source
cannot regress to the direct incompatible call. The focused compatibility
suite passes `18` tests and the full simplified suite passes `136` tests.
Commit `b2c0140` also makes the Lambda doctor construct the installed worker
wrapper through this helper, so the same boundary is checked before another
model launch.

## Process signal

- Detectable offline: yes, if the doctor had instantiated the actual worker
  wrapper rather than checking only the async server and app-state signatures.
- Cheapest gate that caught it: the detached one-step initialization gate,
  before any trajectory or optimizer work.
- Paid time before detection: 56 seconds, approximately `$0.0356`.
- New invariant: the doctor must exercise every vLLM object constructor owned
  by vendored Verl, including the external worker wrapper.
- Retrospective question: should version bridges be centralized around
  behavior-level contract tests instead of being discovered one constructor at
  a time during engine startup?
