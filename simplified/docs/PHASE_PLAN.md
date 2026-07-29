# Small-scale PPP reinforcement-learning phase plan

Last updated: 2026-07-29, after compatibility attempt 12 reloaded the complete
step-1 checkpoint without generating rollouts or running an unintended update.

This is the durable execution plan for the teaching-scale PPP-RL phase. It
tracks what has actually been proved, what remains uncertain, and the exact
next gate. Per-run evidence lives in [`run-reports/`](run-reports/README.md);
the Lambda operating procedure lives in
[`LAMBDA_A6000_GATE.md`](LAMBDA_A6000_GATE.md).

## Objective

Complete and analyze a genuine, low-cost PPP-Agent reproduction using:

- `Qwen/Qwen3-4B`;
- BF16 rank-16 LoRA with learning rate `1e-6`;
- the frozen navigation-v2 read-only environment;
- eight trajectories per prompt;
- FoldGRPO group-relative advantages;
- DAPO token-level loss with asymmetric clipping;
- model-generated-token masking;
- the composite productivity, proactivity, and personalization reward; and
- Gemini Flash-Lite through an AI Studio key for the live simulator stage.

The phase succeeds only after at least 20 real optimizer updates, checkpoint
resume, scorable pre/post evaluation, and a learner-oriented artifact report.
It is a teaching-scale reproduction, not a performance reproduction.

## Frozen decisions

- Keep group size eight and maximum eight logical turns.
- Keep navigation-v2 prompts, tools, duplicate policy, and finish validation
  frozen during RL compatibility and training.
- Do not expose arbitrary shell execution or code editing.
- Use the deterministic simulator until the optimizer/checkpoint gate passes.
- Keep `heldout-v1` sealed until the planned milestone evaluation.
- Do not alter FoldGRPO semantics merely to fit a device.
- Prefer the smallest compatible GPU, but treat GH200 results as architecture-
  specific because its host is ARM64.

## Current state

- Current branch: `codex/ppp-rl-4b`.
- Latest tested commit: `883df51`
  (`Preserve reports during resume probes`).
- Retrospective baseline commit: `e01a29a`
  (`Document GH200 compatibility attempts`).
- Active Lambda instance: one GH200 at the user's request; environment
  bootstrapped, idle, and retaining the complete step-1 checkpoint.
- Live simulator used in compatibility attempts: no.
- Optimizer calls completed: 1; effective nonzero policy updates: 0.
- Checkpoints written: 1 (`global_step_1`).
- Held-out evaluation opened: no.

Attempt 09 completed all eight real Qwen trajectories, overlong masking,
old-policy log-probability processing, and actor preprocessing. It entered
FoldGRPO advantage calculation, then failed because the async postprocessor
dropped the shared question `uid`. Attempt 10 exercised commit `2042e01` in the
real ARM64 Verl/Torch environment without loading a model. All eight synthetic
outputs retained `uid`, `gen_uid`, masks, log probabilities, and reward
metadata; FoldGRPO advantages and DAPO loss were finite and masked-token
invariance passed.

Attempt 11 then completed eight real Qwen trajectories and the full training
mechanics. All eight rewards were zero, however, so the advantages, policy
loss, gradient norm, and effective LoRA update were zero. The step-1
checkpoint and LoRA adapter were saved. The adapter and sanitized evidence are
durable locally; the full actor/optimizer state remains on the active GH200.

Attempt 12 loaded that full state, restored global step 1, and exited because
the requested total was already satisfied. It produced no trajectories,
optimizer calls, or step-2 checkpoint, preserved the report and adapter hashes,
and left no GPU process running.

## Gate status

| Gate | Acceptance evidence | Status | Evidence |
|---|---|---|---|
| Host bootstrap | ARM64 GH200 doctor, CUDA tensor, compatible PyTorch/vLLM/Verl imports | Passed | Attempts 01–06 and Lambda runbook |
| Model load and generation | Real Qwen completion through colocated vLLM | Passed | [Attempt 06](run-reports/attempt-06.md) |
| Eight-rollout group | Eight sanitized real-Qwen trajectories | Passed | [Attempt 08](run-reports/attempt-08.md), [Attempt 09](run-reports/attempt-09.md) |
| Old-policy log probabilities | Non-null generated-token log probabilities reach actor preprocessing | Passed | [Attempt 09](run-reports/attempt-09.md) |
| Adapter-to-trainer schema | Synthetic group contains every FoldGRPO field and invariant | Passed | [Attempt 10](run-reports/attempt-10.md) |
| FoldGRPO advantages | Finite group-relative advantages for eight trajectories | Passed synthetically and on a flat real-Qwen group | [Attempt 10](run-reports/attempt-10.md), [Attempt 11](run-reports/attempt-11.md) |
| DAPO backward/optimizer | One finite loss and real LoRA parameter update | Mechanics passed; zero reward variance caused zero gradient and no effective update | [Attempt 11](run-reports/attempt-11.md) |
| Checkpoint save | LoRA adapter and tracker written at step 1 | Passed | [Attempt 11](run-reports/attempt-11.md) |
| Checkpoint reload/resume | Second guarded command restores step 1 without unintended work | Passed | [Attempt 12](run-reports/attempt-12.md) |
| Live Gemini group | Eight cached, sanitized trajectories without optimizer work | Not started | Pending |
| Short training | 20 genuine optimizer updates within phase budget | Not started | Pending |
| Pre/post evaluation | All episodes scorable across seeds 11, 22, and 33 | Not started | Pending |

## Exact next gate

Checkpoint save and reload are now proved. Before another model-backed command:

1. Commit and push Attempt 12 plus this plan update.
2. Configure `GEMINI_API_KEY` only in the active instance's environment; do not
   copy a local `.env` or persist the key in repository artifacts.
3. Run one eight-trajectory live Gemini group without optimizer work.
4. Preserve simulator cache statistics, call counts, decomposed rewards,
   latency, sanitation evidence, and exact cost.
5. Review whether the live group has reward variance, but do not tune the
   frozen agent or select tasks based on the observed outcome.
6. Only then decide whether the next genuine update should begin the 20-step
   run or whether a generic learning-signal gate is still required.

## Subsequent sequence

1. Run one eight-trajectory live Gemini group without an optimizer update.
2. Review reward variance, simulator caching, sanitation, latency, memory, and
   projected cost.
3. Run 20 RL steps.
4. Extend to 40 only if loss and KL remain finite, at least 25% of groups have
   nonzero reward variance, resume works, and projected phase compute remains
   below $45.
5. Evaluate the untrained and trained 4B adapters across inference seeds
   11, 22, and 33.
6. Contrast results with the frozen 9B reference without presenting it as an
   equivalent control.
7. Produce the learner notebook trace and final development retrospective.

## Budget and artifact ledger

Historical exact cost is unavailable because the terminated GH200's ignored
run directory was not copied off-host. Do not estimate it retroactively.

The current GH200 has `$0.7112` of measured compatibility work before tax:
`$0.6775` for Attempt 11 and `$0.0337` for Attempt 12. One-time bootstrap and
idle-instance time are not included because their exact start/end timestamps
were not preserved.

For every future paid run, record before termination:

| Field | Required value |
|---|---|
| Instance type and hourly price | From the Lambda launch screen |
| UTC start and end | ISO-8601 timestamps |
| Estimated compute cost | Runtime multiplied by hourly price |
| Peak allocated GPU memory | Trainer or `nvidia-smi` evidence |
| Tested commit and command | Exact revision and sanitized arguments |
| Stage reached | One value from the gate table |
| Durable artifacts | Local or remote path copied off the ephemeral host |
| Fix commit | Add after diagnosis, or mark pending |

Keep total paid compute for this phase below $50, with $5 reserved for tax or
one bounded retry.

## After-run update protocol

Before authorizing another paid run:

1. Copy the sanitized log, metrics, summary, and checkpoint manifest to durable
   storage.
2. Create the next numbered report from
   [`run-reports/TEMPLATE.md`](run-reports/TEMPLATE.md).
3. Update the gate table, current state, exact next gate, budget ledger, and
   `Last updated` line in this document.
4. Link the report from [`run-reports/README.md`](run-reports/README.md).
5. Commit the report and plan update with the corrective change when practical;
   otherwise use a dedicated evidence commit.
6. Only then decide whether the next paid run is justified.
