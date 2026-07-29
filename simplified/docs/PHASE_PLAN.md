# Small-scale PPP reinforcement-learning phase plan

Last updated: 2026-07-29, after compatibility attempt 14 completed an ordinary
step-two group but produced a flat clipped reward and no adapter delta.

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
- Latest tested commit: `bcf5e5d`
  (`Add guarded step-two continuation gate`).
- Retrospective baseline commit: `e01a29a`
  (`Document GH200 compatibility attempts`).
- Active Lambda instance: one GH200 at the user's request; environment
  bootstrapped, idle, and retaining the complete step-1 and step-2 checkpoints.
- Live simulator used in compatibility attempts: yes; five live Gemini calls
  served eleven questions across eight trajectories.
- Optimizer calls completed: 2; effective nonzero policy updates: 0.
- Checkpoints written: 2 (`global_step_1` and byte-identical
  `global_step_2` LoRA adapters).
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

Attempt 13 ran eight base-Qwen trajectories against the live Gemini simulator
without an optimizer. Total rewards varied from 0 to 0.1, but all productivity
components were zero and all corrected finishes were empty. The group proves
that the full composite reward can produce nonzero advantages; it does not yet
prove a localization-learning signal.

Attempt 14 resumed the ordinary dataloader for exactly one additional group.
All eight composite rewards were zero after clipping, despite variation in the
proactivity and personalization components. FoldGRPO advantages, policy loss,
and gradient norm were zero, so `global_step_2` contains a LoRA adapter
byte-identical to step 1. A foreground SSH disconnect occurred after the
checkpoint write and before scalar-file logging and repository verification;
the complete scalar record was recovered from the Ray worker log.

## Gate status

| Gate | Acceptance evidence | Status | Evidence |
|---|---|---|---|
| Host bootstrap | ARM64 GH200 doctor, CUDA tensor, compatible PyTorch/vLLM/Verl imports | Passed | Attempts 01–06 and Lambda runbook |
| Model load and generation | Real Qwen completion through colocated vLLM | Passed | [Attempt 06](run-reports/attempt-06.md) |
| Eight-rollout group | Eight sanitized real-Qwen trajectories | Passed | [Attempt 08](run-reports/attempt-08.md), [Attempt 09](run-reports/attempt-09.md) |
| Old-policy log probabilities | Non-null generated-token log probabilities reach actor preprocessing | Passed | [Attempt 09](run-reports/attempt-09.md) |
| Adapter-to-trainer schema | Synthetic group contains every FoldGRPO field and invariant | Passed | [Attempt 10](run-reports/attempt-10.md) |
| FoldGRPO advantages | Finite group-relative advantages for eight trajectories | Passed synthetically and on a flat real-Qwen group | [Attempt 10](run-reports/attempt-10.md), [Attempt 11](run-reports/attempt-11.md) |
| DAPO backward/optimizer | One finite loss and real LoRA parameter update | Mechanics passed twice; both real training groups had zero scalar variance, zero gradient, and no effective update | [Attempt 11](run-reports/attempt-11.md), [Attempt 14](run-reports/attempt-14.md) |
| Checkpoint save | LoRA adapter and tracker written at step 1 | Passed | [Attempt 11](run-reports/attempt-11.md) |
| Checkpoint reload/resume | Second guarded command restores step 1 without unintended work | Passed | [Attempt 12](run-reports/attempt-12.md) |
| Live Gemini group | Eight cached, sanitized trajectories without optimizer work | Passed; composite variance nonzero, productivity flat | [Attempt 13](run-reports/attempt-13.md) |
| Ordinary step-2 continuation | Exactly eight new trajectories, finite loss/KL, nonzero gradient, changed adapter | Failed: finite KL but flat reward, zero gradient, and identical adapter | [Attempt 14](run-reports/attempt-14.md) |
| Short training | 20 genuine optimizer updates within phase budget | Not started | Pending |
| Pre/post evaluation | All episodes scorable across seeds 11, 22, and 33 | Not started | Pending |

## Exact next gate

Checkpoint save/reload, live simulation, and a bounded ordinary continuation
are now proved mechanically. Do not launch the 20-step run through a foreground
SSH command:

1. Commit and push Attempt 14 plus this plan update.
2. Add a detached, reconnectable Lambda launcher with durable PID, start/end,
   exit-status, and cleanup evidence.
3. Make each trainer scalar record durable before checkpoint shutdown can
   interrupt logging.
4. Test launcher disconnect/reconnect and scalar durability without loading a
   model.
5. Treat the 20-step run as the next stochastic learning-signal gate. Do not
   tune or hand-pick a group; measure the fraction of groups with nonzero final
   reward variance and effective adapter deltas.
6. Pause for explicit confirmation before starting that paid run.

## Subsequent sequence

1. Harden detached execution and metrics durability.
2. Run 20 RL steps, retaining flat groups and requiring at least one effective
   adapter delta before claiming genuine learning.
3. Extend to 40 only if loss and KL remain finite, at least 25% of groups have
   nonzero reward variance, resume works, and projected phase compute remains
   below $45.
4. Evaluate the untrained and trained 4B adapters across inference seeds
   11, 22, and 33.
5. Contrast results with the frozen 9B reference without presenting it as an
   equivalent control.
6. Produce the learner notebook trace and final development retrospective.

## Budget and artifact ledger

Historical exact cost is unavailable because the terminated GH200's ignored
run directory was not copied off-host. Do not estimate it retroactively.

The current GH200 has `$1.4428` of measured compatibility work before tax:
`$0.6775` for Attempt 11, `$0.0337` for Attempt 12, `$0.1457` for Attempt 13,
and `$0.5859` for Attempt 14. One-time bootstrap and idle-instance time are not
included because their exact start/end timestamps were not preserved.

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
