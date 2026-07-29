# Small-scale PPP reinforcement-learning phase plan

Last updated: 2026-07-29, after compatibility attempt 09 and its retrospective
report.

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
- Latest implementation fix: `139e5d3` (`Preserve FoldGRPO rollout identities`).
- Retrospective baseline commit: `e01a29a`
  (`Document GH200 compatibility attempts`).
- Active Lambda instance: none.
- Live simulator used in compatibility attempts: no.
- Optimizer steps completed: 0.
- Checkpoints written: 0.
- Held-out evaluation opened: no.

Attempt 09 completed all eight real Qwen trajectories, overlong masking,
old-policy log-probability processing, and actor preprocessing. It entered
FoldGRPO advantage calculation, then failed because the async postprocessor
dropped the shared question `uid`. Commit `139e5d3` now preserves `uid`,
`gen_uid`, and `process_reward_mask`, but that fix has not yet been exercised
by a live GPU run.

## Gate status

| Gate | Acceptance evidence | Status | Evidence |
|---|---|---|---|
| Host bootstrap | ARM64 GH200 doctor, CUDA tensor, compatible PyTorch/vLLM/Verl imports | Passed | Attempts 01–06 and Lambda runbook |
| Model load and generation | Real Qwen completion through colocated vLLM | Passed | [Attempt 06](run-reports/attempt-06.md) |
| Eight-rollout group | Eight sanitized real-Qwen trajectories | Passed | [Attempt 08](run-reports/attempt-08.md), [Attempt 09](run-reports/attempt-09.md) |
| Old-policy log probabilities | Non-null generated-token log probabilities reach actor preprocessing | Passed | [Attempt 09](run-reports/attempt-09.md) |
| Adapter-to-trainer schema | Synthetic group contains every FoldGRPO field and invariant | **Not implemented** | Required before another paid run |
| FoldGRPO advantages | Finite group-relative advantages for eight trajectories | Not proved | Next live gate |
| DAPO backward/optimizer | One finite loss and real LoRA parameter update | Not proved | Pending |
| Checkpoint save | LoRA adapter and tracker written at step 1 | Not proved | Pending |
| Checkpoint reload/resume | Second guarded command resumes from step 1 | Not proved | Pending |
| Live Gemini group | Eight cached, sanitized trajectories without optimizer work | Not started | Pending |
| Short training | 20 genuine optimizer updates within phase budget | Not started | Pending |
| Pre/post evaluation | All episodes scorable across seeds 11, 22, and 33 | Not started | Pending |

## Exact next gate

Do not launch another paid instance yet.

First implement a local synthetic adapter-to-trainer contract test that:

1. Constructs eight Verl-native `AgentLoopOutput` objects without loading a
   model.
2. Assigns one shared `uid` and eight unique `gen_uid` values.
3. Mixes natural, deadline, and turn-limit termination states.
4. Runs the actual async postprocessor into a `DataProto`.
5. Asserts the presence and shape of `response_mask`,
   `rollout_log_probs`, `process_reward_mask`, `mask_rollout`, reward scores,
   `uid`, and `gen_uid`.
6. Calls the actual FoldGRPO advantage function and checks finite results,
   group size eight, and zero loss contribution from overlong/environment
   tokens.
7. Verifies that serialized evidence contains no secrets or raw repository
   observations.

The implementation command is:

```bash
ppp-train contract-gate
```

It uses the actual vendored Verl/Torch path but does not load Qwen, initialize
CUDA model weights, contact Gemini, or inspect a task repository.

After that test passes and is committed:

1. Launch the smallest compatible GPU available.
2. Run one deterministic `--steps 1` gate.
3. Require finite FoldGRPO advantages, backward pass, a changed LoRA parameter,
   and a saved step-1 checkpoint.
4. Export durable sanitized artifacts before terminating the instance.
5. Write the new run report and update this plan before any further paid run.

## Subsequent sequence

1. Rerun the deterministic command and prove checkpoint reload/resume.
2. Run one eight-trajectory live Gemini group without an optimizer update.
3. Review reward variance, simulator caching, sanitation, latency, memory, and
   projected cost.
4. Run 20 RL steps.
5. Extend to 40 only if loss and KL remain finite, at least 25% of groups have
   nonzero reward variance, resume works, and projected phase compute remains
   below $45.
6. Evaluate the untrained and trained 4B adapters across inference seeds
   11, 22, and 33.
7. Contrast results with the frozen 9B reference without presenting it as an
   equivalent control.
8. Produce the learner notebook trace and final development retrospective.

## Budget and artifact ledger

Historical exact cost is unavailable because the terminated GH200's ignored
run directory was not copied off-host. Do not estimate it retroactively.

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
