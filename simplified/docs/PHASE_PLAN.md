# Small-scale PPP reinforcement-learning phase plan

Last updated: 2026-07-30, after the offline action-contract gate restored
finish-correction parity, sanitized parse observability, and schema-constrained
vLLM sampling.

This is the durable execution plan for the teaching-scale PPP-RL phase. It
tracks what has actually been proved, what remains uncertain, and the exact
next gate. Per-run evidence lives in [`run-reports/`](run-reports/README.md);
the Lambda operating procedure lives in
[`LAMBDA_A6000_GATE.md`](LAMBDA_A6000_GATE.md).

## Objective

Complete and analyze a genuine, low-cost PPP-Agent reproduction using:

- `Qwen/Qwen3.5-4B` as the primary compatibility target, matching the local
  model family;
- `Qwen/Qwen3-4B` as the preserved fallback and prior live baseline;
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
- Keep navigation-v2 and the current read-only repository tools frozen while
  restoring action-contract and model parity.
- Treat finish-only correction parity and schema-constrained action generation
  as correctness fixes, not prompt or repository-tool tuning.
- Keep Qwen3.5 and Qwen3 in separate environments. The Qwen3.5 BF16 Hugging
  Face checkpoint matches the local base-model family but is not numerically
  identical to LM Studio's quantized MLX artifact.
- Preserve navigation-v2 as the prompt baseline. Evaluate any
  `navigation-v2.1` candidate only after mechanical parity and only across a
  diverse development set, never this Bokeh group alone.
- Do not expose arbitrary shell execution or code editing.
- Use the deterministic simulator for the new Qwen3.5 mechanical gates until
  its optimizer/checkpoint gate passes; add Gemini only at the subsequent live
  group gate.
- Keep `heldout-v1` sealed until the planned milestone evaluation.
- Do not alter FoldGRPO semantics merely to fit a device.
- Prefer the smallest compatible GPU, but treat GH200 results as architecture-
  specific because its host is ARM64.

## Current state

- Current branch: `codex/ppp-rl-4b`.
- Latest live-tested commit: `bcf5e5d`
  (`Add guarded step-two continuation gate`).
- Latest offline-verified change: detached run management and pre-checkpoint
  scalar durability at commit `213d64d`, followed by action-contract gate
  commit `83d9032`; 122 simplified tests pass.
- Retrospective baseline commit: `e01a29a`
  (`Document GH200 compatibility attempts`).
- Active Lambda instance: none; the user terminated the GH200 after Attempt 14.
- Live simulator used in compatibility attempts: yes; Attempt 13 used five
  live Gemini calls for eleven questions, and Attempt 14 used four live calls
  for four questions.
- Optimizer calls completed: 2; effective nonzero policy updates: 0.
- Checkpoints written: 2 (`global_step_1` and byte-identical
  `global_step_2` LoRA adapters).
- Attempt 14 action contract: 61 model calls, 31 parsed actions, 30 inferred
  invalid actions, two actionless turn-limit trajectories, and zero valid
  finishes.
- Local/training mismatch: local Qwen3.5 used strict JSON Schema at temperature
  0.2; training Qwen3 used prompt-only JSON at temperature 1.0.
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
durable locally; the terminated GH200's full actor/optimizer state is no longer
available.

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

The later trajectory review found that only 31 of 61 model calls became parsed
actions. Two trajectories executed no tool, only one used `list_tree`, all six
finish attempts proposed nonexistent functions, and no correction changed the
rejected target. The expected function was
`src/bokeh/command/subcommands/info.py:Info.invoke`; predictions instead used
invented `show_info`, `show_version`, and `info` symbols in unsupported paths.
One trajectory called `ask_user` after finish validation failed, proving that
the Verl loop did not enforce the ordinary runner's finish-only correction
policy.

The subsequent offline fix adds a `start`/`status`/`logs` controller whose
worker is detached from SSH and whose immutable run directory records PID,
timestamps, exit code, cleanup result, and post-cleanup GPU state. Verl now
fsyncs a scalar snapshot before checkpoint serialization and writes the normal
complete record afterward. Offline tests exercise successful and failed
detached workers, caller exit, reconnectable status/logs, unsafe and duplicate
run IDs, key non-persistence, cleanup evidence, and log/checkpoint ordering.

The offline action-contract gate now passes one shared JSON Schema from the
navigation-v2 loop through `CallLLM` and Ray into vLLM
`StructuredOutputsParams`. A rejected finish narrows the next and only
correction call to `finish`. Sanitized traces record direct invalid-action
categories and parse rates without model prose. The vLLM boundary also rejects
sampled-token/log-probability length mismatches or missing chosen-token
log-probabilities. These results use fake model outputs and constructor-level
compatibility checks; a real Qwen3.5 generation has not yet proved engine-level
schema enforcement.

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
| Detached run lifecycle | SSH-independent worker with durable PID, status, exit, log, and cleanup evidence | Passed offline; live observation pending | Detached-controller tests and Lambda runbook |
| Pre-checkpoint scalar durability | Loss, gradient, and KL survive interruption after checkpoint save | Passed offline; live observation pending | File-logger and ordering tests |
| Sanitized invalid-action observability | Record parse-error categories without exporting model prose | Passed offline; live evidence pending | Focused loop and export tests |
| Finish-correction parity | Verl permits exactly one finish-only correction call | Passed offline; live evidence pending | Scripted invalid/valid and disallowed-correction traces |
| Schema-constrained rollout | vLLM enforces the action schema and preserves chosen-token log probabilities | Transport, constructor, and alignment checks pass offline; real generation pending | Shared-schema and vLLM compatibility tests |
| Qwen3.5 training compatibility | BF16 LoRA actor/rollout completes one effective update and resume in a separate environment | Not started | Qwen3 remains fallback |
| Short training | 20 genuine optimizer updates within phase budget | Not started | Pending |
| Pre/post evaluation | All episodes scorable across seeds 11, 22, and 33 | Not started | Pending |

## Exact next gate

Do not launch the 20-step run yet. The next gate is:

1. Use pushed commit `83d9032` and keep Lambda terminated until the user is
   ready for bounded paid work.
2. On a fresh instance, bootstrap a separate Transformers 5/vLLM Qwen3.5
   environment, run focused offline tests, and verify the Gemini key without
   printing it.
3. Run the host doctor and its finish-only `StructuredOutputsParams` plus
   log-probability constructor check.
4. Pause for explicit confirmation, then run Qwen3.5 gates in order: model
   load, one real constrained action, deterministic eight-rollout group,
   old-policy log probabilities, one optimizer step, LoRA save/reload, and one
   live Gemini group.
5. After each run, inspect every trajectory and update the next numbered report
   before selecting an intervention.
6. Use Qwen3-4B only if Qwen3.5 still requires invasive Verl changes after the
   bounded gate, and record that evidence before falling back.

## Subsequent sequence

1. Run 20 RL steps with the compatible parity model, retaining flat groups and
   requiring at least one effective adapter delta before claiming learning.
2. Report action-parse rate, valid-finish rate, reward variance, and adapter
   deltas per group.
3. Verify checkpoint resume through a second detached lifecycle run.
4. Extend to 40 only if loss and KL remain finite, at least 25% of groups have
   nonzero reward variance, resume works, and projected phase compute remains
   below $45.
5. Evaluate the untrained and trained 4B adapters across inference seeds
   11, 22, and 33.
6. Contrast results with the frozen 9B reference without presenting it as an
   equivalent control.
7. Evaluate a versioned navigation-v2.1 prompt only if mechanical parity leaves
   a broad navigation deficit on diverse development tasks.
8. Produce the learner notebook trace and final development retrospective.

## Budget and artifact ledger

Historical exact cost is unavailable because the terminated GH200's ignored
run directory was not copied off-host. Do not estimate it retroactively.

The terminated GH200 incurred `$1.4428` of measured compatibility work before tax:
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
3. Conduct trajectory-level error analysis before choosing a fix or next gate.
   For an eight-rollout group, inspect all eight trajectories and report action
   parsing, tool use, navigation, finish validation and correction, termination,
   reward components, masking or loss eligibility, and recurring failure
   categories. Label direct evidence separately from inference.
4. Update the gate table, current state, exact next gate, budget ledger, and
   `Last updated` line in this document.
5. Link the report from [`run-reports/README.md`](run-reports/README.md).
6. Commit the report and plan update with the corrective change when practical;
   otherwise use a dedicated evidence commit.
7. Only then decide whether the next paid run is justified.
