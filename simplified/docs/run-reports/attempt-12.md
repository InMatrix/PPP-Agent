# Attempt 12 — checkpoint reload succeeds without an unintended update

## Result

The saved `global_step_1` actor, optimizer, and extra state reloaded
successfully. Verl restored global step 1, recognized that the requested total
was already 1, and exited without generating trajectories or running another
optimizer step.

- eight-rollout group: not run; the prior eight artifacts remained unchanged;
- old-policy log probabilities: not run;
- reward and FoldGRPO advantages: not run;
- backward and optimizer step: not run;
- LoRA checkpoint save: not run; the existing adapter hash was preserved; and
- checkpoint reload/resume: passed.

## Configuration and evidence

- Date and UTC start/resume-marker timestamps: 2026-07-29,
  `05:34:04Z`–`05:34:56.730180Z`.
- Git commit: `883df51` (`Preserve reports during resume probes`).
- Hardware and hourly rate: one Lambda GH200 96 GB at `$2.29/hour`.
- Model and precision: `Qwen/Qwen3-4B`, BF16 rank-16 LoRA, alpha 32.
- Requested work: deterministic `--steps 1` against existing
  `global_step_1`.
- Simulator: not invoked; Gemini was not called.
- Peak GPU memory: not captured for this load-only probe.
- Measured runtime and estimated cost: 53 seconds and `$0.0337` before tax.
- Durable ignored evidence: `simplified/results/training/attempt-12/`.
- Durable tracked summary:
  [`artifacts/attempt-12-resume.json`](artifacts/attempt-12-resume.json).
- Evidence limitations: this proves checkpoint compatibility and identity, not
  a second optimizer update or a nonzero learning signal.

Before and after the probe, the sanitized report hash was
`7984cd85…0013cf3` and the LoRA adapter hash was
`83350623…906f46`. The trajectory count remained eight,
`global_step_2` was absent, and GPU compute-process count was zero after
cleanup.

## Failure and diagnosis

No checkpoint failure occurred. The decisive trainer output was:

```text
Found checkpoint: .../global_step_1
Setting global step to 1
Checkpoint already reached requested total training steps: 1/1.
Resume load verified; no optimizer step will run.
```

vLLM logged that its engine core exited unexpectedly while Ray was shutting
down. The overall command returned zero, the resume marker was written, and no
GPU process remained; this is cleanup noise rather than a failed load.

The reporting fixes behaved correctly for the probe: the original Attempt 11
report was not regenerated with the load-only runtime, and its hash did not
change. Because no new training step logged scalars, the still-empty legacy
metrics file did not exercise the new per-record flush on the GPU path.

## Corrective change

Commit `5d5ecec` added the early completed-total exit and append-only,
immediately flushed file metrics. Commit `883df51` taught the launcher to
preserve the original report during a no-op resume. Local verification was
`100 passed`; this live probe confirms the resume behavior.

No further resume fix is required before the live simulator gate.

## Process signal

- Detectable offline: the off-by-one resume behavior and destructive report
  overwrite were detectable by reading the loop and launcher before running.
- Cheapest gate that would have caught it: the launcher regression test plus
  this 53-second real checkpoint load.
- Paid time spent before detection: approximately `$0.03`.
- New invariant or observability requirement: a resume probe must record loaded
  step, requested total, new-rollout count, new-optimizer count, and unchanged
  artifact hashes.
- Question for the final retrospective: should checkpoint load validation be a
  first-class CLI command so it need not initialize the rollout server at all?
