# Attempt 08 — overlong-rollout batch flag

## Result

All eight real Qwen trajectories completed and passed Verl's output-model
validation. Trainer preprocessing then attempted to build `overlong_mask` from
`batch["mask_rollout"]`, but the adapter had not returned that per-trajectory
flag.

The run stopped before FoldGRPO advantage calculation, backward pass, optimizer
step, or checkpoint save. The GPU was idle after cleanup.

## Configuration and evidence

- Date: 2026-07-29.
- Observed live interval: approximately 03:33:49–03:48:46 UTC, reconstructed
  from terminal output; the raw log is unavailable.
- Observed GPU allocation during rollouts: approximately 35.6 GB.
- Simulator: deterministic; Gemini was not called.
- Artifacts: eight new sanitized trajectories completed.

## Failure and diagnosis

Verl's `mask_overlong` path distinguishes rollouts that exhausted their budget
without a valid finish. The agent adapter already recorded a termination
reason but did not translate it into the framework's `mask_rollout` field.
`response_mask` was not a substitute: it is a token-level model/environment
mask, whereas `mask_rollout` is a trajectory-level overlong flag.

## Corrective change

Commit `0de62a4` (`Mark overlong agent rollouts`) set
`mask_rollout = (termination == "turn_limit")`. Natural and deadline finishes
remain trainable; true turn-limit trajectories receive a zero policy-loss
mask.

## Process signal

- Detectable offline: **yes** with a full batch-schema assertion.
- Cost amplification: the missing scalar was discovered only after an entire
  eight-rollout group.
- Cheaper prevention: generate a synthetic mixed-termination group and run it
  through trainer preprocessing.
- Retrospective question: can every adapter field be traced to its first
  downstream consumer in a checked schema?
