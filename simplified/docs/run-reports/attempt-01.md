# Attempt 01 — rollout import dependency

## Result

The live trainer reached Ray's agent-loop worker import, then failed because
`agents.utils` transitively required `unidiff`, which was absent from the
Lambda environment. No model generation, trajectory, optimizer work, or
checkpoint occurred.

## Configuration and evidence

- Date: 2026-07-29.
- Hardware: one Lambda GH200 96 GB, ARM64.
- Model path: `Qwen/Qwen3-4B`.
- Simulator: deterministic; Gemini was not called.
- Evidence: preserved failure summary and corrective commit. Raw attempt log,
  exact duration, peak memory, and cost are unavailable.

## Failure and diagnosis

The bootstrap doctor covered the declared training stack, but the simplified
loop imported original PPP-Agent utilities whose complete transitive dependency
set was not represented in the Lambda requirements. The immediate missing
module was `unidiff`.

This was a dependency-closure failure, not a model, GPU-memory, or algorithm
failure.

## Corrective change

Commit `5dfd86e` (`Include rollout import dependency`) added
`unidiff==0.7.5` to the Lambda requirements and added bootstrap coverage.

## Process signal

- Detectable offline: **yes**.
- Cheaper prevention: import the exact Hydra-configured agent-loop class inside
  the freshly bootstrapped environment, rather than validating only selected
  top-level packages.
- Retrospective question: should the doctor derive dependency validation from
  the production import graph instead of maintaining a parallel checklist?
