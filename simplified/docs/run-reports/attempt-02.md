# Attempt 02 — shared workspace race

## Result

Eight rollout tasks attempted to prepare the same pinned repository workspace
concurrently. Their Git initialization and snapshot writes overlapped, leaving
an incomplete workspace and aborting the rollout group. No optimizer work or
checkpoint occurred.

The incomplete workspace was quarantined during diagnosis; a subsequent valid
workspace snapshot was prepared successfully.

## Configuration and evidence

- Date: 2026-07-29.
- Hardware: one Lambda GH200 96 GB, ARM64.
- Group size: eight trajectories.
- Simulator: deterministic; Gemini was not called.
- Evidence: preserved failure summary, quarantined-workspace observation, and
  corrective commit. Raw attempt log, exact duration, and cost are unavailable.

## Failure and diagnosis

The adapter correctly shared one repository checkout across trajectories, but
workspace preparation had no process-safe serialization. Unit tests had
covered single-call preparation, not concurrent first use.

This was a concurrency failure in environment setup. It did not justify
reducing the rollout group or changing FoldGRPO semantics.

## Corrective change

Commit `a7b5f6f` (`Serialize shared rollout workspace setup`) added a file lock
around `RepositoryWorkspace.prepare` and concurrency tests proving that
parallel callers share one valid Git snapshot.

## Process signal

- Detectable offline: **yes**.
- Cheaper prevention: exercise first-use workspace preparation from eight
  concurrent workers in the preflight suite.
- Retrospective question: which resources are intentionally shared across a
  rollout group, and does each have an explicit ownership/locking policy?
