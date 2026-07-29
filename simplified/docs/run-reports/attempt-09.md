# Attempt 09 — FoldGRPO group identity

## Result

All eight real Qwen trajectories completed. The repaired overlong flag worked,
old-policy log probabilities were processed, and the actor reported dynamic
microbatch size one. The run entered `compute_advantage`, then failed because
the agent-loop postprocessor had dropped the question-level `uid` needed to
group the eight trajectories.

No FoldGRPO advantages, backward pass, optimizer step, or checkpoint were
completed. The GPU was idle after launcher cleanup.

## Configuration and evidence

- Date: 2026-07-29.
- Observed live interval: approximately 03:56:47–04:11:41 UTC, reconstructed
  from terminal output; the raw log is unavailable.
- Observed GPU allocation during rollouts: approximately 35.6 GB.
- Simulator: deterministic; Gemini was not called.
- Artifacts: eight new sanitized trajectories completed; the directory held 17
  total files including attempts 07 and 08.

## Failure and diagnosis

Before generation, the trainer creates:

- one `uid` per question, repeated across the eight trajectories; and
- one distinct `gen_uid` per trajectory.

The async agent loop passed both into the adapter, but its returned
`extra_fields` omitted them. FoldGRPO therefore could not construct
group-relative advantages. Inspection also showed that the process-reward mask
would be required at the next line and was not yet forwarded.

## Corrective change

Commit `139e5d3` (`Preserve FoldGRPO rollout identities`) validates and returns
both identities and forwards `process_reward_mask`. Missing IDs now fail at
rollout startup instead of after expensive generation. The local suite passed
97 tests.

## Process signal

- Detectable offline: **yes** with an eight-output postprocessing fixture.
- Cost amplification: another full rollout group was spent before discovering
  a non-tensor metadata omission.
- Cheaper prevention: assert one shared `uid`, eight unique `gen_uid` values,
  and all FoldGRPO tensors before the first paid run.
- Retrospective question: should the next live attempt be blocked until a
  synthetic batch reaches finite FoldGRPO advantages locally?
