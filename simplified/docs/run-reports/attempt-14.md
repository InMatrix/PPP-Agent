# Attempt 14 — ordinary step-two group is flat and produces no adapter delta

## Result

The guarded continuation resumed the ordinary dataloader at step 1, generated
exactly eight new live Qwen/Gemini trajectories, calculated old-policy log
probabilities, ran FoldGRPO and DAPO backward, and wrote `global_step_2`.
However, all eight optimized rewards were zero:

```text
[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
```

Consequently, the group-relative advantages, policy loss, gradient norm, and
effective LoRA update were all zero. The step-1 and step-2 adapter SHA-256
hashes are identical. This attempt therefore failed the continuation gate even
though the trainer mechanics and checkpoint write completed.

The controlling SSH connection closed immediately after checkpointing. The
trainer's scalar line survived in the Ray worker log, but the wrapper did not
reach the repository verifier or write its end timestamp.

## Configuration and evidence

- Date and observed run interval: 2026-07-29,
  `06:07:30Z`–`06:22:51Z`.
- Git commit: `bcf5e5d` (`Add guarded step-two continuation gate`).
- Hardware and hourly rate: one Lambda GH200 96 GB at `$2.29/hour`.
- Policy model and update: `Qwen/Qwen3-4B`, BF16 rank-16 LoRA, learning rate
  `1e-6`.
- Simulator: `gemini-3.5-flash-lite` through the instance-local AI Studio key.
- Dataset position: ordinary resumed group
  `vague+first_try+code_func_loc_train-bokeh__bokeh-13422`; it was not
  hand-picked.
- Group and limits: eight trajectories, maximum eight logical turns, 4,096
  generated response tokens per trajectory.
- New/total sanitized trajectories: 8/16.
- Trainer step time: 858.51 seconds; observed wrapper interval: 921 seconds.
- Estimated gate compute: `$0.5859` before tax. Later idle-instance time is not
  included.
- Peak GPU memory: 64.98 GiB allocated and 65.13 GiB reserved.
- Durable ignored evidence:
  `simplified/results/training/attempt-14/`.
- Durable tracked aggregate:
  [`artifacts/attempt-14-step-two.json`](artifacts/attempt-14-step-two.json).
- Simulator cache and hidden replies were not copied or exported.

The ignored evidence directory contains all 16 sanitized trajectories, the
step-two LoRA adapter, the launcher log, and the Ray task-runner scalar log.
The copied adapter hash matches the remote checkpoint.

## Learning-signal diagnosis

All eight productivity rewards were zero and no finish passed validation. Four
trajectories asked no question and received a `-0.10` proactivity adjustment.
The other four asked one question, receiving `+0.05` proactivity but `-0.10`
personalization because the reply did not satisfy this prompt's `first_try`
preference. After the composite reward was clipped at zero, every trajectory
had the same optimized reward.

The trainer reported:

```text
actor/pg_loss       0.0
actor/grad_norm     0.0
rollout_corr/kl     0.0009106665
reward/std_score    0.0
overlong_masked     2
```

The KL and old-policy path were finite, and the actor update call took 24.88
seconds. The unchanged adapter proves that completing an optimizer call is not
equivalent to completing an effective RL update.

Five trajectories finished naturally, two hit the turn limit, and one used
deadline finish. Six attempted finish correction. Across the group there were
61 model calls, four live Gemini calls, 23,259 model-generated tokens, and
seven invalid predictions.

## External interruption and observability

Verl saves the checkpoint before calling its aggregate logger. The SSH session
closed after `global_step_2` was written but before that logger call and before
the launcher's continuation verifier. Ray then terminated its workers. As a
result:

- the complete scalar record exists in the Ray task-runner log;
- `training-metrics.jsonl` is empty;
- no `continuation-step-2.json` was produced; and
- the wrapper's `RUN_END_UTC` is absent.

This interruption did not cause the zero update: the scalar record and
identical adapter already prove the flat reward and zero gradient. It did expose
that a long paid run must not be owned by a foreground SSH session and that
scalar evidence should be made durable before checkpoint shutdown can
intervene.

## Corrective direction

Do not tune the reward, prompt, frozen tools, or agent policy to make this
particular group non-flat. Before a 20-step job:

1. add a detached, reconnectable Lambda launcher with durable PID/exit-status
   files and explicit cleanup;
2. make the scalar record durable before or atomically with checkpointing; and
3. test both behaviors without loading Qwen.

After those observability fixes, the 20-step run is the appropriate stochastic
gate: it samples enough ordinary groups to measure how often reward variance
occurs. It must still report zero-signal groups honestly, and the phase does not
claim a genuine update until at least one adapter delta is observed.

## Process signal

- Detectable offline: the foreground-session ownership and logger ordering are
  testable without a GPU; this particular group's reward collapse required the
  live rollout.
- Cheapest gate that caught it: the bounded one-group continuation used here.
- Paid time spent before detection: 921 observed seconds, approximately
  `$0.59`.
- New invariant: report reward-component variance, final scalar variance,
  gradient norm, and adapter delta separately.
- Retrospective question: should a checkpoint be labeled an optimizer step or
  an effective update when its adapter is byte-identical to its predecessor?
