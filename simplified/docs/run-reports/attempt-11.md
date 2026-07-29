# Attempt 11 — one real optimizer step with a flat reward group

## Result

The first complete model-backed training step reached every mechanical
boundary: eight real Qwen rollouts, old-policy log probabilities, FoldGRPO
advantages, DAPO backward, optimizer step, and step-1 checkpoint save.

The group did not provide a learning signal. All eight terminal rewards were
zero, so all eight advantages, policy loss, gradient norm, and the effective
LoRA parameter update were also zero. The result is therefore a trainer and
checkpoint compatibility pass, but not yet proof that RL changes the policy.

- eight-rollout group: passed with real Qwen trajectories;
- old-policy log probabilities: passed;
- reward and FoldGRPO advantages: computed, but the group was flat at zero;
- backward and optimizer step: completed with zero gradient;
- LoRA checkpoint save: passed; and
- checkpoint reload/resume: not attempted.

## Configuration and evidence

- Date and UTC start/checkpoint timestamps: 2026-07-29,
  `04:58:49Z`–`05:16:34Z`.
- Git commit: `2b6bb1d` (`Record instance-specific training cost`).
- Hardware and hourly rate: one Lambda GH200 96 GB at `$2.29/hour`, from
  Lambda's public instance pricing on the run date.
- Model and precision: `Qwen/Qwen3-4B`, BF16 rank-16 LoRA, alpha 32.
- Group, turn, and token limits: one prompt group, eight rollouts, maximum
  eight logical turns and 4,096 response tokens.
- Simulator: deterministic; Gemini was not called.
- Peak GPU memory: 49.115 GB allocated and 51.059 GB reserved.
- Measured compute and estimated cost: 1,065 seconds and `$0.6775` before tax.
- Durable local ignored bundle:
  `simplified/results/training/attempt-11/ppp-attempt-11-durable.tar.gz`.
- Durable tracked summary:
  [`artifacts/attempt-11-one-step.json`](artifacts/attempt-11-one-step.json).
- Evidence limitations: the file logger remained empty, so the scalar metrics
  line was recovered from the Ray worker log. The outer log stream did not
  close automatically after the worker exited; the sanitized report was
  generated explicitly afterward. The full 17.8 GB actor state remains only
  on the active instance, while the LoRA adapter is also copied locally.

The prepared input archive and its three files matched their local SHA-256
checksums. The host did not contain either held-out parquet.

## Failure and diagnosis

No trainer exception occurred. The learning-signal failure was observed in the
terminal metrics:

```text
reward/avg_score:0.0
reward/std_score:0.0
critic/advantages/mean:0.0
actor/pg_loss:0.0
actor/grad_norm:0.0
```

Four rollouts reached the turn limit with no prediction. The other four
finished naturally or at the deadline, but every proposed function target
failed repository/AST validation. All eight therefore had zero productivity;
reward clipping left every composite reward at zero. FoldGRPO correctly
produced zero relative advantages for the flat group.

The model made 71 calls. Four responses reached the 4,096-token cap, and mean
response length was 3,703 tokens. Generation dominated runtime at 944.7
seconds; old log probabilities took 6.4 seconds, actor update 24.9 seconds, and
checkpoint save 10.2 seconds. The post-update rollout KL was finite at
`0.000795`.

## Corrective change

No algorithm, prompt, tool, termination, or reward change is justified by one
flat group; those interfaces remain frozen. The same evidence commit as this
report flushes every Verl file-logger record immediately and adds best-effort
summary finalization to the launcher's exit/signal trap. Its regression test
reads the metrics file before logger teardown, and the full local suite passes.

Next, exercise those fixes in the real environment, then use the saved
checkpoint for a bounded resume probe. Proof of a changed LoRA parameter still
requires at least one real group with nonzero reward variance; do not
manufacture variance or select a hand-picked task merely to pass the gate.

## Process signal

- Detectable offline: the empty file-logger output and report-finalization
  behavior can be tested without model generation; the flat real reward group
  cannot be predicted from the synthetic contract.
- Cheapest gate that would have caught it: the completed one-group run was
  necessary to observe this agent/reward outcome.
- Paid time spent before detection: 1,065 seconds, approximately `$0.68`.
- New invariant or observability requirement: a run is not a learning update
  merely because the optimizer method returned; require nonzero reward
  variance, nonzero gradient norm, and a before/after adapter hash or tensor
  delta.
- Question for the final retrospective: should the compatibility phase
  distinguish a mechanical optimizer gate from a stochastic learning-signal
  gate, with the latter evaluated across several ordinary training groups?
