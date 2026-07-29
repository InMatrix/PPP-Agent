# Attempt 13 — live Gemini group has style reward variance but zero productivity

## Result

One no-optimizer live group completed with eight Qwen3-4B trajectories and the
Gemini UserVille simulator. Composite rewards had nonzero variance:

```text
[0.10, 0.05, 0.10, 0.00, 0.10, 0.10, 0.00, 0.00]
```

The variance came entirely from proactivity and personalization. All eight
productivity rewards were zero, all eight final predictions were empty after
validation/correction, and no finish passed validation.

- eight-rollout group: passed with eight inference seeds;
- old-policy log probabilities: not run;
- reward and FoldGRPO advantages: computed for analysis only;
- backward and optimizer step: not constructed;
- LoRA checkpoint save: not run; and
- checkpoint reload/resume: not run.

## Configuration and evidence

- Date and UTC server start/stop: 2026-07-29,
  `05:48:25Z`–`05:52:14Z`.
- Live group start/end: `05:50:32Z`–`05:51:48Z`.
- Git commit: `b34c0cd` (`Advance phase plan to live simulator gate`).
- Hardware and hourly rate: one Lambda GH200 96 GB at `$2.29/hour`.
- Policy model: base `Qwen/Qwen3-4B`; the step-1 adapter had a zero gradient
  and is behaviorally unchanged.
- Simulator: `gemini-3.5-flash-lite` through the AI Studio key stored only in
  the instance-local environment.
- Group and limits: eight seeds (`11` through `88`), maximum eight logical
  turns, 4,096 response tokens, and a 64-call cache-wide simulator budget.
- Observed GPU memory: 20,278 MiB while serving; peak was not sampled
  continuously.
- Measured server runtime and estimated cost: 229 seconds and `$0.1457` before
  tax. The live group itself took 76 seconds.
- Durable ignored evidence: `simplified/results/training/attempt-13/`.
- Durable tracked summary:
  [`artifacts/attempt-13-live-gemini.json`](artifacts/attempt-13-live-gemini.json).
- Evidence limitations: the simulator cache is intentionally retained only on
  the instance because it contains hidden UserVille replies.

The exported artifact passed the repository's secret/sanitation validator.
Its trajectories contain redacted question, reasoning, simulator-reply, and
validation placeholders rather than hidden context.

## Failure and diagnosis

The gate itself passed. Five trajectories finished naturally and three used
deadline finish. Each rollout attempted finish correction, but all final
predictions were empty and all validations failed.

The eight trajectories asked eleven questions. Exact-cache reuse reduced those
to five live Gemini calls and five cache rows. Single-question behavior earned
positive proactivity and usually personalization adjustments. Two-question
behavior incurred a proactivity penalty, producing zero total reward after
clipping. Productivity was zero in every rollout.

This proves that a real group can yield nonzero FoldGRPO advantages, but not yet
that the learning signal improves function localization. On this group it
would reinforce preferred question behavior while providing no positive
productivity example.

The initial server wrapper also placed PID/readiness bookkeeping in the wrong
shell context because an ungrouped background operator applied to the preceding
AND-list. The server itself became healthy, so its real PID was recovered and
the gate continued without a restart. No Gemini call occurred before server
health was verified.

## Corrective change

No agent, prompt, tool, simulator, or reward change is justified. The next
training-mechanics probe should resume the ordinary dataloader for exactly one
additional group, record the adapter hash before and after, and require a
nonzero gradient before claiming the first effective RL update.

Do not hand-pick a task or alter the reward merely to obtain variance. A future
cloud run helper should own vLLM startup, readiness, PID capture, and cleanup
instead of relying on an ad hoc background shell expression.

## Process signal

- Detectable offline: the shell grouping error was; the reward decomposition
  required the live group.
- Cheapest gate that would have caught it: a shell lifecycle test for the
  server wrapper and this 76-second no-optimizer group.
- Paid time spent before detection: 229 server-seconds, approximately `$0.15`.
- New invariant or observability requirement: report total-reward variance and
  productivity variance separately; composite variance alone is insufficient
  evidence of task learning.
- Question for the final retrospective: should the phase gate require at least
  one positive-productivity trajectory, or only a nonzero gradient under the
  paper's full composite reward?
