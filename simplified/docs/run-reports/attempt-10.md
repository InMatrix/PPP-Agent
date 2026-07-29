# Attempt 10 — synthetic adapter-to-FoldGRPO contract passes

## Result

The ARM64 GH200 dependency doctor and the no-model adapter-to-trainer contract
gate both passed. Eight synthetic Verl-native trajectories became one
`DataProto` group, produced finite FoldGRPO advantages and a finite DAPO policy
loss, and preserved model-only masking.

This was deliberately not an optimizer run:

- eight-rollout group: passed with synthetic trajectories;
- old-policy log probabilities: passed with synthetic finite values;
- reward and FoldGRPO advantages: passed;
- backward and optimizer step: not attempted;
- LoRA checkpoint save: not attempted; and
- checkpoint reload/resume: not attempted.

## Configuration and evidence

- Date and UTC contract start/end: 2026-07-29,
  `04:51:41Z`–`04:51:45Z`.
- Git commit: `2042e01` (`Add synthetic FoldGRPO contract gate`).
- Hardware and hourly rate: one Lambda GH200 96 GB; launch-screen hourly rate
  was not preserved in the task record.
- Model and precision: no model loaded; synthetic Torch tensors only.
- Group, turn, and token limits: group size eight; response tensor shape
  `[8, 6]`; turn and generation limits were not exercised.
- Simulator: synthetic deterministic data; neither Gemini nor the deterministic
  PPP simulator was called.
- Peak GPU memory: not captured; the gate did not allocate model weights.
- Estimated compute cost: unavailable because the instance launch time and
  hourly rate were not recorded. The measured contract command took four
  seconds after bootstrap.
- Durable sanitized report:
  [`artifacts/attempt-10-contract-gate.json`](artifacts/attempt-10-contract-gate.json).
- Evidence limitations: bootstrap duration and peak memory were not captured.
  The artifact intentionally contains no trajectory IDs, prompts,
  observations, repository contents, or secrets.

The dependency doctor also confirmed Python 3.11, PyTorch `2.9.0+cu128`,
Transformers `4.57.6`, vLLM `0.12.0`, PEFT `0.15.2`, Ray `2.56.1`, and vendored
Verl `0.7.0.dev` on `aarch64`, with the NVIDIA GH200 visible.

## Failure and diagnosis

No failure occurred. The gate proved the previously failing ownership boundary:
the simplified adapter's eight outputs survive the actual Verl async
postprocessor with one shared `uid`, eight unique `gen_uid` values, reward
metadata, masks, and rollout log probabilities intact.

It also proved that changing logits only at environment-token and
overlong-rollout positions does not change the computed policy loss.

## Corrective change

Commit `2042e01` added `ppp-train contract-gate` and its CLI regression test.
Local verification before the GH200 run was `98 passed`; the live Verl/Torch
contract then passed all eleven reported invariants. This is a confirmed
preventive gate, not a diagnostic mitigation.

## Process signal

- Detectable offline: yes, wherever the real vendored Verl/Torch dependencies
  can be installed; no model or GPU is intrinsically required.
- Cheapest gate that caught it: the four-second synthetic contract command.
- Paid time spent before detection: four seconds for this contract, excluding
  one-time environment bootstrap.
- New invariant or observability requirement: run and archive this gate before
  every model-backed compatibility attempt that changes adapter or Verl
  postprocessing behavior.
- Question for the final retrospective: should CI build a CPU-compatible Verl
  environment for this gate, or is a reusable prebuilt cloud image cheaper
  than maintaining that dependency surface?
