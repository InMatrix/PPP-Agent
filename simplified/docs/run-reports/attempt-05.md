# Attempt 05 — eager mode did not repair executor contract

## Result

The eager-mode retry reproduced the same engine-core failure: vLLM still
received `None` where it required a future. This ruled out compilation/capture
mode as the cause. No usable completion, optimizer work, or checkpoint
occurred.

## Configuration and evidence

- Date: 2026-07-29.
- Difference from attempt 04: eager vLLM execution was forced.
- Simulator: deterministic; Gemini was not called.
- Evidence: preserved failure comparison and root-cause fix commit. Raw attempt
  log, exact duration, memory, and cost are unavailable.

## Failure and diagnosis

The ZeroMQ-backed external executor implemented synchronous dispatch but did
not implement vLLM 0.12's asynchronous return contract. Eager execution does
not change that interface requirement.

## Corrective change

Commit `560f8cd` (`Return futures from external vLLM executor`) added a bounded
single-thread executor and made `execute_model` return a real `Future`.
Compatibility-source tests covered the method contract. The following attempt
completed real token generation, confirming this as the root-cause fix.

## Process signal

- Detectable offline: **yes**, once the installed protocol was inspected.
- Cheaper prevention: instantiate the executor with a stub worker and assert
  both return type and resolved value.
- Retrospective question: after one diagnostic retry, should the workflow
  require protocol-level evidence before another paid run?
