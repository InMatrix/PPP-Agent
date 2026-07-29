# Attempt 04 — vLLM executor returned no future

## Result

The engine initialized and reached model execution, but vLLM 0.12 expected the
external executor's `execute_model` method to return a future-like object. The
vendored ZeroMQ executor returned `None`, causing the engine-core request to
fail. No usable completion, optimizer work, or checkpoint occurred.

## Configuration and evidence

- Date: 2026-07-29.
- Runtime: vLLM 0.12 with a colocated Verl actor/rollout worker.
- Simulator: deterministic; Gemini was not called.
- Evidence: preserved traceback summary and diagnostic commit. Raw attempt log,
  exact duration, memory, and cost are unavailable.

## Failure and diagnosis

At this point it was not yet proven whether the failure depended on compiled
execution or on the executor API itself. The next change deliberately tested
the narrower eager-execution hypothesis.

## Diagnostic change

Commit `618737d` (`Use eager vLLM execution for compatibility gate`) set
`enforce_eager=True` and added launcher coverage. This was a bounded diagnostic
mitigation, not a confirmed root-cause fix.

## Process signal

- Detectable offline: **partly**. Signature/return-shape inspection was
  possible locally; real engine behavior still benefited from the GPU.
- Cheaper prevention: assert the executor protocol against the installed vLLM
  class before loading model weights.
- Retrospective question: should diagnostic commits be labeled explicitly in
  their subject so they cannot be mistaken for validated fixes?
