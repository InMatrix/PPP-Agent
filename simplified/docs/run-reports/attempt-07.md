# Attempt 07 — Verl output-model ownership

## Result

The run sustained real multi-turn generation and exported one sanitized
trajectory. Verl then rejected its metrics object during postprocessing:
Pydantic saw an `AgentLoopMetrics` instance owned by `agents.utils`, while the
internal output model required Verl's distinct class of the same name.

No reward-group computation, optimizer work, or checkpoint occurred. GPU
memory returned to idle after failure.

## Configuration and evidence

- Date: 2026-07-29.
- Observed GPU allocation during rollouts: approximately 35.6 GB.
- Simulator: deterministic; Gemini was not called.
- Artifacts: one sanitized trajectory was observed on the instance.
- Evidence: preserved Pydantic traceback and corrective commit. The raw attempt
  log, exact end time, trajectory body, and cost are unavailable.

## Failure and diagnosis

The two models were structurally identical but had different Python identities.
The simplified adapter imported `AgentLoopOutput` and `AgentLoopMetrics` from
the original agent utilities rather than from the framework that validated
them.

## Corrective change

Commit `786f7dc` (`Use Verl rollout output models`) retained `Agent` and
`CallLLM` from the original utilities but constructed the return value using
Verl's native output and metrics classes. The local suite passed 96 tests.

## Process signal

- Detectable offline: **yes** with a postprocessor contract test.
- Cheaper prevention: pass one synthetic adapter output through
  `_agent_loop_postprocess` before loading Qwen.
- Retrospective question: should boundary objects always be constructed from
  the receiving subsystem's types?
