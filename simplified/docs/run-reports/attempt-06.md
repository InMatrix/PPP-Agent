# Attempt 06 — generated-token log probabilities

## Result

The future bridge worked: vLLM completed the first real capped Qwen generation.
Rollout bookkeeping then failed while appending the completion because
`response_log_probs` was `None`. No complete trajectory, optimizer work, or
checkpoint occurred.

This was the first attempt to prove that actor-to-vLLM model execution worked
on the GH200.

## Configuration and evidence

- Date: 2026-07-29.
- Generation: real `Qwen/Qwen3-4B`, capped at 512 tokens for the turn.
- Simulator: deterministic; Gemini was not called.
- Evidence: preserved traceback (`list + NoneType` in `agents/utils.py`),
  observed successful generation, and corrective commit. Raw attempt log,
  exact duration, memory history, and cost are unavailable.

## Failure and diagnosis

The client requested text and token IDs but not chosen-token log probabilities.
The vLLM server therefore correctly returned `log_probs=None`. Substituting
zeros would have made the integration appear to work while invalidating the
old-policy baseline required for PPO/DAPO.

The same inspection also found that the client computed a per-turn token cap
into a different variable and did not reliably send it.

## Corrective change

Commit `39b6031` (`Request rollout token log probabilities`) explicitly set
`logprobs=True`, propagated the effective token cap, and added regression
coverage. The local suite passed 95 tests after the change.

## Process signal

- Detectable offline: **yes** through a request/response contract test.
- Cheaper prevention: validate every field required by the eventual loss
  before beginning model generation.
- Retrospective question: can the adapter declare a typed "training
  completion" contract distinct from an inference-only completion?
