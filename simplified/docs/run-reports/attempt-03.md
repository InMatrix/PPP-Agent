# Attempt 03 — per-turn vLLM token limit

## Result

The run reached vLLM request construction but failed because token limits were
represented in more than one layer of the sampling payload. The per-turn cap
was not being consumed consistently, creating a duplicate `max_tokens`
argument at the compatibility boundary. No completion, optimizer work, or
checkpoint occurred.

## Configuration and evidence

- Date: 2026-07-29.
- Intended limit: at most 512 generated tokens per turn and 4K per trajectory.
- Simulator: deterministic; Gemini was not called.
- Evidence: preserved traceback summary and corrective commit. Raw attempt log,
  exact duration, memory, and cost are unavailable.

## Failure and diagnosis

The simplified client and vendored vLLM server both adapted sampling
parameters. The compatibility layer needed one authoritative operation that
removed any existing token-limit key and returned the effective bounded value.

This was an API-adaptation failure across local client, Verl, and vLLM 0.12.

## Corrective change

Commit `2899cb2` (`Honor per-turn vLLM token limits`) introduced a shared
`pop_max_tokens` compatibility helper, applied the remaining-context cap, and
added tests for duplicate keys and bounded values.

## Process signal

- Detectable offline: **yes**.
- Cheaper prevention: pass representative sampling dictionaries through the
  complete client/server adaptation chain in a no-model test.
- Retrospective question: can sampling-parameter ownership be centralized so
  version shims do not independently mutate the same field?
