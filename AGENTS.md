# Project instructions

## Substantial coding-task handoffs

After completing each substantial coding task, provide a learner-oriented
walkthrough of the implementation. Reserve the full walkthrough for meaningful
features, integrations, or architectural changes; keep routine fixes concise.

The walkthrough should:

- Lead with the outcome and summarize what changed.
- Trace the execution path through the relevant code.
- Link to the key local files and line numbers.
- Clearly distinguish real production or benchmark behavior from mocked,
  scripted, deterministic, or incomplete behavior.
- Explain important design decisions, tradeoffs, and current limitations.
- Show how to run, inspect, and experiment with the implementation.
- Report the relevant verification performed and include representative output
  when useful.

Write the walkthrough for someone learning the codebase. Prefer following one
concrete request, episode, or execution trace end to end over describing files
alphabetically.

## Training and compatibility run reports

After every new compatibility, training, or evaluation run, create or update a
sanitized report under `simplified/docs/run-reports/` using
`TEMPLATE.md`. Write the report before terminating a paid instance whenever
possible, and do not start the next paid run until the previous run is
documented. In the same update, revise `simplified/docs/PHASE_PLAN.md` with the
stage reached, current blocker, evidence or artifact links, budget information
when available, and the exact next gate.

Each report should record the tested commit and configuration, real versus
deterministic or mocked components, stages reached, available timing, memory,
cost and artifact evidence, the minimal failure and diagnosis, and the
resulting fix commit once known. Clearly label unavailable evidence and
inference. Never include credentials, hidden simulator context, raw repository
contents, or unsanitized observations.

Every run must receive trajectory-level error analysis before selecting or
implementing the next intervention. Inspect every trajectory when the group is
small; otherwise inspect all failures plus a representative stratified sample.
At minimum, report action parsing and tool use, navigation behavior, finish
validity and correction behavior, termination reasons, reward components,
masking or loss eligibility, and recurring failure categories. Compare
successful and failed trajectories, distinguish counts derived directly from
artifacts from inferred counts, and state evidence that cannot be recovered.
Do not explain a run using aggregate reward or loss alone.

For a targeted diagnostic run that produces model completions but no agent
trajectories, apply the same rule at completion level: inspect every completion,
validation stage, termination or failure reason, and available token/logprob
evidence before choosing the next intervention.
