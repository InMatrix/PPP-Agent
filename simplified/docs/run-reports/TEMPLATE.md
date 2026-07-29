# Attempt NN — concise failure or outcome

## Result

State the furthest completed stage and the terminal outcome. Explicitly say
whether these gates completed:

- eight-rollout group;
- old-policy log probabilities;
- reward and FoldGRPO advantages;
- backward and optimizer step;
- LoRA checkpoint save; and
- checkpoint reload/resume.

## Configuration and evidence

- Date and UTC start/end timestamps:
- Git commit:
- Hardware and hourly rate:
- Model and precision:
- Group, turn, and token limits:
- Simulator, including whether it was live or deterministic:
- Peak GPU memory:
- Estimated compute cost:
- Durable sanitized log/report location:
- Evidence limitations:

Never include API keys, hidden simulator context, raw repository contents, or
unsanitized observations.

## Failure and diagnosis

Quote only the minimal sanitized exception. Separate observed facts from
inference, and identify the exact ownership boundary that failed.

## Corrective change

Record the fix commit, tests added, verification result, and whether the change
is diagnostic or a confirmed root-cause fix.

## Process signal

- Detectable offline:
- Cheapest gate that would have caught it:
- Paid time spent before detection:
- New invariant or observability requirement:
- Question for the final retrospective:
