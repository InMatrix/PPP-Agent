# GH200 compatibility-run reports

These reports reconstruct the nine one-step compatibility attempts used to
bring the teaching-scale PPP reinforcement-learning stack up on one Lambda
GH200. They are intended as source material for a later retrospective on the
development process, not as benchmark results.

The current execution status and next planned gate live in
[`../PHASE_PLAN.md`](../PHASE_PLAN.md). After every new run, update both its
individual report and that plan.

The Lambda instance was terminated before its ignored run directory was copied
off-host. The individual reports therefore distinguish:

- **Preserved evidence:** commits, source changes, tests, terminal output quoted
  or summarized in the development task, and the final observed artifact counts.
- **Unavailable evidence:** raw `gh200-one-step-attempt*.log` files, exact cost,
  complete per-process memory histories, and generated trajectory bodies.
- **Inference:** conclusions derived from a traceback plus the corrective change.
  Inferences are labeled and are not presented as raw measurements.

All attempts used `Qwen/Qwen3-4B`, BF16 rank-16 LoRA, eight trajectories,
navigation-v2 read-only tools, at most eight logical turns, and the deterministic
simulator. Gemini was not called. Consequently, these runs test integration and
training mechanics—not user-simulator quality or trained-agent performance.

## Attempt index

| Attempt | Furthest boundary reached | Failure | Corrective commit | Failure class |
|---:|---|---|---|---|
| [01](attempt-01.md) | Agent-loop worker import | Missing `unidiff` | `5dfd86e` | Dependency closure |
| [02](attempt-02.md) | Concurrent rollout workspace setup | Shared workspace race | `a7b5f6f` | Concurrency |
| [03](attempt-03.md) | First vLLM request construction | Duplicate/uncapped `max_tokens` | `2899cb2` | API adaptation |
| [04](attempt-04.md) | vLLM engine execution | Executor returned no future | `618737d` | Diagnostic mitigation |
| [05](attempt-05.md) | vLLM engine execution | Eager mode did not fix executor contract | `560f8cd` | Async runtime contract |
| [06](attempt-06.md) | First real generated completion | Token log probabilities absent | `39b6031` | Training-data contract |
| [07](attempt-07.md) | Sanitized trajectory export | Duplicate Pydantic model identity | `786f7dc` | Type ownership |
| [08](attempt-08.md) | Trainer preprocessing after eight rollouts | Missing `mask_rollout` | `0de62a4` | Batch-schema contract |
| [09](attempt-09.md) | FoldGRPO advantage entry after actor preprocessing | Missing group `uid` | `139e5d3` | Grouping/schema contract |

No attempt completed an optimizer step or wrote a checkpoint. Attempts 08 and
09 each completed a full eight-rollout group before failing at the next trainer
boundary. The last observed compatibility directory contained 17 sanitized
trajectory files: one retained from attempt 07 plus eight each from attempts 08
and 09.

## Development-process signals

The sequence exposes three distinct phases:

1. **Host and request-path bring-up (01–06).** Missing dependencies, concurrent
   setup, and vLLM version contracts prevented or corrupted generation.
2. **Rollout-output ownership (07).** The simplified adapter returned a
   structurally identical model owned by the wrong module.
3. **Trainer batch assembly (08–09).** Expensive rollouts succeeded, but fields
   required only downstream were dropped by postprocessing.

The most important candidate improvement for the final retrospective is a
cheap, synthetic **adapter-to-trainer contract gate**. It should construct eight
mock `AgentLoopOutput` objects, run Verl postprocessing, and assert the complete
FoldGRPO batch schema before any model is loaded:

- tensors: `response_mask`, `rollout_log_probs`, `process_reward_mask`,
  `mask_rollout`, and terminal reward scores;
- non-tensors: one shared `uid`, eight distinct `gen_uid` values, and reward
  decomposition metadata;
- invariants: group size eight, model-only loss mask, finite normalized
  advantages, and no repository or secret leakage.

Other retrospective candidates:

- Export the sanitized failure bundle to durable storage automatically before
  launcher cleanup or instance termination.
- Record one machine-readable `stage_reached` value so reports do not rely on
  traceback interpretation.
- Add contract tests at ownership boundaries, not only source-shape tests.
- Separate diagnostic mitigations from confirmed fixes in commit messages and
  reports.
- Use a short synthetic completion for integration gates, reserving full
  multi-turn rollouts for the first test that actually needs them.

## How to use these reports

For the final retrospective, compare:

- failures detectable locally versus failures that genuinely required the GPU;
- time spent generating trajectories before a downstream schema failure;
- diagnostic commits versus root-cause fixes;
- missing observability that made reconstruction harder; and
- which new gate would have prevented each repeated paid attempt.

The per-attempt reports use the same headings so these comparisons can be made
without reopening implementation history.
