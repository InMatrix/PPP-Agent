# GH200 compatibility-run reports

These reports reconstruct the compatibility attempts used to
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

Attempts 01–09 used `Qwen/Qwen3-4B`, BF16 rank-16 LoRA, eight trajectories,
navigation-v2 read-only tools, at most eight logical turns, and the deterministic
simulator. Attempt 10 used eight synthetic trajectories and no model. Gemini was
not called. Consequently, these runs test integration and training mechanics—not
user-simulator quality or trained-agent performance.

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
| [10](attempt-10.md) | Synthetic FoldGRPO advantage and DAPO loss | Passed; no model or optimizer used | `2042e01` | Preventive contract gate |
| [11](attempt-11.md) | Real backward, optimizer, and checkpoint save | Flat zero-reward group; no effective parameter update | Same evidence commit | Learning signal / observability |
| [12](attempt-12.md) | Full checkpoint reload at completed total | Passed without rollouts, optimization, or report overwrite | `5d5ecec`, `883df51` | Resume semantics |
| [13](attempt-13.md) | Eight live Qwen/Gemini trajectories | Composite reward varied, but productivity was zero throughout | Pending continuation gate | Reward interpretation |

Attempt 11 completed the first real optimizer call and wrote the first
checkpoint. Because its eight rewards were all zero, it did not change the
policy. Attempts 08 and 09 each completed a full eight-rollout group before
failing at the next trainer boundary.

Attempt 10 used the actual vendored Verl/Torch postprocessing, FoldGRPO, and
DAPO loss path with synthetic inputs. Its
[`sanitized machine-readable artifact`](artifacts/attempt-10-contract-gate.json)
is durable, but it did not load Qwen or perform an optimizer update.

Attempt 11's LoRA adapter and sanitized evidence bundle are also preserved in
the ignored local results directory. Its compact
[`tracked result`](artifacts/attempt-11-one-step.json) contains hashes and
aggregate metrics without prompts, repository observations, or trajectory IDs.

Attempt 12 proved that the full checkpoint can reload without accidentally
running step 2. Its
[`tracked resume result`](artifacts/attempt-12-resume.json) records unchanged
adapter/report hashes and zero new rollouts.

Attempt 13 exercised the real Gemini UserVille path without constructing an
optimizer. Its
[`tracked live-group result`](artifacts/attempt-13-live-gemini.json) separates
composite reward variance from the all-zero productivity component.

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
