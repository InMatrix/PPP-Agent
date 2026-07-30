# Attempt 14 — ordinary step-two group is flat and produces no adapter delta

## Result

The guarded continuation resumed the ordinary dataloader at step 1, generated
exactly eight new live Qwen/Gemini trajectories, calculated old-policy log
probabilities, ran FoldGRPO and DAPO backward, and wrote `global_step_2`.
However, all eight optimized rewards were zero:

```text
[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
```

Consequently, the group-relative advantages, policy loss, gradient norm, and
effective LoRA update were all zero. The step-1 and step-2 adapter SHA-256
hashes are identical. This attempt therefore failed the continuation gate even
though the trainer mechanics and checkpoint write completed.

The controlling SSH connection closed immediately after checkpointing. The
trainer's scalar line survived in the Ray worker log, but the wrapper did not
reach the repository verifier or write its end timestamp.

## Configuration and evidence

- Date and observed run interval: 2026-07-29,
  `06:07:30Z`–`06:22:51Z`.
- Git commit: `bcf5e5d` (`Add guarded step-two continuation gate`).
- Hardware and hourly rate: one Lambda GH200 96 GB at `$2.29/hour`.
- Policy model and update: `Qwen/Qwen3-4B`, BF16 rank-16 LoRA, learning rate
  `1e-6`.
- Simulator: `gemini-3.5-flash-lite` through the instance-local AI Studio key.
- Dataset position: ordinary resumed group
  `vague+first_try+code_func_loc_train-bokeh__bokeh-13422`; it was not
  hand-picked.
- Group and limits: eight trajectories, maximum eight logical turns, 4,096
  generated response tokens per trajectory.
- New/total sanitized trajectories: 8/16.
- Trainer step time: 858.51 seconds; observed wrapper interval: 921 seconds.
- Estimated gate compute: `$0.5859` before tax. Later idle-instance time is not
  included.
- Peak GPU memory: 64.98 GiB allocated and 65.13 GiB reserved.
- Durable ignored evidence:
  `simplified/results/training/attempt-14/`.
- Durable tracked aggregate:
  [`artifacts/attempt-14-step-two.json`](artifacts/attempt-14-step-two.json).
- Simulator cache and hidden replies were not copied or exported.

The ignored evidence directory contains all 16 sanitized trajectories, the
step-two LoRA adapter, the launcher log, and the Ray task-runner scalar log.
The copied adapter hash matches the remote checkpoint.

## Learning-signal diagnosis

All eight productivity rewards were zero and no finish passed validation. Four
trajectories asked no question and received a `-0.10` proactivity adjustment.
The other four asked one question, receiving `+0.05` proactivity but `-0.10`
personalization because the reply did not satisfy this prompt's `first_try`
preference. After the composite reward was clipped at zero, every trajectory
had the same optimized reward.

The trainer reported:

```text
actor/pg_loss       0.0
actor/grad_norm     0.0
rollout_corr/kl     0.0009106665
reward/std_score    0.0
overlong_masked     2
```

The KL and old-policy path were finite, and the actor update call took 24.88
seconds. The unchanged adapter proves that completing an optimizer call is not
equivalent to completing an effective RL update.

Five trajectories finished naturally, two hit the turn limit, and one used
deadline finish. Six attempted finish correction. Across the group there were
61 model calls, four live Gemini calls, 23,259 model-generated tokens, and
seven invalid predictions.

## Trajectory-level investigation

A 2026-07-30 review compared all eight sanitized step-2 traces with the
training-row ground truth. The vague issue was:

```text
bokeh info shud show jupyter_bokeh version if installed, saves back&forth
```

The expected localization was:

```text
src/bokeh/command/subcommands/info.py:Info.invoke
```

The action-level results were:

| Signal | Count |
|---|---:|
| Model calls | 61 |
| Parsed actions preserved in sanitized traces | 31 |
| Inferred invalid/unparseable actions | 30 |
| Trajectories with no parsed tool action | 2 of 8 |
| Trajectories using `list_tree` | 1 of 8 |
| Initial invalid finish proposals | 6 of 6 finishing trajectories |
| Successful finish corrections | 0 of 6 |
| Valid predicted functions | 0 |

The invalid-action count is inferred as `model_calls - sanitized actions`.
Every successfully parsed action is recorded by the Verl loop, while its parse
error branch feeds the error back to the model without adding a sanitized
record. Two trajectories made nine calls, executed no tool, and hit the turn
limit. The other six also lost between one and five calls to the same
unobserved path. The exact proportions of malformed JSON, extra prose, missing
arguments, and disallowed final-turn tools cannot be reconstructed.

The successfully parsed actions show a separate navigation failure. Only one
trajectory inspected the repository tree. The others jumped to unsupported
paths and symbols derived from the issue wording, including:

```text
bokeh/tools/info.py:show_info
bokeh/commands/info.py:show_info
bokeh/cli/info.py:show_version
bokeh/jupyter_bokeh/__init__.py:show_info
```

The model treated “bokeh info” and “show jupyter_bokeh version” as candidate
function names instead of discovering the `src/bokeh/command/subcommands`
layout, finding class `Info`, and inspecting method `invoke`. The frozen
`list_tree`, `find_symbol`, `search_code`, and `read_file` tools were sufficient
for that path; this trace does not justify adding another repository tool.

All six initial finish attempts failed repository/AST validation. Every
correction ultimately resubmitted the same rejected target. One trajectory
called `ask_user` between its invalid finish and correction, exposing a harness
divergence: the ordinary simplified runner permits only `finish` on that
correction call, while the Verl loop leaves all tools available except on the
deadline turn.

Four trajectories asked one question. The Gemini simulator judged all four
incompatible with the `first_try` preference. Their preceding work consisted
largely of unsupported path probes, so the questions did not demonstrate a
decision-changing repository blocker. The other four asked nothing and
received the no-question proactivity penalty.

The two actionless turn-limit trajectories were also fully masked by the
planned overlong policy. That masking is algorithmically intentional, but it
means those worst action-format failures would not receive a direct policy
gradient even if the group reward had varied.

## Local-versus-training action contract

The local LM Studio agent used `Qwen3.5-4B`, temperature `0.2`, and a strict
JSON Schema passed as `response_format`. The GH200 trainer used
`Qwen/Qwen3-4B`, temperature `1.0`, and raw vLLM token sampling with the JSON
contract present only in prompt text. The run therefore changed the base model,
sampling regime, and output enforcement simultaneously.

The next compatibility target is the Hugging Face
`Qwen/Qwen3.5-4B` checkpoint in BF16 with LoRA. It matches the local model
family, but it is not numerically identical to the quantized MLX artifact used
by LM Studio. Qwen3.5 requires a separate Transformers 5/vLLM environment; the
proven Qwen3 fallback environment must remain untouched.

## External interruption and observability

Verl saves the checkpoint before calling its aggregate logger. The SSH session
closed after `global_step_2` was written but before that logger call and before
the launcher's continuation verifier. Ray then terminated its workers. As a
result:

- the complete scalar record exists in the Ray task-runner log;
- `training-metrics.jsonl` is empty;
- no `continuation-step-2.json` was produced; and
- the wrapper's `RUN_END_UTC` is absent.

This interruption did not cause the zero update: the scalar record and
identical adapter already prove the flat reward and zero gradient. It did expose
that a long paid run must not be owned by a foreground SSH session and that
scalar evidence should be made durable before checkpoint shutdown can
intervene.

## Corrective direction

The detached launcher and pre-checkpoint scalar durability fixes are complete
in commits `ac77e4e` and `213d64d`. Do not launch the 20-step job yet. First:

1. make the Verl loop's correction call finish-only, matching the ordinary
   runner;
2. record sanitized invalid-action categories without preserving model prose;
3. add version-aware schema-constrained vLLM generation while retaining
   chosen-token log probabilities;
4. bootstrap `Qwen/Qwen3.5-4B` in its own environment and pass a bounded
   load/generation, eight-rollout, old-log-probability, optimizer, and
   checkpoint-resume gate; and
5. keep Qwen3-4B as the documented fallback if that bounded gate still requires
   invasive changes to Verl.

Do not tune the reward or repository tools around this Bokeh example. Preserve
`navigation-v2` as the baseline. A compact prompt revision may be evaluated
later as `navigation-v2.1`, but only after output-schema and model parity are
restored and only across a diverse development set.

After those gates, the 20-step run becomes the stochastic learning-signal
test. It must retain flat groups honestly, report malformed-action rate, and
observe at least one effective adapter delta before claiming genuine learning.

## Process signal

- Detectable offline: the foreground-session ownership and logger ordering are
  testable without a GPU; this particular group's reward collapse required the
  live rollout.
- Cheapest gate that caught it: the bounded one-group continuation used here.
- Paid time spent before detection: 921 observed seconds, approximately
  `$0.59`.
- New invariant: report reward-component variance, final scalar variance,
  gradient norm, adapter delta, action-parse rate, and valid-finish rate
  separately.
- Retrospective question: should a checkpoint be labeled an optimizer step or
  an effective update when its adapter is byte-identical to its predecessor,
  and should action-schema enforcement be a required rollout contract?
