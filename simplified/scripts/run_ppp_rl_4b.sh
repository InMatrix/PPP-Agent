#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--print-command|--execute] [--steps 1|20|40] [--simulator deterministic|gemini] [--model qwen35|qwen3] [--projected-compute-usd AMOUNT]"
  echo
  echo "The default is --print-command. --execute requires:"
  echo "  CONFIRM_PAID_TRAINING=I_UNDERSTAND_LAMBDA_IS_BILLING"
}

mode="print"
steps="20"
simulator="gemini"
model="qwen35"
projected_compute_usd=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --print-command)
      mode="print"
      shift
      ;;
    --execute)
      mode="execute"
      shift
      ;;
    --steps)
      steps="${2:?--steps requires a value}"
      shift 2
      ;;
    --simulator)
      simulator="${2:?--simulator requires a value}"
      shift 2
      ;;
    --model)
      model="${2:?--model requires a value}"
      shift 2
      ;;
    --projected-compute-usd)
      projected_compute_usd="${2:?--projected-compute-usd requires a value}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$steps" != "1" && "$steps" != "20" && "$steps" != "40" ]]; then
  echo "--steps must be 1, 20, or 40" >&2
  exit 2
fi
if [[ "$simulator" != "deterministic" && "$simulator" != "gemini" ]]; then
  echo "--simulator must be deterministic or gemini" >&2
  exit 2
fi
case "$model" in
  qwen35)
    model_id="Qwen/Qwen3.5-4B"
    model_slug="qwen35-4b"
    ;;
  qwen3)
    model_id="Qwen/Qwen3-4B"
    model_slug="qwen3-4b"
    ;;
  *)
    echo "--model must be qwen35 or qwen3" >&2
    exit 2
    ;;
esac
if [[ "$steps" == "1" ]]; then
  run_dir="simplified/results/checkpoints/${model_slug}/compatibility"
  save_frequency="1"
else
  run_dir="simplified/results/checkpoints/${model_slug}/training"
  save_frequency="5"
fi
if [[ "$simulator" == "gemini" && "$mode" == "execute" && -z "${GEMINI_API_KEY:-}" ]]; then
  echo "GEMINI_API_KEY is required for a live simulator run." >&2
  exit 2
fi
if [[ "$mode" == "execute" && "${CONFIRM_PAID_TRAINING:-}" != "I_UNDERSTAND_LAMBDA_IS_BILLING" ]]; then
  echo "Paid execution guard not confirmed. Print the command first and review the Lambda console." >&2
  exit 2
fi
if [[ "$mode" == "execute" && "$steps" == "40" ]]; then
  if [[ -z "$projected_compute_usd" ]]; then
    echo "--projected-compute-usd is required before extending to 40 steps." >&2
    exit 2
  fi
  python -m ppp_simplified.training_cli check-extension \
    --run-dir "$run_dir" \
    --projected-compute-usd "$projected_compute_usd"
fi

command=(
  python -m scripts.train_ppp_simplified
  algorithm.adv_estimator=foldgrpo
  algorithm.norm_adv_by_std_in_grpo=True
  actor_rollout_ref.rollout.agent.default_agent_loop=simplified_ppp_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path=simplified/config/verl_agent_loops.yaml
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.agent.num_workers=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35
  actor_rollout_ref.rollout.max_num_seqs=1
  actor_rollout_ref.rollout.max_num_batched_tokens=10240
  actor_rollout_ref.rollout.prompt_length=6144
  actor_rollout_ref.rollout.response_length=4096
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240
  actor_rollout_ref.model.path="$model_id"
  actor_rollout_ref.model.lora_rank=16
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.actor.optim.lr=1e-6
  # Verl expresses this batch size in prompt groups before rollout.n expands
  # the single prompt into eight FoldGRPO trajectories.
  actor_rollout_ref.actor.ppo_mini_batch_size=1
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=10240
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  data.train_files="${PPP_TRAIN_DATA:-simplified/results/training/prepared/train-16x13.parquet}"
  # Validation is disabled during updates; point the required field at the
  # training subset so the sealed held-out parquet is never opened implicitly.
  data.val_files="${PPP_TRAIN_DATA:-simplified/results/training/prepared/train-16x13.parquet}"
  data.train_batch_size=1
  data.seed=42
  data.max_prompt_length=6144
  data.max_response_length=4096
  data.return_raw_chat=True
  +actor_rollout_ref.rollout.plugin.workspace_root=simplified/workspaces
  +actor_rollout_ref.rollout.plugin.max_turn=8
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=512
  +actor_rollout_ref.rollout.plugin.simulator="$simulator"
  +actor_rollout_ref.rollout.plugin.gemini_model=gemini-3.5-flash-lite
  +actor_rollout_ref.rollout.plugin.simulator_cache=simplified/results/simulator-cache.sqlite3
  # Worst case for forty updates: 40 groups × 8 rollouts × 8 questions.
  +actor_rollout_ref.rollout.plugin.simulator_call_budget=2560
  +actor_rollout_ref.rollout.plugin.sanitized_trajectory_dir="${run_dir}/sanitized-trajectories"
  trainer.val_before_train=False
  trainer.val_only=False
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.total_training_steps="$steps"
  trainer.test_freq=-1
  trainer.save_freq="$save_frequency"
  trainer.resume_mode=auto
  trainer.logger='["console","file"]'
  trainer.project_name=ppp-teaching
  trainer.experiment_name="${model_slug}-lora-${simulator}-${steps}step"
  trainer.default_local_dir="$run_dir"
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH="${run_dir}/training-metrics.jsonl"
)

if [[ "$model" == "qwen35" ]]; then
  # Qwen3.5 is a hybrid multimodal architecture. PPP is text-only, so avoid
  # loading its visual tower during rollout on the 48 GB compatibility gate.
  command+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True)
fi

if [[ "$mode" == "print" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

prepared_dir="simplified/results/training/prepared"
resume_probe="false"
if compgen -G "${run_dir}/global_step_*" >/dev/null; then
  resume_probe="true"
fi
for immutable in training-config.json training-subset.json; do
  source_file="${prepared_dir}/${immutable}"
  destination_file="${run_dir}/${immutable}"
  if [[ ! -f "$source_file" ]]; then
    echo "Missing prepared input: $source_file. Run ppp-train prepare first." >&2
    exit 2
  fi
  mkdir -p "$run_dir"
  if [[ -f "$destination_file" ]] && ! cmp -s "$source_file" "$destination_file"; then
    echo "Resume identity mismatch: $destination_file differs from prepared input." >&2
    exit 2
  fi
  if [[ ! -f "$destination_file" ]]; then
    cp "$source_file" "$destination_file"
  fi
done

export VERL_FILE_LOGGER_PATH="${run_dir}/training-metrics.jsonl"
ppp_run_started_at="$(date +%s)"
"${command[@]}"
ppp_run_finished_at="$(date +%s)"
python -m ppp_simplified.training_cli summarize-run \
  --run-dir "$run_dir" \
  --compute-seconds "$((ppp_run_finished_at - ppp_run_started_at))" \
  --hourly-usd 1.09
if [[ "$resume_probe" == "true" ]]; then
  touch "${run_dir}/resume-verified"
fi
