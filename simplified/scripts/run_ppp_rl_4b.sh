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
run_dir="${PPP_RUN_DIR:-$run_dir}"
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
  "${PPP_PYTHON:-python3}" -m ppp_simplified.training_cli check-extension \
    --run-dir "$run_dir" \
    --projected-compute-usd "$projected_compute_usd"
fi

command=(
  "${PPP_PYTHON:-python3}" -m scripts.train_ppp_simplified
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
  # Avoid vLLM's compile/cudagraph path during the single-GPU compatibility
  # gate. The GH200 ARM64 engine reached its first request but returned a null
  # execution future in compiled mode.
  actor_rollout_ref.rollout.enforce_eager=True
  # The colocated 4B actor leaves about 20 GiB free even on an 80 GiB H100.
  # vLLM interprets this fraction against total device memory, so 0.20 keeps
  # its requested KV-cache allocation below the observed free-memory ceiling.
  actor_rollout_ref.rollout.gpu_memory_utilization=0.20
  actor_rollout_ref.rollout.max_num_seqs=1
  actor_rollout_ref.rollout.max_num_batched_tokens=10240
  actor_rollout_ref.rollout.prompt_length=6144
  actor_rollout_ref.rollout.response_length=4096
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240
  actor_rollout_ref.model.path="$model_id"
  # Verl defaults actor construction to FlashAttention 2. Use PyTorch SDPA so
  # the compatibility gate does not depend on a separately compiled extension.
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
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
  # Vendored FoldGRPO writes this compatibility label into each rollout
  # batch even though the simplified agent loop does not branch on it.
  +actor_rollout_ref.rollout.plugin.workflow=search
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
  # Declare keys that vendored Verl mutates so OmegaConf struct mode permits
  # the updates inside main_ppo.run_ppo.
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1=1
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

ppp_python="${PPP_PYTHON:-python3}"
if [[ -n "${PPP_RAY_CLI:-}" ]]; then
  ray_cli="$PPP_RAY_CLI"
elif [[ "$ppp_python" == */* ]]; then
  ray_cli="$(dirname "$ppp_python")/ray"
else
  ray_cli="$(command -v ray || true)"
fi
nvidia_smi="${PPP_NVIDIA_SMI:-$(command -v nvidia-smi || true)}"
gpu_idle_attempts="${PPP_GPU_IDLE_ATTEMPTS:-15}"
gpu_idle_interval="${PPP_GPU_IDLE_INTERVAL_SECONDS:-1}"

stop_ray_runtime() {
  "$ray_cli" stop --force >/dev/null 2>&1
}

wait_for_idle_gpu() {
  [[ -n "$nvidia_smi" ]] || return 0
  local attempt compute_pids raw_pids
  for ((attempt = 1; attempt <= gpu_idle_attempts; attempt++)); do
    if ! raw_pids="$(
      "$nvidia_smi" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null
    )"; then
      echo "Unable to inspect GPU compute processes with $nvidia_smi." >&2
      return 1
    fi
    compute_pids="$(printf '%s' "$raw_pids" | tr '\n' ' ' | xargs)"
    if [[ -z "$compute_pids" ]]; then
      return 0
    fi
    if ((attempt < gpu_idle_attempts)); then
      sleep "$gpu_idle_interval"
    fi
  done
  echo "GPU compute processes remain after Ray cleanup: $compute_pids" >&2
  return 1
}

cleanup_paid_runtime() {
  local status=$?
  local cleanup_failed="false"
  trap - EXIT INT TERM
  if ! stop_ray_runtime; then
    echo "WARN: failed to stop the Ray runtime during cleanup." >&2
    cleanup_failed="true"
  fi
  if ! wait_for_idle_gpu; then
    echo "WARN: GPU cleanup did not complete; terminate the instance." >&2
    cleanup_failed="true"
  fi
  if [[ "$status" -eq 0 && "$cleanup_failed" == "true" ]]; then
    status=2
  fi
  exit "$status"
}

# A failed Ray driver can leave vLLM engine processes alive after the shell
# exits. Start from an idle dedicated host and clean up on every exit path.
if [[ ! -x "$ray_cli" ]]; then
  echo "Ray CLI is unavailable: $ray_cli" >&2
  exit 2
fi
trap cleanup_paid_runtime EXIT INT TERM
if ! stop_ray_runtime; then
  echo "Failed to stop the existing Ray runtime; refusing paid execution." >&2
  exit 2
fi
if ! wait_for_idle_gpu; then
  echo "Refusing to start a paid run on a contaminated GPU." >&2
  exit 2
fi

prepared_dir="${PPP_PREPARED_DIR:-simplified/results/training/prepared}"
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
"$ppp_python" -m ppp_simplified.training_cli summarize-run \
  --run-dir "$run_dir" \
  --compute-seconds "$((ppp_run_finished_at - ppp_run_started_at))" \
  --hourly-usd 1.09
if [[ "$resume_probe" == "true" ]]; then
  touch "${run_dir}/resume-verified"
fi
