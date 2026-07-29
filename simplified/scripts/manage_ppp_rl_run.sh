#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
script_path="${script_dir}/$(basename "${BASH_SOURCE[0]}")"
train_runner="${PPP_TRAIN_RUNNER:-${script_dir}/run_ppp_rl_4b.sh}"
launch_root="${PPP_LAUNCH_ROOT:-${repo_root}/simplified/results/training/launches}"

usage() {
  cat <<'EOF'
Usage:
  manage_ppp_rl_run.sh start RUN_ID --steps 1|2|20|40 \
    --simulator deterministic|gemini --model qwen35|qwen3 \
    [--projected-compute-usd AMOUNT]
  manage_ppp_rl_run.sh status RUN_ID
  manage_ppp_rl_run.sh logs RUN_ID [--lines COUNT] [--follow]

`start` detaches the paid launcher from the current SSH session. Lifecycle
files and the combined log are written below:
  simplified/results/training/launches/RUN_ID/
EOF
}

atomic_write() {
  local path="$1"
  local value="$2"
  local temporary="${path}.tmp.$$"
  printf '%s\n' "$value" > "$temporary"
  mv "$temporary" "$path"
}

validate_run_id() {
  local run_id="$1"
  if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_ID must contain only letters, digits, dots, underscores, or hyphens." >&2
    return 2
  fi
}

state_dir_for() {
  printf '%s/%s\n' "$launch_root" "$1"
}

read_state() {
  local path="$1"
  local fallback="${2:-unknown}"
  if [[ -f "$path" ]]; then
    tr -d '\r\n' < "$path"
  else
    printf '%s' "$fallback"
  fi
}

finish_worker() {
  local exit_code=$?
  trap - EXIT
  atomic_write "${worker_state_dir}/exit-code" "$exit_code"
  atomic_write "${worker_state_dir}/finished-at-utc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$exit_code" -eq 0 ]]; then
    atomic_write "${worker_state_dir}/status" "succeeded"
  else
    atomic_write "${worker_state_dir}/status" "failed"
  fi
  if [[ "$(read_state "${worker_state_dir}/cleanup-status" "pending")" == "pending" ]]; then
    atomic_write "${worker_state_dir}/cleanup-status" "not-observed"
  fi
}

run_worker() {
  worker_state_dir="$1"
  shift
  export PPP_LIFECYCLE_DIR="$worker_state_dir"
  trap finish_worker EXIT
  atomic_write "${worker_state_dir}/status" "running"
  atomic_write "${worker_state_dir}/worker-started-at-utc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cd "$repo_root"
  bash "$train_runner" --execute "$@"
}

start_run() {
  local run_id="$1"
  shift
  validate_run_id "$run_id"

  local steps=""
  local simulator=""
  local model=""
  local projected_compute_usd=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
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
      *)
        echo "Unsupported start argument: $1" >&2
        return 2
        ;;
    esac
  done
  if [[ -z "$steps" || -z "$simulator" || -z "$model" ]]; then
    echo "start requires explicit --steps, --simulator, and --model values." >&2
    return 2
  fi
  if [[ "$steps" != "1" && "$steps" != "2" && "$steps" != "20" && "$steps" != "40" ]]; then
    echo "--steps must be 1, 2, 20, or 40." >&2
    return 2
  fi
  if [[ "$simulator" != "deterministic" && "$simulator" != "gemini" ]]; then
    echo "--simulator must be deterministic or gemini." >&2
    return 2
  fi
  if [[ "$model" != "qwen35" && "$model" != "qwen3" ]]; then
    echo "--model must be qwen35 or qwen3." >&2
    return 2
  fi
  if [[ "${CONFIRM_PAID_TRAINING:-}" != "I_UNDERSTAND_LAMBDA_IS_BILLING" ]]; then
    echo "Paid execution guard not confirmed." >&2
    return 2
  fi
  if [[ "$simulator" == "gemini" && -z "${GEMINI_API_KEY:-}" ]]; then
    echo "GEMINI_API_KEY is required for a live simulator run." >&2
    return 2
  fi
  if [[ ! -f "$train_runner" ]]; then
    echo "Training runner does not exist: $train_runner" >&2
    return 2
  fi

  local state_dir
  state_dir="$(state_dir_for "$run_id")"
  mkdir -p "$launch_root"
  if ! mkdir "$state_dir"; then
    echo "Run state already exists; choose a new RUN_ID: $state_dir" >&2
    return 2
  fi

  local runner_args=(
    --steps "$steps"
    --simulator "$simulator"
    --model "$model"
  )
  if [[ -n "$projected_compute_usd" ]]; then
    runner_args+=(--projected-compute-usd "$projected_compute_usd")
  fi

  atomic_write "${state_dir}/status" "starting"
  atomic_write "${state_dir}/started-at-utc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  atomic_write "${state_dir}/cleanup-status" "pending"
  {
    printf '%q ' "$train_runner" --execute "${runner_args[@]}"
    printf '\n'
  } > "${state_dir}/command.txt"

  nohup "$script_path" _worker "$state_dir" "${runner_args[@]}" \
    </dev/null >>"${state_dir}/run.log" 2>&1 &
  local worker_pid=$!
  atomic_write "${state_dir}/pid" "$worker_pid"

  printf 'run_id=%s\n' "$run_id"
  printf 'pid=%s\n' "$worker_pid"
  printf 'state_dir=%s\n' "$state_dir"
  printf 'status_command=%q status %q\n' "$script_path" "$run_id"
  printf 'logs_command=%q logs %q --follow\n' "$script_path" "$run_id"
}

show_status() {
  local run_id="$1"
  validate_run_id "$run_id"
  local state_dir
  state_dir="$(state_dir_for "$run_id")"
  if [[ ! -d "$state_dir" ]]; then
    echo "Unknown run: $run_id" >&2
    return 2
  fi

  local recorded_status pid observed_status
  recorded_status="$(read_state "${state_dir}/status")"
  pid="$(read_state "${state_dir}/pid")"
  observed_status="$recorded_status"
  if [[ "$recorded_status" == "starting" || "$recorded_status" == "running" ]]; then
    if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
      observed_status="orphaned"
    fi
  fi

  printf 'run_id=%s\n' "$run_id"
  printf 'status=%s\n' "$observed_status"
  printf 'recorded_status=%s\n' "$recorded_status"
  printf 'pid=%s\n' "$pid"
  printf 'started_at_utc=%s\n' "$(read_state "${state_dir}/started-at-utc")"
  printf 'finished_at_utc=%s\n' "$(read_state "${state_dir}/finished-at-utc")"
  printf 'exit_code=%s\n' "$(read_state "${state_dir}/exit-code")"
  printf 'cleanup_status=%s\n' "$(read_state "${state_dir}/cleanup-status")"
  printf 'gpu_processes_after_cleanup=%s\n' \
    "$(read_state "${state_dir}/gpu-processes-after-cleanup")"
  printf 'log=%s\n' "${state_dir}/run.log"

  if [[ "$observed_status" == "failed" || "$observed_status" == "orphaned" ]]; then
    return 1
  fi
}

show_logs() {
  local run_id="$1"
  shift
  validate_run_id "$run_id"
  local lines=80
  local follow="false"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --lines)
        lines="${2:?--lines requires a value}"
        shift 2
        ;;
      --follow)
        follow="true"
        shift
        ;;
      *)
        echo "Unsupported logs argument: $1" >&2
        return 2
        ;;
    esac
  done
  if [[ ! "$lines" =~ ^[1-9][0-9]*$ ]]; then
    echo "--lines must be a positive integer." >&2
    return 2
  fi
  local state_dir log_path
  state_dir="$(state_dir_for "$run_id")"
  log_path="${state_dir}/run.log"
  if [[ ! -f "$log_path" ]]; then
    echo "Run log does not exist: $log_path" >&2
    return 2
  fi
  if [[ "$follow" == "true" ]]; then
    exec tail -n "$lines" -f "$log_path"
  fi
  tail -n "$lines" "$log_path"
}

command="${1:-}"
case "$command" in
  start)
    if [[ $# -lt 2 ]]; then
      usage >&2
      exit 2
    fi
    shift
    start_run "$@"
    ;;
  status)
    if [[ $# -ne 2 ]]; then
      usage >&2
      exit 2
    fi
    show_status "$2"
    ;;
  logs)
    if [[ $# -lt 2 ]]; then
      usage >&2
      exit 2
    fi
    shift
    show_logs "$@"
    ;;
  _worker)
    if [[ $# -lt 2 ]]; then
      exit 2
    fi
    shift
    run_worker "$@"
    ;;
  --help|-h|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
