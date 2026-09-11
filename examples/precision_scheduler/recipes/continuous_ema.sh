#!/usr/bin/env bash
# recipes/continuous_ema.sh -- online EMA policy: a watcher sidecar rebuilds the lookup-table policy
# after every completed rollout while the runner (rollout-only or full step) reloads it each rollout.
#
# Collapses the four archived copies of the pattern (dynamic_tail8k run_continuous_ema30.sh,
# full_rl_policy_matrix run_ema_full_rl.sh, hardmath run_dynamic_ema_training.sh, b32_16k_sensitivity
# run_ema_fullstep_lane.sh): watcher --initialize-only (writes policy revision 0) -> watcher in the
# background -> runner in its own process group with precision_scheduler.reload_policy_each_rollout=true
# -> fail-closed poll: if the watcher dies before online_ema_state.json reports completed_steps >= steps,
# the runner is killed and the recipe exits 1 (the rollout must never continue on a stale policy).
#
# Usage:  RUN_DIR=... MODEL_KEY=... INITIAL_BATCH=64 RESPONSE_CAP=24576 TOTAL_STEPS=30 \
#           BF_TRACE=<bf16 baseline trace> W4_TRACE=<w4 baseline trace> HEATMAP=<heatmap.json> \
#           [DOWNSTREAM_SLOPE=0] [EMA_ALPHA=0.2] [RUNNER=rollout_only|full_step] recipes/continuous_ema.sh [overrides...]
# The watcher is the C6 CLI `python -m verl.experimental.precision_scheduler.cli watch-ema`; WATCHER_CMD
# overrides the whole command (tests use a stub), RUNNER_CMD overrides the runner command.
# DRY_RUN=1 prints the watcher command line and the runner's override list without launching anything.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${here}/../common.sh"

: "${RUN_DIR:?set RUN_DIR}"
steps="${TOTAL_STEPS:-30}"
batch="${INITIAL_BATCH:-64}"
cap="${RESPONSE_CAP:-24576}"
runner="${RUNNER:-rollout_only}"
policy_path="${POLICY_PATH:-${RUN_DIR}/policy.json}"
ps="actor_rollout_ref.rollout.precision_scheduler"

if [[ -n "${WATCHER_CMD:-}" ]]; then
  read -r -a watcher <<<"${WATCHER_CMD}"
else
  : "${BF_TRACE:?set BF_TRACE (pure-BF16 baseline request-lifetime trace)}"
  : "${W4_TRACE:?set W4_TRACE (pure-W4 baseline request-lifetime trace)}"
  : "${HEATMAP:?set HEATMAP (profiler heatmap.json)}"
  watcher=("${PS_PYTHON}" -m verl.experimental.precision_scheduler.cli watch-ema
    --batch "${batch}" --cap "${cap}" --run-dir "${RUN_DIR}" --policy "${policy_path}" --steps "${steps}"
    --bf-trace "${BF_TRACE}" --w4-trace "${W4_TRACE}" --heatmap "${HEATMAP}"
    --alpha "${EMA_ALPHA:-0.2}" --downstream-slope "${DOWNSTREAM_SLOPE:-0}")
  if [[ -n "${WATCHER_EXTRA_ARGS:-}" ]]; then read -r -a extra <<<"${WATCHER_EXTRA_ARGS}"; watcher+=("${extra[@]}"); fi
fi

runner_env=(RUN_DIR="${RUN_DIR}" INITIAL_BATCH="${batch}" RESPONSE_CAP="${cap}" TOTAL_STEPS="${steps}"
  POLICY="${policy_path}" PROJECT_NAME="${PROJECT_NAME:-continuous_ema}")
runner_overrides=("${ps}.reload_policy_each_rollout=true" "$@")
if [[ -n "${RUNNER_CMD:-}" ]]; then
  read -r -a runner_cmd <<<"${RUNNER_CMD}"
else
  case "${runner}" in
    rollout_only|full_step) runner_cmd=(bash "${here}/${runner}.sh") ;;
    *) ps_die "unknown RUNNER '${runner}' (rollout_only | full_step)" ;;
  esac
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'WATCHER\t%s\n' "${watcher[*]}"
  # The runner's dry run needs the policy file to exist (POLICY=<file>.json is validated).
  mkdir -p "$(dirname "${policy_path}")"; [[ -e "${policy_path}" ]] || echo '{}' >"${policy_path}"
  env "${runner_env[@]}" DRY_RUN=1 "${runner_cmd[@]}" "${runner_overrides[@]}"
  exit 0
fi

mkdir -p "${RUN_DIR}/traces" "${RUN_DIR}/logs"
"${watcher[@]}" --initialize-only
[[ -f "${policy_path}" ]] || ps_die "watcher --initialize-only did not write ${policy_path}"

"${watcher[@]}" >"${RUN_DIR}/logs/online_ema_watcher.log" 2>&1 &
watcher_pid=$!
setsid env "${runner_env[@]}" "${runner_cmd[@]}" "${runner_overrides[@]}" &
runner_pid=$!

completed_steps() {
  local state="${RUN_DIR}/online_ema_state.json"
  [[ -f "${state}" ]] || { echo 0; return; }
  "${PS_PYTHON}" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("completed_steps", 0)))' "${state}" 2>/dev/null || echo 0
}
kill_runner() {
  kill -INT -- "-${runner_pid}" 2>/dev/null || true
  for _ in 1 2 3 4 5; do kill -0 "${runner_pid}" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-${runner_pid}" 2>/dev/null || true
  wait "${runner_pid}" 2>/dev/null || true
}
cleanup() {
  if kill -0 "${watcher_pid}" 2>/dev/null; then kill "${watcher_pid}" 2>/dev/null || true; wait "${watcher_pid}" 2>/dev/null || true; fi
  if kill -0 "${runner_pid}" 2>/dev/null; then kill_runner; fi
}
trap cleanup EXIT

while kill -0 "${runner_pid}" 2>/dev/null; do
  if ! kill -0 "${watcher_pid}" 2>/dev/null; then
    completed="$(completed_steps)"
    if (( completed < steps )); then
      echo "FATAL: EMA watcher exited at revision ${completed}/${steps}; terminating the runner" >&2
      kill_runner
      printf 'watcher_exited_early completed=%s steps=%s\n' "${completed}" "${steps}" >"${RUN_DIR}/FAILED"
      exit 1
    fi
  fi
  sleep "${POLL_SECONDS:-2}"
done
wait "${runner_pid}"; runner_rc=$?
wait "${watcher_pid}" 2>/dev/null || true
trap - EXIT
completed="$(completed_steps)"
if (( runner_rc != 0 )); then
  echo "runner exited with rc=${runner_rc}" >&2
  exit "${runner_rc}"
fi
if (( completed < steps )); then
  echo "FATAL: runner finished but the watcher consumed only ${completed}/${steps} steps" >&2
  printf 'watcher_incomplete completed=%s steps=%s\n' "${completed}" "${steps}" >"${RUN_DIR}/FAILED"
  exit 1
fi
touch "${RUN_DIR}/CONTINUOUS_EMA_COMPLETE"
