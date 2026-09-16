#!/usr/bin/env bash
# long_run/train_arm.sh -- one arm of the 100-step protocol (docs/precision_scheduler/recipes_and_evaluation.md).
#
# From the archived hardmath run_training_config.sh and the no_reprefill run_job(): a full-step run
# with periodic checkpoints, an optional shared step-0 LoRA (every arm starts from the same adapter),
# resume from the latest global_step_* on retry, and a bounded attempt loop with an events.log.
#
# Usage:  RUN_DIR=... MODEL_KEY=qwen3_5_9b POLICY=full_w4 DATA_DIR=<bigmath splits> TOTAL_STEPS=100 SAVE_FREQ=10 \
#           [INITIAL_LORA_ADAPTER_PATH=<peft adapter dir>] [MAX_ATTEMPTS=3] long_run/train_arm.sh [overrides...]
# Export the shared step-0 adapter of a protocol with
#   SAVE_INITIAL_CHECKPOINT=1 EXIT_AFTER_INITIAL_CHECKPOINT=1 long_run/train_arm.sh
# (prints VERL_INITIAL_CHECKPOINT_COMPLETE step=0 and leaves checkpoints/global_step_0), then convert it
# with long_run/evaluate_lora_patch.py export-adapter and pass it as INITIAL_LORA_ADAPTER_PATH to every arm.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${RUN_DIR:?set RUN_DIR}"
export TOTAL_STEPS="${TOTAL_STEPS:-100}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export RESPONSE_CAP="${RESPONSE_CAP:-24576}"
export MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-2}"
export CALCULATE_LOG_PROBS="${CALCULATE_LOG_PROBS:-True}"
export PROJECT_NAME="${PROJECT_NAME:-long_run}"
export RUN_TIMEOUT="${RUN_TIMEOUT:-96h}"
max_attempts="${MAX_ATTEMPTS:-3}"
overrides=("trainer.rollout_only=false" "trainer.resume_mode=auto" "actor_rollout_ref.actor.old_log_prob_calculate_entropy=false")
if [[ -n "${INITIAL_LORA_ADAPTER_PATH:-}" ]]; then
  [[ -f "${INITIAL_LORA_ADAPTER_PATH}/adapter_config.json" ]] || { echo "not a PEFT adapter dir: ${INITIAL_LORA_ADAPTER_PATH}" >&2; exit 2; }
  # FSDP reads model.lora_adapter_path, Megatron-Bridge PEFT reads model.lora.adapter_path.
  if [[ "${TRAINER:-megatron}" == "megatron" ]]; then
    overrides+=("actor_rollout_ref.model.lora.adapter_path=${INITIAL_LORA_ADAPTER_PATH}")
  else
    overrides+=("actor_rollout_ref.model.lora_adapter_path=${INITIAL_LORA_ADAPTER_PATH}")
  fi
fi
if [[ "${SAVE_INITIAL_CHECKPOINT:-0}" == "1" ]]; then overrides+=("trainer.save_initial_checkpoint=true"); fi
if [[ "${EXIT_AFTER_INITIAL_CHECKPOINT:-0}" == "1" ]]; then overrides+=("trainer.exit_after_initial_checkpoint=true"); fi
overrides+=("$@")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exec bash "${here}/../run_fullstep.sh" "${overrides[@]}"
fi
mkdir -p "${RUN_DIR}"
for attempt in $(seq 1 "${max_attempts}"); do
  latest=""
  if compgen -G "${RUN_DIR}/checkpoints/global_step_*" >/dev/null; then
    latest="$(find "${RUN_DIR}/checkpoints" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sort -V | tail -1)"
  fi
  printf 'START run=%s policy=%s steps=%s attempt=%s resume=%s %s\n' "${RUN_DIR}" "${POLICY:-bf16}" "${TOTAL_STEPS}" \
    "${attempt}" "${latest:-none}" "$(date -u +%FT%TZ)" | tee -a "${RUN_DIR}/events.log"
  rc=0
  ALLOW_RESUME=1 bash "${here}/../run_fullstep.sh" "${overrides[@]}" || rc=$?
  printf 'END rc=%s attempt=%s %s\n' "${rc}" "${attempt}" "$(date -u +%FT%TZ)" | tee -a "${RUN_DIR}/events.log"
  if [[ ${rc} -eq 0 ]]; then exit 0; fi
  if [[ ${rc} -eq 2 || ${rc} -eq 124 ]]; then exit "${rc}"; fi   # usage error / timeout: do not retry
done
echo "train_arm: giving up after ${max_attempts} attempts" >&2
exit 1
