#!/usr/bin/env bash
# recipes/train_replay.sh -- rollout-free RL steps for timing the trainer: recorded responses are replayed
# through old-log-prob / ref / update (ReplayAgentLoop), the vLLM engine only takes part in the weight sync.
# Used to compare multi-GPU training layouts on identical token batches and to fit the downstream-time
# linear model (time vs tokens) of each layout.
#
# Usage:  RUN_DIR=... MODEL_KEY=qwen3_5_9b REPLAY_FILE=replay.json LAYOUT=megatron_tp|megatron_dp|fsdp_dp \
#           recipes/train_replay.sh [extra hydra overrides...]
# Knobs: N_GPUS (default: count of CUDA_VISIBLE_DEVICES), TP_SIZE (megatron_tp only; default N_GPUS, e.g. 2 on
#   four GPUs gives TP2 x DP2), TRAIN_BATCH_SIZE (default 8 * N_GPUS prompts, i.e. a
#   B32 rollout per GPU at ROLLOUT_N=4), BALANCE_BATCH (default True), TOTAL_STEPS (default 12),
#   RESPONSE_CAP (24576), PPO_MAX_TOKEN_LEN (26624), plus every run_fullstep.sh knob.
# The replay file comes from analysis/build_replay.py; keys are idx-<dataset index>_<n>, so the dataset,
# data.shuffle=False and trainer.stable_sample_uid=true (both set by the drivers) must match the recorded run.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${RUN_DIR:?set RUN_DIR}"
: "${REPLAY_FILE:?set REPLAY_FILE (analysis/build_replay.py output)}"
[[ -f "${REPLAY_FILE}" ]] || { echo "train_replay.sh: replay file not found: ${REPLAY_FILE}" >&2; exit 2; }
layout="${LAYOUT:-megatron_dp}"
visible="${CUDA_VISIBLE_DEVICES:-0}"
n_gpus="${N_GPUS:-$(( $(tr -cd ',' <<<"${visible}" | wc -c) + 1 ))}"
export POLICY=bf16
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((8 * n_gpus))}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export RESPONSE_CAP="${RESPONSE_CAP:-24576}"
export PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-26624}"
export TOTAL_STEPS="${TOTAL_STEPS:-12}"
export CALCULATE_LOG_PROBS=False
export PROJECT_NAME="${PROJECT_NAME:-train_replay}"
tp_size="${TP_SIZE:-${n_gpus}}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_KEY:-qwen3_5_4b}_${layout}_g${n_gpus}}"

case "${layout}" in
  megatron_tp)
    export TRAINER=megatron
    (( n_gpus % tp_size == 0 )) || { echo "train_replay.sh: TP_SIZE ${tp_size} must divide N_GPUS ${n_gpus}" >&2; exit 2; }
    layout_overrides=(
      "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${tp_size}"
      "actor_rollout_ref.rollout.tensor_model_parallel_size=${tp_size}"
    ) ;;
  megatron_dp)
    export TRAINER=megatron
    layout_overrides=(
      actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
      actor_rollout_ref.rollout.tensor_model_parallel_size=1
    ) ;;
  fsdp_dp)
    export TRAINER=fsdp2
    layout_overrides=(
      "actor_rollout_ref.actor.fsdp_config.fsdp_size=${n_gpus}"
      "actor_rollout_ref.ref.fsdp_config.fsdp_size=${n_gpus}"
      actor_rollout_ref.rollout.tensor_model_parallel_size=1
    ) ;;
  *) echo "train_replay.sh: LAYOUT must be megatron_tp | megatron_dp | fsdp_dp (got '${layout}')" >&2; exit 2 ;;
esac

# The replay loop is registered through the agent-loop config path (Hydra instantiate kwargs carry the file).
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  mkdir -p "${RUN_DIR}"
  cat >"${RUN_DIR}/replay_agent.yaml" <<YAML
- name: replay_agent
  _target_: verl.experimental.agent_loop.replay_agent_loop.ReplayAgentLoop
  replay_file: ${REPLAY_FILE}
YAML
fi

exec bash "${here}/../run_fullstep.sh" trainer.rollout_only=false \
  "trainer.n_gpus_per_node=${n_gpus}" \
  "trainer.balance_batch=${BALANCE_BATCH:-True}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=replay_agent \
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=${RUN_DIR}/replay_agent.yaml" \
  "${layout_overrides[@]}" "$@"
