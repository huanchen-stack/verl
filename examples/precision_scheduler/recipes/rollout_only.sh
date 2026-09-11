#!/usr/bin/env bash
# recipes/rollout_only.sh -- generation + reward only (no log-prob / update), N steps, one GPU.
#
# Collapses the archived rollout-only wrappers (dynamic_tail8k run_dynamic.sh, hardmath
# run_rollout_calibration.sh, rl-workflow run_best_t8_verl_rollout_gpu7.sh): the same defaults
# (initial batch B = TRAIN_BATCH_SIZE * ROLLOUT_N, max_num_seqs = B, 8192 batched tokens, GMEM 0.50,
# LoRA 16/16, seed 42, no rollout log-probs) with the policy selected by POLICY.
#
# Usage:  RUN_DIR=... MODEL_KEY=qwen3_5_4b POLICY=tail_t8 INITIAL_BATCH=64 RESPONSE_CAP=24576 TOTAL_STEPS=2 \
#           recipes/rollout_only.sh [extra hydra overrides...]
# INITIAL_BATCH (default 64) sets TRAIN_BATCH_SIZE = INITIAL_BATCH / ROLLOUT_N unless TRAIN_BATCH_SIZE is given.
# Validate afterwards: tools/validate_rollout_run.py RUN_DIR --expected-requests B --steps N
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$(( ${INITIAL_BATCH:-64} / ROLLOUT_N ))}"
export RESPONSE_CAP="${RESPONSE_CAP:-24576}"
export TOTAL_STEPS="${TOTAL_STEPS:-1}"
export CALCULATE_LOG_PROBS=False
export SAVE_FREQ=-1
export PROJECT_NAME="${PROJECT_NAME:-rollout_only}"
exec bash "${here}/../run_fsdp_fullstep.sh" \
  trainer.rollout_only=true "trainer.rollout_only_steps=${TOTAL_STEPS}" "$@"
