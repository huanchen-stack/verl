#!/usr/bin/env bash
# recipes/full_step.sh -- full GRPO steps (rollout + old log-prob + ref + update + weight sync), one GPU.
#
# Collapses the archived full-RL wrappers (full_rl_policy_matrix run_full_rl_policy.sh defaults:
# 30 steps, rollout log-probs on, entropy off, KL loss on; hardmath run_training_config.sh; the
# bf16_learnability run_config.sh LR knobs). The policy is selected by POLICY, the model by MODEL_KEY.
#
# Usage:  RUN_DIR=... MODEL_KEY=qwen3_5_4b POLICY=full_w4 TRAIN_BATCH_SIZE=16 RESPONSE_CAP=24576 TOTAL_STEPS=30 \
#           recipes/full_step.sh [extra hydra overrides...]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TOTAL_STEPS="${TOTAL_STEPS:-30}"
export RESPONSE_CAP="${RESPONSE_CAP:-24576}"
export CALCULATE_LOG_PROBS="${CALCULATE_LOG_PROBS:-True}"
export PROJECT_NAME="${PROJECT_NAME:-full_step}"
exec bash "${here}/../run_fullstep.sh" trainer.rollout_only=false "$@"
