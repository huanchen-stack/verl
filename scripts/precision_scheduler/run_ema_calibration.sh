#!/usr/bin/env bash
# Calibration rollouts for the online EMA policy: one batch of 128 distinct requests
# per precision, no GRPO groups.
#
#   source ~/rl_env.sh && bash run_ema_calibration.sh
#
# The watcher (`cli watch-ema`) prices plans from a pure-BF16 and a pure-W4 baseline
# request-lifetime trace. Those traces must describe single requests, so this runs
# ROLLOUT_N=1 with TRAIN_BATCH_SIZE=128: 128 different prompts, one sample each,
# rather than 32 prompts x 4 samples. A grouped trace would make the survival
# distribution reflect 4 correlated samples of the same prompt.
#
# Everything runs on the fused LoRA fast path, the same kernels the scheduled
# rollout will use.
set -uo pipefail

VERL_PS=${VERL_PS:-$HOME/verl-ps}
VLLM_PS=${VLLM_PS:-$HOME/vllm-ps}
DATA_DIR=${DATA_DIR:-$HOME/ps_data/gsm8k_messages_2048}
RUN_ROOT=${RUN_ROOT:-$HOME/ps_runs/ema_calibration}
RAY_ROOT=${RAY_ROOT:-$HOME/rt}

MODEL_KEY=${MODEL_KEY:-phi4_mini_reasoning}
PHI_NVFP4=${PHI_NVFP4:-$HOME/models/Phi-4-mini-reasoning-NVFP4}
CONFIGS=${CONFIGS:-"bf16 w4a16 nvfp4"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$VLLM_PS:$VERL_PS${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

mkdir -p "$RUN_ROOT"

for precision in $CONFIGS; do
    case "$precision" in
        bf16)  policy=bf16;    int4_path="" ;;
        w4a16) policy=full_w4; int4_path="" ;;               # overlay default
        nvfp4) policy=full_w4; int4_path="$PHI_NVFP4" ;;
        *) echo "unknown precision $precision" >&2; exit 2 ;;
    esac

    run_dir="$RUN_ROOT/${MODEL_KEY}_${precision}"
    [ -f "$run_dir/COMPLETE" ] && { echo "[calib] $precision already COMPLETE"; continue; }
    ray_tmp="$RAY_ROOT/cal$(echo "$precision" | tr -d 'aeiou_')"
    rm -rf "$ray_tmp"; mkdir -p "$ray_tmp"
    echo "[calib] === $precision -> $run_dir ==="

    # POLICY=bf16 leaves the LoRA knobs at their defaults, so the fast path is set
    # explicitly here; the W4 kinds get it from ps_resolve_policy but are pinned
    # anyway so all three calibrations share one kernel path.
    RUN_DIR="$run_dir" \
    RAY_TMPDIR="$ray_tmp" \
    MODEL_KEY="$MODEL_KEY" \
    POLICY="$policy" \
    INT4_MODEL_PATH="$int4_path" \
    EXPERIMENT_NAME="calib_${MODEL_KEY}_${precision}" \
    DATA_DIR="$DATA_DIR" \
    TRAIN_BATCH_SIZE=128 \
    ROLLOUT_N=1 \
    RESPONSE_CAP=16384 \
    TOTAL_STEPS=1 \
    SAVE_FREQ=-1 \
    PROJECT_NAME=ema_calibration \
    bash "$VERL_PS/examples/precision_scheduler/recipes/rollout_only.sh" \
        actor_rollout_ref.rollout.precision_scheduler.lora_fast_path=true \
        actor_rollout_ref.rollout.precision_scheduler.lora_dual_stream=true \
        "$@"
    echo "[calib] $precision exit=$?"
done
echo "[calib] done"
