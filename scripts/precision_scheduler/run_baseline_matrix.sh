#!/usr/bin/env bash
# Baseline matrix: {Qwen3.5-9B, Phi-4-mini-reasoning} x {bf16, w4a16, nvfp4}, one GPU, sequential.
#
#   source ~/rl_env.sh && bash run_baseline_matrix.sh            # all six
#   DRY_RUN=1 bash run_baseline_matrix.sh                        # print overrides, start nothing
#   CONFIGS="qwen3_5_9b:bf16" bash run_baseline_matrix.sh        # one cell
#
# 32 concurrent requests (TRAIN_BATCH_SIZE 8 x ROLLOUT_N 4), responses capped at 16384 tokens,
# 16 GRPO steps per cell: step 1 is warm-up and steps 2-16 are the measurement window.
#
# Every cell runs the *vanilla* punica LoRA path. ps_resolve_policy turns the fused fast path and
# the dual stream on for W4 kinds (matching the archived launchers), so both are forced back off
# here; otherwise the W4 cells would measure a different LoRA kernel than the BF16 cell.
set -uo pipefail

VERL_PS=${VERL_PS:-$HOME/verl-ps}
VLLM_PS=${VLLM_PS:-$HOME/vllm-ps}
DATA_DIR=${DATA_DIR:-$HOME/ps_data/gsm8k_messages_2048}
RUN_ROOT=${RUN_ROOT:-$HOME/ps_runs/baseline_matrix}
# Ray's plasma socket is an AF_UNIX path capped at 107 bytes, and the default
# ${RUN_DIR}/ray_tmp plus Ray's own session_<timestamp>/sockets/... blows past it.
# Keep the root short and per-cell.
RAY_ROOT=${RAY_ROOT:-$HOME/rt}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$VLLM_PS:$VERL_PS${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# The switch / bind / reload contract lines are INFO; verl defaults to WARN (integration defect 3).
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

QWEN_NVFP4=${QWEN_NVFP4:-AxionML/Qwen3.5-9B-NVFP4}
PHI_NVFP4=${PHI_NVFP4:-$HOME/models/Phi-4-mini-reasoning-NVFP4}

# cell = <model key>:<precision>; the overlay supplies bf16 and w4a16 checkpoints, nvfp4 is ours.
CONFIGS=${CONFIGS:-"qwen3_5_9b:bf16 qwen3_5_9b:w4a16 qwen3_5_9b:nvfp4 phi4_mini_reasoning:bf16 phi4_mini_reasoning:w4a16 phi4_mini_reasoning:nvfp4"}

mkdir -p "$RUN_ROOT"

for cell in $CONFIGS; do
    model_key=${cell%%:*}
    precision=${cell##*:}

    case "$precision" in
        bf16)  policy=bf16;    int4_path="" ;;
        w4a16) policy=full_w4; int4_path="" ;;              # overlay default (GPTQ / AutoRound)
        nvfp4) policy=full_w4
               case "$model_key" in
                   qwen3_5_9b)          int4_path="$QWEN_NVFP4" ;;
                   phi4_mini_reasoning) int4_path="$PHI_NVFP4" ;;
                   *) echo "no NVFP4 checkpoint for $model_key" >&2; exit 2 ;;
               esac ;;
        *) echo "unknown precision $precision" >&2; exit 2 ;;
    esac

    run_dir="$RUN_ROOT/${model_key}_${precision}"
    if [ -f "$run_dir/COMPLETE" ]; then
        echo "[matrix] $cell already COMPLETE, skipping"
        continue
    fi
    echo "[matrix] === $cell -> $run_dir ==="

    # Plain assignments, not ${x:+VAR=...}: bash treats a conditional expansion in a
    # command prefix as a command word, not an assignment. An empty INT4_MODEL_PATH is
    # what common.sh already reads as "use the overlay's checkpoint".
    ray_tmp="$RAY_ROOT/$(echo "${model_key}_${precision}" | tr -d 'aeiou_')"
    rm -rf "$ray_tmp"; mkdir -p "$ray_tmp"

    RUN_DIR="$run_dir" \
    RAY_TMPDIR="$ray_tmp" \
    MODEL_KEY="$model_key" \
    POLICY="$policy" \
    INT4_MODEL_PATH="$int4_path" \
    EXPERIMENT_NAME="${model_key}_${precision}" \
    DATA_DIR="$DATA_DIR" \
    TRAIN_BATCH_SIZE=8 \
    ROLLOUT_N=4 \
    RESPONSE_CAP=16384 \
    TOTAL_STEPS=16 \
    SAVE_FREQ=-1 \
    PROJECT_NAME=baseline_matrix \
    DRY_RUN="${DRY_RUN:-}" \
    bash "$VERL_PS/examples/precision_scheduler/recipes/full_step.sh" \
        actor_rollout_ref.rollout.precision_scheduler.lora_fast_path=false \
        actor_rollout_ref.rollout.precision_scheduler.lora_dual_stream=false \
        "$@"
    status=$?
    echo "[matrix] $cell exit=$status"
done
echo "[matrix] all requested cells finished"
