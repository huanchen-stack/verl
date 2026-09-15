#!/usr/bin/env bash
# Qwen3.5-9B decode-TPOT heatmap over three precisions on this box (1x RTX PRO 6000, sm_120).
#
#   source ~/rl_env.sh && bash ~/run_tpot_heatmap_3precision.sh
#
# 8 grid cells (batch 1,8 x context 1024,4096,8192,16384) measured for each of
# bf16, int4 (W4A16) and nvfp4. Every row is vanilla vLLM launched directly on that
# precision's own checkpoint (--standalone-base-precision), so the three rows are
# symmetric; the archived protocol instead routed INT4 through the dual-precision
# runtime, which has no FP4 path and so could not carry the third row.
#
# Resumable: rerun the same command and completed cells are skipped.
set -xeuo pipefail

VLLM_PS=${VLLM_PS:-$HOME/vllm-ps}
VERL_PS=${VERL_PS:-$HOME/verl-ps}
OUT=${OUT:-$HOME/experiments/qwen35_9b_3precision}

BF16_MODEL=${BF16_MODEL:-Qwen/Qwen3.5-9B}
INT4_MODEL=${INT4_MODEL:-sanskar003/Qwen3.5-9B-AWQ}
NVFP4_MODEL=${NVFP4_MODEL:-AxionML/Qwen3.5-9B-NVFP4}

BATCH_SIZES=${BATCH_SIZES:-1,8}
SEQ_LENS=${SEQ_LENS:-1024,4096,8192,16384}
PRECISIONS=${PRECISIONS:-bf16,int4,nvfp4}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$VLLM_PS:$VERL_PS${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT"

python "$VLLM_PS/tools/precision_scheduler/tpot_heatmap.py" \
    --model "$BF16_MODEL" \
    --int4-model "$INT4_MODEL" \
    --nvfp4-model "$NVFP4_MODEL" \
    --standalone-base-precision \
    --precisions "$PRECISIONS" \
    --batch-sizes "$BATCH_SIZES" \
    --seq-lens "$SEQ_LENS" \
    --warmup-steps 2 \
    --measurement-steps 9 \
    --measurement-repetitions 1 \
    --initial-precision-warmup-steps 9 \
    --gpu-memory-utilization 0.5 \
    --tensor-parallel-size 1 \
    --output-dir "$OUT" \
    "$@"
