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

# Protocol knobs.
#   STANDALONE=1 : every row is a vanilla engine on its own checkpoint (no LoRA).
#   STANDALONE=0 : quantized rows run inside the dual-precision runtime against a
#                  shadow, which is how a real rollout decodes.
#   FAST_PATH=1  : fused rollout-LoRA path + dual stream, matching ps_resolve_policy
#                  and the archived launchers. Needs ADAPTER; the cost model that
#                  prices a scheduled rollout has to be measured on the same kernels
#                  the rollout will use.
STANDALONE=${STANDALONE:-1}
FAST_PATH=${FAST_PATH:-0}
ADAPTER=${ADAPTER:-}

#   BF16_LAYERS  : layers the dual-precision runtime keeps in BF16 under the shadow. vLLM's
#                  own default is first:3,last:3, but ps_resolve_policy sets none for every
#                  RL run, so the cost model must be measured with none too (6 of 32 layers
#                  left in BF16 understated the W4 speedup on both models on 2026-09-17).
BF16_LAYERS=${BF16_LAYERS:-none}

EXTRA_ARGS=()
if [ "$STANDALONE" = 1 ]; then
    EXTRA_ARGS+=(--standalone-base-precision)
else
    export VLLM_DUAL_PRECISION_BF16_LAYERS="$BF16_LAYERS"
fi
if [ -n "$ADAPTER" ]; then
    EXTRA_ARGS+=(--adapter "$ADAPTER" --max-lora-rank "${LORA_RANK:-16}")
    # Restrict LoRA to the overlay's targets so the embedding is not wrapped;
    # unrestricted, its vanilla punica path breaks torch.compile.
    EXTRA_ARGS+=(--lora-target-modules "${LORA_TARGETS:-qkv_proj,o_proj,gate_up_proj,down_proj}")
fi
if [ "$FAST_PATH" = 1 ]; then
    [ -n "$ADAPTER" ] || { echo "FAST_PATH=1 needs ADAPTER (tools/rollout_lora/make_zero_lora.py)" >&2; exit 2; }
    export ROLLOUT_QLORA=1
    export VLLM_LORA_ENABLE_DUAL_STREAM=1
    export VLLM_ROLLOUT_LORA_FUSE_PACKED=${VLLM_ROLLOUT_LORA_FUSE_PACKED:-1}
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$VLLM_PS:$VERL_PS${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT"

python "$VLLM_PS/tools/precision_scheduler/tpot_heatmap.py" \
    --model "$BF16_MODEL" \
    --int4-model "$INT4_MODEL" \
    --nvfp4-model "$NVFP4_MODEL" \
    "${EXTRA_ARGS[@]}" \
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
