#!/usr/bin/env bash
# Online-EMA precision scheduler on Phi-4-mini-reasoning, W4A16 then NVFP4, 20 full
# GRPO steps each, with the halt rule watching every run.
#
#   source ~/rl_env.sh && bash run_phi_ema_pair.sh
#
# Trainer is the branch default, Megatron TP1 (decision 10 reversed 2026-09-16); every
# FSDP2 number from this host is deprecated. 32 requests as 8 prompts x 4 samples, 16k
# cap. Rollout is the composed fast path (ps_resolve_policy turns it on for every .json
# policy). The paired BASELINE must itself be a Megatron-trained bf16 run.
#
# The watcher (cli watch-ema) reads bf16_tpot_ms / int4_tpot_ms from the heatmap, so
# the NVFP4 run gets a derived copy with the NVFP4 column in the int4 slot.
set -uo pipefail

VERL_PS=${VERL_PS:-$HOME/verl-ps}
VLLM_PS=${VLLM_PS:-$HOME/vllm-ps}
RUN_ROOT=${RUN_ROOT:-$HOME/ps_runs/ema}
CAL=${CAL:-$HOME/ps_runs/ema_calibration}
HEAT=${HEAT:-$HOME/experiments/phi4_mini_fastpath_heatmap}
BASELINE=${BASELINE:?set BASELINE to a Megatron-trained bf16 metrics jsonl on the same prompts}
PHI_NVFP4=${PHI_NVFP4:-$HOME/models/Phi-4-mini-reasoning-NVFP4}
CONFIGS=${CONFIGS:-"w4a16 nvfp4"}
DECIDE_AT=${DECIDE_AT:-10}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$VLLM_PS:$VERL_PS${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export TRAINER=${TRAINER:-megatron}
export DATA_DIR=${DATA_DIR:-$HOME/ps_data/gsm8k_messages_2048}

mkdir -p "$RUN_ROOT"
for prec in $CONFIGS; do
    case "$prec" in
        w4a16) int4_path=""; heat="$HEAT/heatmap.json"; w4trace="$CAL/phi4_mini_reasoning_w4a16/traces/request_lifetimes_replica000_node000.jsonl" ;;
        nvfp4) int4_path="$PHI_NVFP4"; heat="$HEAT/heatmap_nvfp4_as_int4.json"; w4trace="$CAL/phi4_mini_reasoning_nvfp4/traces/request_lifetimes_replica000_node000.jsonl" ;;
        *) echo "unknown precision $prec" >&2; exit 2 ;;
    esac
    run_dir="$RUN_ROOT/phi4_mini_reasoning_ema_$prec"
    exp="phi4_mini_reasoning_ema_$prec"
    [ -f "$run_dir/COMPLETE" ] && { echo "[ema] $prec already COMPLETE"; continue; }
    ray_tmp="$HOME/rt/ema$(echo $prec | tr -d 'aeiou')"; rm -rf "$ray_tmp"; mkdir -p "$ray_tmp"
    echo "[ema] === $prec -> $run_dir ==="

    RUN_DIR="$run_dir" RAY_TMPDIR="$ray_tmp" MODEL_KEY=phi4_mini_reasoning \
    INT4_MODEL_PATH="$int4_path" EXPERIMENT_NAME="$exp" PROJECT_NAME=continuous_ema \
    INITIAL_BATCH=32 ROLLOUT_N=4 RESPONSE_CAP=16384 TOTAL_STEPS=20 SAVE_FREQ=-1 RUNNER=full_step \
    BF_TRACE="$CAL/phi4_mini_reasoning_bf16/traces/request_lifetimes_replica000_node000.jsonl" \
    W4_TRACE="$w4trace" HEATMAP="$heat" \
    bash "$VERL_PS/examples/precision_scheduler/recipes/continuous_ema.sh" "$@" &
    ema_pid=$!

    python "$VERL_PS/scripts/precision_scheduler/ema_halt_monitor.py" \
        --run-dir "$run_dir" --project continuous_ema --experiment "$exp" \
        --baseline "$BASELINE" --decide-at "$DECIDE_AT" --pid "$ema_pid"
    mon=$?
    wait "$ema_pid"; rc=$?
    echo "[ema] $prec runner exit=$rc monitor exit=$mon $( [ -f "$run_dir/HALTED" ] && echo HALTED )"
done
echo "[ema] done"
