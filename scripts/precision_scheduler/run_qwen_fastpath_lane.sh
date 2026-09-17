#!/usr/bin/env bash
# Qwen3.5-9B fast-path lane on GPU 1 of the PRO 6000 box: calibration, then online EMA for
# both quantized formats, each held to a halt rule against the punica BF16 baseline.
#
#   source ~/rl_env.sh && bash run_qwen_fastpath_lane.sh calib     # stage 2: 3 x 128-request rollouts
#   source ~/rl_env.sh && bash run_qwen_fastpath_lane.sh ema       # stage 3: EMA w4a16, then EMA nvfp4
#   source ~/rl_env.sh && bash run_qwen_fastpath_lane.sh all
#
# Stage 1 (the fast-path TPOT heatmap, run_tpot_heatmap_3precision.sh with STANDALONE=0
# FAST_PATH=1) must already be complete in $HEATMAP_DIR.
#
# Calibration: the watcher prices plans from a pure-BF16 and a pure-W4 request-lifetime trace of
# single requests, so each calibration is ROLLOUT_N=1 x TRAIN_BATCH_SIZE=128: 128 different
# prompts, one sample each, no GRPO groups. Rollout-only, fast path on for all three so the
# traces come from the same kernels the scheduled rollout uses.
#
# EMA: continuous_ema.sh RUNNER=full_step, 8 x 4 = 32 requests per step, cap 16384, 20 steps,
# the same shape as the baseline matrix so step k draws the same prompts as the baseline's step k.
# ema_halt_monitor.py kills the run at step 10 if it is slower than the BF16 baseline over steps 2..10.
#
# The watcher's grid reads only the int4_tpot_ms row of a heatmap, so the NVFP4 arm gets a derived
# heatmap file with the nvfp4 row copied into that slot.
set -uo pipefail

stage="${1:-all}"

VERL_PS=${VERL_PS:-$HOME/verl-ps}
VLLM_PS=${VLLM_PS:-$HOME/vllm-ps}
DATA_DIR=${DATA_DIR:-$HOME/ps_data/gsm8k_messages_2048}
RUN_ROOT=${RUN_ROOT:-$HOME/ps_runs/qwen_fastpath_lane}
RAY_ROOT=${RAY_ROOT:-$HOME/rt}
HEATMAP_DIR=${HEATMAP_DIR:-$HOME/experiments/qwen35_9b_fastpath_heatmap}
BASELINE_BF16=${BASELINE_BF16:-$VERL_PS/docs/precision_scheduler/baselines/matrix_2026-09-16/qwen3_5_9b_bf16.jsonl}

MODEL_KEY=qwen3_5_9b
NVFP4_MODEL=${NVFP4_MODEL:-AxionML/Qwen3.5-9B-NVFP4}
EMA_STEPS=${EMA_STEPS:-20}
HALT_AT=${HALT_AT:-10}

# GPU 1 is ours on this box; GPU 0 belongs to a colleague. Forced, not defaulted: rl_env.sh,
# .bashrc and .profile all export CUDA_VISIBLE_DEVICES=0, so a ${VAR:-1} default silently
# inherits GPU 0 (it did, once). LANE_GPU is the only way to move the lane, and 0 is refused.
export CUDA_VISIBLE_DEVICES="${LANE_GPU:-1}"
case ",${CUDA_VISIBLE_DEVICES}," in *,0,*) echo "[lane] refusing to run on GPU 0 (reserved)" >&2; exit 2;; esac
# The port base keeps the torch distributed range clear of anything launched against GPU 0.
export PS_ALLOW_GPU1=1
export PORT_BASE=${PORT_BASE:-47600}
export PYTHONPATH="$VLLM_PS:$VERL_PS${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export MODEL_KEY DATA_DIR

ps="actor_rollout_ref.rollout.precision_scheduler"
fast_path_overrides=("${ps}.lora_fast_path=true" "${ps}.lora_dual_stream=true")

mkdir -p "$RUN_ROOT"

run_calib() {
    for precision in bf16 w4a16 nvfp4; do
        case "$precision" in
            bf16)  policy=bf16;    int4_path="" ;;
            w4a16) policy=full_w4; int4_path="" ;;             # overlay default: Intel AutoRound
            nvfp4) policy=full_w4; int4_path="$NVFP4_MODEL" ;;
        esac
        run_dir="$RUN_ROOT/calib_${precision}"
        if [ -f "$run_dir/COMPLETE" ]; then echo "[lane] calib $precision already COMPLETE"; continue; fi
        ray_tmp="$RAY_ROOT/qc${precision:0:1}"; rm -rf "$ray_tmp"; mkdir -p "$ray_tmp"
        echo "[lane] === calib $precision -> $run_dir ==="
        RUN_DIR="$run_dir" RAY_TMPDIR="$ray_tmp" POLICY="$policy" INT4_MODEL_PATH="$int4_path" \
        EXPERIMENT_NAME="calib_${MODEL_KEY}_${precision}" \
        TRAIN_BATCH_SIZE=128 ROLLOUT_N=1 RESPONSE_CAP=16384 TOTAL_STEPS=1 SAVE_FREQ=-1 \
        PROJECT_NAME=ema_calibration \
        bash "$VERL_PS/examples/precision_scheduler/recipes/rollout_only.sh" "${fast_path_overrides[@]}"
        echo "[lane] calib $precision exit=$?"
    done
}

derive_nvfp4_heatmap() {
    python - "$HEATMAP_DIR/heatmap.json" "$HEATMAP_DIR/heatmap_nvfp4_as_w4.json" <<'EOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
out = {k: v for k, v in d.items() if k not in ("nvfp4_tpot_ms", "speedup_bf16_over_nvfp4")}
out["int4_tpot_ms"] = d["nvfp4_tpot_ms"]
out["speedup_bf16_over_int4"] = d["speedup_bf16_over_nvfp4"]
out["derived_from"] = {"source": src, "note": "nvfp4 row placed in the int4 slot the TpotGrid reads"}
json.dump(out, open(dst, "w"), indent=2, sort_keys=True)
print("wrote", dst)
EOF
}

run_ema() {
    [ -f "$HEATMAP_DIR/heatmap.json" ] || { echo "[lane] no heatmap at $HEATMAP_DIR/heatmap.json" >&2; return 2; }
    for precision in w4a16 nvfp4; do
        for c in bf16 "$precision"; do
            [ -f "$RUN_ROOT/calib_${c}/COMPLETE" ] || { echo "[lane] calib $c not COMPLETE; run the calib stage first" >&2; return 2; }
        done
        case "$precision" in
            w4a16) int4_path=""; heatmap="$HEATMAP_DIR/heatmap.json" ;;
            nvfp4) int4_path="$NVFP4_MODEL"; heatmap="$HEATMAP_DIR/heatmap_nvfp4_as_w4.json"; derive_nvfp4_heatmap ;;
        esac
        run_dir="$RUN_ROOT/ema_${precision}"
        if [ -f "$run_dir/CONTINUOUS_EMA_COMPLETE" ] || [ -f "$run_dir/HALTED" ]; then
            echo "[lane] ema $precision already finished ($(ls "$run_dir" | grep -E 'COMPLETE|HALTED' | tr '\n' ' '))"; continue
        fi
        ray_tmp="$RAY_ROOT/qe${precision:0:1}"; rm -rf "$ray_tmp"; mkdir -p "$ray_tmp"
        experiment="ema_${MODEL_KEY}_${precision}"
        echo "[lane] === ema $precision -> $run_dir (heatmap $heatmap) ==="
        mkdir -p "$run_dir"
        RUN_DIR="$run_dir" RAY_TMPDIR="$ray_tmp" INT4_MODEL_PATH="$int4_path" \
        EXPERIMENT_NAME="$experiment" PROJECT_NAME=continuous_ema \
        RUNNER=full_step INITIAL_BATCH=32 ROLLOUT_N=4 RESPONSE_CAP=16384 TOTAL_STEPS="$EMA_STEPS" SAVE_FREQ=-1 \
        BF_TRACE="$RUN_ROOT/calib_bf16/traces/request_lifetimes_replica000_node000.jsonl" \
        W4_TRACE="$RUN_ROOT/calib_${precision}/traces/request_lifetimes_replica000_node000.jsonl" \
        HEATMAP="$heatmap" \
        bash "$VERL_PS/examples/precision_scheduler/recipes/continuous_ema.sh" "${fast_path_overrides[@]}" \
            > "$run_dir/lane_driver.log" 2>&1 &
        ema_pid=$!
        python "$VERL_PS/scripts/precision_scheduler/ema_halt_monitor.py" \
            --run-dir "$run_dir" --project continuous_ema --experiment "$experiment" \
            --baseline "$BASELINE_BF16" --decide-at "$HALT_AT" --pid "$ema_pid" \
            2>&1 | tee "$run_dir/halt_monitor.log"
        monitor_rc=${PIPESTATUS[0]}
        wait "$ema_pid"; ema_rc=$?
        echo "[lane] ema $precision exit=$ema_rc monitor=$monitor_rc"
        if [ "$monitor_rc" = 3 ]; then
            echo "[lane] HALTED: EMA $precision slower than BF16 through step $HALT_AT; stopping the lane"
            return 3
        fi
    done
}

case "$stage" in
    calib) run_calib ;;
    ema)   run_ema ;;
    all)   run_calib && run_ema ;;
    *) echo "usage: $0 calib|ema|all" >&2; exit 2 ;;
esac
rc=$?
echo "[lane] stage $stage finished rc=$rc"
exit $rc
