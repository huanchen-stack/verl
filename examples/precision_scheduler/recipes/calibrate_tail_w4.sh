#!/usr/bin/env bash
# recipes/calibrate_tail_w4.sh -- the two-path calibration for the online scheduler on one GPU.
#
# Path 1 (BF16): a rollout-only run of ROLLOUT_STEPS x INITIAL_BATCH distinct prompts (ROLLOUT_N=1) to
#   RESPONSE_CAP under BF16 with token logging on (recipes/rollout_only.sh, POLICY=bf16).
# Path 2 (tail W4): every path-1 request still generating at each cut frontier is re-issued as
#   prompt + its own BF16 prefix and decoded under uniform W4 from there (cli calib-tail-w4). Cuts are
#   population quantiles of the path-1 lengths (CUT_QUANTILES, default 0.667,0.75,0.8,0.9) or explicit
#   CUT_TOKENS (0 = the legacy uniform-W4 path). Cost: sum(1 - q) continuations per request, one batch.
#
# Usage:  RUN_DIR=... MODEL_KEY=qwen3_5_9b MODEL_PATH=... INT4_MODEL_PATH=... \
#           [INITIAL_BATCH=32 ROLLOUT_STEPS=8 RESPONSE_CAP=24576 CUT_QUANTILES=0.667,0.75,0.8,0.9] \
#           [SKIP_BF16=1 (reuse RUN_DIR/bf16)] [DRY_RUN=1] recipes/calibrate_tail_w4.sh
# Outputs: RUN_DIR/bf16/traces/request_lifetimes_replica000_node000.jsonl   (BF_TRACE for the watcher)
#          RUN_DIR/tail_w4/request_lifetimes_replica000_node000.jsonl        (W4_CONT_TRACE for the watcher)
#          RUN_DIR/tail_w4/calibration_manifest.json
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${here}/../common.sh"

: "${RUN_DIR:?set RUN_DIR}"
: "${MODEL_KEY:?set MODEL_KEY}"
: "${MODEL_PATH:?set MODEL_PATH (BF16 checkpoint)}"
: "${INT4_MODEL_PATH:?set INT4_MODEL_PATH (INT4 shadow checkpoint)}"
batch="${INITIAL_BATCH:-32}"
steps="${ROLLOUT_STEPS:-8}"
cap="${RESPONSE_CAP:-24576}"
data="${DATA_FILE:-${PS_DATA_ROOT:?set PS_DATA_ROOT or DATA_FILE}/gsm8k_messages_2048/train.parquet}"
bf_dir="${RUN_DIR}/bf16"
tail_dir="${RUN_DIR}/tail_w4"
bf_trace="${bf_dir}/traces/request_lifetimes_replica000_node000.jsonl"

# Path 1: BF16 rollout-only, ROLLOUT_N=1 so every request is a distinct prompt (a 32x4 layout puts
# the whole tail on a handful of prompts; the archived 9B calibration had 5 prompts behind its 12 tail
# requests). Token logging is on in run_fullstep.sh already (request_trace_log_tokens=true).
if [[ "${SKIP_BF16:-0}" != "1" ]]; then
  RUN_DIR="${bf_dir}" POLICY=bf16 INITIAL_BATCH="${batch}" TRAIN_BATCH_SIZE="${batch}" ROLLOUT_N=1 \
    RESPONSE_CAP="${cap}" TOTAL_STEPS="${steps}" PROJECT_NAME="${PROJECT_NAME:-calib_tail_w4}" \
    bash "${here}/rollout_only.sh" "$@"
fi
[[ -s "${bf_trace}" ]] || ps_die "BF16 calibration trace missing: ${bf_trace}"

# Path 2: tail-W4 continuations (one in-process engine, uniform W4, same shadow / LoRA knobs as the runs).
cuts=()
if [[ -n "${CUT_TOKENS:-}" ]]; then cuts=(--cut-tokens "${CUT_TOKENS}"); else cuts=(--cut-quantiles "${CUT_QUANTILES:-0.667,0.75,0.8,0.9}"); fi
chat_kwargs="${CHAT_TEMPLATE_KWARGS:-$(ps_chat_template_kwargs "${MODEL_KEY}")}"
cmd=("${PS_PYTHON}" -m verl.experimental.precision_scheduler.cli calib-tail-w4
  --bf-trace "${bf_trace}" --output-trace "${tail_dir}/request_lifetimes_replica000_node000.jsonl"
  --calibration-requests "$(( batch * steps ))" --cap "${cap}" --prompt-max "${PROMPT_CAP:-2048}"
  --data "${data}" --model "${MODEL_PATH}" --int4-model "${INT4_MODEL_PATH}"
  --chat-template-kwargs "${chat_kwargs}" --seed "${ROLLOUT_SEED:-42}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.5}" --max-num-seqs "${MAX_NUM_SEQS:-64}" --log-tokens
  "${cuts[@]}")
[[ -n "${LORA_ADAPTER:-}" ]] && cmd+=(--lora-adapter "${LORA_ADAPTER}")
if [[ "${DRY_RUN:-0}" == "1" ]]; then cmd+=(--dry-run); fi
mkdir -p "${tail_dir}"
echo "### calib-tail-w4: ${cmd[*]}"
"${cmd[@]}" 2>&1 | tee "${tail_dir}/calib_tail_w4.log"
echo "### BF_TRACE=${bf_trace}"
echo "### W4_CONT_TRACE=${tail_dir}/request_lifetimes_replica000_node000.jsonl"
