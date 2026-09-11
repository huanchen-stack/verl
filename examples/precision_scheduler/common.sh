#!/usr/bin/env bash
# common.sh -- shared helpers for the precision-scheduler recipes (sourced, never executed).
#
# Decision 9: recipes configure verl and vLLM exclusively through Hydra overrides of YAML keys.
# The only environment the recipes rely on is what scripts/precision_scheduler/env/activate.sh
# exports (PYTHON_BIN, PYTHONPATH, ...) and what the decision-13 launcher run_gpu.sh sets
# (CUDA_VISIBLE_DEVICES, RAY_TMPDIR). Per-model settings come from the YAML overlays in
# examples/precision_scheduler/models/ selected with Hydra's search path + group syntax.
#
# Recipe interface (shell variables read by the helpers; every one has a default except RUN_DIR):
#   RUN_DIR            run directory (metrics/ logs/ checkpoints/ rollouts/ traces/, markers, driver.log)
#   MODEL_KEY          overlay name under models/ (qwen3_5_4b, qwen3_5_9b, phi4_mini_reasoning, gemma4_e2b, nemotron_h)
#   MODEL_PATH         optional local snapshot overriding actor_rollout_ref.model.path (default: overlay HF id)
#   INT4_MODEL_PATH    optional local snapshot overriding rollout.precision_scheduler.int4_model
#   POLICY             bf16 | full_w4 | tail_t<N> | fixed_k<K> | <path to a policy JSON>   (default bf16)
#   DATA_DIR           directory holding train.parquet and test.parquet (default $PS_DATA_ROOT/gsm8k_messages_2048;
#                      PS_DATA_ROOT has no default: export it (README "Data") or set DATA_DIR)
#   REWARD_FN          custom reward file (default examples/precision_scheduler/rewards.py)
#   EXPERIMENT_NAME    trainer.experiment_name (default <model>_<policy>)
#   PORT_BASE          torch-distributed master port range base (default 47000 + 300 * first visible GPU)
#   DRY_RUN=1          print the resolved override list (one `OVERRIDE<TAB>...` line each) and exit 0
#   RUN_TIMEOUT        GNU timeout duration for the trainer (default 12h)
# Extra positional arguments of every recipe are appended verbatim as Hydra overrides.
#
# Output contract of a launched run: STARTED / COMPLETE / FAILED markers, run_config.json,
# driver.log; the trainer writes metrics/<project>/<experiment>.jsonl (FileLogger), rollouts/<step>.jsonl,
# traces/request_lifetimes_replica000_node000.jsonl, checkpoints/global_step_<k>/.

PS_EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PS_REPO_ROOT="$(cd "${PS_EXAMPLES_DIR}/../.." && pwd)"
PS_PYTHON="${PYTHON_BIN:-python}"

ps_die() { echo "$(basename "${BASH_SOURCE[1]:-recipe}"): $*" >&2; exit 2; }

ps_first_gpu() {
  local first="${CUDA_VISIBLE_DEVICES:-0}"
  first="${first%%,*}"
  [[ "${first}" =~ ^[0-9]+$ ]] || first=0
  echo "${first}"
}

# ps_resolve_policy <name> -> fills PS_POLICY_OVERRIDES (array) and PS_POLICY_KIND.
#   bf16      precision_scheduler.enable=false (vanilla vLLM LoRA path)
#   full_w4   enable=true, policy=uniform_w4
#   tail_t<N> enable=true, policy=fixed_threshold:<N>   (switch when the live batch drains to N)
#   fixed_k<K> enable=true, policy=fixed_frontier:<K>   (switch every request at K response tokens)
#   <file>    enable=true, policy=<file>                (calibrated / EMA lookup table JSON)
# The W4 kinds also enable the fused LoRA fast path and the dual stream, exactly as the archived
# launchers set ROLLOUT_QLORA=1 / VLLM_LORA_ENABLE_DUAL_STREAM=1 for every non-BF16 cell.
ps_resolve_policy() {
  local name="${1:-bf16}" spec=""
  local ps="actor_rollout_ref.rollout.precision_scheduler"
  PS_POLICY_OVERRIDES=()
  case "${name}" in
    bf16)
      PS_POLICY_KIND=bf16
      PS_POLICY_OVERRIDES+=("${ps}.enable=false")
      return 0 ;;
    full_w4)   PS_POLICY_KIND=uniform_w4; spec="uniform_w4" ;;
    tail_t*)   PS_POLICY_KIND=fixed_threshold; spec="fixed_threshold:${name#tail_t}"
               [[ "${name#tail_t}" =~ ^[0-9]+$ ]] || ps_die "bad POLICY ${name}" ;;
    fixed_k*)  PS_POLICY_KIND=fixed_frontier; spec="fixed_frontier:${name#fixed_k}"
               [[ "${name#fixed_k}" =~ ^[0-9]+$ ]] || ps_die "bad POLICY ${name}" ;;
    *.json)    PS_POLICY_KIND=lookup_table; spec="${name}"
               [[ -f "${name}" ]] || ps_die "policy JSON not found: ${name}" ;;
    *) ps_die "unknown POLICY '${name}' (bf16 | full_w4 | tail_t<N> | fixed_k<K> | <policy.json>)" ;;
  esac
  PS_POLICY_OVERRIDES+=(
    "${ps}.enable=true"
    "${ps}.policy='${spec}'"
    "${ps}.lora_fast_path=true"
    "${ps}.lora_dual_stream=true"
    "${ps}.bf16_layers=none"
    "${ps}.reprefill=false"
    "${ps}.validate_lifecycle=true"
    "${ps}.online_observations=${RUN_DIR}/switch_observations.jsonl"
  )
  if [[ -n "${INT4_MODEL_PATH:-}" ]]; then
    PS_POLICY_OVERRIDES+=("${ps}.int4_model=${INT4_MODEL_PATH}")
  fi
}

# ps_common_overrides -> fills PS_COMMON_OVERRIDES: model overlay, data, reward, tracing,
# host isolation, logging and run-dir layout. Shared by every driver.
ps_common_overrides() {
  local model_key="${MODEL_KEY:-qwen3_5_4b}"
  local overlay="${PS_EXAMPLES_DIR}/models/${model_key}.yaml"
  [[ -f "${overlay}" ]] || ps_die "no model overlay ${overlay}"
  if [[ -z "${DATA_DIR:-}" && -z "${PS_DATA_ROOT:-}" ]]; then ps_die "set DATA_DIR or PS_DATA_ROOT (see README, Data)"; fi
  local data_dir="${DATA_DIR:-${PS_DATA_ROOT}/gsm8k_messages_2048}"
  local reward="${REWARD_FN:-${PS_EXAMPLES_DIR}/rewards.py}"
  local gpu; gpu="$(ps_first_gpu)"
  local port_base="${PORT_BASE:-$((47000 + gpu * 300))}"
  local ray_tmp="${RAY_TMPDIR:-${RUN_DIR}/ray_tmp}"
  local ps="actor_rollout_ref.rollout.precision_scheduler"
  PS_EXPERIMENT_NAME="${EXPERIMENT_NAME:-${model_key}_${POLICY:-bf16}}"
  PS_COMMON_OVERRIDES=(
    "hydra.searchpath=[file://${PS_EXAMPLES_DIR}]"
    "+models@_global_=${model_key}"
    "data.train_files=${data_dir}/train.parquet"
    "data.val_files=${data_dir}/test.parquet"
    "reward.custom_reward_function.path=${reward}"
    "reward.custom_reward_function.name=compute_score"
    "${ps}.request_trace_dir=${RUN_DIR}/traces"
    "${ps}.request_trace_log_tokens=true"
    "${ps}.zmq_namespace=ps_${model_key}_g${gpu}_$$"
    "${ps}.force_shm_weight_transfer=true"
    "trainer.ray_master_port_range=${port_base}:$((port_base + 299))"
    "trainer.stable_sample_uid=true"
    "trainer.logger=[console,file]"
    "trainer.project_name=${PROJECT_NAME:-precision_scheduler}"
    "trainer.experiment_name=${PS_EXPERIMENT_NAME}"
    "trainer.n_gpus_per_node=1"
    "trainer.nnodes=1"
    "trainer.default_local_dir=${RUN_DIR}/checkpoints"
    "trainer.rollout_data_dir=${RUN_DIR}/rollouts"
    "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS:-12}"
    "ray_kwargs.ray_init.include_dashboard=False"
    "ray_kwargs.ray_init._temp_dir=${ray_tmp}"
  )
  if [[ -n "${MODEL_PATH:-}" ]]; then
    PS_COMMON_OVERRIDES+=("actor_rollout_ref.model.path=${MODEL_PATH}")
  fi
}

# Decision 13: GPU launches go through scripts/precision_scheduler/env/run_gpu.sh, which sets
# CUDA_VISIBLE_DEVICES (never GPU 1), owns the process group and checks for leftovers.
ps_require_launcher_gpu() {
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    ps_die "CUDA_VISIBLE_DEVICES is not set: launch under scripts/precision_scheduler/env/run_gpu.sh"
  fi
  case ",${CUDA_VISIBLE_DEVICES}," in *,1,*) ps_die "GPU 1 is never allowed (decision 13)";; esac
}

# ps_init_run_dir -> creates the layout, refuses to append to a started run, writes run_config.json
# from PS_RUN_CONFIG_JSON (a JSON object string the driver assembles). No-op under DRY_RUN.
ps_init_run_dir() {
  [[ -n "${RUN_DIR:-}" ]] || ps_die "RUN_DIR is required"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then return 0; fi
  ps_require_launcher_gpu
  if [[ -e "${RUN_DIR}/COMPLETE" ]]; then echo "already complete: ${RUN_DIR}"; exit 0; fi
  if [[ -e "${RUN_DIR}/STARTED" && "${ALLOW_RESUME:-0}" != "1" ]]; then
    ps_die "refusing to append to a started run ${RUN_DIR} (set ALLOW_RESUME=1 to resume)"
  fi
  mkdir -p "${RUN_DIR}/metrics" "${RUN_DIR}/logs" "${RUN_DIR}/checkpoints" "${RUN_DIR}/rollouts" "${RUN_DIR}/traces"
  touch "${RUN_DIR}/STARTED"
  rm -f "${RUN_DIR}/FAILED"
  printf '%s\n' "${PS_RUN_CONFIG_JSON:-{\}}" >"${RUN_DIR}/run_config.json"
}

# ps_launch <override>... -> DRY_RUN prints the override list; otherwise runs the trainer under
# GNU timeout, tees driver.log and writes the COMPLETE / FAILED marker. Returns the trainer rc.
ps_launch() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'PYTHON\t%s\n' "${PS_PYTHON}"
    printf 'RUN_DIR\t%s\n' "${RUN_DIR}"
    local o; for o in "$@"; do printf 'OVERRIDE\t%s\n' "${o}"; done
    echo "DRY_RUN_OK"
    return 0
  fi
  ps_require_launcher_gpu
  mkdir -p "${RUN_DIR}/torchinductor_cache" "${RAY_TMPDIR:-${RUN_DIR}/ray_tmp}" "${RUN_DIR}/metrics"
  local rc
  (
    # The FileLogger writes <cwd>/<project>/<experiment>.jsonl: run from metrics/ so no env var is
    # needed. TorchInductor artifacts hold process-local CUDA handles, so the cache is per run
    # (process hygiene, not a project knob). Everything else is a Hydra override.
    cd "${RUN_DIR}/metrics"
    export TORCHINDUCTOR_CACHE_DIR="${RUN_DIR}/torchinductor_cache"
    exec timeout --signal=INT --kill-after=300s "${RUN_TIMEOUT:-12h}" \
      "${PS_PYTHON}" -m verl.trainer.main_ppo "hydra.run.dir=${RUN_DIR}/logs/hydra" "$@"
  ) 2>&1 | tee -a "${RUN_DIR}/driver.log"
  rc=${PIPESTATUS[0]}
  if [[ ${rc} -eq 0 ]]; then touch "${RUN_DIR}/COMPLETE"; else printf '%s\n' "${rc}" >"${RUN_DIR}/FAILED"; fi
  return "${rc}"
}
