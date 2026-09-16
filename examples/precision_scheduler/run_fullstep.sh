#!/usr/bin/env bash
# run_fullstep.sh -- the single-GPU GRPO driver of the precision scheduler (TP=1, DP=1, one vLLM engine).
# TRAINER selects the training engine: megatron (default; decision 10 reversed 2026-09-16: every
# reported RL step trains with Megatron-Core through Megatron-Bridge, LoRA via model.lora.*) or fsdp2
# (kept for the CPU compose tests only; never for a GPU run, decision 10 amended 2026-09-16). run_megatron_fullstep.sh and
# run_fsdp_fullstep.sh are two-line wrappers that pin TRAINER.
#
# Refactored from .codex-report/new-storyline-experiments/eos_hazard_fullstep_b64_cap16k/run_fsdp_fullstep.sh
# and rl-workflow/run_megatron_tp_live_fullstep.sh (the Megatron TP1 driver behind the headline Qwen3.5-9B
# runs; its Megatron block is reconstructed from the replay-regression worker, which recorded the same
# McoreEngineConfig): the model case statement became the YAML overlays (models/*.yaml), the env exports
# became rollout.precision_scheduler.* overrides (docs/precision_scheduler/config.md), the
# resolved_models.json snapshot lookup became MODEL_PATH / INT4_MODEL_PATH, and the study-specific run
# layout became RUN_DIR.
#
# Usage:  RUN_DIR=... MODEL_KEY=qwen3_5_9b POLICY=tail_t8 [knobs...] run_fullstep.sh [extra hydra overrides...]
# Knobs (all optional): TRAINER=megatron TRAIN_BATCH_SIZE=16 ROLLOUT_N=4 RESPONSE_CAP=16384 PROMPT_CAP=2048
#   MAX_MODEL_LEN=cap+prompt GMEM=0.50 TOTAL_STEPS=4 SAVE_FREQ=-1 ACTOR_LR=1e-6 USE_KL_LOSS=True
#   PPO_MAX_TOKEN_LEN=18432 ENFORCE_EAGER=false WEIGHT_BUCKET_MB=4096 ROLLOUT_SEED=42 MAX_CKPT_TO_KEEP=2
#   USE_FUSED_KERNELS=true USE_DYNAMIC_BSZ=true ACTOR_PARAM_OFFLOAD=false
#   CALCULATE_LOG_PROBS=True (rollout log-probs; rollout-only recipes set False)
#   megatron only: RECOMPUTE_GRANULARITY=full RECOMPUTE_METHOD=uniform RECOMPUTE_NUM_LAYERS=1
#     ATTENTION_BACKEND=auto ACTOR_GRAD_OFFLOAD=false ACTOR_OPTIMIZER_OFFLOAD=false
#   fsdp2 only: GRADIENT_CHECKPOINTING=true
# The recipes under recipes/ and long_run/ are thin wrappers that set these and append overrides.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${RUN_DIR:?set RUN_DIR}"
MODEL_KEY="${MODEL_KEY:-qwen3_5_4b}"
POLICY="${POLICY:-bf16}"
train_batch_size="${TRAIN_BATCH_SIZE:-16}"
rollout_n="${ROLLOUT_N:-4}"
requests_per_step="$((train_batch_size * rollout_n))"
response_cap="${RESPONSE_CAP:-16384}"
prompt_cap="${PROMPT_CAP:-2048}"
max_model_len="${MAX_MODEL_LEN:-$((response_cap + prompt_cap))}"
ppo_max_token_len="${PPO_MAX_TOKEN_LEN:-18432}"
gmem="${GMEM:-0.50}"
steps="${TOTAL_STEPS:-4}"
save_freq="${SAVE_FREQ:--1}"
actor_lr="${ACTOR_LR:-1e-6}"
use_kl_loss="${USE_KL_LOSS:-True}"
enforce_eager="${ENFORCE_EAGER:-false}"
weight_bucket_mb="${WEIGHT_BUCKET_MB:-4096}"
rollout_seed="${ROLLOUT_SEED:-42}"
max_ckpt_to_keep="${MAX_CKPT_TO_KEEP:-2}"
use_fused_kernels="${USE_FUSED_KERNELS:-true}"
use_dynamic_bsz="${USE_DYNAMIC_BSZ:-true}"
gradient_checkpointing="${GRADIENT_CHECKPOINTING:-true}"
actor_param_offload="${ACTOR_PARAM_OFFLOAD:-false}"
calculate_log_probs="${CALCULATE_LOG_PROBS:-True}"
actor_micro_batch_size="${ACTOR_MICRO_BATCH_SIZE:-1}"
logprob_micro_batch_size="${LOGPROB_MICRO_BATCH_SIZE:-1}"
trainer_engine="${TRAINER:-megatron}"
recompute_granularity="${RECOMPUTE_GRANULARITY:-full}"
recompute_method="${RECOMPUTE_METHOD:-uniform}"
recompute_num_layers="${RECOMPUTE_NUM_LAYERS:-1}"
attention_backend="${ATTENTION_BACKEND:-auto}"
actor_grad_offload="${ACTOR_GRAD_OFFLOAD:-false}"
actor_optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-false}"

ps_resolve_policy "${POLICY}"
ps_common_overrides

# Training-engine block. Everything else in the override list is engine-neutral.
case "${trainer_engine}" in
  megatron)
    driver_name="run_megatron_fullstep.sh"; backend="megatron"
    # Megatron-Core through Megatron-Bridge (LoRA needs the bridge provider: use_mbridge=True,
    # vanilla_mbridge=False). LoRA targets are the Megatron names in the model overlay
    # (model.lora.target_modules); build_peft_config_for_vllm maps them back to the HF names for the
    # vLLM adapter. Full uniform recompute of one layer at a time is what the headline 9B runs used.
    trainer_overrides=(
      model_engine=megatron
      actor_rollout_ref.actor.megatron.use_mbridge=True actor_rollout_ref.actor.megatron.vanilla_mbridge=False
      actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
      actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
      actor_rollout_ref.actor.megatron.context_parallel_size=1
      actor_rollout_ref.actor.megatron.sequence_parallel=False
      "actor_rollout_ref.actor.megatron.param_offload=${actor_param_offload}"
      "actor_rollout_ref.actor.megatron.grad_offload=${actor_grad_offload}"
      "actor_rollout_ref.actor.megatron.optimizer_offload=${actor_optimizer_offload}"
      "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=${recompute_granularity}"
      "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=${recompute_method}"
      "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=${recompute_num_layers}"
      "actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=${attention_backend}"
      +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False
    ) ;;
  fsdp2)
    driver_name="run_fsdp_fullstep.sh"; backend="fsdp2"
    trainer_overrides=(
      "actor_rollout_ref.model.enable_gradient_checkpointing=${gradient_checkpointing}"
      actor_rollout_ref.actor.strategy=fsdp2
      actor_rollout_ref.actor.fsdp_config.fsdp_size=1 actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
      "actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}"
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
      actor_rollout_ref.ref.strategy=fsdp2
      actor_rollout_ref.ref.fsdp_config.fsdp_size=1
      actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    ) ;;
  *) ps_die "TRAINER must be megatron or fsdp2 (got '${trainer_engine}')" ;;
esac

PS_RUN_CONFIG_JSON=$(cat <<JSON
{"driver":"${driver_name}","backend":"${backend}","model_key":"${MODEL_KEY}","model_path":"${MODEL_PATH:-}","int4_model_path":"${INT4_MODEL_PATH:-}","policy":"${POLICY}","policy_kind":"${PS_POLICY_KIND}","steps":${steps},"train_batch_size":${train_batch_size},"rollout_n":${rollout_n},"requests_per_step":${requests_per_step},"response_cap":${response_cap},"prompt_cap":${prompt_cap},"max_model_len":${max_model_len},"temperature":1.0,"top_p":1.0,"top_k":-1,"rollout_seed":${rollout_seed},"reprefill":false,"lora_rank":16,"gpu_memory_utilization":${gmem},"enforce_eager":${enforce_eager},"update_weights_bucket_megabytes":${weight_bucket_mb},"save_freq":${save_freq},"actor_lr":"${actor_lr}","use_kl_loss":"${use_kl_loss}","data_dir":"${DATA_DIR:-${PS_DATA_ROOT:-}/gsm8k_messages_2048}","experiment_name":"${PS_EXPERIMENT_NAME}"}
JSON
)
ps_init_run_dir

overrides=(
  "${PS_COMMON_OVERRIDES[@]}"
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False
  "data.train_batch_size=${train_batch_size}" "data.max_prompt_length=${prompt_cap}"
  "data.max_response_length=${response_cap}"
  data.filter_overlong_prompts=False data.truncation=error data.shuffle=False data.dataloader_num_workers=1
  "actor_rollout_ref.model.use_fused_kernels=${use_fused_kernels}"
  "${trainer_overrides[@]}"
  "actor_rollout_ref.actor.optim.lr=${actor_lr}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${train_batch_size}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${actor_micro_batch_size}"
  "actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len}"
  "actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}" actor_rollout_ref.actor.kl_loss_coef=0.01
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.calculate_entropy=False
  actor_rollout_ref.actor.use_torch_compile=False
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${logprob_micro_batch_size}"
  "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len}"
  actor_rollout_ref.ref.use_torch_compile=False
  actor_rollout_ref.rollout.name=vllm "actor_rollout_ref.rollout.n=${rollout_n}" actor_rollout_ref.rollout.dtype=bfloat16
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.do_sample=True "actor_rollout_ref.rollout.seed=${rollout_seed}"
  "actor_rollout_ref.rollout.calculate_log_probs=${calculate_log_probs}"
  "actor_rollout_ref.rollout.enforce_eager=${enforce_eager}"
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 "actor_rollout_ref.rollout.gpu_memory_utilization=${gmem}"
  "actor_rollout_ref.rollout.max_model_len=${max_model_len}" "actor_rollout_ref.rollout.max_num_seqs=${requests_per_step}"
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.enable_prefix_caching=False
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${weight_bucket_mb}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${logprob_micro_batch_size}"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len}"
  "${PS_POLICY_OVERRIDES[@]}"
  trainer.critic_warmup=0 trainer.balance_batch=False trainer.val_before_train=False trainer.test_freq=-1
  "trainer.save_freq=${save_freq}" "trainer.max_actor_ckpt_to_keep=${max_ckpt_to_keep}"
  "trainer.total_training_steps=${steps}"
  "$@"
)
ps_launch "${overrides[@]}"
