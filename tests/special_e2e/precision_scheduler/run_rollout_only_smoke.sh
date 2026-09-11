#!/usr/bin/env bash
# One rollout-only step (generation + reward, no training) on a single GPU, driven purely by
# YAML keys: trainer.rollout_only, rollout.precision_scheduler.request_trace_dir, ...
#
# Usage:  RUN_DIR=<dir> [MODEL=<hf snapshot>] [DATA_DIR=<dir with train/test.parquet>] \
#         [REWARD_SCRIPT=<py>] bash run_rollout_only_smoke.sh
# Run it under the decision-13 launcher (scripts/precision_scheduler/env/run_gpu.sh) so
# CUDA_VISIBLE_DEVICES and RAY_TMPDIR are set and every process is killed afterwards.
# Validate the result with validate_rollout_only_run.py.
set -euo pipefail

run="${RUN_DIR:?set RUN_DIR}"
model="${MODEL:-/data/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
data="${DATA_DIR:-/data/huanchen/verl/.codex-report/new-storyline-experiments/eos_hazard_fullstep_b64_cap16k/data/qwen35_4b/gsm8k}"
reward="${REWARD_SCRIPT:-/data/huanchen/verl/.codex-report/new-storyline-experiments/eos_hazard_fullstep_b64_cap16k/universal_reward.py}"
python_bin="${PYTHON_BIN:-python}"
train_batch_size="${TRAIN_BATCH_SIZE:-8}"
rollout_n="${ROLLOUT_N:-4}"
response_cap="${RESPONSE_CAP:-2048}"
prompt_cap="${PROMPT_CAP:-1024}"
gmem="${GMEM:-0.5}"
port_base="${PORT_BASE:-48200}"
ray_tmp="${RAY_TMPDIR:-/tmp/ray_ps_smoke_$$}"

mkdir -p "${run}/metrics" "${run}/rollouts" "${run}/traces" "${ray_tmp}"
export VERL_FILE_LOGGER_ROOT="${run}/metrics"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR="${run}/torchinductor_cache"

"${python_bin}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files="${data}/train.parquet" data.val_files="${data}/test.parquet" \
  data.train_batch_size="${train_batch_size}" data.max_prompt_length="${prompt_cap}" \
  data.max_response_length="${response_cap}" data.filter_overlong_prompts=True data.truncation=error \
  data.shuffle=False data.dataloader_num_workers=1 \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  actor_rollout_ref.model.path="${model}" actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${train_batch_size}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1 actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.strategy=fsdp2 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.use_torch_compile=False actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n="${rollout_n}" \
  actor_rollout_ref.rollout.dtype=bfloat16 actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.seed=42 actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.load_format=dummy actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization="${gmem}" \
  actor_rollout_ref.rollout.max_model_len="$((prompt_cap + response_cap))" \
  actor_rollout_ref.rollout.max_num_seqs="$((train_batch_size * rollout_n))" \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.precision_scheduler.enable=false \
  actor_rollout_ref.rollout.precision_scheduler.request_trace_dir="${run}/traces" \
  actor_rollout_ref.rollout.precision_scheduler.request_trace_log_tokens=true \
  actor_rollout_ref.rollout.precision_scheduler.zmq_namespace="ps_smoke_$$" \
  actor_rollout_ref.rollout.precision_scheduler.force_shm_weight_transfer=true \
  reward.custom_reward_function.path="${reward}" reward.custom_reward_function.name=compute_score \
  trainer.logger='["console","file"]' trainer.project_name=precision_scheduler_smoke \
  trainer.experiment_name=rollout_only_qwen35_4b trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.val_before_train=False trainer.save_freq=-1 trainer.test_freq=-1 trainer.total_training_steps=1 \
  trainer.default_local_dir="${run}/checkpoints" trainer.rollout_data_dir="${run}/rollouts" \
  trainer.rollout_only=true trainer.stable_sample_uid=true \
  trainer.ray_master_port_range="${port_base}:$((port_base + 299))" \
  ray_kwargs.ray_init.num_cpus=12 +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init._temp_dir="${ray_tmp}" 2>&1 | tee "${run}/driver.log"
exit "${PIPESTATUS[0]}"
