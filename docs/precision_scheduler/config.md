# Precision scheduler configuration (`actor_rollout_ref.rollout.precision_scheduler`)

Every control of the rollout precision scheduler is a YAML key (decision 9). The block lives in
`verl/trainer/config/rollout/rollout.yaml` and is materialised as
`verl.workers.config.PrecisionSchedulerConfig`. verl translates it into environment variables for the
vLLM server actor with `verl.workers.rollout.vllm_rollout.precision_scheduler_env.to_vllm_env()`; the
variables are the wire format vLLM's `envs.py` reads inside its own process. Recipes and users never
set them by hand. The table below is the single documentation of that wire format and is pinned by
`tests/workers/rollout/test_precision_scheduler_env_on_cpu.py`.

## Key to environment-variable table

| YAML key | Environment variable | Type / default | Meaning |
|---|---|---|---|
| `enable` | `VLLM_DUAL_PRECISION_ROLLOUT` | bool, `false` | Master switch for dual-precision (BF16 / INT4) rollout. |
| `lora_fast_path` | `ROLLOUT_QLORA` | bool, `false` | Fused torch LoRA path for the single-adapter rollout case (C1). Independent of `enable`. |
| `lora_dual_stream` | `VLLM_LORA_ENABLE_DUAL_STREAM` | bool, `false` | LoRA GEMMs on an auxiliary stream (only effective with `enable: false`, decision 1). |
| `lora_fuse_packed` | `VLLM_ROLLOUT_LORA_FUSE_PACKED` | bool, `true` | Fuse packed qkv / gate_up LoRA slices into one GEMM pair. |
| `int4_model` | `VLLM_DUAL_PRECISION_INT4_MODEL` | str, `null` | Path to the INT4 shadow checkpoint. |
| `policy` | `VLLM_DUAL_PRECISION_POLICY` | str, `""` | `fixed_threshold:<t>`, `fixed_frontier:<K>`, `uniform_w4`, or a path to an EMA policy JSON (decision 6). |
| `bf16_layers` | `VLLM_DUAL_PRECISION_BF16_LAYERS` | str, `"first:3,last:3"` | Layers that stay BF16 while the rest run INT4. |
| `int4_modules` | `VLLM_DUAL_PRECISION_INT4_MODULES` | str, `"all"` | Module classes eligible for INT4. |
| `reprefill` | `VLLM_DUAL_PRECISION_REPREFILL` | bool, `false` | Re-prefill survivors after a switch (decision 5 ablation). |
| `sleep_level` | (none: resolved in verl) | int, `null` | Forced vLLM sleep level. `null` = automatic; forced to 1 when `enable` is true. |
| `reload_policy_each_rollout` | `VLLM_DUAL_PRECISION_RELOAD_POLICY_EACH_ROLLOUT` | bool, `false` | Re-read the policy file at every rollout (online EMA loop). |
| `online_observations` | `VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS` | str, `null` | JSONL path receiving online switch cohorts. |
| `validate_shadow` | `VLLM_DUAL_PRECISION_VALIDATE_SHADOW` | bool, `false` | Validate the INT4 shadow binding at load. |
| `validate_lifecycle` | `VLLM_DUAL_PRECISION_VALIDATE_LIFECYCLE` | bool, `false` | Validate request lifecycle bookkeeping in the scheduler. |
| `request_trace_dir` | `VERL_REQUEST_TRACE_DIR` | str, `null` | Directory of the per-request lifetime trace written by the verl vLLM server actor. |
| `request_trace_log_tokens` | `VERL_REQUEST_TRACE_LOG_TOKENS` | bool, `false` | Add sampled `token_ids` to trace finish rows. |
| `zmq_namespace` | `VERL_ZMQ_NAMESPACE` | str, `null` | Namespace of the colocated weight-transfer socket (sanitised to `[A-Za-z0-9_-]`); independent of `enable`, omitted when null. |
| `force_shm_weight_transfer` | `VERL_FORCE_SHM_WEIGHT_TRANSFER` | bool, `false` | Force the shared-memory weight-transfer path; independent of `enable`, omitted when false. |

## Emission rules

* Booleans are emitted as `"1"` / `"0"`; `null` keys are omitted; every other value is `str(value)`.
* When `enable` is `false`, only the three `lora_*` keys are emitted, and only if at least one of them
  differs from its default. A vanilla config therefore emits an empty dict and the vLLM server sees
  exactly the upstream environment.
* When `enable` is `true`, every non-null key is emitted (including `policy: ""`).
* `zmq_namespace` and `force_shm_weight_transfer` are host-isolation knobs independent of `enable`:
  they are emitted whenever set (a non-empty namespace, `force_shm_weight_transfer: true`) and omitted
  otherwise, so a vanilla config still emits `{}`. The worker-side readers (`ServerAdapter`,
  `vLLMColocateWorkerExtension`) read the env vars: they are the wire, not a second config path.
* `sleep_level` never becomes an env var: `resolve_sleep_level()` applies it inside
  `vLLMHttpServer.sleep()` / `_sleep_hybrid()`. With `enable: true` the level is forced to 1 (logged),
  because level 2 discards the resident INT4 shadow weights; `enable: true` with `sleep_level: 2` is
  rejected at config time.

## Where the variables go

1. `verl.trainer.constants_ppo.get_ppo_ray_runtime_env(precision_scheduler=...)` puts the dict into the
   Ray job runtime env at `ray.init()` (called by `verl/trainer/main_ppo.py`), so every actor inherits it.
2. `vLLMReplica.launch_servers()` merges the same dict into the vLLM server actor's `runtime_env.env_vars`
   for clusters that were initialised outside `main_ppo.py`.

The former hard-coded allowlist in `constants_ppo.py` is replaced by the dict above plus
`collect_forwarded_env()`, a **compatibility pass-through** of `VLLM_DUAL_PRECISION_*`, `VERL_*` and
`TMPDIR` from the driver environment for launchers that still export variables by hand. New recipes
must not rely on it: every knob has a YAML key, and a YAML value is emitted after the pass-through,
so it wins over a hand-set variable of the same name.

## Trainer-side harness keys (not env vars)

| YAML key | Default | Meaning |
|---|---|---|
| `trainer.rollout_only` | `false` | Run generation + reward only, dump rollouts, log `rollout_only/*` metrics, skip logprob / advantage / update. |
| `trainer.rollout_only_steps` | `null` | Number of rollout-only steps; `null` uses `trainer.total_training_steps`. |
| `trainer.save_initial_checkpoint` | `false` | Save the step-0 checkpoint before any training and print `VERL_INITIAL_CHECKPOINT_COMPLETE step=0`. |
| `trainer.exit_after_initial_checkpoint` | `false` | Return right after the initial checkpoint. |
| `trainer.ray_master_port_range` | `null` | `"start:end"` torch-distributed master port range; env fallback `VERL_RAY_MASTER_PORT_RANGE`. |
| `trainer.stable_sample_uid` | `false` | Use `idx-<dataset index>` as the TransferQueue uid so traces are joinable across paired runs. |
| `actor_rollout_ref.actor.old_log_prob_calculate_entropy` | `true` | Compute entropy in the old-logprob pass (`false` skips it and the `actor/entropy` metric). |

The host-isolation knobs are YAML keys too (`precision_scheduler.zmq_namespace`,
`precision_scheduler.force_shm_weight_transfer`, `trainer.ray_master_port_range`); their env forms
(`VERL_ZMQ_NAMESPACE`, `VERL_FORCE_SHM_WEIGHT_TRANSFER`, `VERL_RAY_MASTER_PORT_RANGE`) remain readable
through the `VERL_*` pass-through for hand-set launches only.
