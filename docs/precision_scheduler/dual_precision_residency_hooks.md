# Dual-precision residency: verl hooks

The vLLM side (loading an INT4 GPTQ shadow next to the BF16 base and binding
LoRA wrappers to one of them per forward) is described in vLLM's
`docs/design/dual_precision_residency.md`. verl contributes four small pieces.

## Config keys (`actor_rollout_ref.rollout`)

| Key | Default | Meaning |
|---|---|---|
| `model_path` | `null` | Base checkpoint vLLM serves; `null` reuses `actor_rollout_ref.model.path`. Lets a BF16 actor/ref train while rollout serves a different base. Resolved through `copy_to_local(model_path, use_shm=...)` in `vLLMHttpServer` (`_resolve_rollout_model_path`). |
| `precision_scheduler.sleep_level` | `null` | vLLM sleep level between rollouts (C8 block): 1 offloads weights to CPU and restores them, 2 discards them and relies on the next weight sync. `null` keeps the engine heuristic (1 for LoRA-as-adapter / MTP / NPU and in colocated mode, else 2). |

`verl.workers.config.precision_scheduler.resolve_sleep_level(cfg, default)`
applies the value at both `engine.sleep` sites; with
`precision_scheduler.enable=true` it forces level 1 when unset and refuses 2,
because the INT4 shadow store is loaded once and never re-synced from the
trainer. The rest of the block (`enable`, `int4_model`, `bf16_layers`,
`int4_modules`, `validate_shadow`, `validate_lifecycle`) is translated into
the `VLLM_DUAL_PRECISION_*` env vars for the server actor by C8; see
`docs/precision_scheduler/config.md`.

## Hooks

* `verl/workers/rollout/vllm_rollout/utils.py`:
  `_hide_dual_precision_shadow_model(model)` pops the shadow submodule
  (`vllm.model_executor.dual_precision.SHADOW_MODULE_NAME`, string fallback
  `_vllm_dual_precision_int4_model`) while `update_weights_from_ipc` re-runs
  `process_weights_after_loading`. The Marlin repack is not idempotent; on
  the current vLLM base a second visit asserts inside the kernel, on older
  bases it silently corrupted the packed shadow.
* `verl/workers/engine_workers.py`: `aggressive_empty_cache(force_sync=True)`
  right before `rollout.resume(tags=["weights"])` so the trainer's inactive
  allocator cache does not collide with the remapped BF16 + shadow weights.

Tests: `tests/workers/rollout/rollout_vllm/test_dual_precision_residency_hooks.py`
(CPU, fakes for the engine and the worker).
