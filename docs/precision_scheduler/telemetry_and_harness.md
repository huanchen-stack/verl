# C8: verl telemetry and experiment harness

## Purpose

The verl-side glue that made the precision-scheduling experiments reproducible: a per-request
lifetime trace, a rollout-only measurement mode, an initial-checkpoint gate for paired runs,
multi-experiment isolation on one host, config-to-env translation for the vLLM server actor
(decision 9), and a few opt-in diagnostics. Everything is off by default; a config without the
new keys behaves exactly like upstream verl `2390a3f5`.

## Mechanism

### Configuration (decision 9)

`actor_rollout_ref.rollout.precision_scheduler` (`verl/workers/config/precision_scheduler.py`,
YAML in `verl/trainer/config/rollout/rollout.yaml`) is the source of truth for all vLLM-side
settings. `verl/workers/rollout/vllm_rollout/precision_scheduler_env.py::to_vllm_env()` (implemented in `verl/workers/config/precision_scheduler.py` so the driver never imports the vLLM package) turns it
into the env-var wire format; the key table lives in [`config.md`](config.md). The dict is
injected twice: into the Ray job runtime env at `ray.init()`
(`constants_ppo.get_ppo_ray_runtime_env(precision_scheduler=...)`, called from `main_ppo.py`) and
into the vLLM server actor's `runtime_env.env_vars` in `vLLMReplica.launch_servers()`. The old
hard-coded allowlist is replaced by a `VLLM_DUAL_PRECISION_*` / `VERL_*` / `TMPDIR` pass-through
(`collect_forwarded_env()`), kept only for launchers that still export variables by hand.

### Request lifetime tracer

`verl/workers/rollout/vllm_rollout/request_trace.py::RequestLifetimeTracer` is built once per
`vLLMHttpServer` from `precision_scheduler.request_trace_dir` (env fallback
`VERL_REQUEST_TRACE_DIR`). `generate()` writes one `start` row on entry and one `finish` row on
exit (also for aborted requests, `generation_tokens=0`, `finish_reason="aborted"`). The file
name `request_lifetimes_replica{r:03d}_node{n:03d}.jsonl` and the keys
`event/timestamp/request_id/prompt_tokens` (start) and
`event/timestamp/request_id/generation_tokens/finish_reason[/token_ids]` (finish) are frozen:
`build_policies.py`, `simulate_online_ema.py` and the continuous-EMA watcher parse them by name.
Rows are `json.dumps(sort_keys=True)`; the file handle is opened once per server and lazily on
the first row (the archived code reopened the file for every event; idle `node_rank > 0` servers
create no file). `request_trace_log_tokens: true` adds the sampled `token_ids`, the only lossless
record of the sampled tokens (rollout dumps decode with `skip_special_tokens`).

### Stable trace id

`SingleTurnAgentLoop.run()` passes `trace_request_id="<uid>_<session_id>"` (the July format
`idx-31_1`) to `LLMServerClient.generate()`, which forwards it only when
`rollout.name == "vllm"` (other backends' `generate()` lack the kwarg) while the engine request id
stays a fresh `uuid4().hex` per turn. The tracer echoes the id in both rows. With
`trainer.stable_sample_uid: true` the TransferQueue uid becomes `idx-<extra_info.index>` (row
position as fallback), so dumps and traces are joinable across paired BF16 / INT4 / dynamic runs.
The stash's engine-id override is not ported: the agent-loop id is only the sticky-session key.

### Rollout-only mode

`trainer.rollout_only: true` makes `PPOTrainer.fit()` call `_fit_rollout_only()` after
`on_train_begin()`. Each step runs `step()` (which returns right after reward), prints the frozen
marker `VERL_ROLLOUT_ONLY_COMPLETE step=<k> requests=<n> gen_seconds=<t>`, calls `on_step_end()`
(weight sync + wake-up, exactly what RL does between steps) except after the last step, reads
`responses` / `rm_scores` from TransferQueue, dumps the generations to `trainer.rollout_data_dir`
(required) and logs exactly `timing_s/gen`, `rollout_only/requests`,
`rollout_only/response_tokens`, `critic/rewards/mean` per step (the names the archived
`summarize.py` scripts read from the FileLogger rows). The archived single-step (no metrics,
no `on_step_end`) and multi-step (`on_step_end` after every step including the last) paths are
unified; the number of steps is `trainer.rollout_only_steps` bounded by
`trainer.total_training_steps` (`null` = all training steps).

### Initial checkpoint gate

`trainer.save_initial_checkpoint: true` saves the step-0 checkpoint before validation and prints
`VERL_INITIAL_CHECKPOINT_COMPLETE step=0`; `trainer.exit_after_initial_checkpoint: true` returns
right after it. A resumed run (`global_steps != 0`) is rejected.

### Policy revision barrier

C4's scheduler re-reads the policy JSON at every rollout boundary, and the archived EMA runs
sometimes had the watcher lag a boundary (revision unchanged; vLLM only warns unless
`require_policy_advance: true`). The primary guarantee lives in verl:
`PPOTrainer._policy_revision_barrier()`, called at the top of `_add_batch_to_generate()` (the
single generation entry shared by the sync, colocate-async and separate-async trainers, warmup
batches included). With `precision_scheduler.enable: true`, a file-path `policy` (not an inline
`fixed_threshold:` / `fixed_frontier:` / `uniform_w4` spec) and `policy_barrier_timeout_s > 0`,
the first rollout records `calibration.policy_revision`; every later rollout calls
`wait_for_policy_revision(path, last_revision, timeout_s, poll_s=0.5)` (pure, in
`verl/workers/config/precision_scheduler.py`), which polls the file (tolerating a missing or
half-written file) until the revision exceeds the last one and raises `RuntimeError` naming the
path and the stale revision on timeout.

### Multi-experiment isolation

* `trainer.ray_master_port_range: "start:end"` (env fallback `VERL_RAY_MASTER_PORT_RANGE`) is
  parsed by `verl/utils/net_utils.py::parse_port_range()` (`0 < start < end <= 65536`) and passed
  to `RayWorkerGroup(master_port_range=...)`.
* `precision_scheduler.zmq_namespace` (wire: `VERL_ZMQ_NAMESPACE`) names the colocated
  weight-transfer socket; `weight_sync_namespace()` and `zmq_handle_for()` in
  `verl/workers/rollout/vllm_rollout/utils.py` are the single sanitizer used by the sender
  (`ServerAdapter`) and the receiver (`vLLMColocateWorkerExtension`), replacing the copy-pasted
  sanitizer whose drift would silently break the CUDA-IPC handshake.
* `precision_scheduler.force_shm_weight_transfer` (wire: `VERL_FORCE_SHM_WEIGHT_TRANSFER`,
  `parse_bool_env`) forces the shared-memory path.

Both are emitted by `to_vllm_env()` only when set, so vanilla still emits `{}`; the worker-side
`os.environ` readers are the wire format, and the `VERL_*` pass-through keeps hand-set launches
working.

### Sleep level

`precision_scheduler.sleep_level` (`null`, 1 or 2) overrides vLLM's sleep level in both
`vLLMHttpServer.sleep()` (COLOCATED, default 1) and `_sleep_hybrid()` (upstream rule: 1 for
MTP / LoRA adapters / NPU, else 2) through `resolve_sleep_level()`. With `enable: true` and
`sleep_level: null` level 1 is forced and logged; `enable: true` with `sleep_level: 2` is rejected
at config time because level 2 discards the resident INT4 shadow weights. Every archived run used
level 1.

### update_weights sub-timings

`ActorRolloutRefWorker.update_weights()` returns `{update_weights_materialize,
update_weights_load_merged}` on the LoRA-merge path; `CheckpointEngineManager` max-reduces the
per-rank dicts (`reduce_timing_dicts`) into the `last_update_timing` property, which the sync,
colocate-async and separate-async trainers merge into `timing_raw` after `update_weights`. The
manager's public return type is unchanged; custom manager classes without the property are
tolerated.

### Old-logprob entropy toggle

`actor.old_log_prob_calculate_entropy: false` skips entropy in the old-logprob pass (no
`entropy` field, no `actor/entropy` metric). Default `true` keeps upstream behavior.

### vLLM prompt-logprob extra ids (pending vLLM patch)

`docs/precision_scheduler/pending_vllm_patches/C8_prompt_logprob_extra_ids.diff` adds
`vllm/v1/sample/prompt_logprob_extra.py::inject_extra_prompt_logprobs()` (pure tensor helper, CPU
tested in `tests/v1/sample/test_prompt_logprob_extra_ids.py`), registers
`VLLM_PROMPT_LOGPROB_EXTRA_TOKEN_IDS` (default empty) in `vllm/envs.py`, and calls the helper
from the prompt-logprob path of `gpu_model_runner.py`. It applies cleanly to vLLM
`6bdabbad5b` (`git apply --check` verified) and is not wired to a YAML key because it is an
analysis-script knob (EOS-hazard / layer-sensitivity studies), not a training control.

## Knobs and defaults

See [`config.md`](config.md) for the complete table (precision-scheduler keys, trainer keys, and
the env variables that remain).

## Contracts with neighbors

* C1 / C2 / C4 / C5 / C7 (vLLM) read the env vars emitted by `to_vllm_env()`; the names are
  frozen in `ENV_BY_KEY` and pinned by `tests/workers/rollout/test_precision_scheduler_env_on_cpu.py`.
* C2 adds the env fallback for the sleep level in `vllm_async_server.py`; the config path here
  calls `resolve_sleep_level()` in the two sleep functions only.
* C5 / C6 consume the trace file by name and key (`generation_tokens`, `timestamp`,
  `finish_reason`, `token_ids`).
* C10 recipes set the YAML keys instead of exporting env vars; C10's `tools/validate_rollout_run.py`
  supersedes the minimal validator under `tests/special_e2e/precision_scheduler/`.

## Dropped and why

| Item | Reason |
|---|---|
| `RequestStepRecorder` (per-decode-step rows, stash) | superseded by the lifetime tracer; only live-monitoring aids read it |
| offline rollout replay, raw-string prompts, progress dump (stash) | decision 8; upstream `skip.rollout` covers replay |
| `VERL_CODEX_CUDA_PROFILE_UPDATE_ACTOR`, `old_log_prob_no_lora_adapter` (stash) | dev-only / never set by any launcher |
| nsys bridge thread and `VERL_VLLM_SERVER_NSYS_OPTIONS` | dropped with C4's scheduler counterpart |
| `FlexibleArgumentParser` import shim, `rollout.model_path` | version honesty (decision 11) / owned by C2 |
| vLLM decode profiler window, engine step timing, `request_lifecycle.py` | no launcher sets them; no in-tree consumer |
| `VLLM_DUAL_PRECISION_THRESHOLD`, `_DYNAMIC_POLICY`, `_ASYNC_*`, `VLLM_REPREFILL_ONLY_ROLLOUT`, `VLLM_MARLIN_INPUT_PADDING`, `VLLM_DUAL_PRECISION_NSYS_*` env forwarding | decisions 4, 5, 6; C9 owns Marlin padding |
| global `TMPDIR` forwarding as a feature | now pass-through only (needed by nsys temp dirs only) |

## Measured numbers and provenance

* Archived trace oracle: `/data/huanchen/verl/.codex-report/rl-workflow/raw/e2_24k_fullstep_traces/e2_24k_full_r3/bf16/request_lifetimes_replica000_node000.jsonl`
  (128 rows = 64 start + 64 finish); token-id variant under
  `/data/huanchen/verl/.codex-report/new-storyline-experiments/best_t8_no_reprefill_gpu7/attempts/bf16/set_02/seed_42/traces/`.
  Trimmed 50-row fixture committed under `tests/workers/rollout/fixtures/`.
* Stable-id oracle: `/data/huanchen/verl/.codex-report/rl-workflow/raw/gsm8k_cap16_stable_trace_traces/gsm8k_cap16_stable_20260707_100000/bf16_ours_lora_gmem075/request_steps_replica000.jsonl`
  (`request_id` uuid + `trace_request_id` `idx-12_0`); 6-row fixture committed.
* Metric-name oracle: `runs/b32_cap16384_tail8k_dynamic/metrics/rl_workflow_timing/*.jsonl` rows
  `{timing_s/gen, rollout_only/requests, rollout_only/response_tokens, critic/rewards/mean}` under
  `/data/huanchen/verl/.codex-report/new-storyline-experiments/dynamic_tail8k_heatmap_20260823/`.
* Sub-timing keys read by `.codex-report/rl-workflow/collect_gsm8k_thinking_pageablefix_result.py`
  (`timing_s/update_weights_materialize`, `timing_s/update_weights_load_merged`).
* GPU smoke (this branch, GPU 4, Qwen3.5-4B, GSM8K, `precision_scheduler.enable=false`,
  8 prompts x 4 samples, response cap 2048): see the section "Smoke result" below.

## Smoke result

PASSED (2026-09-11, GPU 4, through the fixed launcher, rc 0): Qwen3.5-4B, GSM8K, 8 prompts x 4
samples, response cap 2048, `precision_scheduler.enable=false`, tracing with token ids.
Marker `VERL_ROLLOUT_ONLY_COMPLETE step=1 requests=32 gen_seconds=95.52`;
`rollout_only/response_tokens=55646`, `critic/rewards/mean=0.625`; 64 trace rows (32 start + 32
finish); `validate_rollout_only_run.py` → `valid=true`. Run dir:
`/tmp/claude-1004/-data-huanchen-verl/eccbbdf5-0e92-4b83-9e1c-f3d6da52e976/scratchpad/c8rev/ps_smoke`.
Earlier attempts in slot B were killed by the launcher's host-wide `ray stop --force` (fixed in C0).

```
RUN_DIR=<run dir> scripts/precision_scheduler/env/run_gpu.sh --gpus 4 --timeout 1800 -- \
  bash tests/special_e2e/precision_scheduler/run_rollout_only_smoke.sh
python tests/special_e2e/precision_scheduler/validate_rollout_only_run.py <run dir> --expected-requests 32
```
