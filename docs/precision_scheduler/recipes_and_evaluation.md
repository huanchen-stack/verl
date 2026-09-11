# C10: datasets, recipes and long-run evaluation

## Purpose

The reproducible entry points of the precision-scheduler work on the verl side: the dataset builders
behind every archived experiment, the single reward module, the single-GPU FSDP2 driver and the four
recipe shapes (rollout-only, full step, continuous EMA, 100-step arm), the run validator, and the
100-step LoRA-accuracy protocol with its deterministic evaluator and summarizer. All of it replaces
launcher `case` statements, hand-exported environment variables and copy-pasted study scripts under
`.codex-report/` with YAML overrides, one `common.sh` and golden-tested Python.

## Mechanism

### Configuration is the override list (decision 9)

`examples/precision_scheduler/common.sh` builds every launch as a list of Hydra overrides of
`ppo_trainer`:

* **model overlay**: `hydra.searchpath=[file://examples/precision_scheduler]` plus
  `+models@_global_=<MODEL_KEY>` merges `models/<key>.yaml` (C9) *after* the base config and *before*
  command-line overrides, so a recipe's own overrides win over the overlay, which wins over the base
  defaults. Composition is verified on CPU for every overlay;
* **policy**: `ps_resolve_policy` turns `POLICY` into the `rollout.precision_scheduler.*` keys
  (table in the README; the vLLM wire format is C8's `to_vllm_env()`, documented in `config.md`).
  `bf16` sets only `enable=false`, so the vanilla path is byte-identical to upstream. Every W4 kind sets
  `validate_shadow=true` (`VALIDATE_SHADOW=0` opts out): the 2026-09-11 integration run had a dummy-loaded
  INT4 shadow (vLLM's shadow loader inherited verl's `load_format: dummy`) that every validator accepted
  (152/152 layers bound, lifecycle probe `exact=True`) and that only showed up as reward 0 and 62/64 cap hits
  in the generations; the cosine check of `validate_shadow` compares the bound shadow against its checkpoint
  at load and fails the launch there instead;
* **experiment name**: `trainer.experiment_name` defaults to `<MODEL_KEY>_<policy name>` where the policy
  name is derived from the policy *kind* (`bf16`, `uniform_w4`, `fixed_threshold_<N>`, `fixed_frontier_<K>`,
  `ema_<JSON basename without .json>`) and sanitized to `[A-Za-z0-9_.-]`; an explicit `EXPERIMENT_NAME` is
  refused outside that charset. The name is a file name (`metrics/<project>/<experiment>.jsonl`): the
  2026-09-11 integration run's first `.json` launch died seven minutes in, after engine init, because the
  policy path's `/` reached the FileLogger `open()`. `test_recipes_on_cpu.py` checks every recipe x policy,
  including a JSON path with characters outside the charset;
* **harness**: `request_trace_dir`, `request_trace_log_tokens`, `zmq_namespace`,
  `force_shm_weight_transfer`, `trainer.ray_master_port_range` (`47000 + 300 * gpu`),
  `trainer.stable_sample_uid`, `trainer.logger=[console,file]`, run-dir paths;
* **DRY_RUN=1** prints `OVERRIDE<TAB><override>` lines and exits 0: the unit tests compose exactly what
  a launch would compose. Every recipe passes extra positional arguments through as overrides.

`ps_launch` refuses to start without `CUDA_VISIBLE_DEVICES` (set by `run_gpu.sh`) or with GPU 1 in it
(decision 13), and `continuous_ema.sh` keeps its runner in the recipe's own process group (killing
descendants via `pgrep -P`) so the launcher's group kill and leftover check still cover the trainer.
`PS_DATA_ROOT` has no default (the README documents the layout); the `/data/huggingface` snapshot
defaults of the prep scripts are kept and documented.
The trainer is started with `cwd=$RUN_DIR/metrics` so upstream's `FileLogger` (which reads
`VERL_FILE_LOGGER_ROOT`, default `.`) writes `metrics/<project>/<experiment>.jsonl` without any
environment variable; `hydra.run.dir` goes to `logs/hydra`. The only exported variable is
`TORCHINDUCTOR_CACHE_DIR=$RUN_DIR/torchinductor_cache` (process hygiene: inductor artifacts hold
process-local CUDA handles; the archived driver comment explains a CUDA-graph profiling failure when
the cache was shared).

### The FSDP2 driver and the recipe shapes

`run_fsdp_fullstep.sh` is the archived `eos_hazard_fullstep_b64_cap16k/run_fsdp_fullstep.sh` with the
model resolution replaced by the overlay, the env block by `precision_scheduler.*` and the study
layout by `RUN_DIR`. Defaults are the archived ones (GRPO, KL loss 0.01 low-var, entropy off, LoRA 16/16,
temperature 1 / top-p 1 / top-k -1, seed 42, 8192 batched tokens, GMEM 0.50, bucket 4096 MB, fsdp2
size 1, ref offloaded). The four recipe shapes are thin wrappers:

| Recipe | Sets | Archived sources collapsed |
|---|---|---|
| `recipes/rollout_only.sh` | `trainer.rollout_only=true`, `rollout_only_steps=N`, `calculate_log_probs=False`, B = `INITIAL_BATCH` (64) with `max_num_seqs = B` | dynamic_tail8k `run_dynamic.sh`, hardmath `run_rollout_calibration.sh`, rl-workflow `run_best_t8_verl_rollout_gpu7.sh` |
| `recipes/full_step.sh` | 30 steps, cap 24576, `calculate_log_probs=True` | full_rl_policy_matrix `run_full_rl_policy.sh`, hardmath `run_training_config.sh`, bf16_learnability `run_config.sh` |
| `recipes/continuous_ema.sh` | C6 `cli.py watch-ema --initialize-only` -> watcher in background -> runner (own process group) with `reload_policy_each_rollout=true`, `POLICY=<policy.json>` and `TRAIN_BATCH_SIZE = INITIAL_BATCH / ROLLOUT_N` (the watcher's `--batch` is `INITIAL_BATCH`; `full_step.sh` reads `TRAIN_BATCH_SIZE`, so deriving it in the recipe keeps the watcher's cohort size equal to the rollout batch; non-divisible or disagreeing values are refused) -> poll every 2 s; if the watcher dies with `online_ema_state.json.completed_steps < steps` the runner group gets SIGINT then SIGKILL, `FAILED` is written and the recipe exits 1; on success `CONTINUOUS_EMA_COMPLETE` | dynamic_tail8k `run_continuous_ema30.sh`, full_rl_policy_matrix `run_ema_full_rl.sh`, hardmath `run_dynamic_ema_training.sh`, b32_16k `run_ema_fullstep_lane.sh` |
| `long_run/train_arm.sh` | 100 steps, `save_freq=10`, `max_actor_ckpt_to_keep=2`, `old_log_prob_calculate_entropy=false`, `resume_mode=auto`, optional `lora_adapter_path`, `save_initial_checkpoint` / `exit_after_initial_checkpoint`, up to `MAX_ATTEMPTS=3` attempts with `events.log` | hardmath `run_training_config.sh`, no_reprefill `run_job()` |

### Data

| Script | Reproduces | How verified |
|---|---|---|
| `data/prepare_gsm8k.py` | `no_reprefill_rl_100step_20260814/data/train.parquet` (2048 rows, token stats 30 / 204 / 69.66), `temporal_guard_120/datasets/manifest.json` (8 x 32 disjoint sets, seed 20260811) | prompts, ground truth, `extra_info.index`, token stats, `source_indices` per set; a 50-row fixture is committed |
| `data/prepare_bigmath.py splits` | `hardmath_.../candidate_data/olympiad_exact1_full/manifest.json` | parquet sha256 of train / calibration / validation / monitor / test alias / pilot_data, `eligible_unique_prompts = 7017` |
| `data/prepare_bigmath.py learnability` | `bf16_learnability_search_20260904/datasets/*/manifest.json` (bands 2-10, 4-18, 8-24 of 64) | parquet sha256 of train and validation |
| `data/prepare_eos_workloads.py render` / `materialize` | `eos_hazard_extensibility/manifests/rendered_prompts.json`, `eos_hazard_fullstep_b64_cap16k/data/<model>/<dataset>/manifest.json` | rendered-prompt sha256 and `prompt_token_max` for Qwen3.5-4B and Phi-4; materialized `row_ids` and template token match |
| `data/prepare_eos_workloads.py source` | `manifests/datasets.json` | needs the HF hub; `PS_RUN_SLOW=1` test only |

The GSM8K source of record is the QeRL JSONL export (`/data/huanchen/vllm/.codex-reports/rollout/datasets/
qerl_gsm8k_train_2048_rollout.jsonl`, `shuffle(seed=0)[:2048]` of `openai/gsm8k` train per its
`repo_faithful_note`). `--hf openai/gsm8k --hf-shuffle-seed 0` implements that selection with
`datasets.shuffle`, but the export has no generator script in either tree, so hub reproduction is not
golden-tested; the archived parquet is reproduced from the export.

### Rewards

`rewards.py` dispatches on `data_source` and returns `{"score", "accuracy"}`:
`openai/gsm8k` (last number of the final 2000 characters, Decimal-equal), `bigmath_math_verify` (last
`<answer>` re-boxed, Math-Verify, binary), `bigmath_qerl_search` / `clean_bigmath_learnability` (same plus a
0.1 format bonus when `</think>` and an answer tag exist), `eos_hazard/*` (substring). Re-scoring archived
rollout dumps of all four families reproduces the stored `score` exactly. The archived
`bf16_learnability_search_20260904` runs crashed on their own `clean_bigmath_learnability` source
(`ValueError: Unsupported data source` in every `driver.log`); the merged module accepts it.

### Validator

`tools/validate_rollout_run.py` (supersedes C8's `tests/special_e2e/precision_scheduler/validate_rollout_only_run.py`,
which now imports from it and keeps its function name and summary keys) checks trace cardinality and
ids, `token_ids` / `trace_request_id` presence, dump rows per step, the `VERL_ROLLOUT_ONLY_COMPLETE`
markers (rollout-only) or the timing keys (`--mode full_step`), FileLogger rows, OOM strings, the
optional INT4 binding proof (`lora_base_layers=N` and `int4_shadow_active=N`, 152 for Qwen3.5-9B) and
the `COMPLETE` marker. The archived `collect_best_t8_verl_rollout.py` asserted the same things for
the best-t8 matrix. The binding proof matches the message text only, so it holds at verl's default
`VLLM_LOGGING_LEVEL=WARN` now that vLLM logs the contract lines at WARNING (`telemetry_and_harness.md`).

### 100-step protocol (from the frozen hardmath `PROTOCOL.md`)

Question: does W4 rollout (uniform, or EMA-scheduled BF16 -> W4) change the learning quality of a
100-step LoRA RL run when actor optimisation stays BF16?

1. **Workload.** `prepare_bigmath.py splits --seed 20260825 --min-solve-rate 0.015625 --max-solve-rate 0.015625
   --sources olympiads,aops_forum,amc_aime,harp,omnimath` (train 4096 / calibration 256 / validation 1024 /
   monitor 256 = first 256 validation rows; 7017 eligible prompts). With 16 prompts per step and no shuffle,
   100 steps consume the first 1600 training prompts once.
2. **Shared step-0 LoRA.** One `train_arm.sh` run with `SAVE_INITIAL_CHECKPOINT=1 EXIT_AFTER_INITIAL_CHECKPOINT=1`
   (prints `VERL_INITIAL_CHECKPOINT_COMPLETE step=0`), then `evaluate_lora_patch.py export-adapter` writes the
   PEFT adapter; every arm loads it through `actor_rollout_ref.model.lora_adapter_path`.
3. **Arms.** `bf16`, `full_w4` (archived name `pure_w4`) and an EMA policy (`continuous_ema.sh RUNNER=full_step`
   with C6's watcher calibrated from BF16 and Tail-W4@8K rollout-only calibration traces), all with seed 42,
   B = 16 x 4, cap 24576, re-prefill off, 100 steps, checkpoints every 10 steps.
   Calibration-acceptance gate of the dynamic arm (PROTOCOL.md): BF16 calibration uses four 64-response
   rollouts; Tail-W4 uses the first four 64-response rollouts for fitting and a fifth, held-out rollout
   for validation. Acceptance requires the held-out cohort's mean suffix length *and* cap-survival fraction
   to fall inside the 95% nonparametric predictive-bootstrap intervals formed from the four fitting
   cohorts. On failure the held-out cohort stays held out; at most one predefined expansion may collect
   256 additional fitting responses plus a new disjoint 64-response held-out cohort, and dynamic training
   cannot start until that second validation passes.
4. **Evaluation.** Every checkpoint on the common BF16 base with `evaluate_lora_patch.py` (temperature 0, seed
   20260825, cap 8192, Wilson 95% CI): monitor (256) at steps 0, 10, ..., 100; validation (1024) at 0, 50, 100.
   The step-0 point loads the shared adapter through the LoRA path (enabling LoRA changes tie-breaking
   even with zero LoRA-B).
5. **Summary.** `summarize_runs.py --timed-steps 2:11 --required-steps '*=100'` for wall time, sampled tokens
   and per-step training reward; the evaluator summaries give the learning curves.
6. **Honesty rules carried over.** No arm is replaced after observing its curve; failed preflights are
   repaired and recorded. Sampling is not bitwise deterministic under async vLLM even with a fixed seed
   (archived: 20/64 vs 23/64 on identical inputs), so GPU-level comparisons are statistical; only
   manifests, policies and summaries are byte-golden.

## Knobs and defaults

Recipe variables and their archived defaults are listed in `examples/precision_scheduler/README.md`
("Recipes"). YAML keys touched: `actor_rollout_ref.rollout.precision_scheduler.{enable,policy,lora_fast_path,
lora_dual_stream,bf16_layers,reprefill,validate_lifecycle,validate_shadow,online_observations,reload_policy_each_rollout,
request_trace_dir,request_trace_log_tokens,zmq_namespace,force_shm_weight_transfer,int4_model}`,
`trainer.{rollout_only,rollout_only_steps,save_initial_checkpoint,exit_after_initial_checkpoint,
ray_master_port_range,stable_sample_uid,resume_mode,save_freq,max_actor_ckpt_to_keep}`,
`actor_rollout_ref.model.lora_adapter_path`, `actor_rollout_ref.actor.old_log_prob_calculate_entropy`.
All default off / vanilla in the YAML; the recipes set them explicitly.

## Contracts with neighbours

* **C8** owns the `precision_scheduler` block, the trainer keys, the trace file format and the markers;
  the recipes only set keys, the validator only reads the frozen names.
* **C9** owns `models/*.yaml`; the recipes select them by `MODEL_KEY` and override paths with
  `MODEL_PATH` / `INT4_MODEL_PATH`.
* **C6** owns `cli.py watch-ema` (`--run-dir --policy --steps --bf-trace --w4-trace --heatmap --alpha
  --downstream-slope --initialize-only`, state file `online_ema_state.json` with `completed_steps`);
  `continuous_ema.sh` depends only on that CLI and file.
* **C4** owns `tools/precision_scheduling/policies/build_static_policy.py` and, on the vLLM side, the
  policy spec strings `fixed_threshold:<t>` / `fixed_frontier:<K>` / `uniform_w4` / `<file>` that
  `ps_resolve_policy` emits.
* **C1 (vLLM)** owns the kernel-ablation driver (`tools/rollout_lora/run_kernel_ablation.sh`); the
  archived `plot_kernel_ablation.py` moved with it and has no verl copy.
* **C0** owns `activate.sh` / `run_gpu.sh`, the only environment the recipes assume.

## Dropped and why

| Item | Reason |
|---|---|
| Megatron driver (`rl-workflow/run_megatron_tp_live_fullstep.sh`) and `run_megatron_fullstep.sh` | Decision 10 (FSDP2 only). It is not a near-copy of the FSDP driver: Megatron LoRA target names (`linear_qkv`, `linear_fc1`, ... instead of the HF names in the overlays), mbridge / recompute / transformer-config blocks, the `fla` / `mamba_ssm` / `causal_conv1d` / TE shims of the dirty environment, and no way to smoke it in the clean environment. C9 keeps `convert_megatron_to_hf_target_modules` so a future port can derive the target lists from the same overlays. |
| eight policy-case wrappers, lane / queue / matrix scripts | collapsed into `POLICY` and the README table |
| `prepare_eurus_*.py`, `flexible_gsm8k_reward.py`, `run_best_t8_rollout_only.py`, hardmath report builders and gates | pre-storyline / duplicate / study-specific (digest verdicts) |
| `VERL_FILE_LOGGER_ROOT`, `HYDRA_FULL_ERROR`, `PYTORCH_CUDA_ALLOC_CONF`, `CUDA_DEVICE_MAX_CONNECTIONS` exports | not project knobs; the file logger is redirected by `cwd`, the others were never load-bearing (the activation script sets `CUDA_DEVICE_MAX_CONNECTIONS`) |
| `+engine_kwargs.vllm.scheduler_reserve_full_isl=False` | False in every final experiment; not a key of the clean vLLM |
| `tools/fit_downstream_slope.py` (digest proposal) | C6's `cli.py fit-downstream` already exists |
| `models.json` of HF ids (card) | superseded by C9's YAML overlays, which carry ids, revisions and LoRA targets |
| b32_16k `prepare_fullstep_data.py` `train_sha256` golden | its 360-row prompt files have a different row schema; `materialize --rows a:b` covers the slicing, the sha is not asserted |

## Measured numbers and provenance

* GSM8K archive: `no_reprefill_rl_100step_20260814/data/summary.json` (2048 rows, prompt tokens min 30, max 204,
  mean 69.66162109375; reproduced with the Qwen3.5-4B tokenizer, which the 27B default of the archived
  script shares).
* BigMath archive: `hardmath_lora_accuracy_100step_20260825/candidate_data/olympiad_exact1_full/manifest.json`
  (`source_sha256 ab7999cd...`, train sha `af7928b3...`); `bf16_learnability_search_20260904/datasets/*/manifest.json`.
* Summaries: `no_reprefill_rl_100step_20260814/efficiency_summary.json` and `efficiency_accuracy_summary.json`
  (15 configurations; every `efficiency_window.aggregates` reproduced exactly; the three B128 arms have
  more steps on disk now than when the archived summaries were generated, so their reward summaries are
  compared on the archived prefix).
* `clean_bigmath_learnability` (the 20260904 search crashed before writing dumps) and `eos_hazard/*`
  (the non-math workloads were screened in the vLLM-only harness, no verl dump) have no archived rollout
  dump: those two branches are unit-tested only.
* Reward dumps re-scored exactly: `no_reprefill.../runs/b64/tail_t8/rollouts/1.jsonl` (256 rows),
  `hardmath.../runs/train_pure_w4_exact1_auto_gate/rollouts/1.jsonl`, `eos_hazard_fullstep.../phi4_mini_reasoning/math500/bf16/main/rollouts/1.jsonl`,
  `bf16_learnability_search_20260827/runs/full_lr3_n8_nokl/rollouts/1.jsonl` (24 rows each for the Math-Verify families).

## Smoke results (2026-09-11, this branch, vLLM clean tree with C1/C2/C4/C5/C6/C8/C9 merged)

Both through `run_gpu.sh`, Qwen3.5-4B local snapshot, `POLICY=bf16` (`precision_scheduler.enable=false`),
GSM8K 8-row parquet from `prepare_gsm8k.py --train-size 8 --test-size 8`, 4 prompts x 4 samples:

* GPU 4, `recipes/rollout_only.sh TOTAL_STEPS=2 RESPONSE_CAP=2048 PROMPT_CAP=1024`: rc 0, `COMPLETE`;
  `validate_rollout_run.py --expected-requests 16 --steps 2 --require-complete` -> `valid=true`, 32 requests,
  55744 output tokens, `gen_seconds` 47.3 / 29.1, `critic/rewards/mean` 0.5 at step 2.
* GPU 5, `recipes/full_step.sh TOTAL_STEPS=1 SAVE_FREQ=1 RESPONSE_CAP=1024 PPO_MAX_TOKEN_LEN=4096
  trainer.save_initial_checkpoint=true`: rc 0, `VERL_INITIAL_CHECKPOINT_COMPLETE step=0`, `global_step_0` and
  `global_step_1` (35 GB total: FSDP shard + optimizer + HF config), `--mode full_step` validation `valid=true`;
  `evaluate_lora_patch.py --adapter .../global_step_0 --limit 8 --max-response-tokens 256` exported
  `actor/lora_adapter/{adapter_config.json,adapter_model.safetensors}` (12 Qwen3.5 targets) and wrote
  `bf16_step0.summary.json` (8 requests, accuracy 0 with the 256-token cap, Wilson [0, 0.324]).
* First attempt of both smokes failed with `AttributeError: module 'vllm.envs' has no attribute
  'VLLM_DUAL_PRECISION_POLICY'`: the C2/C4 merge landed in `/data/huanchen/vllm-clean` while the engine was
  importing (new `scheduler.py`, old `envs.py`). Re-running on the consistent tree passed.
* The same two smokes as pytest: `PS_SMOKE_GPU=4 pytest -m gpu_smoke -k rollout_only tests/precision_scheduler/gpu/test_recipes_gpu_smoke.py`
  -> 1 passed (460 s); `PS_SMOKE_GPU=5 ... -k full_step` -> 1 passed (602 s).
* `nvidia-smi --query-compute-apps` on GPUs 4 and 5 was empty after every run.
