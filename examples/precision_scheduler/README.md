# Precision-scheduler recipes (C10)

Reproducible entry points for the rollout precision scheduler: dataset prep, the reward module, the
single-GPU FSDP2 driver, the recipe wrappers, the 100-step protocol tooling and the run validator.
Design description: [`docs/precision_scheduler/recipes_and_evaluation.md`](../../docs/precision_scheduler/recipes_and_evaluation.md).

Everything is configured through Hydra overrides of YAML keys (decision 9). The recipes export **no**
`VLLM_*` / `VERL_*` / `ROLLOUT_QLORA` variable (`tests/precision_scheduler/test_recipes_on_cpu.py`
pins that); the only environment they rely on is what `scripts/precision_scheduler/env/activate.sh`
exports and what the decision-13 launcher `run_gpu.sh` sets (`CUDA_VISIBLE_DEVICES`, `RAY_TMPDIR`).

## Layout

| Path | Purpose |
|---|---|
| `common.sh` | shared helpers: `ps_resolve_policy` (POLICY name -> `rollout.precision_scheduler.*` overrides), `ps_common_overrides` (model overlay, data, reward, tracing, host isolation, run-dir layout), `ps_init_run_dir` (STARTED / COMPLETE / FAILED markers, `run_config.json`), `ps_launch` (DRY_RUN or `python -m verl.trainer.main_ppo`) |
| `run_fullstep.sh` | the single-GPU (TP=1, DP=1) GRPO driver; `TRAINER=megatron` (default, the reporting trainer: Megatron-Core via Megatron-Bridge, LoRA through `model.lora.*`) or `TRAINER=fsdp2` (CPU compose tests only; never for GPU runs, decision 10); every training knob is a shell variable with the archived default |
| `run_megatron_fullstep.sh`, `run_fsdp_fullstep.sh` | two-line wrappers that pin `TRAINER` |
| `recipes/rollout_only.sh` | generation + reward only, N steps (`trainer.rollout_only=true`) |
| `recipes/full_step.sh` | full GRPO steps (rollout, old log-prob, ref, update, weight sync) |
| `recipes/continuous_ema.sh` | online EMA policy: C6 `watch-ema` sidecar + runner with `reload_policy_each_rollout=true`, fail-closed |
| `long_run/train_arm.sh` | one 100-step arm: periodic checkpoints, shared step-0 LoRA, resume-from-latest, bounded attempts, `events.log` |
| `long_run/evaluate_lora_patch.py` | deterministic held-out evaluator (greedy, fixed seed, Wilson 95% CI); exports a PEFT adapter from an FSDP checkpoint |
| `long_run/summarize_runs.py` | timing aggregates over a warm-up-excluded window + per-step reward curves (archived schema) |
| `data/prepare_gsm8k.py` | GSM8K parquet (QeRL export or HF), slices and disjoint seeded sets, token stats |
| `data/prepare_bigmath.py` | Big-Math difficulty-band splits (`splits`, `learnability`) with sha256 manifests |
| `data/prepare_eos_workloads.py` | 8-workload pipeline `source -> render -> materialize` with sha256 manifests |
| `rewards.py` | one `compute_score` dispatching on `data_source` |
| `models/*.yaml` | per-model overlays (C9): checkpoint pair, LoRA targets, thinking flag, model-specific knobs |
| `../../tools/precision_scheduling/policies/build_static_policy.py` | static baseline policies (C4; frontier / live_threshold / forced_switch) |
| `../../tools/validate_rollout_run.py` | post-run acceptance checks (trace cardinality, dump rows, markers, metrics, OOM, INT4 binding proof) |

## Environment

```bash
export VERL_ROOT=/path/to/verl            # before sourcing
source scripts/precision_scheduler/env/activate.sh clean   # vLLM clean tree + $VERL_ROOT on PYTHONPATH
scripts/precision_scheduler/env/check_env.py --pick-gpus 1
```

Every GPU command runs under `scripts/precision_scheduler/env/run_gpu.sh --gpus <N> --timeout <s> -- <cmd>`
(process-group kill, private Ray temp dir, leftover check); `ps_launch` refuses to start without
`CUDA_VISIBLE_DEVICES` or with GPU 1 in it. Details: `scripts/precision_scheduler/env/SETUP.md`.

## Data

Default `DATA_DIR` is `$PS_DATA_ROOT/gsm8k_messages_2048`; `PS_DATA_ROOT` has no default (export it, e.g.
`/data/huanchen/ps_data` on the measurement host, or set `DATA_DIR`). Nothing under it is committed;
regenerate with the commands below. The prep scripts default to the local hub snapshots under
`/data/huggingface/hub` (`prepare_bigmath.py --source`, `prepare_eos_workloads.py --hub-root`), override
them on another host:

```bash
Q=Qwen/Qwen3.5-4B   # or a local snapshot; only used for the token statistics
# GSM8K, 2048 training rows + 32 test rows (reproduces the archived no_reprefill parquet: 30/204/69.66 tokens)
python examples/precision_scheduler/data/prepare_gsm8k.py --input /data/huanchen/vllm/.codex-reports/rollout/datasets/qerl_gsm8k_train_2048_rollout.jsonl \
    --output-dir /data/huanchen/ps_data/gsm8k_messages_2048 --tokenizer $Q --train-size 2048 --test-size 32
#   (or --hf openai/gsm8k --hf-shuffle-seed 0 --hf-limit 2048 to rebuild from the hub; see the design doc for the caveat)
# 8-row smoke set
python examples/precision_scheduler/data/prepare_gsm8k.py --input ... --output-dir /data/huanchen/ps_data/gsm8k_smoke_8 --tokenizer $Q --train-size 8 --test-size 8
# eight disjoint 32-row guard sets (temporal_guard_120)
python examples/precision_scheduler/data/prepare_gsm8k.py --input ... --output-dir /data/huanchen/ps_data/gsm8k_guard_sets --tokenizer $Q --disjoint-sets 8x32 --selection-seed 20260811
# BigMath (hardmath 100-step protocol; reproduces candidate_data/olympiad_exact1_full/manifest.json)
python examples/precision_scheduler/data/prepare_bigmath.py splits --output-dir /data/huanchen/ps_data/bigmath_olympiad_exact1 \
    --seed 20260825 --min-solve-rate 0.015625 --max-solve-rate 0.015625 --sources olympiads,aops_forum,amc_aime,harp,omnimath
# BigMath learnability bands (2/64..10/64 etc.)
python examples/precision_scheduler/data/prepare_bigmath.py learnability --output-dir /data/huanchen/ps_data/bigmath_band_2_10 --band 2 10
# EOS workloads (needs the HF hub for `source`)
python examples/precision_scheduler/data/prepare_eos_workloads.py source --out /data/huanchen/ps_data/eos/datasets --datasets gsm8k math500
python examples/precision_scheduler/data/prepare_eos_workloads.py render --source .../datasets/gsm8k.jsonl --out .../rendered/qwen3_5_4b/gsm8k.jsonl --tokenizer $Q --enable-thinking
python examples/precision_scheduler/data/prepare_eos_workloads.py materialize --source .../datasets/gsm8k.jsonl --rendered .../rendered/qwen3_5_4b/gsm8k.jsonl \
    --out /data/huanchen/ps_data/eos/qwen3_5_4b/gsm8k --tokenizer $Q --enable-thinking --dataset gsm8k --rows 0:64
```

## Policies

`POLICY` selects the `rollout.precision_scheduler` block (decision 6 spec strings):

| `POLICY` | `enable` | `policy` | LoRA fast path / dual stream |
|---|---|---|---|
| `bf16` (default) | false | — | off (vanilla vLLM LoRA) |
| `full_w4` | true | `uniform_w4` | on |
| `tail_t<N>` | true | `fixed_threshold:<N>` (switch when the live batch drains to N) | on |
| `fixed_k<K>` | true | `fixed_frontier:<K>` (switch every request at K response tokens) | on |
| `<file>.json` | true | that lookup-table policy (static builder or C6 EMA) | on |

Every W4 kind also sets `bf16_layers=none`, `reprefill=false`, `validate_lifecycle=true`,
`validate_shadow=true` (`VALIDATE_SHADOW=0` turns it off) and `online_observations=$RUN_DIR/switch_observations.jsonl`
(the archived final-recipe values plus the shadow check). The INT4 shadow comes from the model overlay
(`INT4_MODEL_PATH` overrides it with a local snapshot).
Static baseline files: `tools/precision_scheduling/policies/build_static_policy.py --kind {frontier,live_threshold,forced_switch}`.

## Recipes

```bash
export RUN_DIR=/data/huanchen/ps_runs/demo MODEL_KEY=qwen3_5_4b MODEL_PATH=<local snapshot>   # MODEL_PATH optional
# rollout-only, B=64 (16 prompts x 4), cap 24576, 2 steps
POLICY=tail_t8 INITIAL_BATCH=64 RESPONSE_CAP=24576 TOTAL_STEPS=2 \
  scripts/precision_scheduler/env/run_gpu.sh --gpus 4 --timeout 7200 -- bash examples/precision_scheduler/recipes/rollout_only.sh
python tools/validate_rollout_run.py $RUN_DIR --expected-requests 64 --steps 2 --require-complete
# full RL, 30 steps
POLICY=full_w4 TRAIN_BATCH_SIZE=16 TOTAL_STEPS=30 bash examples/precision_scheduler/recipes/full_step.sh
# continuous EMA (C6 watcher sidecar; baseline traces + heatmap from the profiling toolkit); the runner
# also gets precision_scheduler.policy_barrier_timeout_s=${POLICY_BARRIER_TIMEOUT_S:-600} (static policies keep 0).
# INITIAL_BATCH (64) is the single batch knob for both RUNNER kinds: the watcher gets --batch INITIAL_BATCH and
# the runner TRAIN_BATCH_SIZE=INITIAL_BATCH/ROLLOUT_N (must divide; a disagreeing TRAIN_BATCH_SIZE is refused)
POLICY_PATH=$RUN_DIR/policy.json BF_TRACE=... W4_TRACE=... HEATMAP=... TOTAL_STEPS=30 RUNNER=rollout_only \
  bash examples/precision_scheduler/recipes/continuous_ema.sh
# any recipe: DRY_RUN=1 prints the override list; extra arguments are appended as Hydra overrides
DRY_RUN=1 POLICY=fixed_k8000 bash examples/precision_scheduler/recipes/full_step.sh trainer.total_training_steps=5
```

Knobs of `run_fullstep.sh` (shell variables, archived defaults): `TRAINER=megatron TRAIN_BATCH_SIZE=16 ROLLOUT_N=4
RESPONSE_CAP=16384 PROMPT_CAP=2048 MAX_MODEL_LEN=cap+prompt PPO_MAX_TOKEN_LEN=18432 GMEM=0.50 TOTAL_STEPS=4
SAVE_FREQ=-1 ACTOR_LR=1e-6 USE_KL_LOSS=True ENFORCE_EAGER=false WEIGHT_BUCKET_MB=4096 ROLLOUT_SEED=42
MAX_CKPT_TO_KEEP=2 USE_FUSED_KERNELS=true USE_DYNAMIC_BSZ=true GRADIENT_CHECKPOINTING=true
ACTOR_PARAM_OFFLOAD=false CALCULATE_LOG_PROBS=True PORT_BASE=47000+300*gpu RUN_TIMEOUT=12h`; Megatron only:
`RECOMPUTE_GRANULARITY=full RECOMPUTE_METHOD=uniform RECOMPUTE_NUM_LAYERS=1 ATTENTION_BACKEND=auto
ACTOR_GRAD_OFFLOAD=false ACTOR_OPTIMIZER_OFFLOAD=false`. The Qwen3.5 overlays carry the Megatron LoRA block
(`model.lora.target_modules` in mcore names); Phi-4-mini and Gemma4 have no Megatron-Bridge mapping yet; per decision 10 they
are not run on FSDP2 — a bridge is written first (C9). `TRAINER=fsdp2` exists for the CPU compose tests only.
`rollout_only.sh` sets `INITIAL_BATCH=64` (-> `TRAIN_BATCH_SIZE=INITIAL_BATCH/ROLLOUT_N`), cap 24576,
`CALCULATE_LOG_PROBS=False`; `full_step.sh` sets 30 steps and cap 24576.

Model overlays are selected with `hydra.searchpath=[file://examples/precision_scheduler]` plus
`+models@_global_=<MODEL_KEY>`; see `models/README.md` for the per-model settings.

Run directory contract: `STARTED` / `COMPLETE` / `FAILED`, `run_config.json`, `driver.log`,
`metrics/<project>/<experiment>.jsonl` (FileLogger; the trainer runs with `cwd=$RUN_DIR/metrics`;
`<experiment>` defaults to `<MODEL_KEY>_<policy name>` with the policy name `bf16` / `uniform_w4` /
`fixed_threshold_<N>` / `fixed_frontier_<K>` / `ema_<policy JSON basename>` sanitized to `[A-Za-z0-9_.-]`, so a
policy path never becomes part of a file name; an explicit `EXPERIMENT_NAME` must match the same charset),
`rollouts/<step>.jsonl`, `traces/request_lifetimes_replica000_node000.jsonl`, `checkpoints/global_step_<k>/`,
`switch_observations.jsonl` (W4 policies), `logs/hydra/`.

## 100-step protocol

See the design doc, section "100-step protocol". In short:

```bash
# 1. shared step-0 LoRA (one run, exits right after the checkpoint)
SAVE_INITIAL_CHECKPOINT=1 EXIT_AFTER_INITIAL_CHECKPOINT=1 RUN_DIR=$ROOT/step0 POLICY=bf16 DATA_DIR=$BIGMATH bash examples/precision_scheduler/long_run/train_arm.sh
python examples/precision_scheduler/long_run/evaluate_lora_patch.py export-adapter $ROOT/step0/checkpoints/global_step_0
# 2. arms (bf16 | full_w4 | <ema policy>), 100 steps, checkpoint every 10
for arm in bf16 full_w4; do RUN_DIR=$ROOT/$arm POLICY=$arm DATA_DIR=$BIGMATH TOTAL_STEPS=100 SAVE_FREQ=10 \
  INITIAL_LORA_ADAPTER_PATH=$ROOT/step0/checkpoints/global_step_0/actor/lora_adapter bash examples/precision_scheduler/long_run/train_arm.sh; done
# 3. held-out evaluation of every checkpoint on the common BF16 base
python examples/precision_scheduler/long_run/evaluate_lora_patch.py --base-model $BASE --data $BIGMATH/monitor.parquet \
  --adapter $ROOT/bf16/checkpoints/global_step_10 --checkpoint-step 10 --configuration bf16 --output $ROOT/eval/bf16_step10.jsonl
# 4. training-side summary
python examples/precision_scheduler/long_run/summarize_runs.py --runs $ROOT --glob '*' --config-id '{policy}' --timed-steps 2:11 --required-steps '*=100' --output $ROOT/summary.json
```

## Tests

| Test | Tier | What |
|---|---|---|
| `tests/precision_scheduler/test_recipes_on_cpu.py` | unit + golden | DRY_RUN override lists of every recipe compose on `ppo_trainer` and validate; per-policy `precision_scheduler` block; every overlay; no env exports; `continuous_ema.sh` fail-closed with stubs; archived `run_config.json` fixture |
| `tests/precision_scheduler/test_data_prep_golden.py` | golden | GSM8K (50-row committed fixture; full archive), BigMath sha256 manifests, EOS render sha256 / materialize row ids |
| `tests/precision_scheduler/test_rewards_and_summaries_golden.py` | unit + golden | reward cases; re-scoring of archived dumps; summarizer vs archived `efficiency_summary.json` |
| `tests/precision_scheduler/gpu/test_recipes_gpu_smoke.py` | gpu-smoke | 2-step rollout-only; 1 full step + initial checkpoint + evaluator |
| `tests/special_e2e/precision_scheduler/` | gpu-smoke (C8) | one rollout-only step; its validator delegates to `tools/validate_rollout_run.py` |

## Kernel ablation drivers

They live on the vLLM side (C1): `tools/rollout_lora/run_kernel_ablation.sh` in the vLLM tree drives
`fixed_token_bench.py` over the 3 backends x 4 kernels; the archived plot generator
`.codex-report/experiment-results/plot_kernel_ablation.py` moved with it (no verl copy).

## Dropped launchers (what they did)

Lane, queue and matrix scripts were host- and date-specific orchestration; the matrices they encoded
are recorded here and in the archived `run_config.json` / manifests. None is ported.

| Archived script | What it did | Replacement |
|---|---|---|
| `dynamic_tail8k_heatmap_20260823/run_all_dynamic.sh`, `run_all_continuous_ema30.sh`, `run_gpu7_continuous_ema30_queue.sh`, `finish_pipeline.sh` | GPU lanes 0/2/3/4/5/6 x batch {128,64,32} x cap {24576,16384}; rollout-only dynamic and continuous-EMA cells | `recipes/rollout_only.sh`, `recipes/continuous_ema.sh`, one invocation per cell |
| `full_rl_policy_matrix_b64_cap24k_20260902/run_matrix.sh`, `run_ema_full_rl.sh` | 8-GPU matrix of bf16 / full_w4 / fixed_t{2,4,8} / fixed_k{6000,8000,10000} / ema, B64 cap 24576, 30 steps | `recipes/full_step.sh` with `POLICY`, static JSONs from `build_static_policy.py` |
| `eos_hazard_fullstep_b64_cap16k/run_priority_lane.sh`, `run_phi_lane.sh`, `run_aux_lane.sh`, `resume_orphaned_preflight.sh` | model x dataset x {bf16, full_w4, tail_t8} cells (B16 x 4, cap 16384, 4 steps) for the extensibility zoo | `run_fsdp_fullstep.sh` with `MODEL_KEY` / `DATA_DIR` |
| `eos_hazard_extensibility/b32_16k_sensitivity/{launch_ema6_fullstep.sh,launch_ema_gpu023.sh,run_ema_heatmap_lane.sh,run_ema_fullstep_lane.sh}` | heatmap lane (vLLM profiler) then EMA full-step lane with the downstream-slope polyfit | C6 `cli.py fit-downstream` + `recipes/continuous_ema.sh RUNNER=full_step` |
| `rl-workflow/run_no_reprefill_rl_100step.sh` | 15 configs (b32/b64/b128 x bf16/full_w4/tail_t*), three 100-step B128 arms with retry/resume | `long_run/train_arm.sh` |
| `rl-workflow/run_heatmap_budget_batch_optimal_t.sh`, `prepare_heatmap_budget_batch_sweep.py` | 72-cell cap x batch x policy sweep with LPT lane packing | not ported (one-off) |
| `rl-workflow/run_best_t8_verl_rollout_gpu7.sh` | 4 guard sets x 3 seeds x 5 policies rollout-only matrix | `prepare_gsm8k.py --disjoint-sets` + `recipes/rollout_only.sh` |
| `hardmath_lora_accuracy_100step_20260825/{launch_main_runs.sh,run_evaluation_matrix.sh,continue_*.sh,run_prelaunch_tests.py,...}` | protocol gates, evaluation matrix over steps 0..100, report builders | `long_run/train_arm.sh`, `long_run/evaluate_lora_patch.py` (loop in the design doc) |
| `rl-workflow/run_megatron_tp_live_fullstep.sh` and every Megatron wrapper | the Megatron TP1 driver behind the headline Qwen3.5-9B runs | being restored as `run_megatron_fullstep.sh` (decision 10 reversed 2026-09-16); until it lands, `full_step.sh` still execs the FSDP2 driver and its numbers are not reportable |
| `rl-workflow/prepare_eurus_*.py`, `data/eurus*` | Eurus/27B timing workloads (pre-storyline) | dropped |
