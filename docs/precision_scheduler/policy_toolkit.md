# Precision-scheduler policy toolkit (`verl.experimental.precision_scheduler`)

Everything that *produces* a switching policy for the rollout precision scheduler lives
here: TPOT grids, response-length hazard tables with an online EMA, the downstream
token-linear regression, the global-search policy builder, the online watcher and the
plots. The package is numpy-only (no torch, no pandas); matplotlib is imported lazily.
The runtime that *consumes* the policy is vLLM (`vllm/v1/core/sched/precision_policy.py`,
component C5); the profiler that measures the TPOT grid is
`vllm/tools/precision_scheduler/tpot_heatmap.py` (documented in its README).

## Purpose and pipeline

```
tpot_heatmap.py  ──heatmap.json──┐
paired BF16 / full-W4 baseline   ├─> build_policy ──dynamic_policy.json (schema 6)──> vLLM scheduler
traces (128 requests each)  ─────┤        ▲                                              │
downstream slope (fit-downstream)┘        │ EMA on switch cohorts                        │
                                          └──── watch-ema <── switch_observations.jsonl ─┘
```

Once per RL step the watcher ingests the switch cohort (entry token + final length per
request that switched), updates the hazard table on the observed bins, rebuilds the whole
table and writes the policy with a bumped `policy_revision`; the scheduler reloads it
before the next rollout.

## Modules

| Module | Contents |
|---|---|
| `cost_model.py` | `PolicyGrid(step=250, cap=16384, batch=32, prompt_step=128, prompt_max=2048)` replaces the STEP/CAP/BATCH/PROMPTS module constants; `make_tpot_cache`, `trajectory_cost`, `trajectory_cost_grid`, `plan_cost`, `switched_plan_cost` (the pseudo code of the C5 card) |
| `hazard.py` | `HazardTable(risk, event, observed)`, `components(entries, finals, grid)`, `ema_update(old, new, alpha)` on observed bins only, `survival(table, start_index)` at bin start |
| `tpot_grid.py` | `TpotGrid` (heatmap.json or legacy Gen1 CSV; log2 interpolation with NaN masking), `matrix_payload` from profiler cells, `validate_heatmap` (median-speedup guard), legacy CSV writer |
| `policy_builder.py` | `build_decisions` (global search), `build_policy`, `fixed_frontier_policy`, `validate_policy`, `lookup`, `decisions_array`, `write_policy_atomic`, `load_policy` |
| `traces.py` | request-lifetime traces, completed-step counting, EngineCore id suffix resolution, switch-cohort rows, metrics rows |
| `calibration.py` | initial BF16 / W4 tables from paired baseline traces or explicit final lengths |
| `downstream_regression.py` | `fit_points`, `fit_split` (120-run), `fit_replay_points` (Megatron replay), `fit_from_metrics` (validation lstsq), `slope_from_metrics` (inline per-model slope) |
| `online_ema.py` | `OnlineEmaWatcher` (poll, gate, rebuild, atomic write, history) and `replay_cohorts` |
| `plots.py` | speedup heatmap, policy switch surface, survival curves, regression scatter |
| `replay_regression/` | `workloads.py` (token workloads from rollout dumps) and `worker.py` (Megatron `TrainingWorker` timing of old-logprob / ref / update; GPU tool) |
| `cli.py` | `build-policy`, `watch-ema`, `fit-downstream`, `grid`, `build-fixed-frontier` |

## The search problem (decision 2: global-search formulation only)

For frontier bins `F[i] = 250, 500, ..., cap-250`, prompt buckets `P[j] = 0, 128, ..., 2048`
and live batch `1..B`:

* hazard per bin `h_i = event[i] / risk[i]`; survival from frontier `f` at the *start* of
  each later bin `S[0] = 1, S[k] = prod_{j<k} (1 - h_{f+j})`;
* `cost(p, fi, pj, live, alive)` = `sum_k TPOT[p](P[pj] + F[fi+k], eff_k) * tokens_k / 1000 * nonempty_k`
  `+ slope * live * sum_k alive_k * tokens_k` with `nonempty_k = 1 - (1 - alive_k)^live`,
  `eff_k = clip(round(live * alive_k / nonempty_k), 1, B)`, `tokens_k = min(250, cap - F[fi+k])`;
* for every observed frontier `fi`: `stay = cost(bf16, fi, ., ., S_bf)`; for every
  candidate `fj >= fi`: `prefix = cost(bf16, fi, ., ., S_bf[:fj-fi])`, `reach = S_bf[fj-fi]`,
  `suffix = cost(w4, fj, ., ., survival(H_w4, fj) * reach)`; the cheapest candidate is
  committed only if strictly cheaper than `stay`; ties between candidates resolve to the
  earliest frontier (strict `<` while iterating upward).

The comparison is plan against plan (BF16 prefix + W4 suffix, downstream token cost
included) against staying BF16, never "is W4 faster right now".

## Policy JSON contract (schema 6)

Top level: `schema_version=6`, `description`, `scan_interval_tokens`, `arm_min_requests`,
`capture_max_batch`, `commitment_enabled`, `receding_horizon_lookup`,
`initial_rollout_batch`, `max_switch_live_batch`, `calibration{kind, ema_alpha,
policy_revision, reprefill, ...}`, `offline_cost_model{response_cap,
downstream_seconds_per_token, switch_overhead_seconds}`, `lookup_table{layout=
"frontier_major,prompt_bucket,live_batch", frontier_start, frontier_step, frontier_count,
prompt_bucket_start, prompt_bucket_step, prompt_bucket_count, live_batch_start,
live_batch_count, committed_frontiers}` where `committed_frontiers` is a flat integer list
(`0` = no switch) of length `frontier_count * prompt_bucket_count * live_batch_count`.
`validate_policy` checks exactly this structurally; the C5 loader in vLLM validates the
same keys (a cross-repo import is not possible from this test-suite).

The fixed-frontier baseline (`fixed_frontier_policy`, `build-fixed-frontier`) emits the
same shape with `receding_horizon_lookup=false` and
`calibration.fixed_response_frontier=K`; the runtime also accepts `fixed_frontier:K`
inline, so the generator exists for archival compatibility and forced-switch calibration.

## Knobs and defaults

| Knob | Default | Where |
|---|---|---|
| `step` (scan interval) | 250 tokens | `PolicyGrid`, `--step` |
| `cap` (response cap) | 16384 | `--cap` |
| `batch` (initial rollout batch = live axis = capture_max_batch = max_switch_live_batch) | 32 | `--batch` |
| `prompt_step` / `prompt_max` | 128 / 2048 (17 buckets) | `--prompt-step`, `--prompt-max` |
| `alpha` | 0.2 | `--alpha` (archived sweeps 0.0-0.2 justify 0.2) |
| `downstream_slope` | 0.0 | `--downstream-slope`; fit per model with `fit-downstream` |
| calibration requests | 128 per baseline | `--calibration-requests` |
| cohort gating | on | `--no-cohort-gate` ingests every resolvable cohort (HEAD watcher behaviour) |
| `steps` | 30 | watcher stops after that many completed rollouts |
| heatmap guard | median speedup must differ from 1.0 by >= 2% | `--skip-heatmap-guard` |

Environment contract with vLLM (wire format only; verl's config layer sets it):
`VLLM_DUAL_PRECISION_POLICY=<policy.json>` (or an inline spec), the scheduler appends
switch cohorts to `VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS` (`switch_observations.jsonl`
in the run directory by default), and `reload_policy_each_rollout` makes it re-read the
policy before every rollout. The watcher reads
`traces/request_lifetimes_replica000_node000.jsonl` written by the verl rollout server.

## Watcher protocol and fail-closed contract

1. `watch-ema --initialize-only` writes revision 0 from the paired baselines.
2. The rollout starts; the watcher polls (0.25 s) the trace and the cohort file.
3. When the number of completed rollouts changes, cohorts with `rollout_index <=
   completed` are replayed in order onto the base W4 table, the policy is rebuilt and
   atomically replaced (`policy_revision = completed_steps`), `online_ema_state.json`
   is replaced and a row is appended to `online_ema_history.jsonl`.
4. The rollout runner must abort if the watcher process dies (a stale policy would
   otherwise be reloaded silently); the watcher exits 0 only after `--steps` rollouts.

Fix relative to the archived watchers: `run_continuous_hazard_ema.py` at HEAD raised
`KeyError('kind')` in its default (forced-8K) mode because the calibration dictionary had
no `kind`; the calibration source now always carries `metadata["kind"]` and the policy
records it under `calibration.base_source`. Cohort gating on completed steps (from the
older watcher) makes revisions deterministic; the newest watcher ingested every
resolvable cohort regardless of timing.

## Measured numbers and goldens (provenance)

* **Qwen3.5-4B global-search EMA run** (`.codex-report/new-storyline-experiments/eos_hazard_extensibility/b32_16k_sensitivity/runs/qwen35_4b/gsm8k/ema_tail_w4/alpha020_ema30_global_search_gpu7_20260910/`):
  B32, cap 16384, alpha 0.2, slope 0.0003561462572540134 s/token, 20 cohorts, revision 30,
  35360 switch states. `build_policy` reproduces `dynamic_policy.json` bit-for-bit from the
  trimmed fixture (15 KB) and from the raw traces (4.7 s).
* **120-run sampled-token regression** (`evidence_bundle/latest_sampled_token_regression*`):
  inference 0.0003699166224487428 s/token + 14.602 s (r2 0.9759), training
  0.0005209837333221172 s/token + 28.509 s (r2 0.9685); reproduced to 1e-9.
* **Megatron replay regression** (`megatron_replay_regression/reports/regression.json`):
  n=9, inference 0.0003506424736647144 s/token (r2 0.99985), training 0.0005039208191747789
  (r2 0.9998); reproduced to 1e-9 including residual columns.
* **Validation downstream fit** (`heatmap_budget_batch_optimal_t_20260820_resume1/analysis/dynamic_policy_validation/artifacts/validation_summary.json`):
  intercept 12.481447564927427 s, slope 0.0008783918670449 s/token, r2 0.998779400507401
  over 320 pure bf16/full_w4 metric rows (caps 4096-24576, batches 16-128, steps 6-15).
* **Legacy Gen1 grid** (`experiment-results/qwen35_9b_tp1_tpot_grid.csv`): 59 ok + 5 oom
  cells regenerated from the `tail_w4_redo` JSONL rows (1.53x at batch 1 x 512 context).
* **Heatmap matrices** for the four valid rep5/corrected runs equal the archived
  `heatmap.json` (Phi-4-mini keeps 4 `null` INT4 cells from capacity failures); the
  2026-09-09 `c1_phi_math` heatmap (median speedup 1.0003) is rejected by the guard.

## Dropped and why

* The earlier EMA/hazard library under `dynamic_tail8k_heatmap_20260823`
  (`build_policies.py`, `simulate_online_hazard.py`, `run_continuous_hazard_ema.py`,
  the schema-4 tail8k builder, `build_forced8k_calibration_policy.py`, the warm5 builder)
  and its goldens: decision 2, the global-search formulation is the only one that ships.
  The docs record that the archived B32/B64 headline dynamic runs used tables from that
  earlier math; the clean branch reproduces the global-search runs instead.
* `simulate_online_ema.py` (scalar multiplicative correction): superseded.
* Study-specific analyses, slides (`python-pptx`) and hard-coded plot scripts.
* pandas (CSV pivot) in favour of the `csv` module; scipy was never used.

## Tests

```bash
python -m pytest -p no:cacheprovider -q tests/experimental/precision_scheduler   # 40 CPU tests
```

Goldens skip nothing: every oracle is a trimmed committed fixture; when the archive is
present the tests additionally rebuild from the raw traces.
