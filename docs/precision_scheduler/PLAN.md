# Rollout Precision Scheduler Clean Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan component-by-component. Each component below becomes its own executable sub-plan (bite-sized TDD steps) generated immediately before that component starts, after the user approves this master plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-implement every feature of the Tail-W4 / precision-scheduling work as a clean, tested branch named `rollout-precision-scheduler-clean` in both repos, with an equivalence test and a design description per component, and nothing from the report directories except curated tooling.

**Architecture:** vLLM owns the runtime (LoRA fast path, dual-precision residency, precision-keyed CUDA graphs, scheduler-side switching, re-prefill). verl owns the training harness and the offline/online policy toolkit (profiling grids, hazard EMA, downstream regression, policy builder, recipes, datasets, long-run evaluation). A torch-free policy module in vLLM defines the policy JSON contract both sides share.

**Tech Stack:** vLLM fork at vanilla upstream `6bdabbad5b` (May 31) with precompiled cu130 kernels; verl fork at `2390a3f5` (June 26); conda env `/data/huanchen/miniforge3/envs/vllm` (torch 2.11+cu130, TE 2.13 with local patch, megatron-core 0.18); numpy-only for the policy toolkit; pytest.

**Spec:** The author's storyline (this conversation, 2026-09-11) plus `.codex-report/new-storyline-experiments/PREDICTION_BASED_DYNAMIC_SWITCHING_HANDOFF.md` sections 3-8 and 11-12. The twelve component inventories that ground this plan are in the session scratchpad under `inv/`.

## Global Constraints

- Branch name in both repos: `rollout-precision-scheduler-clean`.
- vLLM branch base: vanilla `6bdabbad5b`. verl branch base: `2390a3f5` (the measurement base). No rebase onto newer upstream in this effort.
- No change under vLLM `csrc/`, `cmake/`, or `CMakeLists.txt`. The precompiled `.so` payload must stay valid.
- Report directories (`.codex-report/`, `.codex-reports/`, `evidence_bundle/`) never enter a commit. Both branches add them to `.gitignore` in the first commit.
- Every migrated feature ships with: a CPU unit test, a golden or GPU equivalence test where an oracle exists, and a design description under `docs/design/` (vLLM) or `docs/precision_scheduler/` (verl).
- Every commit ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. One component per commit series; no cross-component commits.
- Test runs never write into the repo (`PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`) until the branch is committed.
- GPU tests run only on GPUs 0 and 2 through 7 that a preflight finds unoccupied, under a launcher that kills its process group and verifies the GPUs are clean afterwards (decision 13).

---

## 1. Components at a glance

Eleven components. Sizes are approximate lines of source to port or rewrite, excluding tests.

| # | Component | Repo | LOC to port (approx., excl. tests) | Verdict mix | Wave |
|---|---|---|---|---|---|
| C0 | Environment, worktrees, honest version | host + both | ~300 (scripts) | refactor shims | 0 |
| C1 | LoRA kernel fast path | vLLM | ~600 | copy + refactor | 1 |
| C2 | Dual-precision residency and binding | vLLM (+80 verl) | ~800 | refactor, split file | 1 |
| C5 | Switching policy module (torch-free) | vLLM | ~500 | move + refactor, drop 3 modes | 1 |
| C3 | Precision-keyed CUDA graphs and dispatch | vLLM | ~250 | refactor, drop V2 runner | 2 |
| C4 | Scheduler switching runtime | vLLM | ~450 | extract state machine, drop ~650 | 2 |
| C7 | Re-prefill after switch (ablation) and the reuse-vs-reprefill NLL study | vLLM + verl example | ~120 runtime + ~800 study | copy runtime, drop 2 knobs; curate the NLL study scripts | 3 |
| C8 | verl telemetry and experiment harness | verl (+60 vLLM) | ~500 | refactor into helpers | 1 |
| C9 | Model support | both | ~450 | copy, flag-gate Gemma4/Nemotron | 2 |
| C6 | Profiling and policy-building toolkit | verl pkg + vLLM tools | ~2,500 | consolidate 3 generations | 3 |
| C10 | Datasets, recipes, long-run evaluation | verl examples | ~1,500 | collapse 8 launchers to 5 | 4 |

Coverage check: every hunk in the verl working tree, the verl stash, the vLLM commit `34e66a3`, the vLLM working tree, and the untracked files is assigned above or listed in section 7 as dropped.

---

## 2. Ground rules

### 2.1 Branch and worktree mechanics

```bash
# vLLM: clean worktree from vanilla main, plus a vanilla baseline worktree
git -C /data/huanchen/vllm worktree add -b rollout-precision-scheduler-clean /data/huanchen/vllm-clean 6bdabbad5b
git -C /data/huanchen/vllm worktree add --detach /data/huanchen/vllm-vanilla 6bdabbad5b

# verl: worktree of the fork clone from the measurement base (keeps main checked out)
git -C /data/huanchen/myverl/verl worktree add -b rollout-precision-scheduler-clean /data/huanchen/verl-clean 2390a3f5
```

The dirty checkouts `/data/huanchen/vllm` and `/data/huanchen/verl` are never modified. They remain the reference for side-by-side equivalence runs.

### 2.2 Review protocol

One component at a time, in wave order. For each component:

1. I generate the component sub-plan (bite-sized TDD steps with real code) and show it to you.
2. On your approval, a subagent implements it in the clean worktree: tests first, then code, then the design doc.
3. A second subagent reviews the diff against the sub-plan and the equivalence criteria.
4. I summarize the result, test output, and any deviations. You approve or request changes.
5. Commit. Next component.

Components in the same wave that touch disjoint files run in parallel subagents, each in its own git worktree, then merge in wave order.

### 2.3 Test tiers and what "equivalent" means

| Tier | Runs where | Equivalence criterion |
|---|---|---|
| unit | CPU, every commit | Behavior specified by the sub-plan; ported from existing tests where they exist |
| golden | CPU, skips if archive absent | Byte-equal or 1e-9 numeric match against archived policies, regressions, manifests, log-derived tables |
| gpu-smoke | 1 GPU, tiny model, on request | Kernel presence, counts, log-line contracts, tolerance-based numerics |
| e2e | 1 to 4 GPUs, on request | Archived-run parity on cardinalities and metric names; timing within stated tolerance |

Golden fixtures larger than about 1 MB stay in the archive and are referenced by absolute path with a skip. Small derived fixtures are committed under the test tree.

### 2.4 Descriptions

Each component gets one design document with: purpose, mechanism, the exact knobs and their defaults, the contract with neighboring components, what was dropped from the experimental code and why, and the measured numbers with provenance.

---

## 3. Decisions I need from you

Numbered so you can answer by number. My recommendation is stated for each.

1. **LoRA dual-stream under dual precision.** There are two custom ops in `base_linear.py`. `lora_linear_async` is the dual-stream op: base GEMM on the main stream, LoRA GEMMs on the aux stream. `dual_precision_base_linear` only selects the BF16 or INT4 base and runs that GEMM; LoRA is then applied synchronously on the same stream. `apply()` checks the dual-precision branch first, so with `VLLM_DUAL_PRECISION_ROLLOUT=1` the dual-stream op is never reached. The measured dual-stream gain (ablation stage 4) exists only with dual precision off. Recommendation: preserve that behavior in the clean branch and document it. Making dual-stream active under dual precision is a follow-up that needs re-measurement. **Accepted 2026-09-11.** Author adds: the custom-op dual stream (`lora_linear_async`) is the intended design; dual-stream code written before the custom-op approach may be abandoned. C1 therefore drops the pre-custom-op duplicate path (`_apply_base_forward_async`) instead of unifying it, pending confirmation of which pieces count as legacy.

2. **Policy kinds to keep.** Three policy kinds ship as first-class, all through the same lookup-table contract and the same cohort logging. Heuristic baselines: fixed generation frontier (switch every request at K response tokens, e.g. K=8000) and fixed live-batch threshold (switch when live requests drop to t). Main line: the EMA-based policy, calibrated from the profiler heatmap and updated online from switch cohorts. One sub-question remains for the EMA builder only: two EMA builder implementations exist with different math and both have golden oracles, the one behind the headline B32/B64 runs and the newer global-search one behind the Phi-4 and Qwen3.5-4B runs. **Decided 2026-09-11: global-search only.** The earlier EMA math is dropped entirely; this part is still active research and only the global-search formulation moves forward. C6 loses the legacy mode and its two goldens; the heuristic baselines stay.

3. **Headline dynamic policies.** **Moot after decision 2.** That builder family is dropped. The docs will record that the archived B32/B64 dynamic runs used tables from the earlier math, and that the clean branch reproduces the global-search runs (Phi-4, Qwen3.5-4B) instead.

4. **Dropped runtime modes.** Drop the async cost-model predictor, the in-scheduler online cost model, the lookup hierarchy, and the `switch_thresholds` table mode. Keep the cost model as a pure offline function. Recommendation: drop all four; the profiler's forced-switch use of `switch_thresholds` becomes an explicit `fixed_switch_frontier` field with cohort-free arming. **Accepted 2026-09-11.** The online behavior is unchanged: the EMA rebuilds the table after every RL step, and the scheduler re-indexes it at every 250-response-token frontier. The cost model survives as the offline oracle the table builder is tested against.

5. **Re-prefill.** Keep as a default-off ablation with the committed semantics, drop `VLLM_REPREFILL_ONLY_ROLLOUT`, and drop the dead `num_visible_output_tokens` fold branch. Recommendation: yes to all three. **Accepted 2026-09-11.** Default off; kept only for the minor ablation study.

6. **One policy flag for all policy kinds.** Today fixed threshold t is an integer env var handled by its own latch code, while fixed frontier K and the EMA table are JSON files passed through a second env var and handled by the lookup state machine. The fixed-t path never wrote switch cohorts, so the full-RL matrix emulated it as a degenerate lookup table. **Accepted 2026-09-11.** One flag, `VLLM_DUAL_PRECISION_POLICY`, accepting an inline spec (`fixed_threshold:8`, `fixed_frontier:8000`, `uniform_w4`) or a path to an EMA policy JSON. All kinds run through the same switcher and write the same cohort JSONL; the separate threshold and dynamic-policy env vars go away.

7. **Model support tiers.** First-class: Qwen3.5-9B, Qwen3.5-4B, Phi-4-mini-reasoning (config-only plus the `load_format=auto` note). **Amended 2026-09-16:** every first-class model must train on Megatron TP1 (decision 10); Phi-4-mini needs a Phi3 Megatron-Bridge (Llama-like layout, fused `qkv_proj`/`gate_up_proj`, partial rotary + LongRoPE) under C9 before its RL-step numbers are produced. Behind a flag: Gemma4 dense FFPA path, Nemotron-H hooks and Marlin padding. Dropped: Qwen3.5-27B and everything else in the extensibility zoo. **Amended 2026-09-11:** 27B is no longer needed. The Qwen3.5 GatedDeltaNet LoRA target mapping from the stash still ships, because every Qwen3.5 size has those layers and the headline 9B runs used it under Megatron; only the 27B model, its TP4 configs, and the 27B kernel-ablation launcher are dropped.

8. **verl stash contents.** Port: Qwen3.5 LoRA target mapping, tensordict jagged fallback, stable trace id as trace metadata, optional old-logprob entropy toggle. Drop: offline rollout replay, raw-string prompts, progress dump, update-actor profiler toggle, no-adapter old-logprob knob. Recommendation: as stated. **Accepted 2026-09-11.**

9. **YAML-first configuration, verl and vLLM settings alike.** Make `rollout_only`, `save_initial_checkpoint`, `exit_after_initial_checkpoint`, and `ray_master_port_range` real config keys with defaults instead of `+` overrides. **Accepted and broadened 2026-09-11:** every control in this project is a YAML key. verl gains one config block, `actor_rollout_ref.rollout.precision_scheduler`, holding the vLLM-side settings too: `enable`, `lora_fast_path`, `lora_dual_stream`, `lora_fuse_packed`, `int4_model`, `policy` (the decision-6 spec or JSON path), `bf16_layers`, `int4_modules`, `reprefill`, `sleep_level`, `reload_policy_each_rollout`, `online_observations`, `validate_shadow`, `validate_lifecycle`, `request_trace_dir`, `request_trace_log_tokens`. verl translates that block into the vLLM env vars for the server actor, since vLLM's engine reads settings from `envs.py` in its own process; the env vars remain the wire format and are documented in one table, but recipes and users never set them by hand. Per-model settings move from launcher case statements into YAML overlays under `examples/precision_scheduler/models/`. This replaces the env-forwarding allowlist in C8 with config-to-env translation and removes the env exports from every C10 recipe.

10. **Training driver and GPU scope.** Every headline Qwen3.5-9B result used Megatron TP1; the extensibility results used FSDP2. **Amended 2026-09-11: multi-GPU is out of scope.** Both drivers are restricted to a single GPU (TP=1, DP=1); the TP4 and DP7 configurations, the PP+SP dispatch hunk, and the port-range-per-GPU fan-out for multi-experiment hosts stay only where they cost nothing. Still open: keep Megatron at all? Keeping it preserves the exact training path behind the headline 9B numbers but requires the Megatron env shims (fla, causal_conv1d, mamba_ssm, the TE patch), the HybridStack and RoPE hooks, and the Megatron-to-HF LoRA target mapping. Dropping it leaves FSDP2 only, removes all of those, and changes only the downstream timing fit, which the global-search builder refits per run anyway. Recommendation: FSDP2 only, unless you want the headline 9B training-side timings reproduced exactly. **Reversed 2026-09-16: Megatron TP1 is the training driver for every RL-step experiment.** The FSDP2-only cut was taken without an explicit decision and surfaced only after the 2026-09-15/16 replication (16-step BF16 / W4 on Qwen3.5-9B, 4B, Phi-4-mini; the 8-step live-scheduler full-step run) had been executed with `run_fsdp_fullstep.sh` and the 9B actor CPU-offloaded, which adds a ~111 s fixed cost per step (the archived Megatron fit has a 12 s intercept) and makes the end-to-end step speedups and reward trajectories incomparable with the headline runs. Those FSDP2 runs are discarded; nothing rollout-only (heatmaps, kernel ablation, calibration traces, fixed-K sweep, live EMA loops and their switch points) depends on the trainer and stands. Actions: (a) restore the Megatron TP1 driver as `examples/precision_scheduler/run_megatron_fullstep.sh` (C10 row 'Megatron driver', previously dropped) with the same shell-variable knobs as the FSDP driver, the Megatron LoRA target lists derived from the C9 overlays via `convert_megatron_to_hf_target_modules`, the mbridge / recompute / transformer-config blocks, and the env shims the driver needs (`fla`, `causal_conv1d`, `mamba_ssm`, the TE patch) documented in `SETUP.md`; (b) `recipes/full_step.sh`, `recipes/continuous_ema.sh` (full_step runner) and `long_run/train_arm.sh` exec the Megatron driver; FSDP2 stays only as an explicitly selected `TRAINER=fsdp2` and never appears in reported numbers; (c) `cli.py fit-downstream --kind metrics` is re-run on Megatron rows before any `--downstream-slope` is used; (d) an e2e Megatron one-step GRPO test joins the test tiers. Single GPU (TP=1, DP=1) still holds. **Amended 2026-09-16 (user decision): Megatron TP1 is the trainer for every model, including the extensibility tier (Phi-4-mini, Gemma4, Nemotron-H). FSDP2 is never used for any reported or exploratory RL step; the archived 'extensibility results used FSDP2' line above describes history, not policy. Models without a Megatron-Bridge mapping (Phi3ForCausalLM today) get a bridge written (C9) before they are run; they are not run on FSDP2 as a fallback. `run_fullstep.sh TRAINER=fsdp2` stays only for the CPU compose tests and is removed from the recipes' reachable surface.** **Done 2026-09-16:** (a) `run_fullstep.sh` with `TRAINER=megatron` default (commit acde6425); (b) recipes exec it; (c) refit on the Megatron replication: 9B 8.67e-4 s/token + 5.7 s, R² 0.9988 on steady steps (4B 6.04e-4, R² 0.9992), see `RUN_2026-09-16_megatron_mg20.md`; (d) the Megatron e2e test is still open (the gpu-smoke `test_full_step_initial_checkpoint_and_evaluator` asserts FSDP-only checkpoint files).

11. **Version honesty.** Write an honest `vllm/_version.py` and an honest metadata dist-info into the shim directory, and drop the argument-parser import shim in verl. Recommendation: yes. **Accepted 2026-09-11.**

12. **Routed-experts ring buffer** in the vLLM commit. No archived run enables it. Recommendation: drop unless you remember an MoE run that needed it. **Decided 2026-09-11: drop.**

13. **GPUs for tests.** Which GPUs may the smoke and e2e tiers use, and when. **Decided 2026-09-11: GPUs 0 and 2 through 7, any that are not occupied by another user, and no hanging processes.** GPU 1 is never used. Enforced as a contract, not a habit: a preflight (`check_env.py --pick-gpus N`) lists the allowed GPUs, drops any with a compute process owned by another user or with non-trivial memory in use, and selects N free ones; it refuses to start if fewer than N are free. Every GPU test and recipe runs under a launcher that (a) records the process group it starts, (b) runs Ray with a private temp dir and an explicit `ray stop` in a trap, (c) kills the whole process group on exit, timeout, or Ctrl-C, and (d) verifies afterwards with `nvidia-smi --query-compute-apps` that no process of ours remains on the GPUs it used, failing the run if one does. Tests that spawn vLLM in-process use the same trap.

---

## 4. Component cards

Each card lists sources by origin: `worktree` (uncommitted), `commit` (vLLM `34e66a3`), `stash` (verl `stash@{0}`), `untracked`, `report`.

### C0. Environment, worktrees, honest version

**Purpose.** Make the clean branches importable and testable in the existing conda env without touching the dirty checkouts, and give a vanilla baseline.

**Facts established.** vLLM is an editable install with the precompiled cu130 payload built for `6bdabbad5b`; the payload is gitignored and ABI-compatible with vanilla, dirty, and clean because native code never changed. verl is not installed; the launchers put the repo root and two shim directories on `PYTHONPATH`. The current `vllm/_version.py` reports `0.1.dev`, which makes pristine verl take pre-0.11 code paths, and a fake `vllm-0.18.0` dist-info shadows metadata. TransformerEngine carries a hand patch that a reinstall would silently revert.

**Sources.** `report: rl-workflow/run_megatron_tp_live_fullstep.sh` lines 103-122 (activation contract); `report: rl-workflow/vllm_env_extra_deps/`, `fake_vllm_metadata/`; `worktree: verl vllm_async_server.py` parser shim; `report: env_fix_notes.md`.

**Keep / refactor / drop.**

| Item | Verdict | Reason |
|---|---|---|
| Activation block in the Megatron driver | refactor | becomes `scripts/precision_scheduler/env/activate.sh clean|dirty|vanilla` |
| `fake_vllm_metadata` | refactor | replaced by an honest-version dist-info in `/data/huanchen/envshims/vllm_metadata/` |
| `vllm_env_extra_deps` fla, causal_conv1d, mamba_ssm | move | copied to `/data/huanchen/envshims/pydeps/` (genuinely absent from the env) |
| `vllm_env_extra_deps` mbridge, modelopt, pulp, scipy symlinks | drop | identical packages exist natively |
| `vllm_env_extra_deps` megatron symlink | decide | only difference is a 4-line jit.py env gate; apply to the native copy if the gate is still used |
| verl parser import shim | drop | unnecessary with an honest version |
| TE `max_version` patch | keep as env patch | documented with a RECORD-hash check |
| stale flash_attn wheel, `=1.24.0`, `=1.4.3` files | drop | junk |

**Target layout.**
- `verl: scripts/precision_scheduler/env/{activate.sh, check_env.py, link_vllm_precompiled.sh, SETUP.md, ENVIRONMENT.md}`
- **`SETUP.md` is a from-scratch guide for a new machine** (accepted requirement, 2026-09-11): exact commands in order, no reference to this host's paths. Conda env creation with pinned versions (Python 3.12, torch 2.11+cu130, transformers, ray, flashinfer, peft, compressed-tensors); the flash-attn build command and the cublas re-pin that must follow it; the TransformerEngine version-gate patch as a diff plus how to verify it; vLLM cloned at the pinned base and installed editable with the precompiled cu130 wheel for that exact commit; the extra deps (fla, causal_conv1d, mamba_ssm) with build or wheel instructions; the honest version file and metadata shim; the verl clone and branch; worktree layout; then `check_env.py --expect clean` as the acceptance step. `ENVIRONMENT.md` keeps the rationale and rebuild triggers. The guide is validated by following it on a fresh conda env on this host before C0 is called done.
- `host: /data/huanchen/envshims/{pydeps, vllm_metadata}`
- `host: /data/huanchen/vllm-clean, /data/huanchen/vllm-vanilla, /data/huanchen/verl-clean`

**Tests.**
- unit `test_env_contract.py`: version gates resolve on the honest layout, TE flash-attn gate still patched, precompiled symlinks complete, `check_env.py --expect` refuses the wrong tree, Ray workers inherit the env.
- gpu-smoke: vanilla vs dirty-with-flags-off vs clean greedy token identity on 16 GSM8K prompts. Establishes that "vanilla" equals "dirty with flags off" before any refactor.

**Description.** `ENVIRONMENT.md`: package pins, patches, LD_LIBRARY_PATH rule, wheel provenance, rebuild triggers.

### C1. LoRA kernel fast path

**Purpose.** Replace Punica shrink and expand with a torch-compiled two-GEMM path for single-adapter rollout, packed across QKV, gate-up and Qwen3.5 in_proj slices, with LoRA-first dual-stream ordering and a lazy Punica-metadata path that removes a per-step device-to-host sync.

**Sources.** `commit: vllm/lora/ops/torch_ops/rollout_lora_ops.py` (whole); `commit: vllm/lora/layers/base_linear.py` packed buffers, async ordering; `commit: vllm/lora/layers/column_parallel_linear.py` hooks; `commit + worktree: vllm/lora/punica_wrapper/punica_gpu.py` selection, dispatch, lazy metadata, per-slice path; `commit: vllm/envs.py ROLLOUT_QLORA`; `report: precision-scheduling-validation/fixed_token_bench.py` kernel-ablation mode; `report: tail_w4_redo/run_qwen35_9b_kernel_ablation.sh`; `report: experiment-results/plot_kernel_ablation.py`; `report: eos_hazard_extensibility/make_zero_lora.py`.

**Keep / refactor / drop.**

| Item | Verdict | Reason |
|---|---|---|
| `rollout_lora_matmul` custom op | copy | 57 clean lines; fix fake-impl dtype to match real impl |
| packed A/B buffers and refresh hooks | copy | correct; document that slices sit at multiples of `max_lora_rank` |
| `_execute_lora_async`, `_apply_base_forward_async` | refactor | collapse to one `_apply_async(x, bias, base_fn, lora_first)` helper |
| `_get_rollout_single_lora_index` | refactor | explicit selection table; keep capture-time and replay decisions in lockstep |
| `_try_add_rollout_lora_linear` fused and per-slice | refactor | keep both (ablation stage 2), read `VLLM_ROLLOUT_LORA_FUSE_PACKED` from `envs.py`, keep the dtype cast |
| lazy Punica metadata | copy | the actual D2H fix; all seven entry points guarded |
| `_mcp_apply` early return | drop | unreachable |
| per-step debug f-string | drop | move diagnostics into the fallback branch |
| dual-precision hooks inside `base_linear.py` | move to C2 | C1 must not import `dual_precision` |
| kernel ablation harness and plot | move | `tools/rollout_lora/` with paths as arguments |

**Interfaces.** Produces: `BaseLinearLayerWithLoRA.active_base_layer` hook (default `base_layer`) that C2 overrides through its custom op; env `ROLLOUT_QLORA`, `VLLM_ROLLOUT_LORA_FUSE_PACKED`.

**Tests.**
- unit: fast path vs Punica reference for every layer class including the 4-slice variable-slice layer, at existing tolerances; packed buffer layout; selection table with mocked `prepare_tensors`.
- gpu-smoke: no host sync after warm-up under `set_sync_debug_mode('error')`; kernel presence via torch profiler (no Punica kernels, GEMM kernels present); capture-and-replay identity between eager and graph steps; zero-adapter equals base within tolerance.
- gpu-bench, **technique ablation** (added 2026-09-11): one script, `tools/rollout_lora/run_kernel_ablation.sh`, runs the four stages in order on the same fixed-work cell (batch 1, context 512, 64 decode tokens, 2 warmups, 5 repetitions, synchronized prefill barrier) for BF16 and Marlin INT4 bases: (1) Punica baseline, (2) torch GEMM, (3) plus packed QKV and gate-up fusion, (4) plus dual stream. It writes one JSONL row per cell plus a markdown table with median TPOT and the incremental gain per stage. Pass criteria: each stage's median TPOT is no worse than the previous stage's beyond the run's own 5-repetition spread, and the combined stage 4 beats stage 1 by at least half the archived gain (archived: 8.8% on BF16, 12.2% on Marlin, 33% on BitsAndBytes; new runs recorded alongside). The resulting table goes into the C1 design doc as the "each technique helps, and they compose" evidence. Runs with dual precision off, matching how the archived numbers were taken (decision 1).

**Description.** `docs/design/rollout_lora_fastpath.md`. Must state decision 1.

### C2. Dual-precision residency and binding

**Purpose.** Load a second INT4 checkpoint, attach its quantized linears as a pruned shadow store next to the BF16 model, and bind each LoRA wrapper to the selected base before every forward through an opaque custom op that never mutates module topology.

**Scope change 2026-09-11: GPTQ only.** Supported shadow formats are GPTQ packing (Intel AutoRound `auto_round:auto_gptq`, used by every headline run) and compressed-tensors pack-quantized GPTQ (Phi-4). AWQ checkpoints and the "AWQ-transformed block state" pairing are dropped; the loader refuses an AWQ quant method with a clear error. Verified before dropping: on the Qwen3.5-9B AutoRound shadow, 152 of 176 shared non-linear block tensors are bit-identical to BF16 and the remaining 24 are the 128-dim gated-deltanet norm vectors, differing at bf16 rounding level (max abs 0.004), so the pairing was a near no-op on the headline path. The INT4 path now uses the BF16 model's norms, and the design doc records that residual difference.

**Sources.** `commit + worktree: vllm/model_executor/dual_precision.py` residency half (lines 26-66, 681-787, 790-1399 in the worktree); `worktree: vllm/lora/layers/base_linear.py` custom-op binder; `commit: gpu_model_runner.py` load and register sites; `commit: vllm/forward_context.py base_precision`; `worktree: compressed_tensors_wNa16.py` Marlin padding (assigned to C9 but loaded here); `worktree: verl utils.py _hide_dual_precision_shadow_model`, `vllm_async_server.py _resolve_rollout_model_path` and forced sleep level, `engine_workers.py` empty cache before resume, `config/rollout.py model_path`; `report: GEMMA4_DUAL_PRECISION_AUDIT.md` invariants.

**Keep / refactor / drop.**

| Item | Verdict | Reason |
|---|---|---|
| loader, name matching, shadow store | refactor | split into a package; per-model state instead of module globals; register the two validation env vars; GPTQ formats only |
| AWQ-transformed block state pairing and its shadow-store parameters | drop | GPTQ only; verified near no-op on the headline shadow |
| custom-op binder in `base_linear.py` | copy | the Gemma4 fix; validated on Phi-4 and Gemma4-12B |
| committed `_modules['base_layer']` swap | drop | root cause of the compile failure |
| BF16 layer policy and `mlp_only` module policy | copy | unit-tested; keep default `first:3,last:3` but document that every final run set `none` |
| analysis BF16-layer mask API | refactor | eager-only diagnostic, kept out of the hot bind loop |
| runner load wiring (two `if lora_config` blocks) | refactor | one `attach_dual_precision()` call after LoRA load |
| verl hide-shadow context manager | copy | import the module-name constant from vLLM with a string fallback |
| verl `rollout.model_path` | copy | add a unit test |
| verl forced sleep level | refactor | `rollout.sleep_level` config with env fallback; assert level 1 when dual precision is on |
| verl empty cache before resume | copy | two lines |

**Target layout.** `vllm/model_executor/dual_precision/{__init__, policy_layers, loader, binding, validation}.py`; `docs/design/dual_precision_residency.md`; verl: `config/rollout.py`, `vllm_async_server.py`, `utils.py`, `engine_workers.py`, `tests/workers/rollout/rollout_vllm/test_dual_precision_residency_hooks.py`.

**Interfaces.** Consumes C1 hook. Produces: `attach_dual_precision(model, vllm_config)`, `bind_dual_precision(model, precision, no_compile_layers)`, `BASE_PRECISION_*`, shadow module name constant, `dual_precision_rollout_enabled()`.

**Tests.**
- unit: bind never mutates `_modules`, `_parameters`, `_buffers` (audit invariant 1; the committed swap must fail it); policy and module selection reproduce recorded attach counts (152 attached / 134 fallback for the Qwen3.5-9B AutoRound shadow, 112/27 Nemotron, 70 Gemma E2B MLP-only); the loader rejects an AWQ quant method; config clone keeps every init field; verl hooks with fakes.
- gpu-smoke: custom-op binder under `torch.compile` plus one FULL graph per precision (audit tests 2-3); LoRA delta shared across precisions (audit test 4); lifecycle probes exact after sleep, wake, and weight sync, and failing without the hide helper; shadow numerical validation bounds on Qwen3.5-9B.
- golden: dual-resident W4 greedy ids on Nemotron reproduce `stage0/weight_audit/dual_w4_stream0_greedy/responses.jsonl`.

### C5. Switching policy module (torch-free)

**Purpose.** The decision logic: policy JSON loader and validator, dense lookup table indexed by (frontier, prompt bucket, live batch), commitment rules, live-batch guard, and the receding-horizon cost model kept as a pure offline function.

**The search problem** (from the global-search builder, decision 2):

```text
Inputs
  F[i]        frontier bins: 250, 500, ..., CAP-250 (STEP = 250 response tokens)
  P[j]        prompt buckets: 0, 128, ..., 2048
  B           initial rollout batch; live-batch axis is 1..B
  TPOT[p](ctx, live)   profiler heatmap, log2-interpolated in context and live batch, p in {bf16, w4}
  slope       downstream seconds per sampled token, fitted from RL step timings
  hazard tables H_bf16, H_w4: per bin, risk[i] = P(eligible and alive at bin start),
              event[i] = P(finish inside bin i, not by cap); h_i = event[i] / risk[i]

Online EMA (once per RL step, from that step's switch cohort: entry token and final length per request)
  new = components(entries, finals)                 # risk/event per bin, only bins with eligible requests
  H[k] = (1 - alpha) * H[k] + alpha * new[k]         # k in {risk, event}, only on observed bins
  rebuild table below; bump policy_revision; scheduler reloads before next rollout

survival(H, f)   # P(a request alive at frontier f is still alive at the START of each later bin)
  S[0] = 1;  S[k] = prod_{j < k} (1 - h_{f+j})

cost(p, fi, pj, live, alive[0..n))   # expected cost of decoding bins fi.. under precision p
  for each bin k:
    nonempty_k = 1 - (1 - alive_k)^live           # P(at least one of `live` requests still running)
    eff_k      = clip(round(live * alive_k / nonempty_k), 1, B)   # E[live | nonempty]
    tokens_k   = min(STEP, CAP - F[fi+k])
  rollout    = sum_k TPOT[p](P[pj] + F[fi+k], eff_k) * tokens_k / 1000 * nonempty_k
  downstream = slope * live * sum_k alive_k * tokens_k
  return rollout + downstream

build_policy   # dense table, every (frontier, prompt bucket, live) state
  for each observed frontier fi:
    S_bf = survival(H_bf16, F[fi])
    stay = cost(bf16, fi, ., ., S_bf)                            # never switch
    best = +inf
    for each candidate switch frontier fj >= fi:                 # global search over all later frontiers
      prefix = cost(bf16, fi, ., ., S_bf[0 : fj-fi])             # BF16 until fj
      reach  = S_bf[fj-fi]                                       # P(still alive when fj is reached)
      suffix = cost(w4, fj, ., ., survival(H_w4, F[fj]) * reach)  # W4 from fj on, conditioned on reaching it
      cand   = prefix + suffix
      if cand < best: best, best_fj = cand, fj                   # strict <: earliest frontier wins ties
    table[fi, pj, live] = F[best_fj] if best < stay else 0       # 0 = do not plan a switch

runtime (C4, receding mode; runs when the longest live request crosses a new 250-token frontier)
  state     = (f, decision_live = live + not-yet-arrived cohort members, prompt bucket of median prompt)
  committed = table[f, bucket, decision_live]                    # None if 0
  switch to INT4 (one-way, for the rest of the rollout) when
      committed is not None and max_response_tokens >= committed
      and actual_live <= max_switch_live_batch
```

Two properties the sub-plan must preserve: the comparison is plan against plan (BF16 prefix plus W4 suffix, downstream token cost included, against staying BF16), not "is W4 faster right now"; and ties resolve to the earliest frontier because the update uses strict less-than while iterating from the current frontier upward.

**Sources.** `worktree: dual_precision.py` lines 61-678; `worktree: scheduler.py` inline predicates (min-rule, receding replacement, guard); `untracked: tests/model_executor/test_dual_precision.py` policy tests; `report: dynamic_switch_rollout_20260822/policies/*_validation.json` (87 states); `report: runs/*_online_hazard_warm5_receding_gpuval/logs` (deterministic receding oracle).

**Keep / refactor / drop.**

| Item | Verdict | Reason |
|---|---|---|
| `_lookup_table_frontier`, `lookup_committed_frontier` | move | clean; add layout and schema validation |
| cost model primitives and `predict_receding_horizon` | refactor | `CostModel` class with instance memoization; offline only; ties keep "now" via strict `<` |
| commitment rules | refactor | one helper with `monotone` and `receding` modes; receding is observation-gated |
| loader | refactor | drop `switch_thresholds`, `lookup_hierarchy`; add `fixed_switch_frontier`; validate `schema_version` and layout |
| reload with revision check | refactor | explicit `PolicyStore.load/reload`; fail closed if revision lags when reload is enabled |
| async worker, NOOP profiling knob | drop | profiling-only |
| `max_switch_live_batch` guard | move | evaluated on actual live count |

**Target layout.** `vllm/v1/core/sched/precision_policy.py` importable from the GPU worker (the dispatcher reads `capture_max_batch`); `tools/precision_policy/replay_policy_log.py`; `tests/v1/core/test_precision_policy.py` with trimmed fixtures.

**Interfaces.** Produces: `load_precision_policy(path)`, `LookupTable.committed_frontier()`, `PolicyDecider.observe(...) -> Decision`, `CostModel.predict()`, policy JSON schema 6 contract consumed by C4, C3 (`capture_max_batch`), and C6 (writer).

**Tests.**
- golden: lookup replay reproduces 156/156 commitment lines and 42/42 switches across three static runs; receding replay reproduces 1313/1313 updates and 60/60 switches on the warm5 gpuval runs; cost model reproduces logged predictions to 1e-5 at prompt bucket 0; offline table equals online prediction on the 87 validation states; commitment replay reproduces archived switch frontiers.
- unit: loader accepts schema 2, 4, 5, 6 fixtures and rejects malformed tables; indexing and clamping; commitment modes and guard; cost-model properties; replay tool CLI smoke.

### C3. Precision-keyed CUDA graphs and dispatch

**Purpose.** Make base precision part of the graph key so both precisions have captured graphs, capture INT4 only up to a ceiling, honor the scheduler's override, and fall back to eager with a once-per-key warning.

**Sources.** `commit: vllm/forward_context.py`; `commit + worktree: vllm/v1/cudagraph_dispatcher.py`; `commit + worktree: gpu_model_runner.py` bind-before-execute, `_dummy_run` precision, dispatch plumbing; `worktree: gpu_worker.py`; `commit: vllm/v1/worker/gpu/{cudagraph_utils, dp_utils, model_runner}.py` (V2 runner); `worktree: tests/v1/cudagraph/test_cudagraph_dispatch.py`.

**Keep / refactor / drop.**

| Item | Verdict | Reason |
|---|---|---|
| `BatchDescriptor.base_precision` | copy | the whole mechanism |
| key registration | refactor | register both precisions up to one ceiling in both fixed and dynamic modes (the static path captured one precision per size and fell to eager) |
| `dispatch()` overrides | refactor | keyword-only args; scheduler is the sole owner; drop `precision_num_reqs` and the dead mismatch guard; restore `num_reqs=None` on the bare eager path (4 vanilla tests currently fail) |
| once-per-key missing-graph warning | copy | how the eager int4 steps were noticed |
| runner bind and capture hunks | copy | with the C2 load hunk as a hard dependency |
| V2 runner hunks | drop | never selected by any project model; add an explicit guard |

**Interfaces.** Consumes C2 bind and C5 `capture_max_batch`. Produces: `dispatch(num_tokens, *, num_reqs, uniform_decode, has_lora, num_active_loras, base_precision, ...)`.

**Tests.**
- unit: key sets match the three headline ladders (b32: 29/21, b64: 45/29, b128: 77/45 with LoRA cases [0, 2] and ceiling 32); both precisions registered in fixed mode; override wins over count; eager fallback logs once; vanilla equivalence when disabled (the 10/2 expectations); V2 guard.
- gpu-smoke: wrapper captures and replays two graphs for the same shape under two bindings.
- e2e: engine reports the archived graph counts and zero fallback warnings when the switch happens under the ceiling.

**Risk to resolve in the sub-plan.** Lookup policies are gated only by `max_switch_live_batch`, never by `capture_max_batch`; the hardmath run switched at live 56 with ceiling 32 and ran 11 eager steps. The loader (C5) will reject `max_switch_live_batch > capture_max_batch` and default the guard to the ceiling.

### C4. Scheduler switching runtime

**Purpose.** The scheduler-side state machine that turns the policy into a per-step precision signal: fixed-threshold latch with drain re-arm and uniform-W4, rollout cohorts under asynchronous admission, a monotone response-frontier watermark, lookup with commitment, the live-batch guard, switch-cohort JSONL, and policy reload at rollout boundaries.

**Sources.** `worktree: vllm/v1/core/sched/{scheduler.py, output.py}`, `vllm/v1/request.py` streaming offset; `worktree: tests/v1/core/test_scheduler.py` (12 dynamic tests, 7 asserts call a method that does not exist); `report: runs/b32_cap16384_ema_alpha_a000_30step` (revision-invariant golden produced by the current code); `report: build_fixed_live_threshold_policy.py`, `build_fixed_frontier_policy.py`.

**Keep / refactor / drop.**

| Item | Verdict | Reason |
|---|---|---|
| `SchedulerOutput` fields, request streaming offset | copy | the only cross-process contract |
| fixed-threshold latch | refactor | `FixedThresholdPolicy.tick()`; writes cohorts too; keeps an explicit override seam for the profiler |
| lookup runtime and cohort tracking | refactor | pure `RolloutPrecisionSwitcher` with `on_new_request`, `on_request_output`, `tick`; exact semantics preserved including decision-live inflation and observation-gated receding switches |
| double reload per boundary | fix | reload once; idempotent so archived behavior is unchanged |
| switch-cohort JSONL | refactor | add `prompt_tokens` and trigger fields; always on when a policy is loaded so the "exact switch request states" log line can go |
| online cost loop, async executor, hierarchy branch, `switch_thresholds` fallback | drop | never used by a completed final run |
| NVTX and nsys window code (three copy-pasted blocks) | drop | profiling scaffolding that alters switching; results archived |
| verl nsys bridge thread and server nsight runtime env | drop | counterpart of the above |

**Target layout.** `vllm/v1/core/sched/precision_switch.py`; scheduler integration of about 60 lines; `tests/v1/core/test_precision_switch.py`; `tests/v1/core/golden/precision_switch/`; verl `tools/precision_scheduling/policies/build_static_policy.py` (shared with C10).

**Interfaces.** Consumes C5. Produces: `SchedulerOutput.dual_precision_base_precision`, `num_unfinished_requests`, cohort JSONL schema, `set_forced_precision()` override for the profiler.

**Tests.**
- golden: replay archived lifetimes through the switcher and match the 29 switches and cohorts of `b32_cap16384_ema_alpha_a000_30step`; fixed-frontier replay from the "exact switch request states" lines of `b32_cap16384_fixed_frontier8000_30step`.
- unit: fixed latch, uniform, drain re-arm; env-threshold vs lookup-emulation equivalence; the seven scheduler scenarios re-expressed on fixed-frontier policies and asserted through the public output field; cohort JSONL schema round-trip with the watcher's parser; reload once per boundary and fail-closed on stale revision; vanilla scheduler regression with the fields `None`.
- gpu-smoke: precision signal reaches the dispatcher and the cohort file is written.

### C7. Re-prefill after switch

**Purpose.** Default-off ablation: when the fixed threshold is crossed, preempt every surviving request so its KV is recomputed under INT4.

**Sources.** `commit: scheduler.py` trigger, guards, call site; `commit: request.py` fields; `commit: sched/utils.py`; `worktree` re-arm on drain; `commit: tests` four re-prefill tests; `report: reprefill-study/{run_experiment.py, run_experiment_single_gpu.py, analyze_results.py}`.

**Keep / refactor / drop.**

| Item | Verdict | Reason |
|---|---|---|
| trigger with idle-step semantics | copy | archived evidence depends on it; shared preempt helper is a later cleanup |
| re-arm on drain | copy | untested today; add a test |
| `reprefill_done`, `reprefill_output_offset` fields | copy | renamed |
| `num_visible_output_tokens` fold branch and `check_stop` change | drop | dead in every code path; C4 composes on `num_output_tokens` |
| `VLLM_REPREFILL_ONLY_ROLLOUT` | drop | one archived one-step run |
| NLL study scripts | refactor | `examples/precision_scheduling/analysis/reprefill_nll_study/` with trace selection by explicit id list |

**Tests.** unit: ported four tests plus re-arm and idempotence; async placeholder discard vs vanilla `reset_prefix_cache`. gpu-smoke: one trigger per batch, preempted equals unfinished at crossing. golden: NLL summary reproduced from archived window metrics; full GPU rerun is tolerance-based and optional.

**Description.** One subsection in the C2 design doc plus the study README stating the negative result and that the archived threshold study ran with re-prefill on.

### C8. verl telemetry and experiment harness

**Purpose.** What made the experiments reproducible on the verl side: request lifetime tracing, rollout-only mode, initial-checkpoint gate, multi-experiment isolation, env forwarding, and a few opt-in vLLM diagnostics.

**Sources.** `worktree: vllm_async_server.py` tracer and forced sleep level; `worktree: trainer_base.py` rollout-only, initial checkpoint, port range; `worktree: constants_ppo.py`; `worktree: vllm_rollout.py`, `utils.py` namespace and SHM; `stash: single_turn_agent_loop.py`, `trainer_base.py`, `llm_server.py` stable trace id; `stash: checkpoint_engine/base.py`, `engine_workers.py`, trainer variants timing dict; `stash: actor.py`; `worktree vLLM: gpu_model_runner.py` prompt-logprob extra ids; `commit vLLM: decode profiler window, engine step timing, request_lifecycle.py`.

**Keep / refactor / drop.**

| Item | Verdict | Reason |
|---|---|---|
| lifetime tracer | refactor | `RequestLifetimeTracer` class, one append handle, frozen file name and keys, plus `trace_request_id` |
| rollout-only mode | refactor | `_fit_rollout_only()` with real yaml keys; single and multi-step paths unified; marker and metric names verbatim |
| initial checkpoint gate | copy | add yaml keys |
| master port range | refactor | `parse_port_range()` in `net_utils`; config key with env fallback |
| env forwarding | refactor | per decision 9: a `precision_scheduler` config block is the source of truth; a pure `to_vllm_env(config) -> dict` emits the vLLM env vars into the server actor's runtime env; no hand-set env vars |
| ZMQ namespace and forced SHM | refactor | one sanitizer shared by sender and receiver |
| stable trace id (stash) | refactor | carried as `trace_request_id` through `llm_server`, gated per backend; engine id stays a uuid |
| update-weights sub-timings (stash) | refactor | surfaced via a manager property, no return-type change |
| old-logprob entropy toggle (stash) | refactor | reuse `actor.calculate_entropy`; default unchanged |
| offline replay, raw-string prompts, progress dump, profiler toggle, no-adapter knob (stash) | drop | superseded or unused; upstream `skip.rollout` covers replay |
| nsys bridge and server nsight env | drop | with C4's scheduler counterpart |
| vLLM prompt-logprob extra ids | refactor | pure helper `inject_extra_prompt_logprobs()`; opt-in |
| vLLM decode profiler window, engine step timing, `request_lifecycle.py` | drop | no launcher sets them; no in-tree consumer |

**Tests.** golden: trace schema against an archived trace; runtime-env forwarding key set; stable trace id against the July trace format. unit: rollout-only single and multi-step with mocks (after extraction), initial checkpoint gate, port range parsing, namespace agreement, forced sleep level, prompt-logprob helper. gpu-smoke: one rollout-only step validated by the ported collector.

### C9. Model support

**Purpose.** The model-motivated fixes that let Qwen3.5, Phi-4, Gemma4, and Nemotron-H run.

**Sources.** `worktree verl: agent_loop.py, megatron_utils.py, config/model.py, transformer_impl.py, monkey_patch.py`; `stash verl: megatron_peft_utils.py, tensordict_utils.py`; `untracked verl: two test files`; `worktree vLLM: gemma4_unified registry glue in vllm/transformers_utils/config.py, vllm/transformers_utils/configs/__init__.py, vllm/transformers_utils/model_arch_config_convertor.py, vllm/model_executor/models/config.py, vllm/model_executor/models/registry.py; vllm/model_executor/models/gemma4.py; compressed_tensors_wNa16.py`; `untracked vLLM: vllm/transformers_utils/configs/gemma4_unified.py`; `report: prepare_phi4mini.py, validate_nemotron_marlin_padding.py, run_fsdp_fullstep.sh model blocks`.

**Keep / refactor / drop.**

| Item | Verdict | Reason |
|---|---|---|
| Qwen3.5 mm-token guard + test | copy | add license header; add a positive case |
| rope-theta skip, HybridStack recompute hook + tests | copy / refactor | logger instead of print; importorskip megatron |
| Qwen3.5 GatedDeltaNet LoRA mapping (stash) | refactor | make the `in_proj` expansion architecture-conditional (Nemotron uses the identity name) |
| tensordict jagged fallback (stash) | copy | add a forced-fallback test |
| gemma4_unified normalization | refactor | helper in `verl/utils/model_compat/gemma4.py`, testable with a config fixture |
| `key_mapping` kwarg, padded-path entropy chunking, singleton logits | copy | generic; add CPU equivalence tests |
| Gemma4 dense FFPA | refactor, flag | `verl/models/transformers/gemma4_ffpa.py`, lazy import, config knob instead of env; ffpa-attn as an optional pinned dependency, not vendored |
| vLLM gemma4_unified registry glue, `gemma4.py` fixes | copy | add config and k_norm tests |
| Marlin input padding | refactor, flag | parameter-identity guards; document TP>1 fallthrough |
| per-model launcher blocks | move | `examples/precision_scheduler/models/*.yaml` |
| Falcon-H1, DeepSeek, SmolLM3, GLM, Kimi, Granite, MiniCPM | drop | no code; screening only |

**Tests.** unit as listed per row. gpu-smoke: padded Marlin vs Triton on the real Nemotron layer; FFPA vs SDPA reference; Gemma4 E2B text-only load. e2e: Phi-4 one-step GRPO; Nemotron-H Megatron one-step with non-zero LoRA gradients.

### C6. Profiling and policy-building toolkit

**Purpose.** Everything that produces a policy: the TPOT efficiency heatmap harness (fake KV, decode barrier, warmups and measurements), response-length calibration, the hazard EMA, the downstream token-linear regression, the policy builder, the online watcher, and plots. None of it lives in a package today.

**Sources.** `report vLLM: precision-scheduling-overhead/efficiency_heatmap/{run_heatmap.py, dummy_kv_connector.py, test_heatmap.py}`; `report vLLM: precision-scheduling-validation/fixed_token_bench.py` (Gen1, kernel ablation); `report verl: dynamic_tail8k_heatmap_20260823/{build_policies.py, simulate_online_hazard.py, run_continuous_hazard_ema.py, build_fixed_frontier_policy.py, build_forced8k_calibration_policy.py, test_online_hazard.py, test_frontier_conditioning.py}`; `report verl: eos_hazard_extensibility/b32_16k_sensitivity/{online_ema_policy.py, test_online_ema_policy.py}`; `report verl: heatmap_budget.../generate_validation_report.py` fit lines; `untracked verl: evidence_bundle/build_exports.py` fit; `report verl: rl-workflow/{fit_replay_regression.py, replay_regression_worker.py, build_replay_regression_workloads.py}`.

**Keep / refactor / drop.**

| Item | Verdict | Reason |
|---|---|---|
| `run_heatmap.py` measurement core | refactor | `vllm/tools/precision_scheduler/tpot_heatmap.py`; CLI args instead of baked paths; uses C4's forced-precision seam instead of a stub policy; per-model argument sets documented |
| synthetic KV connector + tests | copy / move | self-contained |
| Gen1 bench | keep kernel-ablation mode only | needed by C1's ablation; shared `SynchronizedPrefillScheduler` |
| global-search builder | refactor | primary `policy_builder.py`; parameters instead of module constants |
| earlier EMA builder and hazard library (tail8k formulation) | drop | decision 2: global-search only; still active research |
| watcher | refactor | one `watch-ema` CLI merging the two watchers; fixes the `KeyError('kind')` at HEAD; pluggable initial calibration |
| downstream regression fits (three variants) | refactor | one `downstream_regression.py` reproducing all three |
| replay regression worker | refactor | `replay_regression/` with CLI args |
| zero-LoRA generator | move | the eos version (auto-discovery) |
| scalar EMA, warm5 builder, study-specific analysis and slides | drop | superseded or evaluation-only |

**Target layout.** `verl/experimental/precision_scheduler/{traces, tpot_grid, hazard, cost_model, policy_builder, calibration, downstream_regression, online_ema, plots, cli}.py`; `vllm/tools/precision_scheduler/`; tests under `tests/experimental/precision_scheduler/` with small fixtures.

**Tests.**
- golden: global-search EMA reproduces the Qwen3.5-4B `dynamic_policy.json` at revision 30 (the tail8k rebuild and continuous-hazard replay goldens are dropped with that builder); the 120-run regression to 1e-9; the Megatron replay regression; the validation downstream fit (intercept 12.4814, slope 0.00087839); legacy CSV grid from Gen1 rows; heatmap matrix from cells for the four valid rep5 runs including the median-speedup validity guard.
- unit: hazard and survival; builder vs brute force; policy JSON contract with the vLLM loader (cross-repo); trace parsing.
- gpu-smoke: 2x2 heatmap on Qwen3.5-9B; forced-switch cohort log plus one watcher revision.
- e2e: 3-step EMA rollout with fail-closed watcher.

### C10. Datasets, recipes, long-run evaluation

**Purpose.** The reproducible entry points: dataset prep, reward functions, the two drivers, the four recipe shapes, the 100-step protocol, and the kernel-ablation drivers.

**Sources.** `report: rl-workflow/run_megatron_tp_live_fullstep.sh`; `report: eos_hazard_fullstep_b64_cap16k/run_fsdp_fullstep.sh`; the eight policy-case wrappers; `report: prepare_gsm8k_thinking_parquet.py, prepare_gsm8k_temporal_guard_sets.py, prepare_bigmath_splits.py, prepare_datasets.py, render_prompts.py, prepare_data.py`; five reward files; `run_no_reprefill_rl_100step.sh`, `summarize_no_reprefill_rl.py`, hardmath `evaluate_bf16_patch.py`, `PROTOCOL.md`; three static policy builders; `collect_best_t8_verl_rollout.py`.

**Keep / refactor / drop.**

| Item | Verdict | Reason |
|---|---|---|
| Megatron driver | refactor | `examples/precision_scheduler/run_megatron_fullstep.sh`; drop Eurus defaults and the missing prep call; `DRY_RUN=1` |
| FSDP2 driver | refactor | `run_fsdp_fullstep.sh` with `models.json` of HF ids and revisions |
| eight policy wrappers | collapse | `common.sh resolve_policy()` and four recipes: `rollout_only.sh`, `full_step.sh`, `continuous_ema.sh`, `long_run/train_arm.sh` |
| six dataset prep scripts | collapse | `data/prepare_gsm8k.py`, `prepare_bigmath.py`, `prepare_eos_workloads.py` with sha256 manifests |
| five reward files | collapse | one `rewards.py` dispatching on `data_source` |
| three static policy builders | collapse | `policies/build_static_policy.py --kind frontier|live_threshold|forced_switch` (shared with C4) |
| 100-step protocol, evaluator, summarizer | refactor | `long_run/{train_arm.sh, evaluate_lora_patch.py, summarize_runs.py}` |
| run validator | refactor | `tools/validate_rollout_run.py` |
| lane and queue scripts, per-GPU matrices | drop | recorded as a README table |
| Eurus prep and data | drop | predates the storyline |
| kernel ablation drivers | move | `vllm/benchmarks/precision_scheduler/` |

**Tests.**
- unit: hydra composition of every recipe's `DRY_RUN` override list for both engines; `continuous_ema.sh` fail-closed with stub watcher and runner.
- golden: dry-run env contract against archived `run_config.json`; GSM8K prep reproduces the archived parquet and token stats; BigMath prep reproduces manifest hashes; eos workload prep reproduces manifests; rewards re-score archived dumps exactly; summarizer reproduces `efficiency_summary.json`; static policies byte-equal the archived fixed-t and fixed-k JSONs; kernel plot reproduces the 12 values.
- gpu-smoke: 2-step rollout-only on a tiny parquet; 1-step full step plus initial checkpoint plus evaluator.

---

## 5. Execution schedule

| Wave | Components | Parallel? | Gate to next wave |
|---|---|---|---|
| 0 | C0 | no | `check_env.py --expect clean` passes; vanilla-vs-clean greedy identity passes |
| 1 | C1, C2, C5, C8 | yes, four worktrees | C1 and C2 merged in that order (C2 consumes C1's hook); C5 and C8 are file-disjoint |
| 2 | C3, C4, C9 | C9 in parallel; C3 before C4 | C4's golden replay passes; dispatcher vanilla tests pass |
| 3 | C7, C6 | yes | C6 goldens pass; C7 unit tests pass |
| 4 | C10 | no | dry-run parity and prep goldens pass; one gpu-smoke rollout-only step |

Each wave ends with the branch pushed to the fork and a short status note. GPU tiers run only after you name GPUs.

---

## 6. Archive and hygiene

- First commit on each branch: `.gitignore` entries for `.codex-report/`, `.codex-reports/`, `evidence_bundle/`, `*.whl`, `=*`.
- Report trees stay where they are. Golden tests reference them by absolute path with a skip; small derived fixtures (about 30 MB total across both repos, mostly one heatmap JSON and a few regression CSVs) are committed.
- Docs to copy in as archive pointers: the handoff document's algorithm sections and the two final EMA tables, into `docs/precision_scheduler/`.
- Not migrated, by decision: Eurus workloads, the vLLM-only rollout harness, the request-lifetime results tree, all nsys captures, the EOS layer-sensitivity scripts, slides and paper figures.

---

## 7. Dropped hunks (coverage ledger)

Listed so nothing disappears silently.

- vLLM commit: `.codex-reports/dual_precision_v2/dual_precision_v2_rollout_bar_plot_temp0.png`; `vllm/v1/engine/core.py` step timing and KV capacity RPC; `vllm/v1/outputs.py` routed-experts copy; `vllm/v1/metrics/request_lifecycle.py` and `tests/v1/metrics/test_request_lifecycle.py`; the V2 runner files `vllm/v1/worker/gpu/cudagraph_utils.py`, `vllm/v1/worker/gpu/dp_utils.py`, `vllm/v1/worker/gpu/model_runner.py`; decode profiler window in `gpu_model_runner.py`.
- vLLM worktree: async predictor and its env knobs; lookup hierarchy; `switch_thresholds`; NVTX and nsys code in the scheduler; `VLLM_REPREFILL_ONLY_ROLLOUT`; the committed module-swap binder.
- vLLM untracked: `scripts/plot_fake_eos_layer_ablation.py`, `scripts/plot_tpot_qwen25_27b.py`, `scripts/plot_tpot_qwen35_9b.py`, `scripts/run_gsm8k_eos_layer_sensitivity.py`, `scripts/run_gsm8k_gold_layer_sensitivity.py`.
- verl worktree: parser import shim; nsys bridge thread and server nsight runtime env; `VERL_GEMMA4_HEAD512_FALLBACK` (never read).
- verl stash: offline rollout replay in `trainer_base.py`, `main_ppo.py`, `trainer_sync.py`, `trainer_colocate_async.py`, `trainer_separate_async.py` (the update-weights timing merge in those three trainer files is kept under C8); raw-string prompts in `single_turn_agent_loop.py` and `rl_dataset.py`; `_dump_step_progress`; `RequestStepRecorder`; `VERL_CODEX_CUDA_PROFILE_UPDATE_ACTOR`; `old_log_prob_no_lora_adapter`.

File-level coverage check (all five sources, 78 files): every file is named in a component card or above. The two untracked verl tests, `tests/experimental/agent_loop/test_text_only_position_ids_on_cpu.py` and `tests/utils/megatron/test_megatron_utils_rope.py`, are copied under C9; `vllm/transformers_utils/model_arch_config_convertor.py` is part of the C9 gemma4 registry glue.

---

## Self-review

- Spec coverage: story items 1 (kernels) → C1; 2 (partial BF16/INT4, both models resident, graphs, scheduler signal) → C2, C3, C4; 3 (profiler and online scheduler) → C5, C6; 4 (re-prefill) → C7; 5 (evaluations and auto-profiling) → C8, C10; 6 (extensibility and models) → C9; datasets and long-run → C10; environment (your addition) → C0.
- Placeholder scan: this master plan intentionally defers step-level code to per-component sub-plans; no task here claims to be executable without one.
- Type consistency: interface names used across cards (`active_base_layer`, `attach_dual_precision`, `bind_dual_precision`, `load_precision_policy`, `PolicyDecider`, `capture_max_batch`, `RolloutPrecisionSwitcher`, `set_forced_precision`, `RequestLifetimeTracer`, `collect_forwarded_env`, `build_static_policy`) are consistent between producing and consuming cards.

---

## 8. Ten-hour execution schedule (supersedes section 5 timing)

Protocol changes required to fit ten hours:

1. **Component cards are the spec.** No separate per-component sub-plan documents. Each implementer subagent receives its section 4 card, its inventory digest, and the ground rules, and does TDD directly.
2. **Batched review.** Components are committed on their own branch as soon as the reviewer subagent passes them. The user reviews per wave, while the next wave runs. Rejections become fix tasks; nothing waits on the user.
3. **One worktree and one GPU per component.** All GPUs from {0,2,3,4,5,6,7}; GPU 1 never. Every subagent runs under the process-cleanup contract (decision 13) and must leave `nvidia-smi --query-compute-apps` empty for its GPU on exit.
4. **Dependency-driven start, not wave-driven.** A component starts the moment its inputs are merged.

| Slot | Hours | Runs in parallel | GPU | Merges at end of slot |
|---|---|---|---|---|
| A | 0-1 | C0 (worktrees, env, honest version, check_env, SETUP.md, greedy identity) | 0 | C0 |
| B | 1-4 | C1, C2, C5, C8, C9, C6 (six worktrees) | C1:2 C2:3 C5:cpu C8:4 C9:5 C6:6 | C1 → C2 → C5 → C8 → C9 → C6 |
| C | 4-6.5 | C3 then C4 (serial), C10, C7 (after C4) | C3/C4:2 C10:4+5 C7:3 | C3 → C4 → C7 → C10 |
| D | 6.5-8.5 | Integration: one full RL step on clean verl + clean vLLM, GSM8K Qwen3.5-4B, EMA policy, compared to the dirty tree | 6,7 | fixes |
| E | 8.5-10 | Fix tasks from user review, push both branches, status note | any | final push |

Golden and gpu-smoke tiers run inside each component's slot on its assigned GPU. The e2e tier runs once, in slot D.
