# The prediction model behind the switch decision (state as of 2026-09-25)

Audience: an agent that has to read, modify or evaluate the scheduler. Every claim points at code on
branch `rollout-precision-scheduler-clean` (verl fork); paths are relative to
`verl/experimental/precision_scheduler/` unless stated. Line numbers are from commit `bc3b3543`.

## 1. What is predicted, and what the decision is

The rollout decodes a batch of `B` requests in BF16 and may switch the **whole batch** to the INT4
shadow ("W4") once, at a frontier `F` on a 250-token grid. The scheduler needs, for every
(current frontier `f`, prompt bucket, live count `n`), the frontier at which to switch — or 0 for
"do not switch from here". That table is the policy (`lookup_table.committed_frontiers`,
`policy_builder.py:349 decisions_array`, consumed in vLLM by
`vllm/v1/core/sched/precision_policy.py:354 LookupTable.committed_frontier`, re-read at every
frontier = receding horizon: a cell equal to the current frontier means "switch now").

The prediction is a **cost comparison of two survival lines** (`policy_builder.py:77 build_decisions`,
loop at lines 106–121):

```
stay(f)        = E[ BF16 decode steps from f  x TPOT_bf16(ctx, live) ] + slope x E[ BF16 tokens ]
switch(f -> F) = BF16 steps f..F + P(alive at F | at f) x E[ W4 steps from F x TPOT_w4 ] + slope x E[ W4 tokens ]
decision(f)    = argmin_F switch(f -> F)  if it beats stay(f), else 0
```

`TPOT` comes from the profiled heatmap (`cost_model.py:78 make_tpot_cache`, `tpot_grid.py`);
`slope` is the downstream (trainer) seconds per generated token (`cli.py --downstream-slope`,
fitted with `fit-downstream`); the expected steps/tokens come from the survival lines
(`cost_model.py:123 trajectory_cost_grid`, which also converts per-request survival into the
expected *non-empty-batch* decode steps for `live` requests).

Two lines feed it:

* **BF16 line**: `P(a request alive at s finishes in bin s)` for every bin, as a hazard table
  (`hazard.py:38 HazardTable`, survival = product of `1 - hazard`, `hazard.py:138 survival`).
* **W4 line(s)**: the same quantity *after a switch*, i.e. what a request that was BF16 up to `F`
  does once it continues in W4.

Everything below is about where these two lines come from and how they move during the run.

## 2. Difference 1 — the W4 line is calibrated per switch point (tail-W4 groups)

Old design: one W4 table from a *uniform* W4 rollout (every request decoded in W4 from token 0,
`calibration.py:115 paired_traces`). Used as the after-switch line at every `F`, it is the wrong
conditional: a request that reached `F` in BF16 and then continues in W4 is not a request generated
in W4 from scratch (uniform W4 inflates length and loses reward; tail-W4 measured 0.9–1.0x at step 0).

New design (`calib-tail-w4`, `cli.py:287`; module `tail_w4_calibration.py`):

1. Run the 256 calibration prompts once in BF16 (recipe path 1,
   `examples/precision_scheduler/recipes/calibrate_tail_w4.sh`).
2. Choose cuts as **population quantiles** of the BF16 lengths, floored to the 250 grid
   (`tail_w4_calibration.py:76 cut_frontiers`; default quantiles `DEFAULT_QUANTILES` = 2/3, 3/4,
   4/5, 9/10 → `--cut-quantiles`, or explicit `--cut-tokens`; the Phi run used 1000/1500/2500, the
   9B run 4250/7000/9000, the 4B run 5000/8250/10750).
3. For every cut `c` and **every request still generating at `c`**, re-issue prompt + its own BF16
   prefix (the first `c` generated tokens) and decode the rest under uniform W4 with
   `max_tokens = cap - c` (`tail_w4_calibration.py:139 plan_continuations`, engine at `:302
   VllmEngine`, trace writer at `:204`). Cost is Σ(1 − q) ≈ 0.88 continuations per request, one
   batch. Long requests appear in every group, by design.
4. The continuation trace becomes one **`W4Group` per cut** (`calibration.py:47 W4Group`;
   `calibration.py:129 grouped_traces`): a hazard table built with *delayed entry* at the cut
   (`hazard.py:67 components(entries=cut, finals)` — bins below the cut stay unobserved), so the
   group is exactly "given BF16 up to `c`, the W4 continuation". The uniform-W4 trace, if given
   (`--w4-trace`), is kept as the cut-0 group.
5. **Routing**: a candidate switch at `F` is priced with the group whose cut is the largest ≤ `F`
   (`calibration.py:72 w4_group_for`; used by `policy_builder.py:65 _w4_tables_by_frontier`, which
   maps every frontier to its group's table before the search at `:102-103`). A single group with
   cut 0 reproduces the legacy behaviour exactly, so the old traces still work.

## 3. Difference 2 — both lines are updated online with the same weighted EMA

### 3.1 The update rule (`hazard.py:113 weighted_update`)

A cohort of `n` new requests is turned into a fraction table (`hazard.py:67 components`) and blended
into the current table on the bins it observed:

```
w     = max( n / (prior_weight + n_seen + n), alpha_min )        # hazard.py:134
table = (1 - w) * old + w * new                                   # on observed bins only, hazard.py:99 ema_update
```

`prior_weight` = how many online requests the calibration is worth (64 in the b32c24 runs;
`--prior-weight`), `n_seen` = online requests already blended into that table, `alpha_min` = 0.05
floor so the table keeps tracking drift. The weight starts near `n/64` and decays — a 4-request
cohort after 200 requests moves a bin by ~2%, not by a fixed 20% (the old fixed-α rule,
`--update ema`, let one cohort flip the table). One rule, applied per rollout to each line:

### 3.2 W4 groups (`online_ema.py:73 replay_cohorts`)

vLLM appends one `switch_cohort` row per rollout to `switch_observations.jsonl` (request ids and
`entry_output_tokens` at the switch). When the rollout completes (all `B` requests finished,
`traces.py:98 completed_steps`), the cohort's `(entry, final length)` pairs
(`traces.py:123 cohort_observation`) are routed to the group with the largest cut ≤ the cohort's
median entry (`online_ema.py:104-105`), built with delayed entry (`:106 components(entries, finals)`)
and blended (`:108-110 weighted_update`). So the group that priced the switch is the one that learns
from it.

### 3.3 BF16 line (`online_ema.py:133 replay_bf16`, opt-in `--bf16-online`)

Every completed rollout is a cohort of `B` BF16 requests, all with entry 0:

* a request that finished under BF16 → event at its length (unless it hit the cap);
* a request that switched at `e` → **censored at `e`**: at risk in every bin below `e`, then out of
  the risk set without an event (`online_ema.py:168-170`: `exits = min(final, e)`,
  `components(..., events=~switched)`; the `events` mask is the one extension made to
  `components`, `hazard.py:67-80`).

Then the same `weighted_update` with the same prior (`online_ema.py:171`;
`--bf16-prior-weight` defaults to `--prior-weight`). The watcher refreshes both lines in
`online_ema.py:231 observe` and rebuilds the policy in `:265 poll`, with hysteresis
(`policy_builder.py:125 limit_decision_step`, ≤ `--max-step-tokens` = 2000 per revision per cell).

### 3.4 Why the BF16 line was added, and its known failure mode

Replaying the 2026-09-19 W4-only arms (`b32c24_*_ema_tailw4_p64`) showed the W4 groups learning
correctly while the BF16 line stayed at its step-0 calibration. On Phi that calibration said 2151
tokens remain from 1750 while the trained policy had ~1600, so the table kept saying "switch at
1750" against a learned W4 tail of 2100 (ratio 1.27 in reality, 0.92 in the table). Updating only
one half of the comparison can make it worse than updating neither.

The asymmetry that remains is in the **data**, not the rule: a switch gives the W4 group a complete
tail, but it removes every BF16 observation beyond `F` (those requests became W4). The BF16 line
therefore learns the current policy below the switch point and keeps the stale calibration above it.
Two consequences, both observed:

* W4-only (BF16 frozen): stuck late when the policy shortens (Phi 0.93x).
* Both lines, censored data only (`b32c24_phi4_mini_reasoning_ema_both_p64`, 2026-09-21): tracked
  the truth through rollout 20, then — as Phi's responses shortened — the W4 groups followed, the
  BF16 tail beyond the switch could not, the ratio flipped to 0.77 and the run locked into switching
  everyone at 750 (12–29 live). 0.957x, better than W4-only, wrong mechanism.

Mitigation in the code: `--bf16-probe-every K` (`online_ema.py:301-308`) publishes an all-zero
(never-switch) table every K-th revision so that rollout is pure BF16 and its `B` requests are
uncensored tail evidence (marked `policy["calibration"]["bf16_probe"]`); cost 1/K of the rollouts
without the W4 gain. A cheaper variant replayed but not implemented: switch at `F + 2000` instead
of never on the exploration rollout (observes the bins that decide "switch or wait", keeps the
gain beyond). The CPU replays are in the session scratchpads (`replay_bf16*.py`, `unified.py`,
`bf16_ema.py`).


### 3.5 Shared survival clock (`--update clock`, 2026-09-21) — the current default candidate

Instead of per-bin updates, each line keeps its calibration shape and learns one termination-intensity
parameter from a censored likelihood (`online_ema.py replay_clock`): `S_t = S_0^a`, `a = (kappa + D)/(kappa + E)`,
`D` = natural finishes, `E` = accumulated cumulative hazard of every observation (BF16: censored at the switch;
W4 group g: exposure `H_g(y) - H_g(s)` from the switch position), discounted by `rho` = 0.9 per rollout;
`kappa` = prior weight (BF16) or prior weight x group size / 256 (W4 groups). Frozen tables are `a = 1`. No bin
can be rewritten by a handful of requests, both lines move, and the CPU replay keeps the Qwen decisions inside
the measured flat band where the per-bin update drifted out. Calibrated with 5 cuts at alive fractions
1/2..1/6 and 64 sampled continuations per cut (`--continuations-per-cut 64`; 9B cuts 1750/4250/5500/6250/7000,
4B 2750/5000/6250/7250/8250).

Results, B32 / cap 24k, steps 9-48 vs pooled BF16 (run dirs `b32c24_<model>_clock_p{64,128}`):

| model | arm | rollout | step | tokens | reward | switch point (live) |
|---|---|---:|---:|---:|---:|---|
| 4B | clock p64 | **1.376x** | **1.240x** | -3% | 0.858 | 2750 (15-20) for 16 rollouts, then 6750-7250 (8) |
| 4B | clock p128 | 1.349x | 1.205x | +1% | 0.861 | 6000 then 2250-2750 (14) |
| 4B | W4-only p64 (previous best) | 1.328x | 1.195x | 0% | 0.856 | 3750 (11.5) |
| 9B | clock p64 | 1.194x | 1.081x | +13% | 0.851 | 5500 (7) |
| 9B | clock p128 | 1.193x | 1.070x | +17% | 0.852 | 3250 (11.5) |
| 9B | W4-only p64 | 1.182x | 1.098x | +5% | 0.858 | 8000 (4) |
| 9B | fixed 7000 | 1.239x | 1.126x | +6% | 0.869 | 7000 (6) |

4B: best arm measured, both priors above the W4-only arm, reward unchanged. 9B: on par with W4-only on rollout,
slightly below on step (inside the CIs) and below fixed 7000: it switches earlier (3250-5500 with 7-12 live) and
pays 13-17% tokens; its W4 group parameters did learn the inflation (a_1750 0.7, a_4250 0.75-0.84) but the cost
model still preferred the early switch -- the remaining 9B question is the price of tokens in the cost model, not
the estimator. (But see 3.6: until 2026-09-25 the group pricing a switch often did not receive that switch's
cohort, and the held-out W4 level on this 9B run was 31.5% low.)

### 3.6 Cohort routing and neighbour sharing (2026-09-25)

**Routing fix.** `replay_clock` and `replay_cohorts` used to route a switch cohort by the median of its
requests' `entry_output_tokens`. Those trail the applied frontier by a few tokens (a switch at 2750 records
2746-2750), so a median of 2748 went to the next lower group while the cut-2750 group priced the switch:
the pricing group did not learn from its own decisions. On the 4B clock_p64 seed-42 run only 2 of the first 14
cohorts reached the 2750 group; the replayed estimate there sat on the calibration and jumped only when a
median happened to equal the cut. Cohorts are now routed by `trigger.applied_response_tokens`
(`traces.cohort_frontier`; cohorts without a trigger keep the median).

**Neighbour sharing (`--w4-share-tokens`, default 0 = off).** A cohort also informs groups whose cut is within
that many tokens of its switch, weight `1 - |F - cut| / share`, with delayed entry at the neighbour's cut. Only
the intensity evidence is pooled; each group keeps its calibration shape.

Held-out evaluation on the existing clock_p64 traces (every rollout priced by revision r-1, same trajectories
for all variants; `examples/precision_scheduler/analysis/eval_cohort_routing.py`, old numbers from the
pre-fix commit's tree): pooled expected tokens after the
switch vs realized --

| run | realized | old routing | fixed | share 1000 | share 2000 |
|---|---:|---:|---:|---:|---:|
| 4B s42 / s43 / s44 / s45 | 6207 / 6142 / 6830 / 7044 | -16.8 / -12.2 / -15.8 / -16.5% | -8.8 / -8.1 / -10.5 / -14.7% | -7.5 / -6.3 / -9.6 / -12.2% | -7.5 / -6.1 / -8.3 / -10.7% |
| 9B | 7757 | -31.5% | -26.3% | -25.4% | -18.4% |
| Phi-mini 24K | 1232 | -17.1% | -0.9% | +15.6% | +19.7% |
| Phi-mini 4K | 817 | -37.9% | -13.9% | -14.7% | -15.1% |
| Phi-4 14B | 1012 | +17.2% | +37.5% | +48.4% | +50.6% |

The fix moves the level (what the switch decision prices) toward the realized value on 7 of 8 runs. Per-request
log-likelihood of the continuation is unchanged on 4B (|diff| < 0.005, CIs span 0), n.s. positive on 9B and
Phi-mini 4K, and lower on Phi-mini 24K (-0.037) and Phi-4 14B (-0.028). Phi-4 14B is a model-class limit, not a
routing one: at its usual frontier 500 the calibration predicts 1412 more tokens, the runs realize 823 (median
256), and one intensity parameter cannot close a shape gap that size. Sharing keeps shrinking the 4B/9B
under-estimate but overshoots on both Phi models, so it stays off. The 9B early-switch question should be
re-measured with the fix before it is attributed to the token price alone.

### 3.7 Switch-candidate floor (`--min-switch-frontier`, 2026-09-25) and what the fix did on the GPU

`build_decisions` searched every frontier as a switch candidate. Below the first tail-W4 cut the calibration has no
continuation measured from a BF16 prefix; a switch there is priced by the cut-0 uniform-W4 group, whose data is W4
from the first token. Once the first tail group learns that its continuations run long (which 3.6 now lets it do),
the search escapes to that unpriced region, and the cut-0 group, with four times the prior weight of a tail group,
is slow to correct it: Qwen3.5-4B, Megatron, seed 42, with the routing fix alone switched at 2250 from rollout 16
on (first cut 2750) and was slower than the pre-fix code (271.9 vs 259.3 s/step, steps 9-48). `build_decisions` /
`build_policy` take `min_switch_frontier` (0 = old behaviour); the watcher defaults it to the first tail cut
(`--min-switch-frontier -1`; `0` turns the floor off; an explicit frontier is honoured) and records it in
`policy.json` (`offline_cost_model.min_switch_frontier`).

The floor removed the escape but not the tendency: with it the seed-42 run sat on the floor 2750 for half of its
rollouts, the same place and step time as "no online update", 8% slower than the pre-fix code
(old / new 0.919 [0.851, 0.988]); seeds 43 and 44 were not different from it. Under FSDP2 (KL off) every adaptive
and fixed arm ties. Full tables, CIs and the run directories:
[`RUN_2026-09-25_qwen4b_efficiency_push.md`](RUN_2026-09-25_qwen4b_efficiency_push.md). The open question is
whether the early-switch push comes from the BF16 line (it never observes the BF16 tail after a switch) or from the
one-parameter W4 group update; `PLAN_two_line_ema.md` is the proposal for the estimator that would separate them.
Runs and figures made before 2026-09-25 used the old routing and no floor (`f8cf7f21`).

## 4. Things that are *not* in the model (checked, so nobody re-derives them)

* **Batch size does not move the switch point.** Saving and cost of a switch both scale with the
  live count, so `argmin_F` depends only on the two lines and the per-step INT4 saving, which is
  flat across batch on Qwen (weight-read bound: ~4 ms 9B, ~2.5 ms 4B, ~1.7 ms Phi ≤ 16 live). The
  decision tables are one number per model across live 1..32 (9B 8000, 4B 3750, Phi 1750).
* **No forgetting factor.** Plain accumulation (weight → `n/N`) reproduced the decisions of the
  weighted rule in replay; a discount γ = 0.9 only added noise at 48 rollouts.
* **No proportional/head-to-tail extrapolation for the BF16 line.** Under training the middle and
  the tail of the length distribution move in opposite directions on all three models; scaling the
  calibration tail by the observed/expected ratio below `F` moved it the wrong way.

## 5. Knobs (all in `cli.py watch-ema`, exported by `recipes/continuous_ema.sh`)

| knob | default | env in recipe |
|---|---|---|
| `--update weighted\|ema` | weighted | `EMA_UPDATE` |
| `--prior-weight` | 32 (runs: 64) | `EMA_PRIOR_WEIGHT` |
| `--alpha-min` | 0.05 | `EMA_ALPHA_MIN` |
| `--max-step-tokens` | 2000 | `EMA_MAX_STEP_TOKENS` |
| `--w4-cont-trace` / `--w4-trace` | one required | `W4_CONT_TRACE` / `W4_TRACE` |
| `--bf16-online` | off | `EMA_BF16_ONLINE=1` |
| `--bf16-prior-weight` | = prior-weight | `EMA_BF16_PRIOR_WEIGHT` |
| `--bf16-probe-every` | 0 | `EMA_BF16_PROBE_EVERY` |
| `--rho` (clock) | 0.9 | `EMA_RHO` |
| `--w4-share-tokens` (clock) | 0 (off) | `EMA_W4_SHARE_TOKENS` |
| `--min-switch-frontier` | -1 (first tail-W4 cut; 0 = no floor) | not exported by the recipe (watcher default; override the command with `WATCHER_CMD`) |

Run directories for the evidence: `/data/huanchen/ps_runs/b32c24_<model>_{ema_tailw4_p64,
ema_both_p64, ema_both_p64_probe4, ema_both_p64_seed43}`; calibration traces
`b32c24_<model>_calib_{bf16,full_w4,tailw4}`; per-revision watcher state in
`online_ema_history.jsonl`, switch cohorts in `switch_observations.jsonl`, request lifetimes in
`traces/request_lifetimes_replica000_node000.jsonl`. The watcher is deterministic given those
files, so any revision's tables can be rebuilt on CPU with `replay_cohorts` / `replay_bf16`.
