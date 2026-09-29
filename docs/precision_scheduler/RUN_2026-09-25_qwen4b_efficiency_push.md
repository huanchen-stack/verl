# Qwen3.5-4B efficiency push (2026-09-25): cohort routing fix, switch floor, and what they did to the step time

Date: 2026-09-25 (runs 10:40-18:00 UTC). Host: 8 GPUs, Megatron TP1 DP1 per run (the
reporting trainer) and FSDP2 single-GPU arms. Model: Qwen3.5-4B, B32, cap 24K, 48 steps,
timing window steps 9-48, every comparison paired step by step (bootstrap CIs, independent
and block length 4). Companion PDF with every table and figure:
[`RUN_2026-09-25_qwen4b_efficiency_report_v2.pdf`](RUN_2026-09-25_qwen4b_efficiency_report_v2.pdf)
(in Chinese; report v2, 17:52 UTC).

Purpose: answer review points W2-W6 (timing definition, preparation cost, prediction validity,
memory, confidence intervals) for the efficiency claims, and fix the online-update bugs found
on 2026-09-25 "in the correct direction". Training / accuracy (W1) was a separate campaign.

## Code

Three commits, developed on a side branch that day and merged into
`rollout-precision-scheduler-clean` on 2026-09-29 (the side branch is gone):

| commit (on the side branch) | what |
|---|---|
| `986916e8` | **Routing fix.** `replay_clock` / `replay_cohorts` route a switch cohort to the W4 group of the frontier it switched at (`trigger.applied_response_tokens`, `traces.cohort_frontier`) instead of the largest cut at or below the cohort's median entry. Entries trail the applied frontier by a few tokens, so a switch at 2750 (median entry 2748) used to train the cut-0 uniform-W4 group while the cut-2750 group priced it. |
| `a703ca55` | **Neighbour sharing** `--w4-share-tokens S` (recipe `EMA_W4_SHARE_TOKENS`), default 0 = off: a cohort also informs groups whose cut is within `S` tokens, weight `1 - |F - cut| / S`, intensity evidence only. Plus `analysis/eval_cohort_routing.py`, the held-out estimator comparison recorded in `prediction_model.md` 3.6. |
| `cc1eea4e` | **Switch floor** `--min-switch-frontier` (`build_decisions(min_switch_frontier=...)`): no candidate switch below the first tail-W4 cut. Watcher default `-1` = first tail cut (0 without tail groups), `0` = old behaviour, or an explicit frontier; recorded in `policy.json` `offline_cost_model.min_switch_frontier`. |

The paper's runs used `f8cf7f21` (old routing, no floor). **The merged branch's watcher
defaults now differ from that code**: routing by switch frontier is unconditional and the
floor is on. To reproduce a paper run, check out `f8cf7f21`; to run the merged code with the
old search space, pass `--min-switch-frontier 0` (there is no switch back to median-entry
routing).

Run-directory suffixes below: `_fsdp` = FSDP2, KL off, `a703ca55`; `_mg` = Megatron, KL on,
`a703ca55` (routing fix, no floor); `_mgf` = Megatron, KL on, `cc1eea4e` (routing fix + floor);
`_old` / `b32c24_*` = `f8cf7f21`. All under `/data/huanchen/ps_runs/eff2_qwen3_5_4b_<arm>_s<seed>_<suffix>`
(BF16, fixed-K, threshold and uniform-W4 arms are the earlier `b32c24_qwen3_5_4b_*` runs; their
behaviour does not depend on the watcher code).

## Results

### FSDP2, KL off (steps 9-48, step time in s; BF16 272.6 s)

| arm | code | step | vs BF16 | switch point |
|---|---|---:|---:|---|
| Tail-W4 (clock p64) | routing fix | 223.4 | 1.220x | 2750 for most rollouts, 2250 at the end |
| Tail-W4 (clock p64) | old routing, s42 / s43 | 227.4 / 228.1 | 1.199x / - | 6000-7250 |
| No online update (beta = 0) | - | 221.1 | 1.233x | 2750 |
| fixed K = 2750 | - | 220.1 | 1.239x | 2750 |
| fixed K = 5000 | - | 223.4 | 1.220x | 5000 |
| fixed K = 7000 | - | 218.7 | 1.247x | 7000 |

Every pairwise difference among the adaptive and fixed arms is not significant: under FSDP2
(no reference pass) the cost landscape is flat, and the calibration-only choice K = 2750 is as
good as Tail-W4. The paper's DP and long-training sections are FSDP2; their speedup over BF16
stands, but they should not suggest an adaptive gain over a simple rule.

### Megatron, KL on (steps 9-48 unless noted; BF16 316.7 s s42, 326.1 s s43)

| arm | code | seed | window | step | switch point (last third) |
|---|---|---|---|---:|---|
| Tail-W4 | `f8cf7f21` (paper) | 42 / 43 / 44 / 45 | 9-48 | 259.3 / 272.5 / 272.1 / 266.4 | 6750-7250 / 2750 / 2750 / 7250 |
| Tail-W4 | routing fix (`_mg`) | 42 / 43 | 9-48 | 271.9 / 272.8 | 2250 (below the first cut 2750) |
| Tail-W4 | routing fix + floor (`_mgf`) | 42 / 43 / 44 | 9-35 / 9-31 / 9-24 | 286.9 / 280.0 / 285.4 | 2750 (51% of rollouts at the floor) / 6750 / 6750 |
| No online update | - | 42 / 43 | 9-48 | 288.9 / 272.2 | 2750 |
| fixed K = 5000 (best fixed, post hoc) | - | 42 / 43 | 9-48 | 270.4 / 276.4 | 5000 |
| fixed K = 3000 (calibration-only pick) | - | 42 | 9-48 | 285.1 | 3000 |

Four seeds of the paper's code give 1.202 +- 0.028x on the full step; the paper's seed 42
(1.24x) is the luckiest. Paired ratios of the new code against the same seed of the old code
(old / new, < 1 means the old code is faster; windows as in the table):

| new code vs | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| old code, same seed | 0.919 [0.851, 0.988] | 0.961 [0.879, 1.037] | 1.031 [0.903, 1.147] |
| routing fix only, same seed | 0.964 [0.902, 1.026] | 1.000 [0.924, 1.086] | - |
| fixed K = 5000, same seed | 0.952 [0.888, 1.009] | 1.004 [0.909, 1.118] | - |

**What happened.** With the routing bug, the group that priced a switch never received its
cohort, so the decision stayed where the calibration plus the BF16-side update put it
(6750-7250), inside the measured flat band. With the fix the pricing group learns that its
continuations run long, and the planner moves the switch earlier: to 2250, below the first
tail cut, where the candidate is priced by the uniform-W4 group that has no BF16-prefix data
(`_mg`); with the floor it sits on the floor 2750 (`_mgf`, seed 42), the same place and the
same step time as "no online update". Under Megatron the fixed-K sweep says early switches
are 5-8% slower than K = 5000, so the fixed code is slower on seed 42 and not different on
seeds 43/44. The three `_mgf` ablations were stopped at steps 30-33 (GPUs returned to the Phi
campaign); in that window "no downstream cost" was faster than the full method
(0.896 [0.816, 0.973]), which points at the learning pushing toward early switches, not at a
missing component. Whether the root cause is the BF16 line (it never sees the BF16 tail after a
switch and extrapolates from data before the switch point) or the one-parameter W4 group
update was not resolved; it needs its own experiment (see `PLAN_two_line_ema.md`).

### The paper's code: ablations and fixed rules against Tail-W4 (review W6)

Ratio = other / Tail-W4 (`f8cf7f21`), > 1 means Tail-W4 is faster, Megatron, KL on, steps 9-48.

| comparison | ratio | 95% CI (per step) | 95% CI (block 4) | seeds |
|---|---:|---|---|---|
| No online update | 1.055 | [1.017, 1.095] | [1.025, 1.086] | 42, 43 |
| No downstream cost | 1.056 | [0.993, 1.123] | [1.005, 1.100] | 42 |
| Constant TPOT | 1.049 | [1.004, 1.098] | [1.006, 1.088] | 42 |
| No state replanning | 1.051 | [0.992, 1.112] | [0.992, 1.082] | 42 |
| fixed K = 5000 (post-hoc best) | 1.028 | [0.984, 1.078] | [0.992, 1.064] | 42, 43 |
| fixed K = 3000 (calibration-only pick) | 1.100 | [1.050, 1.156] | [1.047, 1.153] | 42 |
| fixed K* = 11000 (8-step pilot pick) | 1.063 | [1.000, 1.134] | [1.010, 1.121] | 42 |

Reading: Tail-W4 ties the post-hoc best fixed K and beats the fixed rules obtainable with the
same preparation budget; "no online update" and "constant TPOT" are significant under both
intervals, "no downstream cost" only under the block bootstrap, "no state replanning" is not.

### Other review points (measured, independent of the code change)

* **W2, timing definition.** Table 1's 239 s and Figure 6's 259 s differ by the reference
  (KL) pass, 17 s, plus the weight sync, 4.5 s; the measured components add up to the step
  time within 0.03 s. Measured without any `nvidia-smi` calls the downstream fit slope agrees
  with the scheduler's fit within 1% (0.331 vs 0.328 ms/token), so there is no measurement
  contamination. On the full measured step 4B Tail-W4 is 1.24x (not 1.26x) and uniform W4
  1.00x (not 1.03x).
* **W3, preparation cost.** 87 min for 4B as run (BF16 calibration 34, uniform-W4 24, tail-W4
  continuations 22, TPOT grid 8); break-even about 84 steps. Reusing the first 8 training
  steps' BF16 rollouts and dropping the uniform-W4 trace: 38 min, break-even about 36 steps.
  Picking a fixed K with a pilot costs about 266 min of training, three times the Tail-W4
  preparation.
* **W5, prediction validity.** Calibration alone ranks the optimum at K = 3000 (Spearman
  +0.33 against the 8 measured fixed K), 5% worse than the best; after online learning the
  ranking is +0.57 to +0.67 and the predicted optimum lies inside the measured flat band.
  BF16 drain curves are within 0.6 of 32 requests on average but miss per-rollout variation.
* **W6, memory.** Dual-precision residency cuts KV capacity by 11% (853,580 to 759,883
  tokens); the B32 peak uses 17% of it, no preemption.

## Recommendation recorded that day

Keep the paper's numbers on `f8cf7f21` and report the four-seed mean; state in the
limitations that the online update's cohort routing had an implementation problem whose fix
exposes a tendency to switch early. The three commits were merged on 2026-09-29 so the branch
carries the corrected estimator; the early-switch tendency is the open item.

Scripts (launcher, `analyze.py`, `figs_*.py`, `make_tables.py`, report TeX) live in the
session scratchpad `/tmp/claude-1004/-data-huanchen/4d869d23-67f5-4e71-b172-03a3271d9257/scratchpad/lt/`;
the two PDFs also in `/data/huanchen/efficiency_report_last_try/`.
