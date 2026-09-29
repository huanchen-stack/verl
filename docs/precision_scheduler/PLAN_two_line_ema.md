# Plan: two-line online estimation (BF16 and after-switch W4) with exploration

Status: proposal, 2026-09-20. Nothing below is implemented. Runtime (vLLM scheduler, lookup table, receding-horizon
decider) is unchanged; every change is in the C6 watcher that writes `policy.json` between rollouts.

## 1. What the 48-step B32 / cap-24k runs showed (evidence)

| finding | where |
|---|---|
| The lookup table is a single threshold in practice: every state below F commits to F, every state above says "now"; the live axis carries no information. | 9B table rev 48: 11750 in all 1472 cells below it |
| The threshold never moves earlier because the W4 line below the tried point is never observed; it stays at the uniform-W4 (from-scratch) calibration, the pessimistic extreme. | `components()` only writes bins >= cohort entry |
| A fixed alpha = 0.2 per cohort of 1-5 requests makes the tail estimate a 5-cohort moving average; it jitters +-0.1 around the pooled truth and never tightens. | P(cap \| 11750) by revision: 0.38-0.62 around 0.50 |
| One bad cohort ratchets the threshold later and nothing can bring it back. | 4B: 93 requests said 5750 was fine; rollout 10's 4 cap-runners moved it to 16500 for the remaining 38 steps |
| The BF16 line is never updated; for 9B its calibration deep tail is 1.3-1.5x too heavy, which also argues for a later switch. | P(reach 16k) 0.055 predicted vs 0.041 in 2560 BF16 requests |
| Early switching was not worse where it was tried: 4B rollouts switched at 5750 with 8-15 live gave 1.20x step vs 1.11x at 16500 (9 steps; direction, not proof). | b32c24 4B arm |
| The after-switch behaviour depends on how much runway is left (entry 5-9k: +17-21%; entry 16k+: +1-7%), so one shared stretch factor cannot be extrapolated from late cohorts to early frontiers (emulation: k=1.04 -> "switch at 250", contradicted by uniform W4 at 0.86x). | tail_t4 cohorts; emulate_update.py |

## 2. Design

Two lines per model, both maintained online:

* **P16(s)** — survival of a request generated in BF16. Initialised from the BF16 calibration (256 distinct prompts).
  Updated after every rollout from **all 32 requests**: requests that finished before the switch are complete
  observations; switched requests are right-censored at their entry frontier. Kaplan-Meier increments per 250-token
  bin, blended into the line with a count-weighted alpha (below). Beyond the deepest censoring point the previous
  line's conditional shape is kept.
* **P4(s | entry bucket)** — after-switch continuation, stored per entry bucket of 1000 tokens (survival ratios from the
  entry bin onward), plus per-bucket request counts. Initialised from the uniform-W4 calibration as today, but flagged
  as prior weight n0 = 24 request-equivalents (about one week-one cohort) so that real cohorts dominate quickly.
  Updated from the switched cohort into its entry bucket only. Neighbouring buckets are informed through a smoothness
  prior (shrink toward the average of adjacent buckets weighted by their counts), not through extrapolation from a
  single global factor.
* **Update rule** for both lines: alpha_k = max(n_k / (n0 + N_k), alpha_floor) with N_k the requests seen so far in that
  bucket and alpha_floor = 0.02 (slow drift tracking, about a 150-request memory once converged). Blend the counts
  (risk and event), not the ratios.
* **Decision** — unchanged cost model (heatmap TPOT, downstream slope) and unchanged `build_decisions`, but the W4
  branch for a candidate F uses P4 of F's entry bucket, and buckets with fewer than n_min = 16 observed requests are
  evaluated **optimistically**: their survival ratio is the lower confidence edge (Wilson, 80%) of the estimate, so an
  untried frontier is priced no worse than the data allows. This is what makes the table try earlier points.
* **Exploration (table-level, no runtime change)** — on designated rollouts the watcher publishes the table with the
  committed frontier moved earlier by delta = 1500 tokens from the current argmin (and at most to the earliest bucket
  with < n_min observations). Schedule: every 4th rollout while any bucket between 2000 and the current argmin has
  < n_min requests; afterwards every 8th rollout. Additionally, on any rollout state with live <= 2 the table commits to
  "now" (a drained batch is nearly free to switch and still yields an observation). Exploration rollouts are marked in
  `switch_observations.jsonl` (`"explore": true`) so they can be reported separately.
* **Never-switch must be reachable**: if for every F the optimistic W4 plan is still more expensive than staying (Phi
  case), the table row is 0 (never). Unchanged in the builder; only the inputs change.

## 3. Code changes (C6 only)

* `hazard.py`: `HazardTable` gains `count` per bin; `ema_update(old, new, alpha)` -> `blend_counts(old, new, alpha)`;
  new `km_from_lengths(lengths, censored_at)`; `survival()` unchanged.
* `online_ema.py`: ingest every request of the completed rollout from the trace (lengths + entry frontier from the
  cohort file); maintain `P16` and the bucketed `P4`; exploration schedule; write `explore` flag and per-revision line
  snapshots (`lines_history.jsonl`) for the learnability figure.
* `policy_builder.py`: accept a per-entry-bucket W4 table and an optimistic flag; otherwise unchanged.
* `cli.py`: new flags `--alpha-floor`, `--prior-weight`, `--explore-every`, `--explore-delta`, `--n-min`.
* Tests: unit (count-weighted blend converges to pooled; censoring; optimistic bound; never reachable; explore
  schedule), and a CPU replay test (below) as a regression on the 48-step traces.

## 4. CPU validation before any GPU

Replay environment built from real switched cohorts: tail_t4 (entries 6k-22k, B64/24k), sweep16 (6.5k-10k, 16k cap),
b32c24 EMA arms (11750 / 16500 / 1750 / 3000 / 6500), mg20 fixed_k5000. For a simulated rollout the environment draws
a BF16 batch from the pure-BF16 traces and, when the policy switches at F, replaces each live request's continuation
with one drawn from real switched requests whose entry is within 1000 tokens of F (same model). Pass criteria:

1. On 9B the argmin walks from the calibration's 11750-13250 into 6000-9000 within 15 rollouts and stays; on Phi it
   reaches "never" or a frontier where no request is switched; on 4B one 4-request cap cohort moves the estimate by
   < 0.05, not 0.20.
2. Exploration cost < 1% of simulated step time over 48 rollouts.
3. Predicted rollout + downstream over 48 rollouts is no worse than the fixed-11750 policy on 9B.

## 5. GPU experiment (after 4)

Same protocol as b32c24 (B32, cap 24k, Megatron TP1, 48 steps, first 8 warm-up), reuse the existing BF16 seeds and
uniform-W4 arms as baselines; new arm "Tail-W4 v2" x 2 seeds per model. Order: 9B and 4B on GPUs 1, 3, 4, 6 (~7 h),
then Phi (~1.5 h). Report: the b32c24 table with a v2 row (downstream from the clean fit), the committed frontier per
rollout (the walk), exploration rollouts marked, and the two lines at revisions 0 / 8 / 24 / 48 against the pooled
truth (the learnability figure). Success: 9B >= 1.15x step on steps 9-48 with tokens within +-2% and reward within CI;
frontier trajectory visibly converging rather than constant.
