#!/usr/bin/env python3
"""Held-out score of the W4 continuation prediction under different cohort-routing rules.

For every rollout r of a clock run, the tables of revision r-1 (learned from cohorts 1..r-1 only)
price rollout r's switch at its applied frontier F; every switched request's continuation is scored
by its log-probability under the pricing group's survival curve from F (censored at the cap).
The same realized trajectories are used for every variant, so this compares estimators, not policies.

Run with PYTHONPATH pointing at the tree to evaluate. `--median-routing` strips the triggers so cohorts
fall back to median-entry routing (the pre-2026-09-25 rule, up to the new risk-set filter); run the old
commit's tree for the exact old numbers. `--share` needs a tree with `w4_share_tokens`. The uniform-W4
trace behind each cut-0 group is not recorded in policy.json; the paths below are the matching
calib_full_w4 runs (it only affects the cut-0 group, identically for every variant).

  CUDA_VISIBLE_DEVICES= PYTHONPATH=<tree> python eval_cohort_routing.py --variant fixed --out fixed.json
"""
import argparse, json, logging, sys
from pathlib import Path
import numpy as np

logging.disable(logging.WARNING)
from verl.experimental.precision_scheduler.calibration import grouped_traces, w4_group_for
from verl.experimental.precision_scheduler.cost_model import PolicyGrid
from verl.experimental.precision_scheduler.hazard import survival
from verl.experimental.precision_scheduler.online_ema import replay_clock
from verl.experimental.precision_scheduler.traces import trace_lengths, read_cohorts, cohort_observation

R = Path('/data/huanchen/ps_runs')
TR = 'traces/request_lifetimes_replica000_node000.jsonl'
RUNS = {
    '4B s42': ('b32c24_qwen3_5_4b_clock_p64', 'b32c24_qwen3_5_4b_calib_full_w4'),
    '4B s43': ('b32c24_qwen3_5_4b_clock_p64_seed43', 'b32c24_qwen3_5_4b_calib_full_w4'),
    '4B s44': ('b32c24_qwen3_5_4b_clock_p64_seed44', 'b32c24_qwen3_5_4b_calib_full_w4'),
    '4B s45': ('b32c24_qwen3_5_4b_clock_p64_seed45', 'b32c24_qwen3_5_4b_calib_full_w4'),
    '9B': ('b32c24_qwen3_5_9b_clock_p64', 'b32c24_calib_full_w4'),
    'Phi-mini 24K': ('b32c24_phi4_mini_reasoning_clock_p64', 'b32c24_phi4_mini_reasoning_calib_full_w4'),
    'Phi-mini 4K': ('b32c4_phi4_mini_reasoning_clock_p64', 'b32c24_phi4_mini_reasoning_calib_full_w4'),
    'Phi-4 14B': ('p14b32c24_clock_p64', 'p14b32c24_calib_full_w4'),
}
EPS = 1e-4


def score_run(name, run, uniform, share, median_routing):
    cfg = json.load(open(run / 'run_config.json'))
    cap, batch = int(cfg['response_cap']), int(cfg['requests_per_step'])
    meta = json.load(open(run / 'policy.json'))['calibration']
    src = meta['base_source']
    grid = PolicyGrid(step=250, cap=cap, batch=batch, prompt_step=128, prompt_max=0)
    cal = grouped_traces(Path(src['bf16_trace']), Path(src['w4_trace']), grid, requests=256,
                         include_uniform_w4=uniform / TR)
    starts, finishes = trace_lengths(run / TR)
    cohorts = read_cohorts(run / 'switch_observations.jsonl')
    learn = ([{k: v for k, v in c.items() if k != 'trigger'} for c in cohorts]
             if median_routing else cohorts)
    kwargs = dict(prior_weight=float(meta.get('prior_weight', 64.)), rho=float(meta.get('rho', .9)))
    if share > 0:
        kwargs['w4_share_tokens'] = share
    n_bins = len(grid.frontiers)
    rows = []
    for c in cohorts:
        r = int(c['rollout_index'])
        obs = cohort_observation(c, finishes, cap)
        if obs is None:
            continue
        _, fin, _ = obs
        F = int(c['trigger']['applied_response_tokens'])
        groups = replay_clock(cal, starts, finishes, [x for x in learn if int(x['rollout_index']) <= r - 1],
                              grid, completed=r - 1, **kwargs)[1]
        i0 = min(F // grid.step, n_bins - 1)
        S = survival(w4_group_for(groups, F).table, i0)
        logp, floored = 0.0, 0
        for f in fin:
            if f >= cap:
                p = S[-1]
            else:
                j = int(np.clip(f // grid.step - i0, 0, len(S) - 1))
                p = S[j] - (S[j + 1] if j + 1 < len(S) else 0.0)
            floored += p < EPS
            logp += float(np.log(max(p, EPS)))
        rows.append(dict(rollout=r, frontier=F, n=int(len(fin)), logp=logp, floored=int(floored),
                         predicted=float(S[1:].sum() * grid.step),
                         realized=float(np.mean(fin - F))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', required=True)
    ap.add_argument('--share', type=float, default=0.0)
    ap.add_argument('--median-routing', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    out = {}
    for name, (run, uniform) in RUNS.items():
        out[name] = score_run(name, R / run, R / uniform, a.share, a.median_routing)
        n = sum(x['n'] for x in out[name]); lp = sum(x['logp'] for x in out[name])
        print(f'{a.variant:10s} {name:13s} requests {n:5d}  mean log p {lp / n:+.4f}', flush=True)
    a.out.write_text(json.dumps(dict(variant=a.variant, share=a.share,
                                     median_routing=a.median_routing, runs=out)))


if __name__ == '__main__':
    main()
