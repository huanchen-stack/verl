# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the tail-W4 calibration path, grouped calibration, group routing, the weighted update
and the hysteresis guard.  No engine: the continuation run uses a fake engine."""

import json

import numpy as np
import pytest

from verl.experimental.precision_scheduler import tail_w4_calibration as tw
from verl.experimental.precision_scheduler.calibration import (
    InitialCalibration,
    W4Group,
    from_finals,
    grouped_traces,
    paired_traces,
    w4_group_for,
)
from verl.experimental.precision_scheduler.cost_model import PolicyGrid
from verl.experimental.precision_scheduler.hazard import HazardTable, components, ema_update, survival, weighted_update
from verl.experimental.precision_scheduler.online_ema import replay_cohorts
from verl.experimental.precision_scheduler.policy_builder import build_decisions, limit_decision_step
from verl.experimental.precision_scheduler.traces import continuation_lengths

CAP = 4000
GRID = PolicyGrid(step=250, cap=CAP, batch=8, prompt_step=128, prompt_max=0)


def write_bf16_trace(path, lengths, prompt_tokens=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i, n in enumerate(lengths):
            rid = f"engine-{i}"
            f.write(json.dumps({"event": "start", "request_id": rid, "trace_request_id": f"idx-{i}_0", "prompt_tokens": prompt_tokens, "timestamp": float(i)}) + "\n")
            f.write(json.dumps({"event": "finish", "request_id": rid, "trace_request_id": f"idx-{i}_0", "generation_tokens": n, "finish_reason": "length" if n >= CAP else "stop", "token_ids": list(range(100, 100 + n)), "timestamp": float(i) + 1}) + "\n")


class FakeEngine:
    """Continuation length = min(max_tokens, remaining BF16 length * factor): a controllable inflation."""

    def __init__(self, factor=1.5):
        self.factor = factor
        self.seen = []

    def generate(self, plan):
        out = []
        for req in plan:
            self.seen.append(req.request_id)
            remaining = req.bf16_length - req.cut
            n = int(min(req.max_tokens, max(1, round(remaining * self.factor))))
            out.append(tw.ContinuationResult(req.request_id, n, "length" if n >= req.max_tokens else "stop", list(range(n))))
        return out


def test_cut_frontiers_quantiles_and_tokens():
    lengths = list(range(100, 3300, 100))  # 32 requests
    cuts = tw.cut_frontiers(lengths, quantiles=(0.5, 0.75), step=250, cap=CAP)
    assert cuts == sorted(set(cuts)) and all(c % 250 == 0 for c in cuts) and cuts[0] < cuts[1]
    assert tw.cut_frontiers(lengths, tokens=[0, 1000, 1000], step=250, cap=CAP) == [0, 1000]
    with pytest.raises(ValueError):
        tw.cut_frontiers(lengths, tokens=[CAP], step=250, cap=CAP)
    with pytest.raises(ValueError):
        tw.cut_frontiers(lengths, tokens=[300], step=250, cap=CAP)
    with pytest.raises(ValueError):
        tw.cut_frontiers(lengths, step=250, cap=CAP)


def test_plan_population_alive_at_each_cut(tmp_path):
    lengths = [300, 800, 1200, 2500, 4000, 4000]
    trace = tmp_path / "bf16.jsonl"
    write_bf16_trace(trace, lengths)
    rows = tw.bf16_requests(trace, cap=CAP)
    plan = tw.plan_continuations(rows, [1000, 2000], cap=CAP, prompt_ids_for=lambda tid: list(range(10)))
    per_cut = {c: sorted(r.trace_request_id for r in plan if r.cut == c) for c in (1000, 2000)}
    # alive at 1000: 1200, 2500, 4000, 4000 ; alive at 2000: 2500, 4000, 4000 -> cap-runners appear in every group
    assert per_cut[1000] == ["idx-2_0", "idx-3_0", "idx-4_0", "idx-5_0"]
    assert per_cut[2000] == ["idx-3_0", "idx-4_0", "idx-5_0"]
    req = next(r for r in plan if r.request_id == "idx-3_0@2000")
    assert req.prefix_ids == tuple(range(100, 2100)) and req.max_tokens == CAP - 2000 and req.input_ids[:10] == list(range(10))
    assert tw.split_request_id(req.request_id) == ("idx-3_0", 2000)
    # cut 0 = uniform W4: every request, no prefix
    plan0 = tw.plan_continuations(rows, [0], cap=CAP, prompt_ids_for=lambda tid: list(range(10)))
    assert len(plan0) == len(lengths) and all(r.prefix_ids == () and r.max_tokens == CAP for r in plan0)
    # prompt reconstruction mismatch is an error, not a silent skew
    with pytest.raises(ValueError):
        tw.plan_continuations(rows, [1000], cap=CAP, prompt_ids_for=lambda tid: list(range(11)))


def test_run_writes_trace_that_grouped_traces_reads(tmp_path):
    lengths = [300, 800, 1200, 2500, 3100, 4000, 4000, 900]
    bf_trace = tmp_path / "bf16" / "trace.jsonl"
    write_bf16_trace(bf_trace, lengths)
    rows = tw.bf16_requests(bf_trace, cap=CAP)
    cuts = [1000, 2000]
    plan = tw.plan_continuations(rows, cuts, cap=CAP, prompt_ids_for=lambda tid: list(range(10)))
    out = tmp_path / "tail" / "trace.jsonl"
    writer = tw.ContinuationTraceWriter(out, log_tokens=True, clock=lambda: 0.0)
    engine = FakeEngine(factor=1.5)
    results = tw.run_continuations(engine, plan, writer)
    writer.close()
    assert len(results) == len(plan) == 5 + 4
    obs = continuation_lengths(out, cap=CAP)
    assert len(obs) == len(plan)
    # final = cut + continuation, capped; e.g. idx-3 (2500) cut at 1000 -> 1000 + 1.5*1500 = 3250
    assert (1000, 3250) in obs and all(final <= CAP for _, final in obs)
    manifest = tw.manifest(cuts, plan, quantiles=None, cap=CAP, bf16_requests_count=len(rows))
    assert manifest["continuations_per_cut"] == {1000: 5, 2000: 4}
    cal = grouped_traces(bf_trace, out, GRID, requests=len(lengths))
    assert cal.cuts == [1000, 2000] and [g.requests for g in cal.w4_groups] == [5, 4]
    # a group's table has no observation below its cut and full observation from it
    g2 = cal.w4_groups[1].table
    assert not g2.observed[: GRID.frontier_index(2000)].any() and g2.observed[GRID.frontier_index(2000)]
    # with the legacy pure-W4 trace added it becomes the cut-0 group and w4 aliases it
    cal2 = grouped_traces(bf_trace, out, GRID, requests=len(lengths), include_uniform_w4=bf_trace)
    assert cal2.cuts == [0, 1000, 2000] and cal2.w4 is cal2.w4_groups[0].table


def test_group_routing():
    t = HazardTable.empty(len(GRID.frontiers))
    groups = [W4Group(1000, t, 1), W4Group(2000, t, 1), W4Group(3000, t, 1)]
    assert w4_group_for(groups, 2999).cut == 2000
    assert w4_group_for(groups, 3000).cut == 3000
    assert w4_group_for(groups, 500).cut == 1000  # below the first cut: fallback to the lowest group
    with pytest.raises(ValueError):
        w4_group_for(groups, 500, fallback_lowest=False)
    single = InitialCalibration(bf16=t, w4=t, metadata={"w4_requests": 7})
    assert single.cuts == [0] and single.w4_groups[0].requests == 7


def test_single_group_reproduces_legacy_decisions():
    rng = np.random.default_rng(0)
    bf = rng.integers(200, CAP, 64)
    w4 = np.minimum(CAP, (bf * 1.3).astype(int))
    cal = from_finals(bf, w4, GRID)
    tpot = {"bf16": np.full((len(GRID.frontiers), 1, GRID.batch), 14.0), "w4": np.full((len(GRID.frontiers), 1, GRID.batch), 10.0)}
    legacy = build_decisions(cal.bf16, cal.w4, tpot, GRID, 0.0009)
    grouped = build_decisions(cal.bf16, cal.w4_groups, tpot, GRID, 0.0009)
    assert np.array_equal(legacy, grouped)


def test_groups_change_the_decision_where_the_tail_differs():
    """A tail-W4 group that inflates less than the from-scratch W4 population moves the switch earlier."""
    rng = np.random.default_rng(1)
    bf = rng.integers(200, CAP, 200)
    heavy = np.minimum(CAP, (bf * 1.8).astype(int))  # uniform W4: heavy inflation
    cut = 1000
    alive = bf[bf >= cut]
    light = np.minimum(CAP, cut + ((alive - cut) * 1.05).astype(int))  # switched at 1000: almost no inflation
    cal_heavy = from_finals(bf, heavy, GRID)
    groups = [W4Group(0, cal_heavy.w4, len(heavy)), W4Group(cut, components(np.full(len(light), cut), light, GRID), len(light))]
    tpot = {"bf16": np.full((len(GRID.frontiers), 1, GRID.batch), 14.0), "w4": np.full((len(GRID.frontiers), 1, GRID.batch), 10.0)}
    d_heavy = build_decisions(cal_heavy.bf16, cal_heavy.w4, tpot, GRID, 0.0009)
    d_groups = build_decisions(cal_heavy.bf16, groups, tpot, GRID, 0.0009)
    # the grouped table prices a switch at the cut with the light group and commits exactly there at every live
    # count; the legacy table, which only knows the heavy from-scratch population, decides differently
    assert (d_groups[0, 0, :] == cut).all()
    assert not np.array_equal(d_heavy[0, 0, :], d_groups[0, 0, :])


def test_weighted_update_resists_a_small_cohort_and_replay_routes_groups():
    base = HazardTable(risk=np.full(4, 1.0), event=np.full(4, 0.10), observed=np.ones(4, bool))
    spike = HazardTable(risk=np.full(4, 1.0), event=np.full(4, 0.90), observed=np.ones(4, bool))
    ema, _ = ema_update(base, spike, 0.2), None
    w_early, weight_early = weighted_update(base, spike, 4, 0, prior_weight=32)
    w_late, weight_late = weighted_update(base, spike, 4, 90, prior_weight=32)
    assert weight_early == pytest.approx(4 / 36) and weight_late == pytest.approx(4 / 126)
    assert w_late.event[0] < w_early.event[0] < ema.event[0]
    _, floor = weighted_update(base, spike, 1, 10_000, prior_weight=32, alpha_min=0.05)
    assert floor == pytest.approx(0.05)
    # replay: cohorts entering at 1300 go to the 1000 group, at 2200 to the 2000 group
    t = HazardTable.empty(len(GRID.frontiers))
    groups = [W4Group(1000, t.copy(), 0), W4Group(2000, t.copy(), 0)]
    finishes = {"a": 3000, "b": 3500, "c": 2600}
    cohorts = [
        {"event": "switch_cohort", "rollout_index": 1, "requests": [{"request_id": "a-deadbeef", "entry_output_tokens": 1300}, {"request_id": "b-deadbeef", "entry_output_tokens": 1300}]},
        {"event": "switch_cohort", "rollout_index": 2, "requests": [{"request_id": "c-deadbeef", "entry_output_tokens": 2200}]},
    ]
    out, processed = replay_cohorts(groups, cohorts, finishes, GRID, 0.2, update="weighted", prior_weight=8)
    assert [p["group_cut"] for p in processed] == [1000, 2000]
    assert processed[0]["weight"] == pytest.approx(2 / 10) and processed[1]["weight"] == pytest.approx(1 / 9)
    # delayed entry: a cohort entering at 1300 informs bins from 1500 on, and only its own group
    assert out[0].table.observed[GRID.frontier_index(1500)] and not out[0].table.observed[GRID.frontier_index(1250)]
    assert not out[1].table.observed[GRID.frontier_index(1500)]


def test_limit_decision_step():
    prev = np.array([[[5000, 5000, 0]]])
    new = np.array([[[16000, 4000, 9000]]])
    out = limit_decision_step(prev, new, 2000)
    assert out.tolist() == [[[7000, 4000, 9000]]]  # +2000 cap, -1000 allowed, never-switch passes through
    assert np.array_equal(limit_decision_step(None, new, 2000), new)
    assert np.array_equal(limit_decision_step(prev, new, 0), new)


@pytest.mark.parametrize("requests", [4])
def test_paired_traces_still_single_group(tmp_path, requests):
    bf = tmp_path / "bf.jsonl"
    w4 = tmp_path / "w4.jsonl"
    write_bf16_trace(bf, [500, 900, 1500, 3000])
    write_bf16_trace(w4, [700, 1200, 2500, 4000])
    cal = paired_traces(bf, w4, GRID, requests=requests)
    assert cal.cuts == [0] and cal.w4_groups[0].requests == requests
    sv = survival(cal.w4, 0)
    assert 0 < sv[GRID.frontier_index(1000)] < 1


def test_replay_bf16_uses_the_w4_rule_with_switched_requests_censored_and_probe_publishes_never_switch(tmp_path):
    from verl.experimental.precision_scheduler.hazard import weighted_update
    from verl.experimental.precision_scheduler.online_ema import OnlineEmaWatcher, replay_bf16
    from verl.experimental.precision_scheduler.tpot_grid import TpotGrid
    from verl.experimental.precision_scheduler.policy_builder import decisions_array, load_policy

    grid = PolicyGrid(step=250, cap=4096, batch=4, prompt_step=128, prompt_max=0)
    base = from_finals([500, 900, 1500, 3000], [600, 1200, 2500, 4000], grid).bf16
    starts = [{"request_id": f"r{i}"} for i in range(4)]
    finishes = {"r0": 700, "r1": 2500, "r2": 3900, "r3": 4000}
    cohorts = [{"event": "switch_cohort", "rollout_index": 1, "requests": [{"request_id": "r1-deadbeef", "entry_output_tokens": 1000}, {"request_id": "r2-deadbeef", "entry_output_tokens": 1000}]}]
    table, stats = replay_bf16(base, starts, finishes, cohorts, grid, prior_weight=4.0)
    assert stats == {"requests": 4, "censored": 2}  # r1, r2 left BF16 at 1000
    # identical to one weighted_update with the rollout's fraction table: r1/r2 alive up to 1000 and no event,
    # r0 an event in [500,750), r3 a cap-runner
    expected = components(np.zeros(4, dtype=np.int64), np.array([700, 1000, 1000, 4000]), grid, events=np.array([True, False, False, True]))
    manual, weight = weighted_update(base, expected, 4, 0, prior_weight=4.0, alpha_min=0.05)
    assert weight == pytest.approx(0.5)
    np.testing.assert_allclose(table.risk, manual.risk)
    np.testing.assert_allclose(table.event, manual.event)
    i_1250 = grid.frontier_index(1250)
    assert expected.risk[i_1250] == pytest.approx(0.25) and expected.event[i_1250] == 0.0  # only r3 still at risk, no event
    # bins beyond every request are unobserved by the rollout and keep the base hazard
    i_tail = grid.frontier_index(3000)
    assert expected.risk[i_tail] == pytest.approx(0.25)
    # watcher with probes: revision K publishes a never-switch table and marks it
    tpot = TpotGrid([1, 8], [512, 4096], np.full((2, 2), 14.0), np.full((2, 2), 10.0))
    cal = from_finals([500, 900, 1500, 3000, 3500, 4000, 4000, 4000], [600, 1200, 2500, 4000, 4000, 4000, 4000, 4000], GRID)
    w = OnlineEmaWatcher(run_dir=tmp_path, policy_path=tmp_path / "p.json", calibration=cal, tpot=tpot, grid=GRID, alpha=0.2, slope=0.0009, steps=4, bf16_probe_every=2)
    assert w.bf16_online
    w.run(initialize_only=True, quiet=True)
    assert decisions_array(load_policy(tmp_path / "p.json")).any()
    (tmp_path / "traces").mkdir()
    with (tmp_path / "traces" / "request_lifetimes_replica000_node000.jsonl").open("w") as f:
        for step in (1, 2):
            for k in range(8):
                rid = f"s{step}r{k}"
                f.write(json.dumps({"event": "start", "request_id": rid, "prompt_tokens": 5, "timestamp": 0}) + "\n")
                f.write(json.dumps({"event": "finish", "request_id": rid, "generation_tokens": 800 + 300 * k, "finish_reason": "stop", "timestamp": 1}) + "\n")
    state = w.poll()
    assert state["policy_revision"] == 2
    pol = load_policy(tmp_path / "p.json")
    assert pol["calibration"]["bf16_probe"] is True and not decisions_array(pol).any()


def test_replay_clock_learns_one_parameter_per_line_and_frozen_is_a_equals_one(tmp_path):
    from verl.experimental.precision_scheduler.online_ema import OnlineEmaWatcher, replay_clock
    from verl.experimental.precision_scheduler.tpot_grid import TpotGrid
    from verl.experimental.precision_scheduler.policy_builder import decisions_array, load_policy

    grid = PolicyGrid(step=250, cap=4096, batch=4, prompt_step=128, prompt_max=0)
    rng = np.random.default_rng(0)
    bf_cal = list(rng.integers(600, 3000, 64)); w4_cal = list(rng.integers(700, 3200, 64))
    cal = from_finals(bf_cal, w4_cal, grid)
    # no data -> a = 1 on both lines: tables reproduce the calibration survival
    bf, groups, info = replay_clock(cal, [], {}, [], grid, prior_weight=16.0)
    assert info["a16"] == 1.0 and all(v == 1.0 for v in info["a_w4"].values())
    np.testing.assert_allclose(survival(bf, 0), survival(cal.bf16, 0), atol=1e-6)
    # a run whose BF16 requests finish much earlier than the calibration -> a16 > 1 (shorter), survival below calibration
    starts = [{"request_id": f"r{i}"} for i in range(16)]
    finishes = {f"r{i}": int(v) for i, v in enumerate(rng.integers(300, 900, 16))}
    bf2, _, info2 = replay_clock(cal, starts, finishes, [], grid, prior_weight=4.0)
    assert info2["a16"] > 1.5
    assert survival(bf2, 0)[grid.frontier_index(1000)] < survival(cal.bf16, 0)[grid.frontier_index(1000)]
    # switched requests with long W4 tails -> the group's a < 1 (longer), BF16 sees them only as exposure up to the switch
    cohorts = [{"event": "switch_cohort", "rollout_index": 1, "requests": [{"request_id": f"r{i}-deadbeef", "entry_output_tokens": 500} for i in range(4)]}]
    fin3 = {f"r{i}": 3800 for i in range(4)}
    _, groups3, info3 = replay_clock(cal, starts[:4], fin3, cohorts, grid, prior_weight=4.0)
    assert list(info3["a_w4"].values())[0] < 1.0
    # watcher end to end with update="clock"
    tpot = TpotGrid([1, 8], [512, 4096], np.full((2, 2), 14.0), np.full((2, 2), 10.0))
    w = OnlineEmaWatcher(run_dir=tmp_path, policy_path=tmp_path / "p.json", calibration=cal, tpot=tpot, grid=grid, alpha=0.2, slope=0.0009, steps=4, update="clock", prior_weight=16.0)
    assert w.bf16_online
    w.run(initialize_only=True, quiet=True)
    pol = load_policy(tmp_path / "p.json")
    assert pol["calibration"]["update_rule"] == "clock" and pol["calibration"]["bf16_online_requests"]["a16"] == 1.0
    (tmp_path / "traces").mkdir()
    with (tmp_path / "traces" / "request_lifetimes_replica000_node000.jsonl").open("w") as f:
        for k in range(4):
            f.write(json.dumps({"event": "start", "request_id": f"s1r{k}", "prompt_tokens": 5, "timestamp": 0}) + "\n")
            f.write(json.dumps({"event": "finish", "request_id": f"s1r{k}", "generation_tokens": 400 + 100 * k, "finish_reason": "stop", "timestamp": 1}) + "\n")
    state = w.poll()
    assert state["policy_revision"] == 1
    assert load_policy(tmp_path / "p.json")["calibration"]["bf16_online_requests"]["a16"] > 1.0
