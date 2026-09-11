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
"""Unit tests for the global-search policy builder (ported from test_online_ema_policy.py)."""

import numpy as np
import pytest

from verl.experimental.precision_scheduler.cost_model import (
    PolicyGrid,
    make_tpot_cache,
    plan_cost,
    switched_plan_cost,
)
from verl.experimental.precision_scheduler.hazard import HazardTable
from verl.experimental.precision_scheduler.policy_builder import (
    build_decisions,
    build_policy,
    decisions_array,
    fixed_frontier_policy,
    lookup,
    policy_from_decisions,
    validate_policy,
)


class SyntheticHeat:
    """BF16 wins early; W4 wins sufficiently late."""

    def get(self, precision, context, batch):
        if precision == "bf16":
            return 2.0
        return 4.0 if context < 4000 else 1.0


class SlowW4:
    def get(self, precision, context, batch):
        return 2.0 if precision == "bf16" else 3.0


def no_exit_table(grid: PolicyGrid) -> HazardTable:
    n = len(grid.frontiers)
    return HazardTable(risk=np.ones(n), event=np.zeros(n), observed=np.ones(n, dtype=bool))


def brute_force(bf, w4, cache, grid, slope, fi, pi, live):
    baseline = plan_cost(bf, cache, grid, "bf16", fi, pi, live, slope)
    candidates = [
        (switched_plan_cost(bf, w4, cache, grid, fi, fj, pi, live, slope), int(grid.frontiers[fj]))
        for fj in range(fi, len(grid.frontiers))
    ]
    cost, frontier = min(candidates)
    return (frontier if cost < baseline else 0), cost, baseline


def test_future_frontier_matches_brute_force_and_is_not_forced_immediate():
    grid = PolicyGrid()
    bf = no_exit_table(grid)
    w4 = no_exit_table(grid)
    heat = SyntheticHeat()
    policy, _ = build_policy(bf, w4, heat, grid, slope=0.0, revision=0, alpha=0.2)
    decisions = decisions_array(policy)
    assert decisions.shape == grid.shape
    cache = make_tpot_cache(grid, heat)
    expected, _, _ = brute_force(bf, w4, cache, grid, 0.0, fi=0, pi=0, live=1)
    assert decisions[0, 0, 0] == expected
    assert expected > 250


def test_builder_matches_brute_force_on_tiny_grid_with_hazard():
    grid = PolicyGrid(step=250, cap=3000, batch=4, prompt_step=512, prompt_max=1024)
    n = len(grid.frontiers)
    rng = np.random.default_rng(0)
    bf = HazardTable(risk=np.ones(n), event=rng.uniform(0.0, 0.3, n), observed=np.ones(n, bool))
    w4 = HazardTable(risk=np.ones(n), event=rng.uniform(0.0, 0.3, n), observed=np.ones(n, bool))

    class Heat:
        def get(self, precision, context, batch):
            base = 2.0 + 0.001 * context + 0.1 * batch
            return base if precision == "bf16" else base * (1.4 if context < 1500 else 0.7)

    cache = make_tpot_cache(grid, Heat())
    decisions = build_decisions(bf, w4, cache, grid, slope=1e-4)
    for fi in range(n):
        for pi in range(len(grid.prompts)):
            for live in range(1, grid.batch + 1):
                expected, _, _ = brute_force(bf, w4, cache, grid, 1e-4, fi, pi, live)
                assert decisions[fi, pi, live - 1] == expected, (fi, pi, live)


def test_no_switch_when_all_future_w4_plans_are_slower():
    grid = PolicyGrid()
    table = no_exit_table(grid)
    policy, count = build_policy(table, table, SlowW4(), grid, slope=0.0, revision=0, alpha=0.2)
    assert count == 0
    assert not any(policy["lookup_table"]["committed_frontiers"])


def test_strictly_faster_w4_switches_now_and_equal_speeds_never_switch():
    grid = PolicyGrid(step=250, cap=2000, batch=2, prompt_step=1024, prompt_max=1024)
    table = no_exit_table(grid)

    class Faster:
        def get(self, precision, context, batch):
            return 1.0 if precision == "bf16" else 0.5

    class Equal:
        def get(self, precision, context, batch):
            return 1.0

    decisions = build_decisions(table, table, make_tpot_cache(grid, Faster()), grid, slope=0.0)
    assert np.all(decisions[0] == 250)
    assert np.all(decisions[3] == 1000)
    # plan-vs-plan comparison uses strict <, so an equal-cost plan never commits a switch
    decisions = build_decisions(table, table, make_tpot_cache(grid, Equal()), grid, slope=0.0)
    assert not decisions.any()


def test_policy_json_contract_round_trip(tmp_path):
    grid = PolicyGrid(step=250, cap=2000, batch=3, prompt_step=512, prompt_max=1024)
    table = no_exit_table(grid)
    policy, _ = build_policy(table, table, SyntheticHeat(), grid, slope=0.0, revision=3, alpha=0.2)
    validate_policy(policy)
    assert policy["schema_version"] == 6
    assert policy["lookup_table"]["layout"] == "frontier_major,prompt_bucket,live_batch"
    assert policy["lookup_table"]["frontier_count"] == 7
    assert policy["lookup_table"]["prompt_bucket_count"] == 3
    assert policy["lookup_table"]["live_batch_count"] == 3
    assert policy["calibration"]["policy_revision"] == 3
    assert policy["calibration"]["ema_alpha"] == 0.2
    for key in (
        "description",
        "scan_interval_tokens",
        "arm_min_requests",
        "capture_max_batch",
        "commitment_enabled",
        "receding_horizon_lookup",
        "initial_rollout_batch",
        "max_switch_live_batch",
        "calibration",
        "offline_cost_model",
        "lookup_table",
    ):
        assert key in policy
    for key in ("response_cap", "downstream_seconds_per_token", "switch_overhead_seconds"):
        assert key in policy["offline_cost_model"]
    # indexing helper mirrors the flat layout, including clamping
    decisions = decisions_array(policy)
    assert lookup(policy, 250, 0, 1) == decisions[0, 0, 0]
    assert lookup(policy, 1750, 1024, 3) == decisions[-1, -1, -1]
    assert lookup(policy, 1900, 5000, 99) == decisions[-1, -1, -1]
    assert lookup(policy, 0, 0, 0) == decisions[0, 0, 0]


def test_validate_policy_rejects_malformed_tables():
    grid = PolicyGrid(step=250, cap=1000, batch=2, prompt_step=512, prompt_max=512)
    table = no_exit_table(grid)
    policy, _ = build_policy(table, table, SlowW4(), grid, slope=0.0, revision=0, alpha=0.1)
    validate_policy(policy)
    bad = dict(policy)
    bad["lookup_table"] = dict(policy["lookup_table"], committed_frontiers=[0])
    with pytest.raises(ValueError, match="committed_frontiers"):
        validate_policy(bad)
    bad["lookup_table"] = dict(policy["lookup_table"], layout="live_batch,prompt_bucket,frontier_major")
    with pytest.raises(ValueError, match="layout"):
        validate_policy(bad)
    bad["lookup_table"] = dict(policy["lookup_table"], committed_frontiers=[0.5] * 12)
    with pytest.raises(ValueError, match="integer"):
        validate_policy(bad)
    missing = {k: v for k, v in policy.items() if k != "capture_max_batch"}
    with pytest.raises(ValueError, match="capture_max_batch"):
        validate_policy(missing)
    with pytest.raises(ValueError, match="schema_version"):
        validate_policy(dict(policy, schema_version=4))


def test_fixed_frontier_policy_matches_archived_generator():
    policy = fixed_frontier_policy(batch=32, cap=16384, frontier=8000)
    validate_policy(policy)
    table = policy["lookup_table"]
    assert table["frontier_count"] == 65 and table["prompt_bucket_count"] == 17 and table["live_batch_count"] == 32
    decisions = decisions_array(policy)
    assert np.all(decisions[:32] == 8000)  # frontiers 250..8000 inclusive commit to 8000
    assert np.all(decisions[32:] == 0)
    assert policy["receding_horizon_lookup"] is False
    assert policy["calibration"]["fixed_response_frontier"] == 8000
    with pytest.raises(ValueError):
        fixed_frontier_policy(batch=32, cap=16384, frontier=8001)


def test_fixed_frontier_guard_never_exceeds_capture_max_batch():
    policy = fixed_frontier_policy(batch=64, cap=16384, frontier=8000)
    validate_policy(policy)
    assert policy["capture_max_batch"] == 32 and policy["max_switch_live_batch"] == 32
    assert policy["lookup_table"]["live_batch_count"] == 64
    assert (
        fixed_frontier_policy(batch=64, cap=16384, frontier=8000, capture_max_batch=64)["max_switch_live_batch"] == 64
    )
    with pytest.raises(ValueError, match="capture_max_batch"):
        validate_policy(dict(policy, max_switch_live_batch=64))
    # the EMA builder path caps the guard the same way and refuses an explicit oversize guard
    grid = PolicyGrid(step=250, cap=1000, batch=8, prompt_step=512, prompt_max=512)
    decisions = np.zeros(grid.shape, dtype=np.int64)
    capped = policy_from_decisions(
        decisions, grid, description="d", calibration={"kind": "k"}, slope=0.0, capture_max_batch=4
    )
    assert capped["max_switch_live_batch"] == 4 and capped["capture_max_batch"] == 4
    with pytest.raises(ValueError, match="exceeds capture_max_batch"):
        policy_from_decisions(
            decisions,
            grid,
            description="d",
            calibration={"kind": "k"},
            slope=0.0,
            capture_max_batch=4,
            max_switch_live_batch=8,
        )
