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
"""Hazard components, delayed-entry EMA and bin-start survival."""

import numpy as np
import pytest

from verl.experimental.precision_scheduler.cost_model import PolicyGrid
from verl.experimental.precision_scheduler.hazard import HazardTable, components, ema_update, survival

GRID = PolicyGrid(step=250, cap=2000, batch=4)  # frontiers 250..1750 (7 bins)


def test_components_risk_event_per_bin():
    entries = np.zeros(4, dtype=int)
    finals = np.asarray([300, 300, 900, 2000])  # two finish in bin 250, one in bin 750, one hits the cap
    table = components(entries, finals, GRID)
    assert table.observed.all()
    # bin 250: all four alive at the start, two finish inside it
    assert table.risk[0] == 1.0 and table.event[0] == 0.5
    # bin 500: two alive, none finish
    assert table.risk[1] == 0.5 and table.event[1] == 0.0
    # bin 750: two alive, one finishes
    assert table.risk[2] == 0.5 and table.event[2] == 0.25
    # last bin 1750: the capped request is alive but the cap is right-censored, not an event
    assert table.risk[-1] == 0.25 and table.event[-1] == 0.0


def test_cap_is_right_censored_not_eos():
    table = components(np.zeros(2, dtype=int), np.asarray([2000, 2000]), GRID)
    assert table.event.sum() == 0.0
    assert np.all(table.risk == 1.0)


def test_delayed_entry_leaves_earlier_bins_unobserved():
    table = components(np.asarray([800, 800]), np.asarray([1300, 2000]), GRID)
    assert not table.observed[:3].any()  # bins 250, 500, 750 precede the entry
    assert table.observed[3:].all()
    assert table.risk[3] == 1.0  # both eligible and alive at 1000


def test_ema_updates_only_observed_bins():
    n = len(GRID.frontiers)
    old = HazardTable(risk=np.full(n, 0.5), event=np.full(n, 0.1), observed=np.ones(n, bool))
    new = components(np.asarray([800]), np.asarray([1300]), GRID)
    updated = ema_update(old, new, alpha=0.5)
    np.testing.assert_allclose(updated.risk[:3], 0.5)
    np.testing.assert_allclose(updated.event[:3], 0.1)
    np.testing.assert_allclose(updated.risk[3], 0.5 * 0.5 + 0.5 * 1.0)
    assert updated.observed.all()
    # inputs are not mutated
    assert old.risk[3] == 0.5


def test_ema_alpha_bounds():
    n = len(GRID.frontiers)
    table = HazardTable.empty(n)
    with pytest.raises(ValueError):
        ema_update(table, table, alpha=1.5)
    with pytest.raises(ValueError):
        ema_update(table, table, alpha=-0.1)


def test_survival_is_one_at_observed_frontier():
    n = len(GRID.frontiers)
    table = HazardTable(risk=np.ones(n), event=np.zeros(n), observed=np.ones(n, bool))
    table.event[0] = 0.25
    got = survival(table, 0)
    assert got[0] == 1.0
    assert got[1] == 0.75
    assert len(got) == n
    # from a later frontier the earlier hazard no longer matters
    later = survival(table, 1)
    assert later[0] == 1.0 and len(later) == n - 1


def test_survival_decreases_after_eos_bin_and_handles_zero_risk():
    n = len(GRID.frontiers)
    table = HazardTable(risk=np.zeros(n), event=np.zeros(n), observed=np.zeros(n, bool))
    table.risk[2] = 1.0
    table.event[2] = 1.0
    got = survival(table, 0)
    assert got[2] == 1.0 and got[3] == 0.0 and got[-1] == 0.0
    assert survival(table, n).size == 0
