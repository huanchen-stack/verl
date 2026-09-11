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
"""Plot primitives render PNGs (skipped without matplotlib)."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from verl.experimental.precision_scheduler import plots  # noqa: E402
from verl.experimental.precision_scheduler.cost_model import PolicyGrid  # noqa: E402
from verl.experimental.precision_scheduler.downstream_regression import fit_points  # noqa: E402
from verl.experimental.precision_scheduler.hazard import HazardTable  # noqa: E402
from verl.experimental.precision_scheduler.policy_builder import fixed_frontier_policy  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def test_plots_write_pngs(tmp_path):
    heatmap = json.loads((FIXTURES / "heatmaps" / "model_phi4_mini_rep5_20260910" / "heatmap.json").read_text())
    plots.plot_speedup_heatmap(heatmap, tmp_path / "speedup.png")
    plots.plot_policy_surface(fixed_frontier_policy(8, 4000, 2000), tmp_path / "surface.png")
    grid = PolicyGrid(step=250, cap=4000, batch=8)
    n = len(grid.frontiers)
    tables = {
        "bf16": HazardTable(np.ones(n), np.full(n, 0.1), np.ones(n, bool)),
        "w4": HazardTable(np.ones(n), np.full(n, 0.2), np.ones(n, bool)),
    }
    plots.plot_survival(tables, grid.frontiers, tmp_path / "survival.png")
    x = np.arange(1, 11, dtype=float) * 1000
    plots.plot_regression(x, 0.001 * x + 2, fit_points(x, 0.001 * x + 2), tmp_path / "regression.png")
    for name in ("speedup", "surface", "survival", "regression"):
        assert (tmp_path / f"{name}.png").stat().st_size > 1000
