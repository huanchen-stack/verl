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
"""TPOT grids: heatmap matrix from cells (four valid rep5 runs), validity guard, legacy Gen1 CSV grid."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from verl.experimental.precision_scheduler.tpot_grid import (
    TpotGrid,
    legacy_grid_rows,
    matrix_payload,
    median_speedup,
    validate_heatmap,
    write_legacy_grid_csv,
)

FIXTURES = Path(__file__).parent / "fixtures"
VALID_RUNS = (
    "model_phi4_mini_rep5_20260910",
    "model_qwen35_4b_20260910_corrected_selector",
    "model_gemma_e2b_rep5_20260910",
    "model_gemma_e4b_rep5_20260910",
)


@pytest.mark.parametrize("run", VALID_RUNS)
def test_heatmap_matrix_from_cells_matches_archive(run):
    rows = json.loads((FIXTURES / "heatmaps" / run / "cells_trimmed.json").read_text())
    archived = json.loads((FIXTURES / "heatmaps" / run / "heatmap.json").read_text())
    payload = matrix_payload(rows, archived["batch_sizes"], archived["seq_lens"])
    for key in ("batch_sizes", "seq_lens", "required_kv_bytes", "bf16_tpot_ms", "int4_tpot_ms"):
        assert payload[key] == archived[key], key
    for got, exp in zip(payload["speedup_bf16_over_int4"], archived["speedup_bf16_over_int4"], strict=True):
        for g, e in zip(got, exp, strict=True):
            assert (g is None and e is None) or g == pytest.approx(e, rel=1e-12)
    assert validate_heatmap(payload) > 1.05
    if run.startswith("model_phi4"):
        assert sum(v is None for row in payload["int4_tpot_ms"] for v in row) == 4  # capacity failures stay null


def test_validity_guard_catches_selector_bug_heatmap():
    invalid = json.loads((FIXTURES / "heatmaps" / "invalid_c1_phi_math_20260909T1954Z" / "heatmap.json").read_text())
    assert abs(median_speedup(invalid) - 1.0) < 0.01
    with pytest.raises(ValueError, match="indistinguishable from 1.0"):
        validate_heatmap(invalid)
    valid = json.loads((FIXTURES / "heatmaps" / VALID_RUNS[0] / "heatmap.json").read_text())
    assert validate_heatmap(valid) == pytest.approx(median_speedup(valid))


def test_legacy_csv_grid_from_gen1_rows(tmp_path):
    bf16 = json.loads((FIXTURES / "gen1_qwen35_9b_tp1_grid" / "bf16_rows.json").read_text())
    w4 = json.loads((FIXTURES / "gen1_qwen35_9b_tp1_grid" / "w4_rows.json").read_text())
    rows = legacy_grid_rows(bf16, w4)
    write_legacy_grid_csv(tmp_path / "grid.csv", rows)
    expected = list(
        csv.DictReader((FIXTURES / "gen1_qwen35_9b_tp1_grid" / "expected_qwen35_9b_tp1_tpot_grid.csv").open())
    )
    got = list(csv.DictReader((tmp_path / "grid.csv").open()))
    assert len(got) == len(expected) == 64
    for g, e in zip(got, expected, strict=True):
        assert (g["batch_size"], g["context_len"], g["benchmark_status"], g["aggregation"]) == (
            e["batch_size"],
            e["context_len"],
            e["benchmark_status"],
            e["aggregation"],
        )
        for key in ("bf16_tpot_ms", "partial_int4_tpot_ms", "speedup", "speedup_percent"):
            if e[key] == "":
                assert g[key] == ""
            else:
                assert float(g[key]) == pytest.approx(float(e[key]), rel=1e-12)
    assert sum(r["benchmark_status"] == "oom" for r in got) == 5
    # the CSV round-trips into a grid whose interpolation matches the heatmap-json path
    grid = TpotGrid.from_legacy_csv(tmp_path / "grid.csv")
    via_json = TpotGrid.from_heatmap_json(grid.to_heatmap_json())
    rng = np.random.default_rng(1)
    for _ in range(50):
        context, batch = float(rng.uniform(300, 40000)), float(rng.uniform(1, 160))
        for precision in ("bf16", "w4"):
            assert grid.get(precision, context, batch) == pytest.approx(via_json.get(precision, context, batch))
    assert grid.get("bf16", 512, 1) == pytest.approx(13.853461841963941)
    assert grid.get("w4", 512, 1) == pytest.approx(9.058247492205174)


def test_interpolation_masks_nan_cells_and_is_log2():
    grid = TpotGrid([1, 4], [1024, 4096], [[1.0, 3.0], [2.0, math.nan]], [[0.5, 1.5], [1.0, 2.0]])
    assert grid.get("bf16", 2048, 1) == pytest.approx(2.0)  # log2 midpoint of 1024..4096
    assert grid.get("bf16", 2048, 2) == pytest.approx(2.0)  # batch row 4 is NaN at 4096 -> masked to 2.0
    assert grid.get("bf16", 4096, 4) == pytest.approx(2.0)
    assert grid.get("w4", 100, 1) == pytest.approx(0.5)  # clamps below the grid
    assert grid.get("w4", 1e6, 100) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        TpotGrid([1, 4], [1024, 4096], [[1.0]], [[1.0]])
