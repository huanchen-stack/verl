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
"""Goldens for the three archived downstream-time fits (to 1e-9) plus the inline slope helper."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from verl.experimental.precision_scheduler.downstream_regression import (
    fit_from_metrics,
    fit_points,
    fit_replay_points,
    fit_split,
    load_replay_runs,
    read_points_csv,
    slope_from_metrics,
)

FIXTURES = Path(__file__).parent / "fixtures"
ARCHIVE_120 = Path("/data/huanchen/verl/evidence_bundle/latest_sampled_token_regression.json")
ARCHIVE_MEGATRON = Path("/data/huanchen/verl/.codex-report/new-storyline-experiments/megatron_replay_regression")


def test_120_run_regression_to_1e9():
    points = read_points_csv(FIXTURES / "sampled_token_regression_120" / "points.csv")
    expected = json.loads((FIXTURES / "sampled_token_regression_120" / "expected.json").read_text())
    got = fit_split(points)
    assert got["n"] == expected["n"] == 120
    for name in ("inference", "training"):
        for key in ("slope_seconds_per_sampled_token", "intercept_seconds", "r2"):
            assert got[name][key] == pytest.approx(expected[name][key], abs=1e-9, rel=1e-9), (name, key)
    if ARCHIVE_120.exists():
        assert json.loads(ARCHIVE_120.read_text())["inference"] == expected["inference"]


def test_megatron_replay_regression_golden():
    runs = sorted((FIXTURES / "megatron_replay_regression").glob("gpu[4-7].json"))
    expected = json.loads((FIXTURES / "megatron_replay_regression" / "expected_regression.json").read_text())
    summary, rows = fit_replay_points(load_replay_runs(runs))
    assert summary["n"] == expected["n"] == 9
    assert summary["token_range"] == expected["token_range"]
    assert summary["batch_sizes"] == expected["batch_sizes"]
    for name in ("inference", "training"):
        for key in ("slope_s_per_token", "intercept_s", "r2", "mape_pct", "rmse_s"):
            assert summary[name][key] == pytest.approx(expected[name][key], abs=1e-9, rel=1e-9), (name, key)
    for batch, stats in expected["by_batch"].items():
        assert summary["by_batch"][batch]["n"] == stats["n"]
        assert summary["by_batch"][batch]["inference_mape_pct"] == pytest.approx(stats["inference_mape_pct"], rel=1e-9)
    measured = list(csv.DictReader((FIXTURES / "megatron_replay_regression" / "expected_measured_points.csv").open()))
    assert [int(r["point"]) for r in measured] == [r["point"] for r in rows]
    for archived, row in zip(measured, rows, strict=True):
        for key in ("pred_inference_s", "pred_training_s", "inference_residual_pct", "training_residual_pct"):
            assert float(archived[key]) == pytest.approx(row[key], rel=1e-9, abs=1e-9)
    if (ARCHIVE_MEGATRON / "reports" / "regression.json").exists():
        archived = json.loads((ARCHIVE_MEGATRON / "reports" / "regression.json").read_text())
        assert archived["inference"] == expected["inference"]


def test_validation_downstream_fit_golden():
    rows = read_points_csv(FIXTURES / "validation_downstream_fit" / "points.csv")
    expected = json.loads((FIXTURES / "validation_downstream_fit" / "expected.json").read_text())
    metrics = [
        {
            "step": r["step"],
            "data": {
                "perf/total_num_tokens": r["total_num_tokens"],
                "timing_s/step": r["timing_s_step"],
                "timing_s/gen": r["timing_s_gen"],
            },
        }
        for r in rows
    ]
    assert len(metrics) == 320  # 4 caps x 4 batches x 2 pure policies x steps 6..15
    got = fit_from_metrics(metrics, steps=set(range(6, 16)))
    assert got["intercept_seconds"] == pytest.approx(12.481447564927427, abs=1e-9)
    assert got["seconds_per_token"] == pytest.approx(0.0008783918670449, abs=1e-12)
    assert got["r2"] == pytest.approx(0.998779400507401, abs=1e-9)
    assert got["seconds_per_token"] == pytest.approx(expected["seconds_per_token"], abs=1e-12)
    # step filtering is applied
    assert fit_from_metrics(metrics, steps={6})["n"] == 32


def test_fit_points_and_inline_slope():
    x = np.asarray([1000.0, 2000.0, 3000.0, 4000.0])
    fit = fit_points(x, 0.5 * x + 3.0)
    assert fit["slope_s_per_token"] == pytest.approx(0.5) and fit["intercept_s"] == pytest.approx(3.0)
    assert fit["r2"] == pytest.approx(1.0) and fit["mape_pct"] == pytest.approx(0.0, abs=1e-9)
    metrics = [
        {"step": i, "data": {"perf/total_num_tokens": t, "timing_s/step": 0.001 * t + 20.0, "timing_s/gen": 10.0}}
        for i, t in enumerate((1e4, 2e4, 3e4, 4e4))
    ]
    slope, n = slope_from_metrics(metrics)
    assert n == 4 and slope == pytest.approx(0.001)
    # a negative fitted slope is clipped to zero; too few rows yield zero
    metrics_neg = [
        dict(m, data=dict(m["data"], **{"timing_s/step": -0.001 * m["data"]["perf/total_num_tokens"] + 50}))
        for m in metrics
    ]
    assert slope_from_metrics(metrics_neg) == (0.0, 4)
    assert slope_from_metrics(metrics[:2]) == (0.0, 2)
    with pytest.raises(RuntimeError):
        fit_replay_points([{"warmup": True}])
