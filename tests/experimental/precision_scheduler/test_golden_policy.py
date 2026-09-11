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
"""Golden: the global-search builder reproduces the archived Qwen3.5-4B ``dynamic_policy.json`` (rev 30).

The trimmed inputs (128 BF16 finals, 128 W4 finals, the corrected-selector heatmap, the 20 switch
cohorts with their resolved final lengths) are committed under ``fixtures/``; the raw archive is
consulted additionally when present.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from verl.experimental.precision_scheduler.calibration import from_finals, paired_traces
from verl.experimental.precision_scheduler.cost_model import PolicyGrid
from verl.experimental.precision_scheduler.hazard import components, ema_update
from verl.experimental.precision_scheduler.online_ema import replay_cohorts
from verl.experimental.precision_scheduler.policy_builder import build_policy, validate_policy
from verl.experimental.precision_scheduler.tpot_grid import TpotGrid
from verl.experimental.precision_scheduler.traces import completed_steps, read_cohorts, trace_lengths

FIXTURE = Path(__file__).parent / "fixtures" / "qwen35_4b_global_search_rev30"


@pytest.fixture(scope="module")
def inputs():
    return json.loads((FIXTURE / "inputs.json").read_text())


@pytest.fixture(scope="module")
def expected():
    with gzip.open(FIXTURE / "expected_committed_frontiers.json.gz", "rt") as handle:
        return np.asarray(json.load(handle), dtype=np.int64)


def test_global_search_reproduces_qwen35_4b_revision_30(inputs, expected):
    grid = PolicyGrid(**inputs["grid"])
    calibration = from_finals(inputs["bf16_finals"], inputs["w4_finals"], grid)
    table = calibration.w4
    finals = inputs["trace_finals"]
    for cohort in inputs["cohorts"]:
        entries = np.asarray([r["entry_output_tokens"] for r in cohort["requests"]])
        ids = [str(r["request_id"]).rpartition("-")[0] for r in cohort["requests"]]
        table = ema_update(table, components(entries, np.asarray([finals[i] for i in ids]), grid), inputs["alpha"])
    tpot = TpotGrid.from_heatmap_json(inputs["heatmap_json"])
    policy, switch_states = build_policy(
        calibration.bf16,
        table,
        tpot,
        grid,
        inputs["downstream_slope"],
        inputs["completed_steps"],
        inputs["alpha"],
    )
    validate_policy(policy)
    assert switch_states == inputs["expected_switch_states"] == 35360
    np.testing.assert_array_equal(np.asarray(policy["lookup_table"]["committed_frontiers"]), expected)
    assert policy["calibration"]["policy_revision"] == 30


def test_replay_cohorts_helper_matches_manual_ema(inputs):
    grid = PolicyGrid(**inputs["grid"])
    calibration = from_finals(inputs["bf16_finals"], inputs["w4_finals"], grid)
    finals = {rid: int(v) for rid, v in inputs["trace_finals"].items()}
    table, processed = replay_cohorts(calibration.w4, inputs["cohorts"], finals, grid, inputs["alpha"])
    assert len(processed) == 20
    manual = calibration.w4
    for cohort in inputs["cohorts"]:
        entries = np.asarray([r["entry_output_tokens"] for r in cohort["requests"]])
        ids = [str(r["request_id"]).rpartition("-")[0] for r in cohort["requests"]]
        manual = ema_update(manual, components(entries, np.asarray([finals[i] for i in ids]), grid), inputs["alpha"])
    np.testing.assert_allclose(table.risk, manual.risk)
    np.testing.assert_allclose(table.event, manual.event)


def test_archive_matches_committed_fixture(inputs, expected):
    run = Path(inputs["source_run"])
    if not (run / "dynamic_policy.json").exists():
        pytest.skip(f"archived run not available: {run}")
    archived = json.loads((run / "dynamic_policy.json").read_text())
    np.testing.assert_array_equal(np.asarray(archived["lookup_table"]["committed_frontiers"]), expected)
    assert archived["calibration"]["policy_revision"] == 30
    # full rebuild from the raw traces (the trimmed fixture must be a faithful extraction)
    bf16_trace, w4_trace, heatmap = Path(inputs["bf16_trace"]), Path(inputs["w4_trace"]), Path(inputs["heatmap"])
    if not (bf16_trace.exists() and w4_trace.exists() and heatmap.exists()):
        pytest.skip("archived baseline traces or heatmap not available")
    grid = PolicyGrid(**inputs["grid"])
    calibration = paired_traces(bf16_trace, w4_trace, grid, requests=128)
    starts, finishes = trace_lengths(run / "traces" / "request_lifetimes_replica000_node000.jsonl")
    assert completed_steps(starts, finishes, grid.batch) == 30
    table, processed = replay_cohorts(
        calibration.w4, read_cohorts(run / "switch_observations.jsonl"), finishes, grid, inputs["alpha"]
    )
    assert len(processed) == 20
    policy, _ = build_policy(
        calibration.bf16, table, TpotGrid.from_heatmap_json(heatmap), grid, inputs["downstream_slope"], 30, 0.2
    )
    np.testing.assert_array_equal(np.asarray(policy["lookup_table"]["committed_frontiers"]), expected)
