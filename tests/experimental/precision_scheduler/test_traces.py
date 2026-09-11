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
"""Request-lifetime trace and switch-cohort parsing."""

import json

import numpy as np
import pytest

from verl.experimental.precision_scheduler.calibration import paired_traces
from verl.experimental.precision_scheduler.cost_model import PolicyGrid
from verl.experimental.precision_scheduler.traces import (
    cohort_observation,
    completed_steps,
    final_lengths,
    read_cohorts,
    read_jsonl,
    resolve_request_id,
    trace_lengths,
)


def write_lines(path, rows, *, truncate_last=False):
    text = "\n".join(json.dumps(r) for r in rows) + "\n"
    if truncate_last:
        text = text[:-10]
    path.write_text(text)


def test_trace_lengths_and_completed_steps(tmp_path):
    rows = []
    for step in (1, 2):
        for i in range(2):
            rows.append({"event": "start", "request_id": f"s{step}r{i}", "prompt_tokens": 10, "timestamp": 1.0})
    rows += [
        {"event": "finish", "request_id": "s1r0", "generation_tokens": 500, "finish_reason": "stop"},
        {"event": "finish", "request_id": "s1r1", "generation_tokens": 9000, "finish_reason": "length"},
        {"event": "finish", "request_id": "s2r0", "generation_tokens": 20, "finish_reason": "stop"},
        {"event": "finish", "request_id": "s2r1", "generation_tokens": 30, "finish_reason": "stop"},
    ]
    path = tmp_path / "trace.jsonl"
    write_lines(path, rows, truncate_last=True)  # writer still appending the final line
    starts, finishes = trace_lengths(path, cap=4096)
    assert [s["request_id"] for s in starts] == ["s1r0", "s1r1", "s2r0", "s2r1"]
    assert finishes == {"s1r0": 500, "s1r1": 4096, "s2r0": 20}
    assert completed_steps(starts, finishes, batch=2) == 1
    assert len(read_jsonl(path)) == 7
    write_lines(path, rows)
    starts, finishes = trace_lengths(path)
    assert completed_steps(starts, finishes, batch=2) == 2
    assert completed_steps(starts, finishes, batch=3) == 1
    np.testing.assert_array_equal(final_lengths(starts[:2], finishes), [500, 9000])
    # calibration limit: enough starts but a missing finish is an error
    limited, _ = trace_lengths(path, limit=2)
    assert len(limited) == 2
    with pytest.raises(RuntimeError, match="incomplete calibration trace"):
        trace_lengths(path, limit=5)


def test_resolve_request_id_strips_engine_suffix():
    finishes = {"abc-123": 5, "plain": 7}
    assert resolve_request_id("plain", finishes) == "plain"
    assert resolve_request_id("plain-deadbeef", finishes) == "plain"
    assert resolve_request_id("abc-123-0badf00d", finishes) == "abc-123"
    assert resolve_request_id("plain-dead", finishes) is None  # suffix must be 8 hex characters
    assert resolve_request_id("unknown-deadbeef", finishes) is None


def test_cohort_observation_and_read_cohorts(tmp_path):
    finishes = {"a": 900, "b": 5000}
    grid = PolicyGrid(step=250, cap=4096, batch=2)
    cohort = {
        "event": "switch_cohort",
        "rollout_index": 3,
        "requests": [
            {"request_id": "a-12345678", "entry_output_tokens": 750},
            {"request_id": "b", "entry_output_tokens": 800},
        ],
    }
    entries, finals, skipped = cohort_observation(cohort, finishes, grid.cap)
    np.testing.assert_array_equal(entries, [750, 800])
    np.testing.assert_array_equal(finals, [900, 4096])
    assert skipped == 0
    assert cohort_observation(dict(cohort, requests=[]), finishes, grid.cap) is None
    unresolved = dict(cohort, requests=[{"request_id": "zzz", "entry_output_tokens": 1}])
    assert cohort_observation(unresolved, finishes, grid.cap) is None
    # a request whose final length precedes its switch entry is skipped, not fatal
    aborted = {"request_id": "a", "entry_output_tokens": 1000}
    entries, finals, skipped = cohort_observation(
        dict(cohort, requests=[*cohort["requests"], aborted]), finishes, grid.cap
    )
    np.testing.assert_array_equal(entries, [750, 800])
    assert skipped == 1
    assert cohort_observation(dict(cohort, requests=[aborted]), finishes, grid.cap) is None
    path = tmp_path / "switch_observations.jsonl"
    write_lines(path, [cohort, {"event": "other"}, cohort])
    assert len(read_cohorts(path)) == 2
    assert read_cohorts(tmp_path / "missing.jsonl") == []


def test_paired_trace_calibration(tmp_path):
    grid = PolicyGrid(step=250, cap=1000, batch=2)
    for name, lengths in (("bf16", [300, 600, 1000, 200]), ("w4", [100, 100, 900, 1000])):
        rows = [{"event": "start", "request_id": f"{name}{i}", "prompt_tokens": 1} for i in range(4)]
        rows += [{"event": "finish", "request_id": f"{name}{i}", "generation_tokens": n} for i, n in enumerate(lengths)]
        write_lines(tmp_path / f"{name}.jsonl", rows)
    calibration = paired_traces(tmp_path / "bf16.jsonl", tmp_path / "w4.jsonl", grid, requests=4)
    assert calibration.metadata["bf16_requests"] == 4
    assert calibration.bf16.risk[0] == 0.75 and calibration.bf16.event[0] == 0.25  # 200 ended before 250; 300 in bin
    assert calibration.w4.risk[0] == 0.5  # two W4 requests ended before 250
    with pytest.raises(RuntimeError):
        paired_traces(tmp_path / "bf16.jsonl", tmp_path / "w4.jsonl", grid, requests=8)
