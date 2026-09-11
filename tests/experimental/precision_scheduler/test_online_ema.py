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
"""Watcher: one revision step against a fake trace and cohort file."""

import json

import numpy as np

from verl.experimental.precision_scheduler.calibration import from_finals
from verl.experimental.precision_scheduler.cost_model import PolicyGrid
from verl.experimental.precision_scheduler.online_ema import OnlineEmaWatcher
from verl.experimental.precision_scheduler.policy_builder import load_policy
from verl.experimental.precision_scheduler.traces import read_jsonl

GRID = PolicyGrid(step=250, cap=2000, batch=2, prompt_step=512, prompt_max=512)


class Heat:
    def get(self, precision, context, batch):
        return 2.0 if precision == "bf16" else (3.0 if context < 800 else 1.0)


def write_trace(path, steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for step, lengths in enumerate(steps, start=1):
            ids = [f"s{step}r{i}" for i in range(len(lengths))]
            for rid in ids:
                f.write(json.dumps({"event": "start", "request_id": rid, "prompt_tokens": 40, "timestamp": 0.0}) + "\n")
            for rid, length in zip(ids, lengths, strict=True):
                if length is None:
                    continue
                row = {"event": "finish", "request_id": rid, "generation_tokens": length, "finish_reason": "stop"}
                f.write(json.dumps(row) + "\n")


def make_watcher(tmp_path):
    calibration = from_finals([400, 900, 1500, 2000], [300, 700, 1200, 2000], GRID)
    return OnlineEmaWatcher(
        run_dir=tmp_path,
        policy_path=tmp_path / "dynamic_policy.json",
        calibration=calibration,
        tpot=Heat(),
        grid=GRID,
        alpha=0.5,
        slope=1e-4,
        steps=2,
    )


def test_initialize_only_writes_revision_zero(tmp_path):
    watcher = make_watcher(tmp_path)
    state = watcher.run(initialize_only=True, quiet=True)
    assert state["policy_revision"] == 0 and state["ema_updates"] == 0
    policy = load_policy(tmp_path / "dynamic_policy.json")
    assert policy["calibration"]["policy_revision"] == 0
    assert policy["calibration"]["base_source"]["kind"] == "explicit final lengths"
    assert json.loads((tmp_path / "online_ema_state.json").read_text())["completed_steps"] == 0
    assert len(read_jsonl(tmp_path / "online_ema_history.jsonl")) == 1


def test_one_revision_step_with_fake_cohort(tmp_path):
    watcher = make_watcher(tmp_path)
    watcher.run(initialize_only=True, quiet=True)
    base_w4 = watcher.calibration.w4.copy()
    # step 1 complete, step 2 still running (one request unfinished)
    write_trace(tmp_path / "traces" / "request_lifetimes_replica000_node000.jsonl", [[600, 1300], [500, None]])
    cohorts = [
        {
            "event": "switch_cohort",
            "rollout_index": 1,
            "requests": [
                {"request_id": "s1r0-deadbeef", "entry_output_tokens": 500},
                {"request_id": "s1r1-deadbeef", "entry_output_tokens": 500},
            ],
        },
        {
            "event": "switch_cohort",
            "rollout_index": 2,
            "requests": [
                {"request_id": "s2r0-deadbeef", "entry_output_tokens": 250},
            ],
        },
    ]
    with (tmp_path / "switch_observations.jsonl").open("w") as f:
        for c in cohorts:
            f.write(json.dumps(c) + "\n")
    state = watcher.poll()
    assert state is not None
    assert state["completed_steps"] == 1 and state["policy_revision"] == 1
    # the rollout-2 cohort is gated out until step 2 completes
    assert state["ema_updates"] == 1
    assert state["processed_observations"][0]["rollout_index"] == 1
    policy = load_policy(tmp_path / "dynamic_policy.json")
    assert policy["calibration"]["policy_revision"] == 1
    assert policy["calibration"]["updates"] == 1
    # the EMA moved the observed bins (entries at 500 -> bins from 500 on) and left bin 250 alone
    _, table, _ = watcher.observe()
    assert table.risk[0] == base_w4.risk[0] and table.event[0] == base_w4.event[0]
    assert not np.array_equal(table.event[1:], base_w4.event[1:])
    assert watcher.poll() is None  # nothing changed
    assert len(read_jsonl(tmp_path / "online_ema_history.jsonl")) == 2
    # finishing step 2 yields revision 2 with both cohorts and run() returns at steps=2
    write_trace(tmp_path / "traces" / "request_lifetimes_replica000_node000.jsonl", [[600, 1300], [500, 900]])
    final = watcher.run(quiet=True)
    assert final["policy_revision"] == 2 and final["ema_updates"] == 2
