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
"""CLI smoke: every subcommand runs end to end on the committed fixtures."""

import json
from pathlib import Path

import pytest

from verl.experimental.precision_scheduler.cli import main
from verl.experimental.precision_scheduler.policy_builder import load_policy

FIXTURES = Path(__file__).parent / "fixtures"


def write_trace(path, lengths):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"event": "start", "request_id": f"r{i}", "prompt_tokens": 20} for i in range(len(lengths))]
    rows += [{"event": "finish", "request_id": f"r{i}", "generation_tokens": n} for i, n in enumerate(lengths)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_build_policy_and_watch_ema_initialize(tmp_path, capsys):
    inputs = json.loads((FIXTURES / "qwen35_4b_global_search_rev30" / "inputs.json").read_text())
    heatmap = tmp_path / "heatmap.json"
    heatmap.write_text(json.dumps(inputs["heatmap_json"]))
    write_trace(tmp_path / "bf16.jsonl", inputs["bf16_finals"])
    write_trace(tmp_path / "w4.jsonl", inputs["w4_finals"])
    common = [
        "--bf-trace",
        str(tmp_path / "bf16.jsonl"),
        "--w4-trace",
        str(tmp_path / "w4.jsonl"),
        "--heatmap",
        str(heatmap),
        "--batch",
        "4",
        "--cap",
        "4000",
        "--downstream-slope",
        "0.0003",
    ]
    assert main(["build-policy", *common, "--output", str(tmp_path / "policy.json"), "--revision", "2"]) == 0
    policy = load_policy(tmp_path / "policy.json")
    assert policy["calibration"]["policy_revision"] == 2
    assert policy["lookup_table"]["live_batch_count"] == 4 and policy["offline_cost_model"]["response_cap"] == 4000
    run_dir = tmp_path / "run"
    assert (
        main(
            [
                "watch-ema",
                *common,
                "--run-dir",
                str(run_dir),
                "--policy",
                str(run_dir / "dynamic_policy.json"),
                "--initialize-only",
                "--steps",
                "1",
            ]
        )
        == 0
    )
    state = json.loads((run_dir / "online_ema_state.json").read_text())
    assert state["policy_revision"] == 0 and state["completed_steps"] == 0
    assert load_policy(run_dir / "dynamic_policy.json")["calibration"]["updates"] == 0
    out = capsys.readouterr().out
    assert '"policy_revision": 0' in out


def test_fit_downstream_grid_and_fixed_frontier(tmp_path, capsys):
    runs = sorted(str(p) for p in (FIXTURES / "megatron_replay_regression").glob("gpu[4-7].json"))
    assert main(["fit-downstream", "--kind", "replay", "--inputs", *runs, "--output", str(tmp_path / "reg.json")]) == 0
    assert json.loads((tmp_path / "reg.json").read_text())["n"] == 9
    assert (tmp_path / "reg.points.csv").exists()
    capsys.readouterr()
    assert (
        main(
            [
                "fit-downstream",
                "--kind",
                "points",
                "--inputs",
                str(FIXTURES / "sampled_token_regression_120" / "points.csv"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["n"] == 120

    cells = tmp_path / "cells.jsonl"
    rows = json.loads((FIXTURES / "heatmaps" / "model_gemma_e2b_rep5_20260910" / "cells_trimmed.json").read_text())
    cells.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert main(["grid", "--kind", "cells", "--inputs", str(cells), "--output", str(tmp_path / "heatmap.json")]) == 0
    archived = json.loads((FIXTURES / "heatmaps" / "model_gemma_e2b_rep5_20260910" / "heatmap.json").read_text())
    assert json.loads((tmp_path / "heatmap.json").read_text())["bf16_tpot_ms"] == archived["bf16_tpot_ms"]
    bad = tmp_path / "bad_cells.jsonl"
    bad.write_text("\n".join(json.dumps(dict(r, median_tpot_ms=1.0)) for r in rows) + "\n")
    with pytest.raises(ValueError, match="indistinguishable"):
        main(["grid", "--kind", "cells", "--inputs", str(bad)])
    assert main(["grid", "--kind", "cells", "--inputs", str(bad), "--skip-heatmap-guard"]) == 0

    gen1 = FIXTURES / "gen1_qwen35_9b_tp1_grid"
    for name in ("bf16", "w4"):
        (tmp_path / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in json.loads((gen1 / f"{name}_rows.json").read_text())) + "\n"
        )
    assert (
        main(
            [
                "grid",
                "--kind",
                "gen1-csv",
                "--inputs",
                str(tmp_path / "bf16.jsonl"),
                str(tmp_path / "w4.jsonl"),
                "--output",
                str(tmp_path / "grid.csv"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "grid",
                "--kind",
                "csv-to-heatmap",
                "--inputs",
                str(tmp_path / "grid.csv"),
                "--output",
                str(tmp_path / "grid_heatmap.json"),
            ]
        )
        == 0
    )
    assert len(json.loads((tmp_path / "grid_heatmap.json").read_text())["batch_sizes"]) == 8

    assert (
        main(
            [
                "build-fixed-frontier",
                "--batch",
                "32",
                "--cap",
                "16384",
                "--frontier",
                "8000",
                "--output",
                str(tmp_path / "fixed.json"),
            ]
        )
        == 0
    )
    policy = load_policy(tmp_path / "fixed.json")
    assert policy["calibration"]["fixed_response_frontier"] == 8000
    assert sum(policy["lookup_table"]["committed_frontiers"]) == 8000 * 32 * 17 * 32
