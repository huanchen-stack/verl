# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for tools/precision_scheduling/policies/build_static_policy.py.

Golden tier: the three kinds reproduce the archived fixed-k, fixed-t and forced-switch
policy files byte for byte (skipped when the archive is absent). Unit tier: table
shape, cell semantics, CLI and argument validation.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "precision_scheduling" / "policies" / "build_static_policy.py"
ARCHIVE = Path("/data/huanchen/verl/.codex-report/new-storyline-experiments")
MATRIX = ARCHIVE / "full_rl_policy_matrix_b64_cap24k_20260902" / "policies"
HEATMAP = ARCHIVE / "dynamic_tail8k_heatmap_20260823" / "policies"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("build_static_policy", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _archived(path: Path) -> bytes:
    if not path.exists():
        pytest.skip(f"archive missing: {path}")
    return path.read_bytes()


GOLDEN_CASES = [
    ("frontier", dict(batch=64, cap=24576, frontier=6000), MATRIX / "fixed_k6000.json"),
    ("frontier", dict(batch=64, cap=24576, frontier=8000), MATRIX / "fixed_k8000.json"),
    ("frontier", dict(batch=64, cap=24576, frontier=10000), MATRIX / "fixed_k10000.json"),
    ("frontier", dict(batch=32, cap=16384, frontier=8000), HEATMAP / "b32_cap16384_fixed_frontier8000_30step.json"),
    ("frontier", dict(batch=32, cap=24576, frontier=6000), HEATMAP / "b32_cap24576_fixed_frontier6000_30step.json"),
    ("frontier", dict(batch=64, cap=16384, frontier=10000), HEATMAP / "b64_cap16384_fixed_frontier10000_30step.json"),
    ("live_threshold", dict(batch=64, cap=24576, threshold=2), MATRIX / "fixed_t2.json"),
    ("live_threshold", dict(batch=64, cap=24576, threshold=4), MATRIX / "fixed_t4.json"),
    ("live_threshold", dict(batch=64, cap=24576, threshold=8), MATRIX / "fixed_t8.json"),
    (
        "forced_switch",
        dict(batch=128, cap=24576, frontier=8000),
        HEATMAP / "b128_cap24576_forced_tail8k_calibration.json",
    ),
]


@pytest.mark.parametrize("kind, kwargs, archived", GOLDEN_CASES, ids=[c[2].name for c in GOLDEN_CASES])
def test_golden_byte_equal_to_archived_policy(tool, kind, kwargs, archived):
    expected = _archived(archived)
    produced = tool.serialize(tool.build_policy(kind, **kwargs)).encode()
    assert hashlib.sha256(produced).hexdigest() == hashlib.sha256(expected).hexdigest(), archived.name
    assert produced == expected


def test_frontier_table_shape_and_cells(tool):
    policy = tool.build_policy("frontier", batch=4, cap=2000, frontier=1000)
    table = policy["lookup_table"]
    frontiers = list(range(250, 2000, 250))
    assert table["frontier_count"] == len(frontiers) == 7
    assert table["prompt_bucket_count"] == 17 and table["live_batch_count"] == 4
    cells = table["committed_frontiers"]
    assert len(cells) == 7 * 17 * 4
    per_frontier = 17 * 4
    for index, frontier in enumerate(frontiers):
        block = set(cells[index * per_frontier : (index + 1) * per_frontier])
        assert block == ({1000} if frontier <= 1000 else {0})
    assert policy["initial_rollout_batch"] == 4 and policy["receding_horizon_lookup"] is False
    assert policy["schema_version"] == 6


def test_live_threshold_table_is_constant_first_frontier_with_guard(tool):
    policy = tool.build_policy("live_threshold", batch=8, cap=1000, threshold=2)
    assert set(policy["lookup_table"]["committed_frontiers"]) == {250}
    assert policy["max_switch_live_batch"] == 2
    assert policy["calibration"]["fixed_live_batch_threshold"] == 2


def test_forced_switch_is_schema_4_without_receding_or_guard_keys(tool):
    policy = tool.build_policy("forced_switch", batch=8, cap=4096, frontier=2000)
    assert policy["schema_version"] == 4
    assert "receding_horizon_lookup" not in policy and "max_switch_live_batch" not in policy
    assert policy["description"].endswith("force BF16->W4 at response length 2K")


@pytest.mark.parametrize(
    "kind, kwargs",
    [
        ("frontier", dict(batch=4, cap=2000, frontier=1001)),
        ("frontier", dict(batch=4, cap=2000, frontier=2000)),
        ("frontier", dict(batch=4, cap=2000)),
        ("live_threshold", dict(batch=4, cap=2000, threshold=4)),
        ("live_threshold", dict(batch=4, cap=2000, threshold=0)),
        ("live_threshold", dict(batch=4, cap=2000)),
        ("forced_switch", dict(batch=4, cap=2000, frontier=100)),
        ("bogus", dict(batch=4, cap=2000, frontier=1000)),
    ],
)
def test_invalid_arguments_are_rejected(tool, kind, kwargs):
    with pytest.raises(ValueError):
        tool.build_policy(kind, **kwargs)


def test_cli_writes_the_file_and_loads_in_the_vllm_policy_module(tmp_path):
    output = tmp_path / "nested" / "fixed_k1000.json"
    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--kind",
            "frontier",
            "--batch",
            "4",
            "--cap",
            "2000",
            "--frontier",
            "1000",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    text = output.read_text()
    assert text.endswith("\n") and "\n" not in text[:-1]
    raw = json.loads(text)
    assert raw["lookup_table"]["frontier_count"] == 7
    precision_policy = pytest.importorskip("vllm.v1.core.sched.precision_policy")
    policy = precision_policy.load_precision_policy(str(output))
    assert policy.kind == precision_policy.KIND_LOOKUP
    assert policy.initial_rollout_batch == 4
    assert policy.table.committed_frontier(250, 0, 4) == 1000
    assert policy.table.committed_frontier(1250, 0, 4) is None
