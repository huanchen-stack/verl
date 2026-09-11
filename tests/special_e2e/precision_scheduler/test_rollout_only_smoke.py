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
"""gpu-smoke: one rollout-only step (Qwen3.5-4B, GSM8K, precision_scheduler.enable=false).

The GPU run itself is launched out of band under the decision-13 launcher::

    RUN_DIR=<dir> run_gpu.sh --gpus <N> --timeout 1800 -- bash run_rollout_only_smoke.sh

This test validates that run directory with the ported collector when
``PS_ROLLOUT_ONLY_RUN_DIR`` points at it; otherwise it is skipped. The validator itself is
exercised on a synthetic run directory in ``test_validator_on_synthetic_run``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_rollout_only_run import validate_rollout_only_run  # noqa: E402


def _write_synthetic_run(run_dir: Path, n: int, drop_finish: bool = False) -> None:
    (run_dir / "traces").mkdir(parents=True)
    (run_dir / "rollouts").mkdir()
    (run_dir / "metrics").mkdir()
    with open(run_dir / "traces" / "request_lifetimes_replica000_node000.jsonl", "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "event": "start",
                        "prompt_tokens": 5,
                        "request_id": f"r{i}",
                        "timestamp": 1.0,
                        "trace_request_id": f"idx-{i}_0",
                    }
                )
                + "\n"
            )
        for i in range(n - (1 if drop_finish else 0)):
            f.write(
                json.dumps(
                    {
                        "event": "finish",
                        "finish_reason": "stop",
                        "generation_tokens": 3,
                        "request_id": f"r{i}",
                        "timestamp": 2.0,
                        "token_ids": [1, 2, 3],
                        "trace_request_id": f"idx-{i}_0",
                    }
                )
                + "\n"
            )
    with open(run_dir / "rollouts" / "1.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"uid": f"idx-{i}", "score": 1.0}) + "\n")
    (run_dir / "driver.log").write_text(f"...\nVERL_ROLLOUT_ONLY_COMPLETE step=1 requests={n} gen_seconds=12.5\n")
    with open(run_dir / "metrics" / "rows.jsonl", "w") as f:
        f.write(
            json.dumps(
                {
                    "step": 1,
                    "data": {
                        "timing_s/gen": 12.5,
                        "rollout_only/requests": n,
                        "rollout_only/response_tokens": 3 * n,
                        "critic/rewards/mean": 1.0,
                    },
                }
            )
            + "\n"
        )


def test_validator_on_synthetic_run(tmp_path):
    _write_synthetic_run(tmp_path / "ok", 8)
    summary = validate_rollout_only_run(tmp_path / "ok", expected_requests=8)
    assert summary["valid"], summary["errors"]
    assert summary["request_count"] == 8 and summary["rollout_seconds"] == [12.5]
    _write_synthetic_run(tmp_path / "bad", 8, drop_finish=True)
    summary = validate_rollout_only_run(tmp_path / "bad", expected_requests=8)
    assert not summary["valid"] and any("cardinality" in e for e in summary["errors"])


@pytest.mark.gpu_smoke
def test_rollout_only_smoke_run_dir():
    run_dir = os.environ.get("PS_ROLLOUT_ONLY_RUN_DIR")
    if not run_dir:
        pytest.skip("set PS_ROLLOUT_ONLY_RUN_DIR to a run produced by run_rollout_only_smoke.sh")
    expected = int(os.environ.get("PS_ROLLOUT_ONLY_EXPECTED_REQUESTS", "32"))
    summary = validate_rollout_only_run(Path(run_dir), expected_requests=expected)
    assert summary["valid"], summary["errors"]
