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
"""Validate one rollout-only run directory (ported from the archived collect_best_t8_verl_rollout.py checks).

Checks, for ``expected_requests`` = train_batch_size * rollout_n per step:
  * traces/request_lifetimes_replica*_node*.jsonl: N start rows and N finish rows with identical id sets;
  * rollouts/<step>.jsonl: N dump rows per step;
  * driver.log: one ``VERL_ROLLOUT_ONLY_COMPLETE step=<k> requests=N gen_seconds=<t>`` marker per step;
  * metrics/**/*.jsonl (FileLogger): one row per step with the four rollout_only metrics;
  * no OOM strings in the driver log.
Usage: python validate_rollout_only_run.py <run_dir> --expected-requests N [--steps K]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MARKER_RE = re.compile(r"VERL_ROLLOUT_ONLY_COMPLETE step=(\d+) requests=(\d+) gen_seconds=([0-9.eE+-]+)")
ROLLOUT_ONLY_METRICS = ("timing_s/gen", "rollout_only/requests", "rollout_only/response_tokens", "critic/rewards/mean")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_rollout_only_run(run_dir: Path, expected_requests: int, steps: int = 1) -> dict:
    errors: list[str] = []
    starts, finishes = {}, {}
    for path in sorted((run_dir / "traces").glob("request_lifetimes_replica*_node*.jsonl")):
        for row in _jsonl(path):
            (starts if row["event"] == "start" else finishes)[row["request_id"]] = row
    total = expected_requests * steps
    if len(starts) != total or len(finishes) != total or set(starts) != set(finishes):
        errors.append(f"trace cardinality starts={len(starts)} finishes={len(finishes)} expected={total}")
    if any("token_ids" not in row for row in finishes.values()):
        errors.append("finish rows without token_ids (request_trace_log_tokens not honored)")
    if any("trace_request_id" not in row for row in starts.values()):
        errors.append("start rows without trace_request_id")

    dump_rows = {}
    for step in range(1, steps + 1):
        path = run_dir / "rollouts" / f"{step}.jsonl"
        dump_rows[step] = len(_jsonl(path)) if path.exists() else -1
        if dump_rows[step] != expected_requests:
            errors.append(f"rollout dump rows step {step}: {dump_rows[step]} != {expected_requests}")

    log = (run_dir / "driver.log").read_text(encoding="utf-8", errors="replace")
    markers = MARKER_RE.findall(log)
    if [int(s) for s, _, _ in markers] != list(range(1, steps + 1)):
        errors.append(f"markers found for steps {[int(s) for s, _, _ in markers]}, expected 1..{steps}")
    if any(int(r) != expected_requests for _, r, _ in markers):
        errors.append(f"marker request counts {[r for _, r, _ in markers]} != {expected_requests}")
    if "out of memory" in log.lower() or "outofmemoryerror" in log.lower():
        errors.append("OOM string in driver log")

    metric_rows = [row for path in sorted((run_dir / "metrics").rglob("*.jsonl")) for row in _jsonl(path)]
    rollout_rows = [row for row in metric_rows if all(k in row.get("data", {}) for k in ROLLOUT_ONLY_METRICS)]
    if len(rollout_rows) != steps:
        errors.append(f"FileLogger rows with rollout_only metrics: {len(rollout_rows)} != {steps}")
    elif any(row["data"]["rollout_only/requests"] != expected_requests for row in rollout_rows):
        errors.append("rollout_only/requests metric disagrees with expected_requests")

    summary = {
        "valid": not errors,
        "errors": errors,
        "request_count": len(finishes),
        "total_output_tokens": sum(int(r["generation_tokens"]) for r in finishes.values()),
        "rollout_seconds": [float(t) for _, _, t in markers],
        "dump_rows": dump_rows,
        "metrics_rows": len(rollout_rows),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args(argv)
    summary = validate_rollout_only_run(args.run_dir, args.expected_requests, args.steps)
    print(json.dumps(summary, indent=2))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
