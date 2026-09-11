#!/usr/bin/env python3
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
"""Post-run acceptance checks for a precision-scheduler run directory.

Ported from the archived ``collect_best_t8_verl_rollout.py`` (its asserts were the acceptance
contract of every rollout-only cell) and the C8 ``validate_rollout_only_run.py`` (whose function
name and summary keys are kept; that file now delegates here).

Checks, for ``expected_requests`` = train_batch_size * rollout_n per step:
  * traces/request_lifetimes_replica*_node*.jsonl: N start rows and N finish rows per step with
    identical id sets, ``trace_request_id`` on start rows, ``token_ids`` on finish rows (unless
    ``--no-token-ids``);
  * rollouts/<step>.jsonl: N dump rows per step;
  * driver.log: one ``VERL_ROLLOUT_ONLY_COMPLETE step=<k> requests=N gen_seconds=<t>`` marker per
    step (rollout-only runs) or the FileLogger timing keys (``--mode full_step``);
  * metrics/**/*.jsonl: one FileLogger row per step with the rollout-only metrics, or with the
    full-step timing keys;
  * no OOM strings in the driver log;
  * optional INT4 binding proof for W4 policies: ``lora_base_layers=<n>`` and
    ``int4_shadow_active=<n>`` in the log equal ``--expected-lora-layers`` (152 for Qwen3.5-9B);
  * optional COMPLETE marker (``--require-complete``).

Usage: validate_rollout_run.py <run_dir> --expected-requests N [--steps K] [--mode rollout_only|full_step]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MARKER_RE = re.compile(r"VERL_ROLLOUT_ONLY_COMPLETE step=(\d+) requests=(\d+) gen_seconds=([0-9.eE+-]+)")
ROLLOUT_ONLY_METRICS = ("timing_s/gen", "rollout_only/requests", "rollout_only/response_tokens", "critic/rewards/mean")
FULL_STEP_METRICS = (
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "timing_s/step",
    "perf/total_num_tokens",
)
OOM_STRINGS = ("out of memory", "outofmemoryerror")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_traces(run_dir: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    starts: dict[str, dict] = {}
    finishes: dict[str, dict] = {}
    for path in sorted((run_dir / "traces").glob("request_lifetimes_replica*_node*.jsonl")):
        for row in _jsonl(path):
            (starts if row.get("event") == "start" else finishes)[row["request_id"]] = row
    return starts, finishes


def read_metric_rows(run_dir: Path) -> list[dict]:
    return [row for path in sorted((run_dir / "metrics").rglob("*.jsonl")) for row in _jsonl(path)]


def validate_rollout_run(
    run_dir: Path,
    expected_requests: int,
    steps: int = 1,
    *,
    mode: str = "rollout_only",
    require_token_ids: bool = True,
    expected_lora_layers: int | None = None,
    require_complete: bool = False,
) -> dict:
    errors: list[str] = []
    starts, finishes = read_traces(run_dir)
    total = expected_requests * steps
    if len(starts) != total or len(finishes) != total or set(starts) != set(finishes):
        errors.append(f"trace cardinality starts={len(starts)} finishes={len(finishes)} expected={total}")
    if require_token_ids and any("token_ids" not in row for row in finishes.values()):
        errors.append("finish rows without token_ids (request_trace_log_tokens not honored)")
    if any("trace_request_id" not in row for row in starts.values()):
        errors.append("start rows without trace_request_id")

    dump_rows: dict[int, int] = {}
    for step in range(1, steps + 1):
        path = run_dir / "rollouts" / f"{step}.jsonl"
        dump_rows[step] = len(_jsonl(path)) if path.exists() else -1
        if dump_rows[step] != expected_requests:
            errors.append(f"rollout dump rows step {step}: {dump_rows[step]} != {expected_requests}")

    log_path = run_dir / "driver.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if not log:
        errors.append("driver.log missing or empty")
    markers = MARKER_RE.findall(log)
    if mode == "rollout_only":
        if [int(s) for s, _, _ in markers] != list(range(1, steps + 1)):
            errors.append(f"markers found for steps {[int(s) for s, _, _ in markers]}, expected 1..{steps}")
        if any(int(r) != expected_requests for _, r, _ in markers):
            errors.append(f"marker request counts {[r for _, r, _ in markers]} != {expected_requests}")
    lowered = log.lower()
    if any(s in lowered for s in OOM_STRINGS):
        errors.append("OOM string in driver log")

    metric_keys = ROLLOUT_ONLY_METRICS if mode == "rollout_only" else FULL_STEP_METRICS
    metric_rows = read_metric_rows(run_dir)
    step_rows = [row for row in metric_rows if all(k in row.get("data", {}) for k in metric_keys)]
    if len(step_rows) != steps:
        errors.append(f"FileLogger rows with {mode} metrics: {len(step_rows)} != {steps}")
    elif mode == "rollout_only" and any(row["data"]["rollout_only/requests"] != expected_requests for row in step_rows):
        errors.append("rollout_only/requests metric disagrees with expected_requests")

    if expected_lora_layers is not None:
        wrappers = [int(x) for x in re.findall(r"lora_base_layers=(\d+)", log)]
        if not wrappers or any(x != expected_lora_layers for x in wrappers):
            errors.append(f"unexpected LoRA wrapper counts: {wrappers} (expected {expected_lora_layers})")
        if not re.search(rf"precision=int4[^\n]*int4_shadow_active={expected_lora_layers}\b", log):
            errors.append(f"missing {expected_lora_layers}-layer INT4 binding proof")
    if require_complete and not (run_dir / "COMPLETE").exists():
        errors.append("COMPLETE marker missing")

    return {
        "valid": not errors,
        "errors": errors,
        "mode": mode,
        "steps": steps,
        "request_count": len(finishes),
        "total_output_tokens": sum(int(r.get("generation_tokens", 0)) for r in finishes.values()),
        "cap_hits": sum(1 for r in finishes.values() if r.get("finish_reason") == "length"),
        "rollout_seconds": [float(t) for _, _, t in markers],
        "dump_rows": dump_rows,
        "metrics_rows": len(step_rows),
        "reward_mean": [
            row["data"].get("critic/rewards/mean", row["data"].get("critic/score/mean")) for row in step_rows
        ],
    }


# Backward-compatible name used by tests/special_e2e/precision_scheduler (C8).
def validate_rollout_only_run(run_dir: Path, expected_requests: int, steps: int = 1) -> dict:
    return validate_rollout_run(run_dir, expected_requests, steps, mode="rollout_only")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--mode", choices=("rollout_only", "full_step"), default="rollout_only")
    parser.add_argument("--no-token-ids", action="store_true", help="do not require token_ids on finish rows")
    parser.add_argument("--expected-lora-layers", type=int, default=None, help="INT4 binding proof (W4 policies)")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    summary = validate_rollout_run(
        args.run_dir,
        args.expected_requests,
        args.steps,
        mode=args.mode,
        require_token_ids=not args.no_token_ids,
        expected_lora_layers=args.expected_lora_layers,
        require_complete=args.require_complete,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
