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
"""Build the deterministic static baseline policies of the precision scheduler.

Three kinds, all dense lookup tables in the policy JSON contract shared with vLLM
(``vllm/v1/core/sched/precision_policy.py``), reproduced byte-for-byte from the
archived experiment builders:

* ``frontier``       switch every rollout at one response frontier K (the fixed-k
                     baselines ``fixed_k{6000,8000,10000}.json`` and the
                     ``*_fixed_frontier<K>_30step.json`` sweeps): every cell at or
                     before K is K, later cells 0; monotone commitment.
* ``live_threshold`` switch when the live batch drains to t (the fixed-t baselines
                     ``fixed_t{2,4,8}.json``): every cell is the first 250-token
                     frontier and ``max_switch_live_batch = t`` is the sole switching
                     condition, so the fixed baseline writes the same switch-cohort
                     JSONL as the calibrated policies.
* ``forced_switch``  schema-4 rollout-only calibration policy that forces one
                     BF16->W4 switch at K for a B128 cohort
                     (``b128_cap24576_forced_tail8k_calibration.json``).

The inline specs ``fixed_frontier:K`` / ``fixed_threshold:t`` of the vLLM policy flag
express the same two baselines without a file (cohort-free arming); this builder is
for runs that need the explicit ``initial_rollout_batch`` cohort or a checked-in file.

    build_static_policy.py --kind frontier --batch 64 --cap 24576 --frontier 8000 --output fixed_k8000.json
    build_static_policy.py --kind live_threshold --batch 64 --cap 24576 --threshold 8 --output fixed_t8.json
    build_static_policy.py --kind forced_switch --batch 128 --cap 24576 --frontier 8000 --output forced8k.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCAN_STEP = 250
PROMPT_STEP = 128
PROMPT_COUNT = 17
CAPTURE_MAX_BATCH = 32
LAYOUT = "frontier_major,prompt_bucket,live_batch"

KIND_FRONTIER = "frontier"
KIND_LIVE_THRESHOLD = "live_threshold"
KIND_FORCED_SWITCH = "forced_switch"
KINDS = (KIND_FRONTIER, KIND_LIVE_THRESHOLD, KIND_FORCED_SWITCH)


def _frontiers(cap: int) -> list[int]:
    return list(range(SCAN_STEP, cap, SCAN_STEP))


def _lookup_table(frontiers: list[int], batch: int, values: list[int]) -> dict[str, Any]:
    return {
        "layout": LAYOUT,
        "frontier_start": frontiers[0],
        "frontier_step": SCAN_STEP,
        "frontier_count": len(frontiers),
        "prompt_bucket_start": 0,
        "prompt_bucket_step": PROMPT_STEP,
        "prompt_bucket_count": PROMPT_COUNT,
        "live_batch_start": 1,
        "live_batch_count": batch,
        "committed_frontiers": values,
    }


def _cells(frontiers: list[int], batch: int, value_at) -> list[int]:
    values: list[int] = []
    for current in frontiers:
        value = value_at(current)
        for _prompt in range(PROMPT_COUNT):
            for _live in range(1, batch + 1):
                values.append(value)
    return values


def build_frontier_policy(batch: int, cap: int, frontier: int) -> dict[str, Any]:
    """``build_fixed_frontier_policy.py`` (dynamic_tail8k_heatmap_20260823)."""
    if frontier % SCAN_STEP:
        raise ValueError("frontier must be aligned to the 250-token scan grid")
    if not (SCAN_STEP <= frontier < cap):
        raise ValueError("frontier must be within the response range")
    frontiers = _frontiers(cap)
    values = _cells(frontiers, batch, lambda current: frontier if current <= frontier else 0)
    return {
        "schema_version": 6,
        "description": f"B{batch} cap{cap} fixed global BF16-to-W4 response frontier at {frontier} tokens",
        "scan_interval_tokens": SCAN_STEP,
        "arm_min_requests": batch,
        "capture_max_batch": CAPTURE_MAX_BATCH,
        "commitment_enabled": True,
        "receding_horizon_lookup": False,
        "initial_rollout_batch": batch,
        "calibration": {
            "kind": "fixed generation-length baseline",
            "fixed_response_frontier": frontier,
            "reprefill": False,
        },
        "lookup_table": _lookup_table(frontiers, batch, values),
    }


def build_live_threshold_policy(batch: int, cap: int, threshold: int) -> dict[str, Any]:
    """``build_fixed_live_threshold_policy.py`` (full_rl_policy_matrix_b64_cap24k_20260902)."""
    if threshold <= 0 or threshold >= batch:
        raise ValueError("threshold must be in [1, batch)")
    frontiers = _frontiers(cap)
    # Always return the already-passed first scan boundary: the
    # max_switch_live_batch gate is the sole switching condition.
    values = _cells(frontiers, batch, lambda _current: SCAN_STEP)
    return {
        "schema_version": 6,
        "description": f"B{batch} cap{cap} fixed live-batch threshold t={threshold} with exact switch-cohort logging",
        "scan_interval_tokens": SCAN_STEP,
        "arm_min_requests": batch,
        "capture_max_batch": CAPTURE_MAX_BATCH,
        "commitment_enabled": True,
        "receding_horizon_lookup": False,
        "initial_rollout_batch": batch,
        "max_switch_live_batch": threshold,
        "calibration": {
            "kind": "fixed live-batch threshold baseline",
            "fixed_live_batch_threshold": threshold,
            "reprefill": False,
        },
        "lookup_table": _lookup_table(frontiers, batch, values),
    }


def build_forced_switch_policy(batch: int, cap: int, frontier: int) -> dict[str, Any]:
    """``build_forced8k_calibration_policy.py`` (dynamic_tail8k_heatmap_20260823): schema 4."""
    if frontier % SCAN_STEP or not (SCAN_STEP <= frontier < cap):
        raise ValueError("frontier must be a 250-aligned value inside the response range")
    frontiers = _frontiers(cap)
    values = _cells(frontiers, batch, lambda current: frontier if current <= frontier else 0)
    cap_label = f"{cap // 1024}K"
    frontier_label = f"{frontier // 1000}K" if frontier % 1000 == 0 else str(frontier)
    return {
        "schema_version": 4,
        "description": (
            f"B{batch} cap{cap_label} rollout-only calibration: force BF16->W4 at response length {frontier_label}"
        ),
        "scan_interval_tokens": SCAN_STEP,
        "arm_min_requests": batch,
        "capture_max_batch": CAPTURE_MAX_BATCH,
        "commitment_enabled": True,
        "initial_rollout_batch": batch,
        "lookup_table": _lookup_table(frontiers, batch, values),
    }


def build_policy(kind: str, *, batch: int, cap: int, frontier: int | None = None, threshold: int | None = None):
    if kind == KIND_FRONTIER:
        if frontier is None:
            raise ValueError("--frontier is required for --kind frontier")
        return build_frontier_policy(batch, cap, frontier)
    if kind == KIND_LIVE_THRESHOLD:
        if threshold is None:
            raise ValueError("--threshold is required for --kind live_threshold")
        return build_live_threshold_policy(batch, cap, threshold)
    if kind == KIND_FORCED_SWITCH:
        if frontier is None:
            raise ValueError("--frontier is required for --kind forced_switch")
        return build_forced_switch_policy(batch, cap, frontier)
    raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")


def serialize(policy: dict[str, Any]) -> str:
    """Exactly the archived encoding: compact separators, one trailing newline."""
    return json.dumps(policy, separators=(",", ":")) + "\n"


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--batch", type=int, required=True, help="rollout cohort size B (live-batch axis 1..B)")
    parser.add_argument("--cap", type=int, required=True, help="response cap; frontiers run 250..cap-250")
    parser.add_argument("--frontier", type=int, help="switch frontier K for frontier / forced_switch")
    parser.add_argument("--threshold", type=int, help="live-batch threshold t for live_threshold")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch <= 0 or args.cap <= SCAN_STEP:
        raise SystemExit("--batch must be positive and --cap larger than the 250-token scan step")
    policy = build_policy(args.kind, batch=args.batch, cap=args.cap, frontier=args.frontier, threshold=args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialize(policy))
    print(args.output)
    return args.output


if __name__ == "__main__":
    main()
