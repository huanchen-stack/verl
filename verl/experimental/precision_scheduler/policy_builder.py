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
"""Global-search receding-horizon policy builder and the schema-6 policy JSON contract.

For every observed state ``(frontier fi, prompt bucket, live)`` the builder compares the plan
"stay BF16" against every plan "BF16 until fj >= fi, then W4" and commits the cheapest switch
frontier only when it is strictly cheaper than staying.  Ties between candidate frontiers resolve
to the earliest one because the search iterates upward with a strict ``<``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .cost_model import PolicyGrid, TpotSource, make_tpot_cache, trajectory_cost_grid
from .hazard import HazardTable, survival

SCHEMA_VERSION = 6
LAYOUT = "frontier_major,prompt_bucket,live_batch"
REQUIRED_TOP_LEVEL = (
    "schema_version",
    "description",
    "scan_interval_tokens",
    "arm_min_requests",
    "capture_max_batch",
    "commitment_enabled",
    "receding_horizon_lookup",
    "initial_rollout_batch",
    "max_switch_live_batch",
    "calibration",
    "offline_cost_model",
    "lookup_table",
)
REQUIRED_LOOKUP = (
    "layout",
    "frontier_start",
    "frontier_step",
    "frontier_count",
    "prompt_bucket_start",
    "prompt_bucket_step",
    "prompt_bucket_count",
    "live_batch_start",
    "live_batch_count",
    "committed_frontiers",
)
REQUIRED_COST_MODEL = ("response_cap", "downstream_seconds_per_token", "switch_overhead_seconds")


def build_decisions(
    bf: HazardTable, w4: HazardTable, cache: dict[str, np.ndarray], grid: PolicyGrid, slope: float
) -> np.ndarray:
    """Dense committed-frontier table ``[frontier, prompt bucket, live]`` (0 = never switch)."""
    frontiers = grid.frontiers
    decisions = np.zeros(grid.shape, dtype=np.int64)
    for fi in range(len(frontiers)):
        bf_alive = survival(bf, fi)
        stay = trajectory_cost_grid(cache, grid, "bf16", fi, bf_alive, slope)
        best_cost = np.full_like(stay, np.inf)
        best_frontier = np.zeros_like(decisions[fi])
        for future_fi in range(fi, len(frontiers)):
            prefix_bins = future_fi - fi
            prefix = trajectory_cost_grid(cache, grid, "bf16", fi, bf_alive[:prefix_bins], slope)
            reach = float(bf_alive[prefix_bins])
            w4_alive = survival(w4, future_fi) * reach
            candidate = prefix + trajectory_cost_grid(cache, grid, "w4", future_fi, w4_alive, slope)
            improve = candidate < best_cost
            best_cost[improve] = candidate[improve]
            best_frontier[improve] = int(frontiers[future_fi])
        eligible = best_cost < stay
        decisions[fi][eligible] = best_frontier[eligible]
    return decisions


def policy_from_decisions(
    decisions: np.ndarray,
    grid: PolicyGrid,
    *,
    description: str,
    calibration: dict[str, Any],
    slope: float,
    receding_horizon_lookup: bool = True,
    capture_max_batch: int | None = None,
    max_switch_live_batch: int | None = None,
    switch_overhead_seconds: float = 0.0,
) -> dict[str, Any]:
    """Wrap a decision array in the schema-6 JSON contract consumed by the vLLM scheduler."""
    if decisions.shape != grid.shape:
        raise ValueError(f"decisions shape {decisions.shape} does not match grid {grid.shape}")
    frontiers, prompts = grid.frontiers, grid.prompts
    return {
        "schema_version": SCHEMA_VERSION,
        "description": description,
        "scan_interval_tokens": grid.step,
        "arm_min_requests": grid.batch,
        "capture_max_batch": grid.batch if capture_max_batch is None else int(capture_max_batch),
        "commitment_enabled": True,
        "receding_horizon_lookup": bool(receding_horizon_lookup),
        "initial_rollout_batch": grid.batch,
        "max_switch_live_batch": grid.batch if max_switch_live_batch is None else int(max_switch_live_batch),
        "calibration": dict(calibration),
        "offline_cost_model": {
            "response_cap": grid.cap,
            "downstream_seconds_per_token": float(slope),
            "switch_overhead_seconds": float(switch_overhead_seconds),
        },
        "lookup_table": {
            "layout": LAYOUT,
            "frontier_start": int(frontiers[0]),
            "frontier_step": grid.step,
            "frontier_count": len(frontiers),
            "prompt_bucket_start": int(prompts[0]),
            "prompt_bucket_step": grid.prompt_step,
            "prompt_bucket_count": len(prompts),
            "live_batch_start": 1,
            "live_batch_count": grid.batch,
            "committed_frontiers": decisions.reshape(-1).tolist(),
        },
    }


def build_policy(
    bf: HazardTable,
    w4: HazardTable,
    tpot: TpotSource,
    grid: PolicyGrid,
    slope: float,
    revision: int,
    alpha: float,
    *,
    description: str | None = None,
    calibration_kind: str = "paired BF16/full-W4 baseline traces plus online delayed-entry EMA",
    cache: dict[str, np.ndarray] | None = None,
    extra_calibration: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Run the global search and return ``(policy_json, switch_states)``."""
    if cache is None:
        cache = make_tpot_cache(grid, tpot)
    decisions = build_decisions(bf, w4, cache, grid, slope)
    if description is None:
        description = f"B{grid.batch} cap{grid.cap // 1024}K model-specific EMA future-frontier global full-cost lookup"
    calibration = {
        "kind": calibration_kind,
        "ema_alpha": float(alpha),
        "policy_revision": int(revision),
        "reprefill": False,
    }
    if extra_calibration:
        calibration.update(extra_calibration)
    policy = policy_from_decisions(decisions, grid, description=description, calibration=calibration, slope=slope)
    return policy, int(np.count_nonzero(decisions))


def fixed_frontier_policy(
    batch: int,
    cap: int,
    frontier: int,
    *,
    step: int = 250,
    prompt_step: int = 128,
    prompt_count: int = 17,
    capture_max_batch: int = 32,
) -> dict[str, Any]:
    """Dense policy that switches every request at one fixed response frontier (heuristic baseline).

    Byte-compatible with the archived ``build_fixed_frontier_policy.py`` output; the runtime also
    accepts the inline spec ``fixed_frontier:K`` so this generator exists for archival compatibility
    and for forced-switch calibration rollouts.
    """
    if frontier % step:
        raise ValueError("frontier must be aligned to the scan grid")
    if not step <= frontier < cap:
        raise ValueError("frontier must be within the response range")
    if batch <= 0 or prompt_count <= 0:
        raise ValueError("batch and prompt_count must be positive")
    frontiers = list(range(step, cap, step))
    values = [
        frontier if current <= frontier else 0
        for current in frontiers
        for _prompt in range(prompt_count)
        for _live in range(1, batch + 1)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "description": f"B{batch} cap{cap} fixed global BF16-to-W4 response frontier at {frontier} tokens",
        "scan_interval_tokens": step,
        "arm_min_requests": batch,
        "capture_max_batch": capture_max_batch,
        "commitment_enabled": True,
        "receding_horizon_lookup": False,
        "initial_rollout_batch": batch,
        "max_switch_live_batch": batch,
        "calibration": {
            "kind": "fixed generation-length baseline",
            "fixed_response_frontier": frontier,
            "reprefill": False,
        },
        "offline_cost_model": {
            "response_cap": cap,
            "downstream_seconds_per_token": 0.0,
            "switch_overhead_seconds": 0.0,
        },
        "lookup_table": {
            "layout": LAYOUT,
            "frontier_start": frontiers[0],
            "frontier_step": step,
            "frontier_count": len(frontiers),
            "prompt_bucket_start": 0,
            "prompt_bucket_step": prompt_step,
            "prompt_bucket_count": prompt_count,
            "live_batch_start": 1,
            "live_batch_count": batch,
            "committed_frontiers": values,
        },
    }


def validate_policy(policy: dict[str, Any]) -> None:
    """Structural validation mirroring the vLLM loader (schema 6, dense frontier-major table)."""
    if not isinstance(policy, dict):
        raise ValueError("policy must be a JSON object")
    for key in REQUIRED_TOP_LEVEL:
        if key not in policy:
            raise ValueError(f"policy is missing required key {key!r}")
    if policy["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}, got {policy['schema_version']!r}")
    for key in ("scan_interval_tokens", "arm_min_requests", "capture_max_batch", "initial_rollout_batch"):
        if not isinstance(policy[key], int) or isinstance(policy[key], bool) or policy[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if policy["max_switch_live_batch"] is not None and (
        not isinstance(policy["max_switch_live_batch"], int) or policy["max_switch_live_batch"] <= 0
    ):
        raise ValueError("max_switch_live_batch must be a positive integer or null")
    for key in ("commitment_enabled", "receding_horizon_lookup"):
        if not isinstance(policy[key], bool):
            raise ValueError(f"{key} must be a boolean")
    calibration = policy["calibration"]
    if not isinstance(calibration, dict) or "kind" not in calibration:
        raise ValueError("calibration must be an object with a 'kind'")
    cost_model = policy["offline_cost_model"]
    if not isinstance(cost_model, dict):
        raise ValueError("offline_cost_model must be an object")
    for key in REQUIRED_COST_MODEL:
        if key not in cost_model:
            raise ValueError(f"offline_cost_model is missing {key!r}")
    table = policy["lookup_table"]
    if not isinstance(table, dict):
        raise ValueError("lookup_table must be an object")
    for key in REQUIRED_LOOKUP:
        if key not in table:
            raise ValueError(f"lookup_table is missing {key!r}")
    if table["layout"] != LAYOUT:
        raise ValueError(f"lookup_table layout must be {LAYOUT!r}")
    for key in ("frontier_step", "frontier_count", "prompt_bucket_step", "prompt_bucket_count", "live_batch_count"):
        if not isinstance(table[key], int) or table[key] <= 0:
            raise ValueError(f"lookup_table.{key} must be a positive integer")
    if table["frontier_start"] != table["frontier_step"] or table["live_batch_start"] != 1:
        raise ValueError("lookup_table must start at the first frontier bin and live batch 1")
    if policy["scan_interval_tokens"] != table["frontier_step"]:
        raise ValueError("scan_interval_tokens must equal lookup_table.frontier_step")
    values = table["committed_frontiers"]
    expected = table["frontier_count"] * table["prompt_bucket_count"] * table["live_batch_count"]
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError(f"committed_frontiers must be a flat list of {expected} entries")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise ValueError("committed_frontiers must contain non-negative integer frontiers (0 = no switch)")
    step = table["frontier_step"]
    cap = cost_model["response_cap"]
    if any(v and (v % step or v >= cap) for v in values):
        raise ValueError("committed_frontiers must be on the scan grid and below the response cap")


def decisions_array(policy: dict[str, Any]) -> np.ndarray:
    """``committed_frontiers`` reshaped to ``[frontier, prompt bucket, live]``."""
    table = policy["lookup_table"]
    shape = (table["frontier_count"], table["prompt_bucket_count"], table["live_batch_count"])
    return np.asarray(table["committed_frontiers"], dtype=np.int64).reshape(shape)


def lookup(policy: dict[str, Any], frontier: int, prompt_tokens: float, live: int) -> int:
    """Committed frontier for a runtime state, clamping every axis like the scheduler does."""
    table = policy["lookup_table"]
    fi = int(np.clip((frontier - table["frontier_start"]) // table["frontier_step"], 0, table["frontier_count"] - 1))
    pi = int(
        np.clip(
            round((prompt_tokens - table["prompt_bucket_start"]) / table["prompt_bucket_step"]),
            0,
            table["prompt_bucket_count"] - 1,
        )
    )
    li = int(np.clip(live - table["live_batch_start"], 0, table["live_batch_count"] - 1))
    flat = (fi * table["prompt_bucket_count"] + pi) * table["live_batch_count"] + li
    return int(table["committed_frontiers"][flat])


def write_policy_atomic(path: Path, policy: dict[str, Any]) -> None:
    """Write compact JSON through a temp file so a concurrent reader never sees a partial policy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(policy, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text())
    validate_policy(policy)
    return policy
