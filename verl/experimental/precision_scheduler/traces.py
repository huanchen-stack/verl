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
"""Parsers for the rollout artefacts the toolkit consumes.

* request-lifetime traces (``request_lifetimes_replica*_node*.jsonl``) written by the verl rollout
  server: ``{"event": "start", "request_id", "prompt_tokens", "timestamp"}`` and
  ``{"event": "finish", "request_id", "generation_tokens", "finish_reason", "timestamp"}``
* switch-cohort JSONL written by the vLLM scheduler (``VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS``):
  ``{"event": "switch_cohort", "rollout_index", "requests": [{"request_id", "entry_output_tokens"}]}``
* metrics JSONL rows ``{"step": int, "data": {"timing_s/step": ..., "perf/total_num_tokens": ...}}``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ENGINE_SUFFIX_LENGTH = 8


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL tolerantly: a writer may still be appending the last line."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(errors="replace") as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def trace_lengths(
    path: Path, *, cap: int | None = None, limit: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Start rows (in file order, at most ``limit``) and ``request_id -> generation_tokens`` for them.

    With ``limit`` the trace must contain that many complete requests (calibration baselines).
    """
    starts: list[dict[str, Any]] = []
    finishes: dict[str, int] = {}
    wanted: set[str] = set()
    for item in read_jsonl(path):
        rid = str(item.get("request_id", ""))
        event = item.get("event")
        if event == "start" and (limit is None or len(starts) < limit):
            starts.append(item)
            wanted.add(rid)
        elif event == "finish" and rid in wanted and "generation_tokens" in item:
            length = int(item["generation_tokens"])
            finishes[rid] = length if cap is None else min(length, cap)
    if limit is not None and (len(starts) < limit or len(finishes) < limit):
        raise RuntimeError(f"incomplete calibration trace {path}: {len(starts)} starts / {len(finishes)} finishes")
    return starts, finishes


def final_lengths(starts: list[dict[str, Any]], finishes: dict[str, int]) -> np.ndarray:
    return np.asarray([finishes[str(row["request_id"])] for row in starts], dtype=np.int64)


def completed_steps(starts: list[dict[str, Any]], finishes: dict[str, int], batch: int) -> int:
    """Number of leading rollout steps (groups of ``batch`` starts) whose requests all finished."""
    complete = 0
    for offset in range(0, len(starts), batch):
        group = starts[offset : offset + batch]
        if len(group) < batch or any(str(row["request_id"]) not in finishes for row in group):
            break
        complete += 1
    return complete


def resolve_request_id(request_id: str, finishes: dict[str, int]) -> str | None:
    """Map EngineCore's ``<client-id>-<8 hex>`` back to the trace's client id."""
    if request_id in finishes:
        return request_id
    client_id, separator, engine_suffix = request_id.rpartition("-")
    if separator and len(engine_suffix) == ENGINE_SUFFIX_LENGTH and client_id in finishes:
        return client_id
    return None


def read_cohorts(path: Path) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(path) if row.get("event") == "switch_cohort"]


def cohort_observation(
    cohort: dict[str, Any], finishes: dict[str, int], cap: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """``(entry_tokens, final_lengths)`` for a cohort, or ``None`` if any request is unresolved."""
    requests = cohort.get("requests", [])
    if not requests:
        return None
    ids = [resolve_request_id(str(row["request_id"]), finishes) for row in requests]
    if any(rid is None for rid in ids):
        return None
    entries = np.asarray([int(row["entry_output_tokens"]) for row in requests], dtype=np.int64)
    finals = np.asarray([min(finishes[rid], cap) for rid in ids], dtype=np.int64)
    if np.any(finals < entries):
        raise RuntimeError(f"rollout {cohort.get('rollout_index')}: final length precedes switch entry")
    return entries, finals


def read_metrics(paths: list[Path]) -> list[dict[str, Any]]:
    """Metric rows (``step`` + ``data``) from one or more rl_workflow_timing JSONL files."""
    rows = []
    for path in paths:
        rows.extend(row for row in read_jsonl(path) if isinstance(row.get("data"), dict))
    return rows
