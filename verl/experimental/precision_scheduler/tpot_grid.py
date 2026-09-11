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
"""Decode TPOT grids: the profiler heatmap (Gen2 ``heatmap.json``) and the legacy Gen1 CSV grid."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

PRECISION_KEYS = {"bf16": "bf16_tpot_ms", "w4": "int4_tpot_ms"}
# Row labels used by the profiler harness (vLLM tools/precision_scheduler/tpot_heatmap.py).
HEATMAP_PRECISIONS = ("bf16", "int4")


class TpotGrid:
    """Measured decode milliseconds/token on a (batch size x context length) grid, log2-interpolated."""

    def __init__(self, batch_sizes, seq_lens, bf16: np.ndarray, w4: np.ndarray) -> None:
        self.bs = np.asarray(batch_sizes, dtype=float)
        self.seq = np.asarray(seq_lens, dtype=float)
        self.grid = {"bf16": np.asarray(bf16, dtype=float), "w4": np.asarray(w4, dtype=float)}
        shape = (len(self.bs), len(self.seq))
        for precision, values in self.grid.items():
            if values.shape != shape:
                raise ValueError(f"{precision} grid has shape {values.shape}, expected {shape}")
            if not np.any(np.isfinite(values)):
                raise ValueError(f"{precision} grid has no finite cells")
        if np.any(np.diff(self.bs) <= 0) or np.any(np.diff(self.seq) <= 0):
            raise ValueError("batch sizes and sequence lengths must be strictly increasing")

    @classmethod
    def from_heatmap_json(cls, source: Path | dict[str, Any]) -> TpotGrid:
        data = json.loads(Path(source).read_text()) if not isinstance(source, dict) else source
        to_array = lambda rows: np.asarray([[math.nan if v is None else v for v in row] for row in rows], float)  # noqa: E731
        return cls(
            data["batch_sizes"], data["seq_lens"], to_array(data["bf16_tpot_ms"]), to_array(data["int4_tpot_ms"])
        )

    @classmethod
    def from_legacy_csv(cls, path: Path) -> TpotGrid:
        """Gen1 grid CSV (``batch_size, context_len, bf16_tpot_ms, partial_int4_tpot_ms, ...``)."""
        rows = list(csv.DictReader(Path(path).open(newline="")))
        batches = sorted({int(r["batch_size"]) for r in rows})
        contexts = sorted({int(r["context_len"]) for r in rows})
        bf16 = np.full((len(batches), len(contexts)), math.nan)
        w4 = np.full((len(batches), len(contexts)), math.nan)
        for r in rows:
            i, j = batches.index(int(r["batch_size"])), contexts.index(int(r["context_len"]))
            if r.get("bf16_tpot_ms"):
                bf16[i, j] = float(r["bf16_tpot_ms"])
            if r.get("partial_int4_tpot_ms"):
                w4[i, j] = float(r["partial_int4_tpot_ms"])
        return cls(batches, contexts, bf16, w4)

    def get(self, precision: str, context: float, batch: float) -> float:
        """Log2 interpolation in context then batch; NaN cells are masked per row."""
        grid = self.grid[precision]
        along = []
        for row in grid:
            valid = np.isfinite(row)
            if not np.any(valid):
                along.append(np.nan)
            else:
                along.append(np.interp(np.log2(max(context, 1)), np.log2(self.seq[valid]), row[valid]))
        along = np.asarray(along)
        valid = np.isfinite(along)
        return float(np.interp(np.log2(max(batch, 1)), np.log2(self.bs[valid]), along[valid]))

    def to_heatmap_json(self) -> dict[str, Any]:
        nan_to_none = lambda a: [[None if not np.isfinite(v) else float(v) for v in row] for row in a]  # noqa: E731
        bf16, w4 = nan_to_none(self.grid["bf16"]), nan_to_none(self.grid["w4"])
        return {
            "batch_sizes": [int(b) for b in self.bs],
            "seq_lens": [int(s) for s in self.seq],
            "bf16_tpot_ms": bf16,
            "int4_tpot_ms": w4,
            "speedup_bf16_over_int4": _speedup(bf16, w4),
        }


def _ratio(left, right):
    return left / right if left is not None and right not in (None, 0.0) else None


def _speedup(bf16, int4):
    return [[_ratio(left, right) for left, right in zip(a, b, strict=True)] for a, b in zip(bf16, int4, strict=True)]


def matrix_payload(
    rows: list[dict[str, Any]],
    batch_sizes: list[int],
    seq_lens: list[int],
    required_kv_bytes: dict[tuple[int, int], int] | None = None,
) -> dict[str, Any]:
    """``heatmap.json`` payload from profiler ``cells.jsonl`` rows (missing cells stay ``null``).

    Mirrors ``matrix_payload`` in the vLLM harness so the archived heatmaps can be regenerated here.
    """
    lookup = {
        (int(r["batch_size"]), int(r["seq_len"]), str(r["precision"])): r for r in rows if r.get("status") == "ok"
    }

    def matrix(precision):
        return [
            [
                float(lookup[(b, s, precision)]["median_tpot_ms"]) if (b, s, precision) in lookup else None
                for s in seq_lens
            ]
            for b in batch_sizes
        ]

    required = dict(required_kv_bytes or {})
    for r in rows:
        required.setdefault((int(r["batch_size"]), int(r["seq_len"])), int(r["required_kv_bytes"]))
    bf16, int4 = matrix("bf16"), matrix("int4")
    return {
        "batch_sizes": list(batch_sizes),
        "seq_lens": list(seq_lens),
        "required_kv_bytes": [[required.get((b, s)) for s in seq_lens] for b in batch_sizes],
        "bf16_tpot_ms": bf16,
        "int4_tpot_ms": int4,
        "speedup_bf16_over_int4": _speedup(bf16, int4),
    }


def median_speedup(payload: dict[str, Any]) -> float:
    values = [v for row in payload["speedup_bf16_over_int4"] for v in row if v is not None]
    if not values:
        raise ValueError("heatmap has no paired BF16/INT4 cells")
    return float(statistics.median(values))


def validate_heatmap(payload: dict[str, Any], *, min_median_speedup: float = 1.02) -> float:
    """Guard against a profiler run whose INT4 rows silently executed BF16.

    All heatmaps dated 2026-09-09 were produced before the precision-selector fix and show a median
    speedup of exactly 1.000; a valid W4 heatmap has a clearly non-unit median.  Returns the median.
    """
    median = median_speedup(payload)
    if abs(median - 1.0) < (min_median_speedup - 1.0):
        raise ValueError(
            f"median BF16/INT4 speedup {median:.4f} is indistinguishable from 1.0; "
            "the INT4 rows most likely ran at BF16 (precision selection did not take effect)"
        )
    return median


def legacy_grid_rows(bf16_rows: list[dict[str, Any]], w4_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gen1 ``fixed_token_bench`` JSONL rows -> rows of the legacy ``*_tpot_grid.csv``.

    Cells where either precision is ``oom`` are emitted with empty numbers and ``benchmark_status=oom``.
    """
    bf16 = {(int(r["batch_size"]), int(r["context_len"])): r for r in bf16_rows}
    w4 = {(int(r["batch_size"]), int(r["context_len"])): r for r in w4_rows}
    batches = sorted({k[0] for k in set(bf16) | set(w4)})
    contexts = sorted({k[1] for k in set(bf16) | set(w4)})
    out = []
    for b in batches:
        for c in contexts:
            left, right = bf16.get((b, c)), w4.get((b, c))
            row = {"batch_size": b, "context_len": c, "bf16_tpot_ms": "", "partial_int4_tpot_ms": "", "speedup": ""}
            row["speedup_percent"] = ""
            row["aggregation"] = "median_of_5_repetitions"
            if left is None or right is None:
                row["benchmark_status"] = "missing"
            elif left["status"] == "oom" or right["status"] == "oom":
                row["benchmark_status"] = "oom"
            elif left["status"] == "ok" and right["status"] == "ok":
                speedup = left["tpot_ms"] / right["tpot_ms"]
                row.update(
                    bf16_tpot_ms=left["tpot_ms"],
                    partial_int4_tpot_ms=right["tpot_ms"],
                    speedup=speedup,
                    speedup_percent=(speedup - 1.0) * 100.0,
                    benchmark_status="ok",
                )
            else:
                row["benchmark_status"] = "missing"
            out.append(row)
    return out


LEGACY_GRID_FIELDS = (
    "batch_size",
    "context_len",
    "bf16_tpot_ms",
    "partial_int4_tpot_ms",
    "speedup",
    "speedup_percent",
    "aggregation",
    "benchmark_status",
)


def write_legacy_grid_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEGACY_GRID_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
