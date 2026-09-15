#!/usr/bin/env python3
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
"""Print a three-precision ``heatmap.json`` as readable tables.

The vLLM harness (``tools/precision_scheduler/tpot_heatmap.py``) writes BF16, INT4 (W4A16)
and NVFP4 decode-TPOT matrices plus a speedup matrix per quantized row.  This renders them
as text so a run can be read without opening the PNGs.

    python summarize_tpot_heatmap.py <run dir or heatmap.json>
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

TPOT_KEYS = (("bf16", "bf16_tpot_ms"), ("int4 (W4A16)", "int4_tpot_ms"), ("nvfp4", "nvfp4_tpot_ms"))
SPEEDUP_KEYS = (
    ("BF16 / INT4 (W4A16)", "speedup_bf16_over_int4"),
    ("BF16 / NVFP4", "speedup_bf16_over_nvfp4"),
)


def _table(matrix: list[list[float | None]], batch_sizes: list[int], seq_lens: list[int], fmt: str) -> str:
    header = "  batch " + "".join(f"{s:>9}" for s in seq_lens)
    lines = [header]
    for batch, row in zip(batch_sizes, matrix, strict=True):
        cells = "".join("     None" if v is None else format(v, fmt) for v in row)
        lines.append(f"{batch:>7} " + cells)
    return "\n".join(lines)


def _finite(matrix: list[list[float | None]]) -> list[float]:
    return [v for row in matrix for v in row if v is not None]


def summarize(payload: dict[str, Any]) -> str:
    batch_sizes, seq_lens = payload["batch_sizes"], payload["seq_lens"]
    out: list[str] = []
    for label, key in TPOT_KEYS:
        matrix = payload.get(key)
        if not matrix or not _finite(matrix):
            continue
        values = _finite(matrix)
        out.append(f"\n{label} decode TPOT (ms/token), rows = batch, cols = context")
        out.append(_table(matrix, batch_sizes, seq_lens, ">9.2f"))
        out.append(f"  min {min(values):.2f}  median {statistics.median(values):.2f}  max {max(values):.2f}")
    for label, key in SPEEDUP_KEYS:
        matrix = payload.get(key)
        if not matrix or not _finite(matrix):
            continue
        values = _finite(matrix)
        out.append(f"\n{label} speedup (>1 means the quantized row decodes faster)")
        out.append(_table(matrix, batch_sizes, seq_lens, ">9.3f"))
        out.append(f"  min {min(values):.3f}  median {statistics.median(values):.3f}  max {max(values):.3f}")
    return "\n".join(out)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    target = Path(sys.argv[1])
    path = target / "heatmap.json" if target.is_dir() else target
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = path.parent / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        print(f"run status : {data.get('status')}")
        for precision, model in (data.get("base_models") or {}).items():
            print(f"  {precision:6s} <- {model}")
    print(summarize(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
