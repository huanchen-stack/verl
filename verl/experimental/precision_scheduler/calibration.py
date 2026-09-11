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
"""Initial (pre-rollout) hazard calibration sources for the online EMA.

Two sources are supported, both producing a BF16 table and a W4 base table:

* ``paired_traces``: the first ``N`` requests of a pure-BF16 and a pure-W4 baseline rollout.  Both
  start decoding at token zero, so their empirical lifetime distributions condition cleanly at any
  later frontier without assuming a switch point.
* ``from_finals``: explicit final-length arrays (used by tests and by the ``build-policy`` CLI when
  lengths were extracted elsewhere).  Entries default to zero (full-length observation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cost_model import PolicyGrid
from .hazard import HazardTable, components
from .traces import final_lengths, trace_lengths


@dataclass
class InitialCalibration:
    bf16: HazardTable
    w4: HazardTable
    metadata: dict[str, Any]


def from_finals(
    bf16_finals: np.ndarray,
    w4_finals: np.ndarray,
    grid: PolicyGrid,
    *,
    bf16_entries: np.ndarray | None = None,
    w4_entries: np.ndarray | None = None,
    kind: str = "explicit final lengths",
) -> InitialCalibration:
    bf16_finals = np.asarray(bf16_finals, dtype=np.int64)
    w4_finals = np.asarray(w4_finals, dtype=np.int64)
    bf16_entries = np.zeros(len(bf16_finals), dtype=np.int64) if bf16_entries is None else bf16_entries
    w4_entries = np.zeros(len(w4_finals), dtype=np.int64) if w4_entries is None else w4_entries
    return InitialCalibration(
        bf16=components(bf16_entries, bf16_finals, grid),
        w4=components(w4_entries, w4_finals, grid),
        metadata={"kind": kind, "bf16_requests": int(len(bf16_finals)), "w4_requests": int(len(w4_finals))},
    )


def paired_traces(bf16_trace: Path, w4_trace: Path, grid: PolicyGrid, *, requests: int = 128) -> InitialCalibration:
    """Calibrate from the first ``requests`` completed requests of paired BF16 / full-W4 baselines."""
    bf_starts, bf_finishes = trace_lengths(bf16_trace, cap=grid.cap, limit=requests)
    w4_starts, w4_finishes = trace_lengths(w4_trace, cap=grid.cap, limit=requests)
    calibration = from_finals(
        final_lengths(bf_starts, bf_finishes),
        final_lengths(w4_starts, w4_finishes),
        grid,
        kind=f"paired {requests}-request BF16/full-W4 baseline traces",
    )
    calibration.metadata.update({"bf16_trace": str(bf16_trace), "w4_trace": str(w4_trace)})
    return calibration
