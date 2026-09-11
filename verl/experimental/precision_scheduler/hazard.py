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
"""Discrete response-length hazard tables with delayed entry and an EMA update.

Every table is indexed by the frontier bins of a :class:`PolicyGrid` (``250, 500, ...``).
For bin ``i`` starting at frontier ``F[i]``:

* ``risk[i]``  = P(request is eligible and still alive at the start of the bin)
* ``event[i]`` = P(request finishes inside the bin, not by hitting the cap)
* ``observed[i]`` is set only when at least one eligible request contributes.

A request is *eligible* for a bin when it entered (switched precision) at or before the bin start,
so switch cohorts observed late in a rollout never contaminate the earlier bins.  The response cap is
treated as right-censoring, never as an end-of-sequence event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cost_model import PolicyGrid


@dataclass
class HazardTable:
    risk: np.ndarray
    event: np.ndarray
    observed: np.ndarray

    @classmethod
    def empty(cls, size: int) -> HazardTable:
        return cls(risk=np.zeros(size), event=np.zeros(size), observed=np.zeros(size, dtype=bool))

    def copy(self) -> HazardTable:
        return HazardTable(self.risk.copy(), self.event.copy(), self.observed.copy())

    def hazard(self) -> np.ndarray:
        """Per-bin exit probability ``event / risk`` (zero where nothing was at risk)."""
        hazard = np.divide(self.event, self.risk, out=np.zeros_like(self.risk), where=self.risk > 0)
        return np.clip(hazard, 0.0, 1.0)

    def to_json(self) -> dict:
        return {"risk": self.risk.tolist(), "event": self.event.tolist(), "observed": self.observed.tolist()}

    @classmethod
    def from_json(cls, payload: dict) -> HazardTable:
        return cls(
            risk=np.asarray(payload["risk"], dtype=float),
            event=np.asarray(payload["event"], dtype=float),
            observed=np.asarray(payload["observed"], dtype=bool),
        )


def components(entries: np.ndarray, finals: np.ndarray, grid: PolicyGrid) -> HazardTable:
    """Empirical risk/event fractions per bin from per-request entry tokens and final lengths."""
    entries = np.asarray(entries, dtype=np.int64)
    finals = np.minimum(np.asarray(finals, dtype=np.int64), grid.cap)
    if entries.shape != finals.shape:
        raise ValueError("entries and finals must have the same shape")
    frontiers = grid.frontiers
    table = HazardTable.empty(len(frontiers))
    for i, start in enumerate(frontiers):
        eligible = entries <= start
        denominator = int(eligible.sum())
        if not denominator:
            continue
        table.observed[i] = True
        alive = eligible & (finals >= start)
        table.risk[i] = alive.sum() / denominator
        bin_end = min(int(start) + grid.step, grid.cap)
        table.event[i] = np.sum(alive & (finals < bin_end) & (finals < grid.cap)) / denominator
    return table


def ema_update(old: HazardTable, new: HazardTable, alpha: float) -> HazardTable:
    """Blend ``new`` into ``old`` on the bins ``new`` observed; other bins are untouched."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if old.risk.shape != new.risk.shape:
        raise ValueError("hazard tables must share the frontier grid")
    mask = new.observed
    result = old.copy()
    result.risk[mask] = (1.0 - alpha) * old.risk[mask] + alpha * new.risk[mask]
    result.event[mask] = (1.0 - alpha) * old.event[mask] + alpha * new.event[mask]
    result.observed |= mask
    return result


def survival(table: HazardTable, start_index: int) -> np.ndarray:
    """P(alive at the *start* of each bin from ``start_index``) for a request alive at that frontier.

    ``S[0] = 1`` and ``S[k] = prod_{j<k} (1 - h[start_index + j])``.  Survival is evaluated at the
    bin start so that a BF16 prefix and a W4 suffix compose exactly at the switch frontier.
    """
    selected = table.hazard()[start_index:]
    if not len(selected):
        return np.empty(0, dtype=float)
    return np.concatenate(([1.0], np.cumprod(1.0 - selected[:-1])))
