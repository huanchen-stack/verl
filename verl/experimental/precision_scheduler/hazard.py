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


def components(
    entries: np.ndarray, finals: np.ndarray, grid: PolicyGrid, events: np.ndarray | None = None
) -> HazardTable:
    """Empirical risk/event fractions per bin from per-request entry tokens and final lengths.

    A request is at risk in every bin from its entry to its final length. ``events`` marks which
    finals are real finishes (default: every final below the cap); a censored request -- one that
    left the observed precision at ``final`` without finishing, e.g. a BF16 request switched to W4
    there -- passes ``False`` so it drops out of the risk set without counting as an event.
    """
    entries = np.asarray(entries, dtype=np.int64)
    finals = np.minimum(np.asarray(finals, dtype=np.int64), grid.cap)
    if entries.shape != finals.shape:
        raise ValueError("entries and finals must have the same shape")
    events_arr = finals < grid.cap if events is None else (np.asarray(events, dtype=bool) & (finals < grid.cap))
    if events_arr.shape != finals.shape:
        raise ValueError("events must have the same shape as finals")
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
        table.event[i] = np.sum(alive & (finals < bin_end) & events_arr) / denominator
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


def weighted_update(
    old: HazardTable,
    new: HazardTable,
    n_new: int,
    n_seen: int,
    *,
    prior_weight: float,
    alpha_min: float = 0.0,
) -> tuple[HazardTable, float]:
    """Count-weighted blend: ``w = max(n_new / (prior_weight + n_seen + n_new), alpha_min)`` on observed bins.

    ``n_seen`` counts the online requests already blended into ``old`` (the calibration itself is
    worth ``prior_weight`` requests). The weight therefore starts near ``n_new / prior_weight`` (fast
    learning while evidence is thin) and decays toward ``alpha_min`` (a slow EMA that tracks drift);
    a 4-request cohort after 90 online requests moves a bin by ~3 %, not by a fixed 20 %.
    Returns the blended table and the weight used.
    """
    if n_new <= 0:
        return old.copy(), 0.0
    if prior_weight < 0 or n_seen < 0 or not 0.0 <= alpha_min <= 1.0:
        raise ValueError("prior_weight and n_seen must be >= 0 and alpha_min in [0, 1]")
    weight = max(float(n_new) / float(prior_weight + n_seen + n_new), float(alpha_min))
    return ema_update(old, new, min(weight, 1.0)), min(weight, 1.0)


def survival(table: HazardTable, start_index: int) -> np.ndarray:
    """P(alive at the *start* of each bin from ``start_index``) for a request alive at that frontier.

    ``S[0] = 1`` and ``S[k] = prod_{j<k} (1 - h[start_index + j])``.  Survival is evaluated at the
    bin start so that a BF16 prefix and a W4 suffix compose exactly at the switch frontier.
    """
    selected = table.hazard()[start_index:]
    if not len(selected):
        return np.empty(0, dtype=float)
    return np.concatenate(([1.0], np.cumprod(1.0 - selected[:-1])))
