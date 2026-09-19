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

Three sources are supported, all producing a BF16 table and one or more W4 *groups*:

* ``paired_traces``: the first ``N`` requests of a pure-BF16 and a pure-W4 baseline rollout.  Both
  start decoding at token zero, so their empirical lifetime distributions condition cleanly at any
  later frontier without assuming a switch point.  One W4 group with cut 0.
* ``grouped_traces``: the BF16 baseline plus a *tail-W4 continuation* trace (``calib-tail-w4``):
  every BF16 request still alive at a cut frontier ``F_k`` was continued under W4 from its own
  BF16 prefix.  One W4 group per cut, describing what a request switched at ``F_k`` does next.
  The uniform-W4 baseline is the special case of a single cut at 0.
* ``from_finals``: explicit final-length arrays (used by tests and by the ``build-policy`` CLI when
  lengths were extracted elsewhere).  Entries default to zero (full-length observation).

A group's table is a hazard table over the whole grid whose bins below the cut carry no
observation; :func:`w4_group_for` picks the group that prices a candidate switch frontier (the
largest cut at or below it) and :func:`w4_table_for` returns that group's table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cost_model import PolicyGrid
from .hazard import HazardTable, components
from .traces import continuation_lengths, final_lengths, trace_lengths


@dataclass
class W4Group:
    """After-switch (tail-W4) hazard table for requests that entered W4 at ``cut`` tokens."""

    cut: int
    table: HazardTable
    requests: int


@dataclass
class InitialCalibration:
    bf16: HazardTable
    w4: HazardTable
    metadata: dict[str, Any]
    w4_groups: list[W4Group] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.w4_groups:
            self.w4_groups = [W4Group(cut=0, table=self.w4, requests=int(self.metadata.get("w4_requests", 0)))]
        self.w4_groups.sort(key=lambda g: g.cut)

    @property
    def cuts(self) -> list[int]:
        return [g.cut for g in self.w4_groups]


def w4_group_for(groups: list[W4Group], frontier: int, *, fallback_lowest: bool = True) -> W4Group:
    """The group pricing a switch at ``frontier``: the largest cut <= frontier.

    Below the first cut there is no after-switch data; with ``fallback_lowest`` the first group is
    used (its conditional from its own cut), otherwise ``ValueError``.
    """
    if not groups:
        raise ValueError("no W4 groups")
    chosen = None
    for group in groups:  # sorted ascending by cut
        if group.cut <= frontier:
            chosen = group
    if chosen is None:
        if not fallback_lowest:
            raise ValueError(f"frontier {frontier} precedes the first cut {groups[0].cut}")
        chosen = groups[0]
    return chosen


def w4_table_for(groups: list[W4Group], frontier: int) -> HazardTable:
    return w4_group_for(groups, frontier).table


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


def grouped_traces(
    bf16_trace: Path,
    continuation_trace: Path,
    grid: PolicyGrid,
    *,
    requests: int = 128,
    include_uniform_w4: Path | None = None,
) -> InitialCalibration:
    """Calibrate from a BF16 baseline plus a tail-W4 continuation trace (one group per cut).

    ``continuation_trace`` rows carry ``cut_frontier`` (tokens generated under BF16 before the
    switch) and ``generation_tokens`` (tokens generated under W4 after it); a request's final
    length is their sum, capped.  Each group's table is built with delayed entry at its cut, so
    bins below the cut stay unobserved.  ``include_uniform_w4`` adds the pure-W4 baseline as the
    cut-0 group (the legacy prior for switches below the first cut).
    """
    bf_starts, bf_finishes = trace_lengths(bf16_trace, cap=grid.cap, limit=requests)
    bf16 = components(np.zeros(len(bf_starts), dtype=np.int64), final_lengths(bf_starts, bf_finishes), grid)
    groups: list[W4Group] = []
    by_cut: dict[int, list[int]] = {}
    for cut, final in continuation_lengths(continuation_trace, cap=grid.cap):
        by_cut.setdefault(int(cut), []).append(int(final))
    for cut in sorted(by_cut):
        finals = np.asarray(by_cut[cut], dtype=np.int64)
        entries = np.full(len(finals), cut, dtype=np.int64)
        groups.append(W4Group(cut=cut, table=components(entries, finals, grid), requests=int(len(finals))))
    if include_uniform_w4 is not None:
        w4_starts, w4_finishes = trace_lengths(include_uniform_w4, cap=grid.cap, limit=requests)
        finals = final_lengths(w4_starts, w4_finishes)
        groups.append(W4Group(cut=0, table=components(np.zeros(len(finals), dtype=np.int64), finals, grid), requests=int(len(finals))))
    if not groups:
        raise ValueError(f"no continuation observations in {continuation_trace}")
    groups.sort(key=lambda g: g.cut)
    calibration = InitialCalibration(
        bf16=bf16,
        w4=groups[0].table,
        metadata={
            "kind": f"{requests}-request BF16 baseline plus tail-W4 continuations at cuts {[g.cut for g in groups]}",
            "bf16_requests": int(len(bf_starts)),
            "w4_requests": int(sum(g.requests for g in groups)),
            "bf16_trace": str(bf16_trace),
            "w4_trace": str(continuation_trace),
            "w4_groups": [{"cut": g.cut, "requests": g.requests} for g in groups],
        },
        w4_groups=groups,
    )
    return calibration
