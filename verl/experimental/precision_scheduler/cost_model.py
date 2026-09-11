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
"""Offline receding-horizon cost model shared by the policy builder and its tests.

The model prices a decode trajectory from frontier bin ``fi`` onwards for a rollout with ``live``
requests still running, given per-bin alive probabilities ``alive[k]``:

* ``nonempty_k = 1 - (1 - alive_k) ** live``            P(at least one request still decoding)
* ``eff_k = clip(round(live * alive_k / nonempty_k), 1, B)``  E[live | nonempty], rounded because
  TPOT is only measured at integer batch sizes
* ``tokens_k = min(STEP, CAP - F[fi + k])``
* rollout seconds  = sum_k TPOT[p](prompt + F[fi + k], eff_k) * tokens_k / 1000 * nonempty_k
* downstream seconds = slope * live * sum_k alive_k * tokens_k

Ties between plans are broken by the caller with a strict ``<``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

PRECISIONS = ("bf16", "w4")


class TpotSource(Protocol):
    def get(self, precision: str, context: float, batch: float) -> float: ...


@dataclass(frozen=True)
class PolicyGrid:
    """Frontier bins, prompt buckets and the live-batch axis of one lookup table."""

    step: int = 250
    cap: int = 16384
    batch: int = 32
    prompt_step: int = 128
    prompt_max: int = 2048

    def __post_init__(self) -> None:
        if self.step <= 0 or self.cap <= self.step or self.batch <= 0:
            raise ValueError("PolicyGrid requires step > 0, cap > step and batch > 0")
        if self.prompt_step <= 0 or self.prompt_max < 0:
            raise ValueError("PolicyGrid requires prompt_step > 0 and prompt_max >= 0")

    @property
    def frontiers(self) -> np.ndarray:
        return np.arange(self.step, self.cap, self.step, dtype=np.int64)

    @property
    def prompts(self) -> np.ndarray:
        return np.arange(0, self.prompt_max + self.prompt_step, self.prompt_step, dtype=float)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.frontiers), len(self.prompts), self.batch)

    def frontier_index(self, frontier: int) -> int:
        """Index of an on-grid frontier token count."""
        if frontier % self.step or not self.step <= frontier < self.cap:
            raise ValueError(f"frontier {frontier} is not on the {self.step}-token grid below cap {self.cap}")
        return frontier // self.step - 1


def make_tpot_cache(grid: PolicyGrid, tpot: TpotSource) -> dict[str, np.ndarray]:
    """Dense TPOT table ``[frontier, prompt bucket, live-1]`` per precision (milliseconds/token)."""
    frontiers, prompts = grid.frontiers, grid.prompts
    cache = {}
    for precision in PRECISIONS:
        values = np.empty((len(frontiers), len(prompts), grid.batch), dtype=float)
        for fi, frontier in enumerate(frontiers):
            for pi, prompt in enumerate(prompts):
                for live in range(1, grid.batch + 1):
                    values[fi, pi, live - 1] = tpot.get(precision, float(prompt + frontier), float(live))
        if not np.all(np.isfinite(values)):
            raise ValueError(f"TPOT source produced non-finite values for {precision}")
        cache[precision] = values
    return cache


def _chunks(grid: PolicyGrid, fi: int, count: int) -> np.ndarray:
    return np.minimum(grid.step, grid.cap - grid.frontiers[fi : fi + count])


def trajectory_cost(
    cache: dict[str, np.ndarray],
    grid: PolicyGrid,
    precision: str,
    fi: int,
    pi: int,
    live: int,
    alive_probability: np.ndarray,
    slope: float,
) -> float:
    """Expected seconds (rollout + downstream) of decoding bins ``fi..`` for one state."""
    alive_probability = np.asarray(alive_probability, dtype=float)
    if not len(alive_probability):
        return 0.0
    nonempty = 1.0 - np.power(1.0 - alive_probability, live)
    conditional = np.divide(live * alive_probability, nonempty, out=np.ones_like(alive_probability), where=nonempty > 0)
    effective = np.clip(np.rint(conditional).astype(int), 1, grid.batch)
    chunks = _chunks(grid, fi, len(alive_probability))
    segment = cache[precision][fi : fi + len(alive_probability), pi, :]
    tpot = np.take_along_axis(segment, effective[:, None] - 1, axis=1)[:, 0]
    rollout = float(np.sum(tpot * chunks / 1000.0 * nonempty))
    downstream = slope * live * float(np.sum(alive_probability * chunks))
    return rollout + downstream


def trajectory_cost_grid(
    cache: dict[str, np.ndarray],
    grid: PolicyGrid,
    precision: str,
    fi: int,
    alive_probability: np.ndarray,
    slope: float,
) -> np.ndarray:
    """Vectorized :func:`trajectory_cost` for every prompt bucket and initial live count ``[P, B]``."""
    alive_probability = np.asarray(alive_probability, dtype=float)
    prompt_count = len(grid.prompts)
    if not len(alive_probability):
        return np.zeros((prompt_count, grid.batch), dtype=float)
    live = np.arange(1, grid.batch + 1, dtype=float)
    nonempty = 1.0 - np.power(1.0 - alive_probability[:, None], live[None, :])
    conditional = np.divide(
        live[None, :] * alive_probability[:, None], nonempty, out=np.ones_like(nonempty), where=nonempty > 0
    )
    effective = np.clip(np.rint(conditional).astype(int), 1, grid.batch)
    chunks = _chunks(grid, fi, len(alive_probability))
    # [bin, prompt, effective-live] -> [bin, prompt, initial-live]
    tpot = np.take_along_axis(cache[precision][fi : fi + len(alive_probability)], effective[:, None, :] - 1, axis=2)
    rollout = np.sum(tpot * chunks[:, None, None] / 1000.0 * nonempty[:, None, :], axis=0)
    downstream_tokens = live * np.sum(alive_probability * chunks)
    return rollout + slope * downstream_tokens[None, :]


def plan_cost(table, cache: dict[str, np.ndarray], grid: PolicyGrid, precision: str, fi, pi, live, slope) -> float:
    """Cost of staying on ``precision`` from bin ``fi`` on (survival taken from ``table``)."""
    from .hazard import survival

    return trajectory_cost(cache, grid, precision, fi, pi, live, survival(table, fi), slope)


def switched_plan_cost(bf, w4, cache: dict[str, np.ndarray], grid: PolicyGrid, fi, future_fi, pi, live, slope) -> float:
    """Cost of BF16 until ``future_fi`` and W4 thereafter, conditioned on reaching the switch."""
    from .hazard import survival

    if future_fi < fi:
        raise ValueError("switch frontier precedes observed frontier")
    bf_alive = survival(bf, fi)
    prefix_bins = future_fi - fi
    prefix = trajectory_cost(cache, grid, "bf16", fi, pi, live, bf_alive[:prefix_bins], slope)
    reach = float(bf_alive[prefix_bins])
    w4_alive = survival(w4, future_fi) * reach
    suffix = trajectory_cost(cache, grid, "w4", future_fi, pi, live, w4_alive, slope)
    return prefix + suffix
