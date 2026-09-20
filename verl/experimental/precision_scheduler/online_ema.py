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
"""Online EMA watcher: rebuild the switching policy after every RL step from its switch cohort.

Protocol (one watcher process per run directory, started before the rollout with
``--initialize-only`` so revision 0 exists, then kept alive next to the trainer):

1. Poll the request-lifetime trace and count completed rollout steps (groups of ``batch`` starts
   whose requests all finished).
2. Read the switch-cohort JSONL the vLLM scheduler appends under
   ``VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS``; resolve EngineCore request ids to trace ids; ingest
   every cohort whose rollout index is at most the number of completed steps (deterministic replay).
3. Rebuild the hazard table from the initial calibration by re-applying all ingested cohorts in
   order (``ema_update`` on observed bins only), run the global search, and atomically write the
   policy JSON with ``policy_revision = completed_steps``.  The scheduler reloads it before the
   next rollout when ``reload_policy_each_rollout`` is enabled.
4. Append the state to ``online_ema_history.jsonl`` and stop once ``steps`` rollouts completed.

Update rules (``update``):

* ``"weighted"`` (default): each cohort blends into the W4 group that priced its switch (the group
  with the largest cut at or below the cohort's entry) with weight
  ``max(n / (prior_weight + n_seen + n), alpha_min)`` -- fast while evidence is thin, then decaying
  toward ``alpha_min`` (a slow EMA for drift).  A 4-request cohort can no longer flip the table.
* ``"ema"``: the legacy fixed-``alpha`` blend into the single W4 table.

Hysteresis: ``max_step_tokens`` limits how far each committed frontier may move per revision.

BF16 line (``bf16_online``): the same rule applied to the other precision. Every completed rollout is
one cohort of ``batch`` BF16 requests -- a request that finished under BF16 is an event at its length,
one that switched is censored at its entry (at risk up to there, then gone) -- and it blends into the
BF16 table with the same weighted update and prior (``bf16_prior_weight``, default the W4 prior).
Caveat that keeps it off by default: a run that switches at ``F`` on every rollout censors exactly the
requests that would inform the BF16 tail beyond ``F``, so the online line beyond ``F`` rests on the few
rollouts that switched later, a thin and biased sample (replayed on 4B it drifted away from the pure-BF16
truth). ``bf16_probe_every=K`` fixes that: every K-th revision publishes a never-switch table, so that
rollout is pure BF16 and its 32 requests are uncensored tail evidence (cost: one rollout in K without the
switch gain); it turns ``bf16_online`` on.

The rollout runner must fail closed when the watcher dies (the policy would otherwise go stale).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import InitialCalibration, W4Group, w4_group_for
from .cost_model import PolicyGrid, TpotSource, make_tpot_cache
from .hazard import HazardTable, components, ema_update, weighted_update
from .policy_builder import build_policy, decisions_array, limit_decision_step, write_policy_atomic
from .traces import cohort_observation, completed_steps, read_cohorts, resolve_request_id, trace_lengths

DEFAULT_TRACE = "traces/request_lifetimes_replica000_node000.jsonl"
DEFAULT_COHORTS = "switch_observations.jsonl"


def replay_cohorts(
    base: HazardTable | list[W4Group],
    cohorts: list[dict[str, Any]],
    finishes: dict[str, int],
    grid: PolicyGrid,
    alpha: float,
    *,
    max_rollout_index: int | None = None,
    update: str = "ema",
    prior_weight: float = 32.0,
    alpha_min: float = 0.05,
) -> tuple[HazardTable | list[W4Group], list[dict[str, Any]]]:
    """Re-apply switch cohorts (in file order) to ``base``; returns the updated table(s) and a processing log.

    ``base`` may be one W4 table (legacy) or the calibration's W4 groups; with groups each cohort is
    routed to the group whose cut is the largest at or below the cohort's median entry frontier.
    ``update="weighted"`` uses :func:`weighted_update` with per-group online request counts,
    ``"ema"`` the fixed-``alpha`` blend.
    """
    grouped = not isinstance(base, HazardTable)
    groups: list[W4Group] = [W4Group(g.cut, g.table, g.requests) for g in base] if grouped else [W4Group(0, base, 0)]
    seen = [0] * len(groups)
    processed = []
    for cohort in cohorts:
        index = cohort.get("rollout_index")
        if max_rollout_index is not None and index is not None and int(index) > max_rollout_index:
            continue
        observation = cohort_observation(cohort, finishes, grid.cap)
        if observation is None:
            continue
        entries, finals, skipped = observation
        entry = int(np.median(entries))
        gi = groups.index(w4_group_for(groups, entry)) if grouped else 0
        new = components(entries, finals, grid)
        if update == "weighted":
            table, weight = weighted_update(
                groups[gi].table, new, int(len(entries)), seen[gi], prior_weight=prior_weight, alpha_min=alpha_min
            )
        elif update == "ema":
            table, weight = ema_update(groups[gi].table, new, alpha), float(alpha)
        else:
            raise ValueError(f"unknown update rule {update!r}")
        groups[gi] = W4Group(groups[gi].cut, table, groups[gi].requests)
        seen[gi] += int(len(entries))
        processed.append(
            {
                "rollout_index": index,
                "requests": int(len(entries)),
                "skipped_requests": skipped,
                "entry_tokens_mean": float(np.mean(entries)),
                "final_tokens_mean": float(np.mean(finals)),
                "cap_requests": int(np.sum(finals >= grid.cap)),
                "group_cut": int(groups[gi].cut),
                "weight": float(weight),
                "group_seen": int(seen[gi]),
            }
        )
    return (groups if grouped else groups[0].table), processed


def replay_bf16(
    base: HazardTable,
    starts: list[dict[str, Any]],
    finishes: dict[str, int],
    cohorts: list[dict[str, Any]],
    grid: PolicyGrid,
    *,
    completed: int | None = None,
    prior_weight: float = 64.0,
    alpha_min: float = 0.05,
) -> tuple[HazardTable, dict[str, int]]:
    """BF16 line: blend each completed rollout into ``base`` with the W4 groups' weighted update.

    A rollout is a cohort of ``grid.batch`` requests that all entered at 0.  One that finished under
    BF16 is an event at its length (unless it hit the cap); one that switched is censored at its entry
    -- at risk until there, then out of the risk set without an event -- so bins beyond the switch see
    only the requests that were still BF16 there.  Same :func:`weighted_update` and prior semantics as
    :func:`replay_cohorts`; returns the table and ``{"requests", "censored"}`` counts.
    """
    entries: dict[str, int] = {}
    for cohort in cohorts:
        for row in cohort.get("requests", []):
            rid = resolve_request_id(str(row["request_id"]), finishes)
            if rid is not None:
                entries[rid] = min(entries.get(rid, 10**9), int(row["entry_output_tokens"]))
    rows = starts if completed is None else starts[: completed * grid.batch]
    table = base.copy()
    seen = 0
    censored = 0
    for offset in range(0, len(rows), grid.batch):
        group = rows[offset : offset + grid.batch]
        ids = [str(start["request_id"]) for start in group]
        if len(group) < grid.batch or any(rid not in finishes for rid in ids):
            break
        finals = np.array([min(int(finishes[rid]), grid.cap) for rid in ids], dtype=np.int64)
        switched = np.array([rid in entries and entries[rid] <= finishes[rid] for rid in ids], dtype=bool)
        exits = np.where(switched, [entries.get(rid, 0) for rid in ids], finals)
        new = components(np.zeros(len(ids), dtype=np.int64), exits, grid, events=~switched)
        table, _ = weighted_update(table, new, len(ids), seen, prior_weight=prior_weight, alpha_min=alpha_min)
        seen += len(ids)
        censored += int(switched.sum())
    return table, {"requests": int(seen), "censored": int(censored)}


class OnlineEmaWatcher:
    def __init__(
        self,
        *,
        run_dir: Path,
        policy_path: Path,
        calibration: InitialCalibration,
        tpot: TpotSource,
        grid: PolicyGrid,
        alpha: float,
        slope: float,
        w4_token_penalty: float = 0.0,
        steps: int,
        trace_name: str = DEFAULT_TRACE,
        cohort_name: str = DEFAULT_COHORTS,
        gate_cohorts_by_completed_steps: bool = True,
        description: str | None = None,
        update: str = "weighted",
        prior_weight: float = 32.0,
        alpha_min: float = 0.05,
        max_step_tokens: int = 2000,
        bf16_online: bool = False,
        bf16_prior_weight: float | None = None,
        bf16_probe_every: int = 0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.policy_path = Path(policy_path)
        self.calibration = calibration
        self.grid = grid
        self.alpha = float(alpha)
        self.slope = float(slope)
        self.w4_token_penalty = float(w4_token_penalty)
        self.steps = int(steps)
        self.trace_path = self.run_dir / trace_name
        self.cohort_path = self.run_dir / cohort_name
        self.state_path = self.run_dir / "online_ema_state.json"
        self.history_path = self.run_dir / "online_ema_history.jsonl"
        self.gate = gate_cohorts_by_completed_steps
        self.description = description
        self._cache = make_tpot_cache(grid, tpot)
        self._tpot = tpot
        self.last_revision = -1
        self.last_state: dict[str, Any] | None = None
        self.bf16_table: HazardTable = calibration.bf16
        self.update = update
        self.prior_weight = float(prior_weight)
        self.alpha_min = float(alpha_min)
        self.max_step_tokens = int(max_step_tokens)
        self._previous_decisions: np.ndarray | None = None
        self.bf16_online = bool(bf16_online) or int(bf16_probe_every) > 0
        self.bf16_prior_weight = float(prior_weight if bf16_prior_weight is None else bf16_prior_weight)
        self.bf16_probe_every = int(bf16_probe_every)
        self._bf16_stats: dict[str, int] = {}

    def observe(self) -> tuple[int, list[W4Group], list[dict[str, Any]]]:
        """Completed steps, the updated W4 groups and the cohort log, without writing anything.

        Also refreshes ``self.bf16_table`` (the online BF16 line) when ``bf16_online`` is set.
        """
        starts, finishes = trace_lengths(self.trace_path) if self.trace_path.exists() else ([], {})
        complete = completed_steps(starts, finishes, self.grid.batch)
        cohorts = read_cohorts(self.cohort_path)
        table, processed = replay_cohorts(
            self.calibration.w4_groups,
            cohorts,
            finishes,
            self.grid,
            self.alpha,
            max_rollout_index=complete if self.gate else None,
            update=self.update,
            prior_weight=self.prior_weight,
            alpha_min=self.alpha_min,
        )
        if self.bf16_online:
            self.bf16_table, self._bf16_stats = replay_bf16(
                self.calibration.bf16,
                starts,
                finishes,
                [c for c in cohorts if not self.gate or int(c.get("rollout_index", 0)) <= complete],
                self.grid,
                completed=complete if self.gate else None,
                prior_weight=self.bf16_prior_weight,
                alpha_min=self.alpha_min,
            )
        else:
            self.bf16_table = self.calibration.bf16
        return complete, table, processed

    def poll(self) -> dict[str, Any] | None:
        """Rebuild and publish when the completed-step count changed; returns the new state or None."""
        complete, table, processed = self.observe()
        revision = complete
        if revision == self.last_revision:
            return None
        policy, switch_states = build_policy(
            self.bf16_table,
            table,
            self._tpot,
            self.grid,
            self.slope,
            revision,
            self.alpha,
            description=self.description,
            cache=self._cache,
            calibration_kind=str(self.calibration.metadata.get("kind", "unknown"))
            + f" plus online delayed-entry {self.update} update",
            extra_calibration={
                "base_source": self.calibration.metadata,
                "updates": len(processed),
                "update_rule": self.update,
                "prior_weight": self.prior_weight,
                "alpha_min": self.alpha_min,
                "max_step_tokens": self.max_step_tokens,
                "bf16_online": self.bf16_online,
                "bf16_prior_weight": self.bf16_prior_weight,
                "bf16_probe_every": self.bf16_probe_every,
                "bf16_online_requests": self._bf16_stats,
                "w4_groups": [{"cut": g.cut, "requests": g.requests} for g in self.calibration.w4_groups],
            },
            w4_token_penalty=self.w4_token_penalty,
        )
        decisions = decisions_array(policy)
        limited = limit_decision_step(self._previous_decisions, decisions, self.max_step_tokens)
        self._previous_decisions = limited
        probe = self.bf16_probe_every > 0 and revision > 0 and revision % self.bf16_probe_every == 0
        if probe:
            # BF16 probe: the next rollout never switches, so every request is an uncensored BF16 observation
            # (the only way the run can learn the BF16 tail beyond its own switch point). The limited table is
            # kept as the hysteresis anchor and published again on the following revision.
            limited = np.zeros_like(limited)
        policy["lookup_table"]["committed_frontiers"] = limited.reshape(-1).tolist()
        policy["calibration"]["bf16_probe"] = bool(probe)
        write_policy_atomic(self.policy_path, policy)
        state = {
            "completed_steps": complete,
            "policy_revision": revision,
            "ema_updates": len(processed),
            "switch_states": switch_states,
            "alpha": self.alpha,
            "downstream_seconds_per_token": self.slope,
            "processed_observations": processed,
            "policy_path": str(self.policy_path),
            "updated_at": time.time(),
        }
        write_policy_atomic(self.state_path, state)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a") as stream:
            stream.write(json.dumps(state) + "\n")
        self.last_revision = revision
        self.last_state = state
        return state

    def run(self, *, initialize_only: bool = False, poll_interval: float = 0.25, quiet: bool = False) -> dict[str, Any]:
        while True:
            state = self.poll()
            if state is not None and not quiet:
                print(json.dumps({k: v for k, v in state.items() if k != "processed_observations"}), flush=True)
            complete = self.last_state["completed_steps"] if self.last_state else 0
            if initialize_only or complete >= self.steps:
                assert self.last_state is not None
                return self.last_state
            time.sleep(poll_interval)
