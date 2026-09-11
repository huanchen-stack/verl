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

The rollout runner must fail closed when the watcher dies (the policy would otherwise go stale).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import InitialCalibration
from .cost_model import PolicyGrid, TpotSource, make_tpot_cache
from .hazard import HazardTable, components, ema_update
from .policy_builder import build_policy, write_policy_atomic
from .traces import cohort_observation, completed_steps, read_cohorts, trace_lengths

DEFAULT_TRACE = "traces/request_lifetimes_replica000_node000.jsonl"
DEFAULT_COHORTS = "switch_observations.jsonl"


def replay_cohorts(
    base: HazardTable,
    cohorts: list[dict[str, Any]],
    finishes: dict[str, int],
    grid: PolicyGrid,
    alpha: float,
    *,
    max_rollout_index: int | None = None,
) -> tuple[HazardTable, list[dict[str, Any]]]:
    """Re-apply switch cohorts (in file order) to ``base``; returns the table and a processing log."""
    table = base
    processed = []
    for cohort in cohorts:
        index = cohort.get("rollout_index")
        if max_rollout_index is not None and index is not None and int(index) > max_rollout_index:
            continue
        observation = cohort_observation(cohort, finishes, grid.cap)
        if observation is None:
            continue
        entries, finals = observation
        table = ema_update(table, components(entries, finals, grid), alpha)
        processed.append(
            {
                "rollout_index": index,
                "requests": int(len(entries)),
                "entry_tokens_mean": float(np.mean(entries)),
                "final_tokens_mean": float(np.mean(finals)),
                "cap_requests": int(np.sum(finals >= grid.cap)),
            }
        )
    return table, processed


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
        steps: int,
        trace_name: str = DEFAULT_TRACE,
        cohort_name: str = DEFAULT_COHORTS,
        gate_cohorts_by_completed_steps: bool = True,
        description: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.policy_path = Path(policy_path)
        self.calibration = calibration
        self.grid = grid
        self.alpha = float(alpha)
        self.slope = float(slope)
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

    def observe(self) -> tuple[int, HazardTable, list[dict[str, Any]]]:
        """Completed steps, the EMA-updated W4 table and the cohort log, without writing anything."""
        starts, finishes = trace_lengths(self.trace_path) if self.trace_path.exists() else ([], {})
        complete = completed_steps(starts, finishes, self.grid.batch)
        table, processed = replay_cohorts(
            self.calibration.w4,
            read_cohorts(self.cohort_path),
            finishes,
            self.grid,
            self.alpha,
            max_rollout_index=complete if self.gate else None,
        )
        return complete, table, processed

    def poll(self) -> dict[str, Any] | None:
        """Rebuild and publish when the completed-step count changed; returns the new state or None."""
        complete, table, processed = self.observe()
        revision = complete
        if revision == self.last_revision:
            return None
        policy, switch_states = build_policy(
            self.calibration.bf16,
            table,
            self._tpot,
            self.grid,
            self.slope,
            revision,
            self.alpha,
            description=self.description,
            cache=self._cache,
            calibration_kind=str(self.calibration.metadata.get("kind", "unknown")) + " plus online delayed-entry EMA",
            extra_calibration={"base_source": self.calibration.metadata, "updates": len(processed)},
        )
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
