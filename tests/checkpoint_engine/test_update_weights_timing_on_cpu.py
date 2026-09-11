# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""update_weights sub-timings surface through CheckpointEngineManager.last_update_timing (no return-type change)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("ray")

from verl.checkpoint_engine import base as ckpt_base
from verl.checkpoint_engine.base import CheckpointEngineManager, reduce_timing_dicts
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


def test_reduce_timing_dicts_max_merges_and_ignores_non_dicts():
    ranks = [
        {"update_weights_materialize": 1.0, "update_weights_load_merged": 2.5},
        None,
        {"update_weights_materialize": 1.5, "update_weights_load_merged": 2.0},
        "garbage",
    ]
    assert reduce_timing_dicts(ranks) == {"update_weights_materialize": 1.5, "update_weights_load_merged": 2.5}
    assert reduce_timing_dicts({"a": 1}) == {"a": 1.0}
    assert reduce_timing_dicts(None) == {}
    assert reduce_timing_dicts([None, None]) == {}


def test_manager_naive_path_exposes_last_update_timing(monkeypatch):
    manager = object.__new__(CheckpointEngineManager)
    manager.backend = "naive"
    manager._last_update_timing = {}
    manager.actor_wg = SimpleNamespace(update_weights=MagicMock(return_value="refs"))
    monkeypatch.setattr(
        ckpt_base.ray, "get", lambda refs: [{"update_weights_materialize": 0.5}, {"update_weights_materialize": 0.7}]
    )
    assert manager.update_weights(3) is None  # return type unchanged
    manager.actor_wg.update_weights.assert_called_once_with(global_steps=3, mode="naive")
    assert manager.last_update_timing == {"update_weights_materialize": 0.7}
    # The property returns a copy.
    manager.last_update_timing["x"] = 1.0
    assert "x" not in manager.last_update_timing


def test_trainer_sync_merges_sub_timings_into_timing_raw():
    trainer = object.__new__(PPOTrainerSync)
    trainer.global_steps = 2
    trainer.timing_raw = {}
    trainer.checkpoint_manager = SimpleNamespace(
        update_weights=MagicMock(),
        last_update_timing={"update_weights_materialize": 1.25, "update_weights_load_merged": 0.75},
    )
    trainer.on_step_end()
    assert "update_weights" in trainer.timing_raw
    assert trainer.timing_raw["update_weights_materialize"] == 1.25
    assert trainer.timing_raw["update_weights_load_merged"] == 0.75

    # Managers without the property (custom checkpoint_manager_class) and empty timings are fine.
    trainer.timing_raw = {}
    trainer.checkpoint_manager = SimpleNamespace(update_weights=MagicMock())
    trainer.on_step_end()
    assert set(trainer.timing_raw) == {"update_weights"}
