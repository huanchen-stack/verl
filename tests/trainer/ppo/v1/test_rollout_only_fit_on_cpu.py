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
"""CPU tests for the rollout-only harness, the initial-checkpoint gate, the master port range and stable uids.

``PPOTrainer`` is instantiated with ``object.__new__`` and every collaborator that needs Ray or
TransferQueue is replaced by a mock, so ``fit()`` can be driven end to end on CPU.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("ray")
pytest.importorskip("transfer_queue")

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer

# Marker regex used by the archived collector (.codex-report/rl-workflow/collect_best_t8_verl_rollout.py L33).
MARKER_RE = re.compile(r"VERL_ROLLOUT_ONLY_COMPLETE step=(\d+) requests=(\d+) gen_seconds=([0-9.eE+-]+)")


class _FakeBatch:
    def __init__(self, n: int):
        self.keys = [f"k{i}" for i in range(n)]
        self.partition_id = "train"
        self.extra_info = {}

    def __len__(self):
        return len(self.keys)


def _rollout_data(lengths: list[int], scores: list[float]):
    responses = torch.nested.nested_tensor([torch.zeros(n, dtype=torch.long) for n in lengths], layout=torch.jagged)
    rm_scores = torch.zeros(len(lengths), max(lengths))
    for i, score in enumerate(scores):
        rm_scores[i, lengths[i] - 1] = score
    return {"responses": responses, "rm_scores": rm_scores}


class _Trainer(PPOTrainer):
    """Concrete subclass so the abstract hooks can be replaced by mocks."""

    def on_step_end(self):
        raise AssertionError("replaced by a mock in _make_trainer")

    def on_sample_end(self):
        raise AssertionError("replaced by a mock in _make_trainer")


def _make_trainer(tmp_path, **trainer_overrides) -> PPOTrainer:
    trainer = object.__new__(_Trainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "p",
                "experiment_name": "e",
                "logger": ["console"],
                "val_before_train": False,
                "total_epochs": 1,
                "total_training_steps": 3,
                "rollout_data_dir": str(tmp_path / "rollouts"),
                "rollout_only": False,
                "rollout_only_steps": None,
                "save_initial_checkpoint": False,
                "exit_after_initial_checkpoint": False,
                "ray_master_port_range": None,
                "stable_sample_uid": False,
                "device": "cpu",
                **trainer_overrides,
            },
            "global_profiler": {"steps": None},
            "data": {"train_batch_size": 4},
            "actor_rollout_ref": {"rollout": {"temperature": 1.0, "n": 2}},
        }
    )
    trainer.global_steps = 0
    trainer.total_training_steps = 3
    trainer.train_dataloader = [None] * 10
    trainer.timing_raw = {}
    for name in (
        "on_train_begin",
        "on_train_end",
        "on_step_begin",
        "on_step_end",
        "on_validate_begin",
        "on_validate_end",
        "_start_profiling",
        "_stop_profiling",
        "_save_checkpoint",
        "_validate",
        "_log_rollout_data",
        "_shutdown_dump_executor",
        "_compute_metrics",
    ):
        setattr(trainer, name, MagicMock(name=name))
    return trainer


@pytest.fixture
def patched_module(monkeypatch):
    logger = MagicMock(name="Tracking")
    monkeypatch.setattr(trainer_base, "Tracking", MagicMock(return_value=logger))
    monkeypatch.setattr(trainer_base, "ValidationGenerationsLogger", MagicMock())
    monkeypatch.setattr(trainer_base, "tqdm", MagicMock())
    tq = SimpleNamespace(kv_batch_get=MagicMock(), kv_clear=MagicMock())
    monkeypatch.setattr(trainer_base, "tq", tq)
    return SimpleNamespace(logger=logger, tq=tq)


def test_rollout_only_single_step_exit(tmp_path, patched_module, capsys):
    trainer = _make_trainer(tmp_path, rollout_only=True, rollout_only_steps=1)
    batch = _FakeBatch(8)

    def fake_step(metrics, timing_raw):
        timing_raw["gen"] = 1.5
        return batch

    trainer.step = MagicMock(side_effect=fake_step)
    patched_module.tq.kv_batch_get.return_value = _rollout_data([3, 5, 2, 4, 1, 1, 6, 2], [1, 0, 1, 0, 0, 1, 1, 0])

    trainer.fit(agent_loop_manager=MagicMock())

    out = capsys.readouterr().out
    match = MARKER_RE.search(out)
    assert match and match.group(1) == "1" and match.group(2) == "8" and float(match.group(3)) == 1.5
    assert trainer.step.call_count == 1
    trainer._log_rollout_data.assert_called_once()
    assert trainer._log_rollout_data.call_args.args[2] == str(tmp_path / "rollouts")
    trainer._shutdown_dump_executor.assert_called_once()
    trainer.on_step_end.assert_not_called()  # skipped after the last step
    trainer._save_checkpoint.assert_not_called()
    trainer._validate.assert_not_called()
    patched_module.tq.kv_clear.assert_called_once_with(keys=batch.keys, partition_id="train")
    logged = patched_module.logger.log.call_args.kwargs
    assert logged["step"] == 1
    assert logged["data"] == {
        "timing_s/gen": 1.5,
        "rollout_only/requests": 8,
        "rollout_only/response_tokens": 24,
        "critic/rewards/mean": 0.5,
    }
    assert trainer.global_steps == 2


def test_rollout_only_multi_step_metrics(tmp_path, patched_module, capsys):
    trainer = _make_trainer(tmp_path, rollout_only=True, rollout_only_steps=None)  # -> total_training_steps = 3
    gen_times = iter([1.0, 2.0, 3.0])

    def fake_step(metrics, timing_raw):
        timing_raw["gen"] = next(gen_times)
        return _FakeBatch(4)

    trainer.step = MagicMock(side_effect=fake_step)
    patched_module.tq.kv_batch_get.side_effect = [
        _rollout_data([1, 2, 3, 4], [1, 1, 1, 1]),
        _rollout_data([5, 5, 5, 5], [0, 0, 0, 0]),
        _rollout_data([2, 2, 2, 2], [1, 0, 1, 0]),
    ]

    trainer.fit(agent_loop_manager=MagicMock())

    markers = MARKER_RE.findall(capsys.readouterr().out)
    assert [(int(s), int(r)) for s, r, _ in markers] == [(1, 4), (2, 4), (3, 4)]
    assert trainer.step.call_count == 3
    assert trainer.on_step_end.call_count == 2  # between steps, not after the last one
    assert patched_module.tq.kv_clear.call_count == 3
    assert trainer._log_rollout_data.call_count == 3
    trainer._shutdown_dump_executor.assert_called_once()
    rows = [(c.kwargs["step"], c.kwargs["data"]) for c in patched_module.logger.log.call_args_list]
    assert [step for step, _ in rows] == [1, 2, 3]
    assert [d["rollout_only/response_tokens"] for _, d in rows] == [10, 20, 8]
    assert [d["critic/rewards/mean"] for _, d in rows] == [1.0, 0.0, 0.5]
    assert [d["timing_s/gen"] for _, d in rows] == [1.0, 2.0, 3.0]
    assert all(d["rollout_only/requests"] == 4 for _, d in rows)
    assert trainer.global_steps == 4


def test_rollout_only_steps_bounded_by_total_training_steps(tmp_path, patched_module):
    trainer = _make_trainer(tmp_path, rollout_only=True, rollout_only_steps=10)
    trainer.step = MagicMock(side_effect=lambda m, t: _FakeBatch(2))
    patched_module.tq.kv_batch_get.return_value = _rollout_data([1, 1], [0, 0])
    trainer.fit(agent_loop_manager=MagicMock())
    assert trainer.step.call_count == 3


def test_rollout_only_requires_rollout_data_dir(tmp_path, patched_module):
    trainer = _make_trainer(tmp_path, rollout_only=True, rollout_data_dir=None)
    trainer.step = MagicMock()
    with pytest.raises(RuntimeError, match="rollout_data_dir"):
        trainer.fit(agent_loop_manager=MagicMock())
    trainer.step.assert_not_called()


def test_step_returns_after_reward_in_rollout_only_mode(tmp_path):
    trainer = _make_trainer(tmp_path, rollout_only=True)
    trainer.global_steps = 1
    trainer._add_batch_to_generate = MagicMock()
    trainer.on_sample_begin = MagicMock()
    trainer.on_sample_end = MagicMock()
    batch = _FakeBatch(4)
    trainer.replay_buffer = SimpleNamespace(sample=MagicMock(return_value=(batch, {})))
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    trainer._compute_reward_colocate = MagicMock(return_value=batch)
    trainer._balance_batch = MagicMock()
    trainer._compute_old_log_prob = MagicMock()
    timing_raw = {}
    out = trainer.step({}, timing_raw)
    assert out is batch
    trainer._compute_reward_colocate.assert_called_once()
    trainer._balance_batch.assert_not_called()
    trainer._compute_old_log_prob.assert_not_called()
    assert "gen" in timing_raw and "reward" in timing_raw


def test_save_initial_checkpoint_gate(tmp_path, patched_module, capsys):
    trainer = _make_trainer(tmp_path, save_initial_checkpoint=True, exit_after_initial_checkpoint=True)
    trainer.config.trainer.val_before_train = True
    trainer.step = MagicMock()
    trainer.fit(agent_loop_manager=MagicMock())
    trainer._save_checkpoint.assert_called_once()
    assert "VERL_INITIAL_CHECKPOINT_COMPLETE step=0" in capsys.readouterr().out
    trainer._validate.assert_not_called()
    trainer.step.assert_not_called()
    trainer._shutdown_dump_executor.assert_called_once()


def test_save_initial_checkpoint_requires_fresh_run(tmp_path, patched_module):
    trainer = _make_trainer(tmp_path, save_initial_checkpoint=True)
    trainer.global_steps = 5
    with pytest.raises(RuntimeError):
        trainer.fit(agent_loop_manager=MagicMock())


def test_save_initial_checkpoint_without_exit_continues(tmp_path, patched_module):
    trainer = _make_trainer(tmp_path, save_initial_checkpoint=True, rollout_only=True, rollout_only_steps=1)
    trainer.step = MagicMock(side_effect=lambda m, t: _FakeBatch(2))
    patched_module.tq.kv_batch_get.return_value = _rollout_data([1, 1], [0, 0])
    trainer.fit(agent_loop_manager=MagicMock())
    trainer._save_checkpoint.assert_called_once()
    assert trainer.step.call_count == 1


def test_master_port_range_from_config_and_env(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, ray_master_port_range="35000:35199")
    assert trainer._resolve_master_port_range() == [35000, 35199]
    trainer = _make_trainer(tmp_path)
    monkeypatch.delenv("VERL_RAY_MASTER_PORT_RANGE", raising=False)
    assert trainer._resolve_master_port_range() is None
    monkeypatch.setenv("VERL_RAY_MASTER_PORT_RANGE", "40000:40199")
    assert trainer._resolve_master_port_range() == [40000, 40199]
    trainer = _make_trainer(tmp_path, ray_master_port_range="10:5")
    with pytest.raises(ValueError):
        trainer._resolve_master_port_range()


def test_setup_passes_master_port_range_to_worker_group(tmp_path, monkeypatch):
    """Only the worker-group kwargs part of _setup is exercised: everything before it is stubbed."""
    trainer = _make_trainer(tmp_path, ray_master_port_range="35000:35199")
    trainer.config.global_profiler = OmegaConf.create({"steps": None})
    trainer.use_critic = False
    trainer.role_worker_mapping = {trainer_base.Role.ActorRollout: object}
    pool = object()
    trainer.resource_pool_manager = SimpleNamespace(
        create_resource_pool=MagicMock(),
        resource_pool_dict={"p": pool},
        get_resource_pool=MagicMock(return_value=pool),
    )
    for name in ("_init_tokenizer", "_init_dataloader", "_init_dump_executor", "_init_resource_pool_mgr"):
        setattr(trainer, name, MagicMock())
    monkeypatch.setattr(trainer_base, "RayClassWithInitArgs", MagicMock())
    monkeypatch.setattr(trainer_base, "create_colocated_worker_cls", MagicMock())
    wg = MagicMock()

    class _Stop(Exception):
        """Raised right after the worker group is constructed; the rest of _setup needs Ray."""

    wg.return_value.spawn.side_effect = _Stop
    monkeypatch.setattr(trainer_base, "RayWorkerGroup", wg)
    trainer.config.actor_rollout_ref = OmegaConf.create({})
    with pytest.raises(_Stop):
        trainer._setup()
    assert wg.call_args.kwargs["master_port_range"] == [35000, 35199]
    assert wg.call_args.kwargs["device_name"] == "cpu"


def test_stable_sample_uid(tmp_path):
    batch_dict = {
        "raw_prompt": np.array([[], [], []], dtype=object),
        "extra_info": np.array([{"index": 31}, {"index": 7}, "not-a-dict"], dtype=object),
    }
    trainer = _make_trainer(tmp_path, stable_sample_uid=True)
    assert list(trainer._make_sample_uids(batch_dict)) == ["idx-31", "idx-7", "idx-2"]
    # Without extra_info the row position is used.
    assert list(trainer._make_sample_uids({"raw_prompt": np.array([[], []], dtype=object)})) == ["idx-0", "idx-1"]
    trainer = _make_trainer(tmp_path, stable_sample_uid=False)
    uids = trainer._make_sample_uids(batch_dict)
    assert len(uids) == 3 and all(re.fullmatch(r"[0-9a-f-]{36}", u) for u in uids)
