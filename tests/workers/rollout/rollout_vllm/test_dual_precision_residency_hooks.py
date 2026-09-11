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
"""CPU tests for the verl side of dual-precision residency (C2 hooks)."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("ray")
pytest.importorskip("vllm")

from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.vllm_rollout import vllm_async_server
from verl.workers.rollout.vllm_rollout.utils import (
    DUAL_PRECISION_SHADOW_MODULE_NAME,
    _hide_dual_precision_shadow_model,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import _resolve_rollout_model_path

# --------------------------------------------------------------------------- #
# (a) hide helper                                                               #
# --------------------------------------------------------------------------- #


def test_shadow_module_name_matches_vllm_contract():
    from vllm.model_executor.dual_precision import SHADOW_MODULE_NAME

    assert DUAL_PRECISION_SHADOW_MODULE_NAME == SHADOW_MODULE_NAME == "_vllm_dual_precision_int4_model"


def test_hide_helper_pops_shadow_during_context_and_restores_it():
    model = torch.nn.Module()
    model.add_module("layers", torch.nn.Module())
    shadow = torch.nn.Module()
    model.add_module(DUAL_PRECISION_SHADOW_MODULE_NAME, shadow)
    other_before = dict(model._modules)
    other_before.pop(DUAL_PRECISION_SHADOW_MODULE_NAME)

    with _hide_dual_precision_shadow_model(model):
        assert DUAL_PRECISION_SHADOW_MODULE_NAME not in model._modules
        assert [n for n, _ in model.named_modules()] == ["", "layers"]
        # Everything else stays in place, by identity.
        assert {k: id(v) for k, v in model._modules.items()} == {k: id(v) for k, v in other_before.items()}
    assert model._modules[DUAL_PRECISION_SHADOW_MODULE_NAME] is shadow


def test_hide_helper_restores_shadow_when_body_raises():
    model = torch.nn.Module()
    shadow = torch.nn.Module()
    model.add_module(DUAL_PRECISION_SHADOW_MODULE_NAME, shadow)
    with pytest.raises(RuntimeError):
        with _hide_dual_precision_shadow_model(model):
            raise RuntimeError("repack failed")
    assert model._modules[DUAL_PRECISION_SHADOW_MODULE_NAME] is shadow


def test_hide_helper_is_noop_without_shadow():
    model = torch.nn.Module()
    model.add_module("layers", torch.nn.Module())
    before = dict(model._modules)
    with _hide_dual_precision_shadow_model(model):
        assert dict(model._modules) == before
    assert dict(model._modules) == before
    with _hide_dual_precision_shadow_model(object()):
        pass


def test_weight_sync_repack_never_sees_the_shadow(monkeypatch):
    """The real call site: process_weights_after_loading runs under the helper."""
    seen = []

    def fake_process(model, model_config, device):
        seen.append(list(model._modules))

    import vllm.model_executor.model_loader.utils as loader_utils

    monkeypatch.setattr(loader_utils, "process_weights_after_loading", fake_process)
    model = torch.nn.Module()
    model.add_module("layers", torch.nn.Module())
    model.add_module(DUAL_PRECISION_SHADOW_MODULE_NAME, torch.nn.Module())
    with _hide_dual_precision_shadow_model(model):
        loader_utils.process_weights_after_loading(model, None, "cpu")
    assert seen == [["layers"]]
    assert DUAL_PRECISION_SHADOW_MODULE_NAME in model._modules


# --------------------------------------------------------------------------- #
# (b) rollout.model_path                                                        #
# --------------------------------------------------------------------------- #


def test_resolve_rollout_model_path_defaults_to_actor_local_path(monkeypatch):
    calls = []
    monkeypatch.setattr(vllm_async_server, "copy_to_local", lambda p, use_shm: calls.append((p, use_shm)) or "/local/x")
    model_config = SimpleNamespace(local_path="/local/actor", use_shm=True)

    assert _resolve_rollout_model_path(SimpleNamespace(model_path=None), model_config) == "/local/actor"
    assert calls == []
    assert _resolve_rollout_model_path(SimpleNamespace(model_path="hf://int4"), model_config) == "/local/x"
    assert calls == [("hf://int4", True)]


def test_rollout_config_accepts_model_path():
    assert RolloutConfig(name="vllm", model_path="/ckpt/int4").model_path == "/ckpt/int4"
    assert RolloutConfig(name="vllm").model_path is None


# --------------------------------------------------------------------------- #
# (c) sleep level (resolved by C8's precision_scheduler.resolve_sleep_level)     #
# --------------------------------------------------------------------------- #


class _RecordingEngine:
    def __init__(self):
        self.levels = []

    async def sleep(self, level: int):
        self.levels.append(level)

    async def reset_encoder_cache(self):
        pass


def _server(sleep_level, lora_rank, mode):
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(
        mtp=None,
        precision_scheduler=SimpleNamespace(enable=False, sleep_level=sleep_level),
        free_cache_engine=True,
    )
    server.model_config = SimpleNamespace(lora_rank=lora_rank, lora={})
    server.engine = _RecordingEngine()
    server.node_rank = 0
    server.rollout_mode = mode
    return server


@pytest.mark.parametrize(
    ("sleep_level", "lora_rank", "expected"),
    [(None, 0, 2), (None, 16, 1), (1, 0, 1), (2, 16, 2)],
)
def test_hybrid_sleep_honors_config_over_lora_heuristic(monkeypatch, sleep_level, lora_rank, expected):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)
    server = _server(sleep_level, lora_rank, vllm_async_server.RolloutMode.HYBRID)
    asyncio.run(server.sleep())
    assert server.engine.levels == [expected]


@pytest.mark.parametrize(("sleep_level", "expected"), [(None, 1), (1, 1), (2, 2)])
def test_colocated_sleep_honors_config(monkeypatch, sleep_level, expected):
    server = _server(sleep_level, 0, vllm_async_server.RolloutMode.COLOCATED)
    asyncio.run(server.sleep())
    assert server.engine.levels == [expected]


def test_dual_precision_forces_sleep_level_one_at_both_sites(monkeypatch):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)
    for mode in (vllm_async_server.RolloutMode.HYBRID, vllm_async_server.RolloutMode.COLOCATED):
        server = _server(None, 0, mode)
        server.config.precision_scheduler.enable = True
        asyncio.run(server.sleep())
        assert server.engine.levels == [1]


# --------------------------------------------------------------------------- #
# (d) empty cache before resume                                                 #
# --------------------------------------------------------------------------- #


def test_update_weights_empties_cache_before_resuming_weights(monkeypatch):
    from verl.workers import engine_workers

    events = []
    monkeypatch.setattr(engine_workers, "aggressive_empty_cache", lambda force_sync=False: events.append("empty_cache"))
    monkeypatch.setattr(engine_workers, "set_expandable_segments", lambda enable: None)
    monkeypatch.setattr(engine_workers, "log_gpu_memory_usage", lambda *a, **k: None)

    class _Rollout:
        sleep_level = None

        async def resume(self, tags):
            events.append(("resume", tuple(tags)))

        async def update_weights(self, params, **kwargs):
            events.append("update_weights")

    class _Engine:
        is_param_offload_enabled = False

        def get_per_tensor_param(self, **kwargs):
            return iter(()), None

    fake = SimpleNamespace(
        config=SimpleNamespace(
            rollout=SimpleNamespace(free_cache_engine=True, checkpoint_engine=SimpleNamespace(backend="naive"))
        ),
        actor=SimpleNamespace(engine=_Engine()),
        rollout=_Rollout(),
        layered_summon=False,
        peft_merge=False,
        base_sync_done=False,
    )
    asyncio.run(engine_workers.ActorRolloutRefWorker.update_weights(fake, global_steps=1, mode="naive"))

    assert events[:2] == ["empty_cache", ("resume", ("weights",))]
    assert events[-1] == ("resume", ("kv_cache",))
