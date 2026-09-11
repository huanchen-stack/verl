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
"""CPU tests for ``rollout.precision_scheduler`` and its env-var wire format (decision 9)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.workers.config import PrecisionSchedulerConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.precision_scheduler_env import (
    ENV_BY_KEY,
    collect_forwarded_env,
    resolve_sleep_level,
    to_vllm_env,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# The allowlist archived from the dirty tree (verl/trainer/constants_ppo.py, worktree), minus knobs dropped by
# decisions 4 (async predictor, threshold / dynamic-policy env pairs), 5 (VLLM_REPREFILL_ONLY_ROLLOUT), 6 (one
# policy flag), the nsys bridge (dropped with C4), the Marlin padding knob (C9), the sleep-level / trace / namespace
# / port-range envs that became config keys or VERL_* pass-through, and TMPDIR (pass-through only).
ARCHIVED_ALLOWLIST_MINUS_DROPPED = frozenset(
    {
        "ROLLOUT_QLORA",
        "VLLM_LORA_ENABLE_DUAL_STREAM",
        "VLLM_DUAL_PRECISION_ROLLOUT",
        "VLLM_DUAL_PRECISION_INT4_MODEL",
        "VLLM_DUAL_PRECISION_BF16_LAYERS",
        "VLLM_DUAL_PRECISION_INT4_MODULES",
        "VLLM_DUAL_PRECISION_REPREFILL",
        "VLLM_DUAL_PRECISION_VALIDATE_SHADOW",
        "VLLM_DUAL_PRECISION_VALIDATE_LIFECYCLE",
    }
)
# Added by decision 6 (single policy flag), the continuous-EMA runs (reload / online observations, which the
# archived allowlist forgot), the C1 fuse-packed knob, and the tracer's two config keys.
ADDED_BY_DECISIONS = frozenset(
    {
        "VLLM_DUAL_PRECISION_POLICY",
        "VLLM_DUAL_PRECISION_RELOAD_POLICY_EACH_ROLLOUT",
        "VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS",
        "VLLM_ROLLOUT_LORA_FUSE_PACKED",
        "VERL_REQUEST_TRACE_DIR",
        "VERL_REQUEST_TRACE_LOG_TOKENS",
    }
)


def _headline_config() -> PrecisionSchedulerConfig:
    """The headline dynamic-policy cell: dual precision on, LoRA fast path, tracing with token ids."""
    return PrecisionSchedulerConfig(
        enable=True,
        lora_fast_path=True,
        int4_model="/models/qwen3.5-9b-w4",
        policy="/runs/policy.json",
        online_observations="/runs/online_switch_cohorts.jsonl",
        request_trace_dir="/runs/traces",
        request_trace_log_tokens=True,
    )


def test_defaults_are_off_and_vanilla_emits_empty_env():
    cfg = PrecisionSchedulerConfig()
    assert cfg.enable is False
    assert cfg.lora_fast_path is False
    assert cfg.lora_dual_stream is False
    assert cfg.lora_fuse_packed is True
    assert cfg.int4_model is None
    assert cfg.policy == ""
    assert cfg.bf16_layers == "first:3,last:3"
    assert cfg.int4_modules == "all"
    assert cfg.reprefill is False
    assert cfg.sleep_level is None
    assert cfg.reload_policy_each_rollout is False
    assert cfg.online_observations is None
    assert cfg.validate_shadow is False
    assert cfg.validate_lifecycle is False
    assert cfg.request_trace_dir is None
    assert cfg.request_trace_log_tokens is False
    assert to_vllm_env(cfg) == {}


def test_rollout_config_hangs_precision_scheduler_from_yaml_and_dict():
    cfg = RolloutConfig(name="vllm", precision_scheduler={"enable": True, "int4_model": "/m"})
    assert isinstance(cfg.precision_scheduler, PrecisionSchedulerConfig)
    assert cfg.precision_scheduler.int4_model == "/m"
    default = RolloutConfig(name="vllm")
    assert default.precision_scheduler == PrecisionSchedulerConfig()
    yaml_cfg = OmegaConf.load(REPO_ROOT / "verl/trainer/config/rollout/rollout.yaml")
    block = OmegaConf.to_container(yaml_cfg.precision_scheduler, resolve=True)
    assert block.pop("_target_") == "verl.workers.config.PrecisionSchedulerConfig"
    assert PrecisionSchedulerConfig(**block) == PrecisionSchedulerConfig()
    assert set(block) == set(ENV_BY_KEY) | {"sleep_level"}


def test_disabled_emits_only_lora_keys_when_set():
    cfg = PrecisionSchedulerConfig(lora_fast_path=True, int4_model="/ignored", policy="fixed_frontier:8000")
    assert to_vllm_env(cfg) == {
        "ROLLOUT_QLORA": "1",
        "VLLM_LORA_ENABLE_DUAL_STREAM": "0",
        "VLLM_ROLLOUT_LORA_FUSE_PACKED": "1",
    }


def test_headline_env_matches_archived_allowlist_golden():
    env = to_vllm_env(_headline_config())
    assert set(env) == ARCHIVED_ALLOWLIST_MINUS_DROPPED | ADDED_BY_DECISIONS
    assert env["VLLM_DUAL_PRECISION_ROLLOUT"] == "1"
    assert env["ROLLOUT_QLORA"] == "1"
    assert env["VLLM_LORA_ENABLE_DUAL_STREAM"] == "0"
    assert env["VLLM_ROLLOUT_LORA_FUSE_PACKED"] == "1"
    assert env["VLLM_DUAL_PRECISION_INT4_MODEL"] == "/models/qwen3.5-9b-w4"
    assert env["VLLM_DUAL_PRECISION_POLICY"] == "/runs/policy.json"
    assert env["VLLM_DUAL_PRECISION_BF16_LAYERS"] == "first:3,last:3"
    assert env["VLLM_DUAL_PRECISION_INT4_MODULES"] == "all"
    assert env["VLLM_DUAL_PRECISION_REPREFILL"] == "0"
    assert env["VLLM_DUAL_PRECISION_RELOAD_POLICY_EACH_ROLLOUT"] == "0"
    assert env["VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS"] == "/runs/online_switch_cohorts.jsonl"
    assert env["VLLM_DUAL_PRECISION_VALIDATE_SHADOW"] == "0"
    assert env["VLLM_DUAL_PRECISION_VALIDATE_LIFECYCLE"] == "0"
    assert env["VERL_REQUEST_TRACE_DIR"] == "/runs/traces"
    assert env["VERL_REQUEST_TRACE_LOG_TOKENS"] == "1"
    # sleep_level is not an env var: it is resolved inside the verl server actor.
    assert "sleep_level" not in ENV_BY_KEY


def test_enabled_omits_null_keys():
    env = to_vllm_env(PrecisionSchedulerConfig(enable=True))
    absent_keys = (
        "VLLM_DUAL_PRECISION_INT4_MODEL",
        "VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS",
        "VERL_REQUEST_TRACE_DIR",
    )
    for absent in absent_keys:
        assert absent not in env
    assert env["VLLM_DUAL_PRECISION_POLICY"] == ""


def test_config_md_table_matches_env_by_key():
    doc = (REPO_ROOT / "docs/precision_scheduler/config.md").read_text(encoding="utf-8")
    for key, env in ENV_BY_KEY.items():
        assert f"| `{key}` | `{env}` |" in doc, f"docs/precision_scheduler/config.md misses {key} -> {env}"


@pytest.mark.parametrize("level", [0, 3, "x"])
def test_sleep_level_validation(level):
    with pytest.raises(ValueError):
        PrecisionSchedulerConfig(sleep_level=level)


def test_enable_with_level_2_rejected():
    with pytest.raises(ValueError):
        PrecisionSchedulerConfig(enable=True, sleep_level=2)


def test_resolve_sleep_level(caplog):
    assert resolve_sleep_level(None, 2) == 2
    assert resolve_sleep_level(PrecisionSchedulerConfig(), 2) == 2
    assert resolve_sleep_level(PrecisionSchedulerConfig(sleep_level=2), 1) == 2
    assert resolve_sleep_level(PrecisionSchedulerConfig(sleep_level=1), 2) == 1
    with caplog.at_level(logging.INFO, logger="verl.workers.config.precision_scheduler"):
        assert resolve_sleep_level(PrecisionSchedulerConfig(enable=True), 2) == 1
    assert "forces vLLM sleep level 1" in caplog.text
    assert resolve_sleep_level(PrecisionSchedulerConfig(enable=True, sleep_level=1), 2) == 1


def test_collect_forwarded_env_prefixes():
    environ = {
        "VLLM_DUAL_PRECISION_NEW_KNOB": "7",
        "VERL_ZMQ_NAMESPACE": "a/b",
        "VERL_RAY_MASTER_PORT_RANGE": "35000:35199",
        "TMPDIR": "/dev/shm/x",
        "VLLM_LOGGING_LEVEL": "INFO",
        "HOME": "/root",
    }
    assert collect_forwarded_env(environ) == {
        "VLLM_DUAL_PRECISION_NEW_KNOB": "7",
        "VERL_ZMQ_NAMESPACE": "a/b",
        "VERL_RAY_MASTER_PORT_RANGE": "35000:35199",
        "TMPDIR": "/dev/shm/x",
    }


def test_get_ppo_ray_runtime_env_forwards_config_and_environ():
    environ = {k: v for k, v in os.environ.items() if not k.startswith(("VERL_", "VLLM_DUAL_PRECISION_"))}
    environ.pop("TMPDIR", None)
    environ["VERL_ZMQ_NAMESPACE"] = "dynro_bf16_g3_12345"
    environ["VLLM_DUAL_PRECISION_EXTRA"] = "x"
    with patch.dict(os.environ, environ, clear=True):
        env_vars = get_ppo_ray_runtime_env()["env_vars"]
        assert env_vars["VERL_ZMQ_NAMESPACE"] == "dynro_bf16_g3_12345"
        assert env_vars["VLLM_DUAL_PRECISION_EXTRA"] == "x"
        assert "TMPDIR" not in env_vars
        assert "VLLM_DUAL_PRECISION_ROLLOUT" not in env_vars
        # Determinism keys keep their call-time defaults.
        assert env_vars["PYTHONHASHSEED"] == "0"
        assert env_vars["VERL_FULL_DETERMINISM"] == "0"
        assert env_vars["VLLM_BATCH_INVARIANT"] == "0"

        env_vars = get_ppo_ray_runtime_env(precision_scheduler=_headline_config())["env_vars"]
        assert env_vars["VLLM_DUAL_PRECISION_ROLLOUT"] == "1"
        assert env_vars["VERL_REQUEST_TRACE_DIR"] == "/runs/traces"
        assert env_vars["VERL_ZMQ_NAMESPACE"] == "dynro_bf16_g3_12345"

        # A DictConfig block (as hydra hands it over) is accepted too.
        block = OmegaConf.create({"enable": True, "lora_fast_path": True})
        env_vars = get_ppo_ray_runtime_env(precision_scheduler=block)["env_vars"]
        assert env_vars["ROLLOUT_QLORA"] == "1"
