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
"""Translate ``rollout.precision_scheduler`` into the vLLM environment-variable wire format.

vLLM's engine reads its settings from ``vllm/envs.py`` inside its own process, so the
config block has to reach the server actor as environment variables.  This module is
the only place that knows the mapping; the resulting table is documented in
``docs/precision_scheduler/config.md``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

from verl.workers.config.precision_scheduler import PrecisionSchedulerConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ENV_BY_KEY",
    "FORWARDED_ENV_PREFIXES",
    "FORWARDED_ENV_KEYS",
    "LORA_KEYS",
    "collect_forwarded_env",
    "resolve_sleep_level",
    "to_vllm_env",
]

# Config key -> environment variable.  Frozen: analysis tooling and the vLLM side depend on the names.
ENV_BY_KEY: dict[str, str] = {
    "enable": "VLLM_DUAL_PRECISION_ROLLOUT",
    "lora_fast_path": "ROLLOUT_QLORA",
    "lora_dual_stream": "VLLM_LORA_ENABLE_DUAL_STREAM",
    "lora_fuse_packed": "VLLM_ROLLOUT_LORA_FUSE_PACKED",
    "int4_model": "VLLM_DUAL_PRECISION_INT4_MODEL",
    "policy": "VLLM_DUAL_PRECISION_POLICY",
    "bf16_layers": "VLLM_DUAL_PRECISION_BF16_LAYERS",
    "int4_modules": "VLLM_DUAL_PRECISION_INT4_MODULES",
    "reprefill": "VLLM_DUAL_PRECISION_REPREFILL",
    "reload_policy_each_rollout": "VLLM_DUAL_PRECISION_RELOAD_POLICY_EACH_ROLLOUT",
    "online_observations": "VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS",
    "validate_shadow": "VLLM_DUAL_PRECISION_VALIDATE_SHADOW",
    "validate_lifecycle": "VLLM_DUAL_PRECISION_VALIDATE_LIFECYCLE",
    "request_trace_dir": "VERL_REQUEST_TRACE_DIR",
    "request_trace_log_tokens": "VERL_REQUEST_TRACE_LOG_TOKENS",
}

# Keys emitted even when ``enable`` is false (the LoRA fast path is independent of dual precision).
LORA_KEYS: tuple[str, ...] = ("lora_fast_path", "lora_dual_stream", "lora_fuse_packed")

# Backward-compatible pass-through from the driver environment into Ray actors.
FORWARDED_ENV_PREFIXES: tuple[str, ...] = ("VLLM_DUAL_PRECISION_", "VERL_")
FORWARDED_ENV_KEYS: tuple[str, ...] = ("TMPDIR",)


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def to_vllm_env(cfg: PrecisionSchedulerConfig) -> dict[str, str]:
    """Return the environment variables encoding ``cfg``.

    Rules: booleans become ``"1"``/``"0"``; ``None`` values are omitted; when ``enable`` is
    false only the ``lora_*`` keys are emitted, so a vanilla config yields an empty dict
    (the LoRA defaults ``false, false, true`` are emitted only when one of them is set
    away from its default or ``enable`` is true).
    """
    env: dict[str, str] = {}
    if cfg.enable:
        keys = list(ENV_BY_KEY)
    else:
        defaults = PrecisionSchedulerConfig()
        if all(getattr(cfg, key) == getattr(defaults, key) for key in LORA_KEYS):
            return env
        keys = list(LORA_KEYS)
    for key in keys:
        value = getattr(cfg, key)
        if value is None:
            continue
        env[ENV_BY_KEY[key]] = _format(value)
    return env


def resolve_sleep_level(cfg: Optional[PrecisionSchedulerConfig], default: int) -> int:
    """Return the vLLM sleep level to use.

    ``cfg.sleep_level`` wins when set.  With dual precision enabled and no explicit level,
    level 1 is forced (level 2 discards the resident INT4 shadow weights) and the reason
    is logged once per call site.  Otherwise ``default`` (the upstream rule) applies.
    """
    if cfg is None:
        return default
    if cfg.sleep_level is not None:
        if cfg.enable and cfg.sleep_level != 1:
            raise ValueError(f"precision_scheduler.enable=true requires sleep_level 1 (got {cfg.sleep_level!r})")
        return int(cfg.sleep_level)
    if cfg.enable:
        logger.info(
            "precision_scheduler.enable=true forces vLLM sleep level 1 (auto rule would pick %d): "
            "level 2 would discard the resident INT4 shadow weights.",
            default,
        )
        return 1
    return default


def collect_forwarded_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Pick the ``VLLM_DUAL_PRECISION_*``, ``VERL_*`` and ``TMPDIR`` entries of ``environ``.

    Pure helper used by :func:`verl.trainer.constants_ppo.get_ppo_ray_runtime_env` for
    backward compatibility with launchers that export env vars directly.
    """
    forwarded: dict[str, str] = {}
    for key, value in environ.items():
        if key in FORWARDED_ENV_KEYS or key.startswith(FORWARDED_ENV_PREFIXES):
            forwarded[key] = value
    return forwarded
