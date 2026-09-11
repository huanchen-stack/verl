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
"""Config block for the rollout precision scheduler (``actor_rollout_ref.rollout.precision_scheduler``).

The block is the single source of truth for every vLLM-side precision-scheduling
setting.  verl translates it into environment variables for the vLLM server actor
(see :mod:`verl.workers.rollout.vllm_rollout.precision_scheduler_env`); users and
recipes never set those variables by hand.  All defaults are OFF so a vanilla config
behaves exactly like upstream verl.
"""

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from verl.base_config import BaseConfig

logger = logging.getLogger(__name__)

__all__ = [
    "PrecisionSchedulerConfig",
    "ENV_BY_KEY",
    "FORWARDED_ENV_PREFIXES",
    "FORWARDED_ENV_KEYS",
    "HOST_KEYS",
    "INLINE_POLICY_PREFIXES",
    "LORA_KEYS",
    "collect_forwarded_env",
    "is_policy_path",
    "read_policy_revision",
    "resolve_sleep_level",
    "to_vllm_env",
    "wait_for_policy_revision",
]


@dataclass
class PrecisionSchedulerConfig(BaseConfig):
    """Knobs of the dual-precision (BF16 / INT4) rollout scheduler.

    Args:
        enable: Master switch for dual-precision rollout (``VLLM_DUAL_PRECISION_ROLLOUT``).
        lora_fast_path: Use the fused rollout LoRA path (``ROLLOUT_QLORA``). Independent of ``enable``.
        lora_dual_stream: Run LoRA GEMMs on an auxiliary CUDA stream (``VLLM_LORA_ENABLE_DUAL_STREAM``).
        lora_fuse_packed: Fuse packed (qkv / gate_up) LoRA GEMMs (``VLLM_ROLLOUT_LORA_FUSE_PACKED``).
        int4_model: Path to the INT4 shadow checkpoint (``VLLM_DUAL_PRECISION_INT4_MODEL``).
        policy: Switching policy spec: ``fixed_threshold:<t>``, ``fixed_frontier:<K>``, ``uniform_w4``,
            or a path to an EMA policy JSON (``VLLM_DUAL_PRECISION_POLICY``). Empty string means unset.
        bf16_layers: Layers kept in BF16 under INT4 (``VLLM_DUAL_PRECISION_BF16_LAYERS``).
        int4_modules: Module classes eligible for INT4 (``VLLM_DUAL_PRECISION_INT4_MODULES``).
        reprefill: Re-prefill survivors after a precision switch (``VLLM_DUAL_PRECISION_REPREFILL``).
        sleep_level: Forced vLLM sleep level. ``null`` means auto (forced to 1 when ``enable`` is true).
        reload_policy_each_rollout: Re-read the policy file at every rollout
            (``VLLM_DUAL_PRECISION_RELOAD_POLICY_EACH_ROLLOUT``).
        online_observations: JSONL path for online switch cohorts (``VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS``).
        validate_shadow: Validate the INT4 shadow binding at load (``VLLM_DUAL_PRECISION_VALIDATE_SHADOW``).
        validate_lifecycle: Validate request lifecycle bookkeeping (``VLLM_DUAL_PRECISION_VALIDATE_LIFECYCLE``).
        request_trace_dir: Directory for the per-request lifetime trace (``VERL_REQUEST_TRACE_DIR``).
        request_trace_log_tokens: Record sampled token ids in the trace (``VERL_REQUEST_TRACE_LOG_TOKENS``).
        zmq_namespace: Host-wide namespace of the colocated weight-transfer socket (``VERL_ZMQ_NAMESPACE``);
            lets several independent Ray clusters share one host. ``null`` uses the Ray job id.
        force_shm_weight_transfer: Force the shared-memory weight-transfer path even where CUDA IPC works
            (``VERL_FORCE_SHM_WEIGHT_TRANSFER``).
        require_policy_advance: vLLM strict mode: fail instead of warn when the policy file's revision did
            not advance at a rollout boundary (``VLLM_DUAL_PRECISION_REQUIRE_POLICY_ADVANCE``).
        policy_barrier_timeout_s: verl-only. When > 0 and ``policy`` is a file path, the trainer waits (up to
            this many seconds) before every rollout after the first until ``calibration.policy_revision`` in
            the policy file exceeds the revision seen at the previous rollout. 0 disables the barrier.
    """

    enable: bool = False
    lora_fast_path: bool = False
    lora_dual_stream: bool = False
    lora_fuse_packed: bool = True
    int4_model: Optional[str] = None
    policy: str = ""
    bf16_layers: str = "first:3,last:3"
    int4_modules: str = "all"
    reprefill: bool = False
    sleep_level: Optional[int] = None
    reload_policy_each_rollout: bool = False
    online_observations: Optional[str] = None
    validate_shadow: bool = False
    validate_lifecycle: bool = False
    request_trace_dir: Optional[str] = None
    request_trace_log_tokens: bool = False
    zmq_namespace: Optional[str] = None
    force_shm_weight_transfer: bool = False
    require_policy_advance: bool = False
    policy_barrier_timeout_s: float = 0

    def __post_init__(self) -> None:
        if self.policy_barrier_timeout_s < 0:
            raise ValueError(
                f"precision_scheduler.policy_barrier_timeout_s must be >= 0; got {self.policy_barrier_timeout_s!r}"
            )
        if self.sleep_level is not None and self.sleep_level not in (1, 2):
            raise ValueError(f"precision_scheduler.sleep_level must be null, 1 or 2; got {self.sleep_level!r}")
        if self.enable and self.sleep_level == 2:
            raise ValueError(
                "precision_scheduler.sleep_level=2 is incompatible with precision_scheduler.enable=true: "
                "level 2 discards the INT4 shadow weights that dual precision keeps resident."
            )


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
    "require_policy_advance": "VLLM_DUAL_PRECISION_REQUIRE_POLICY_ADVANCE",
    "request_trace_dir": "VERL_REQUEST_TRACE_DIR",
    "request_trace_log_tokens": "VERL_REQUEST_TRACE_LOG_TOKENS",
    "zmq_namespace": "VERL_ZMQ_NAMESPACE",
    "force_shm_weight_transfer": "VERL_FORCE_SHM_WEIGHT_TRANSFER",
}

# Keys emitted even when ``enable`` is false (the LoRA fast path is independent of dual precision).
LORA_KEYS: tuple[str, ...] = ("lora_fast_path", "lora_dual_stream", "lora_fuse_packed")
# Host-isolation keys: independent of ``enable``, emitted only when set (null / false are omitted).
HOST_KEYS: tuple[str, ...] = ("zmq_namespace", "force_shm_weight_transfer")

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
    false only the ``lora_*`` keys are emitted (and only if one of them differs from its
    default), so a vanilla config yields an empty dict. The host-isolation keys
    ``zmq_namespace`` / ``force_shm_weight_transfer`` are independent of ``enable`` and are
    emitted only when set (``null`` / ``false`` are omitted).
    """
    env: dict[str, str] = {}
    if cfg.enable:
        keys = [key for key in ENV_BY_KEY if key not in HOST_KEYS]
    else:
        defaults = PrecisionSchedulerConfig()
        keys = [] if all(getattr(cfg, k) == getattr(defaults, k) for k in LORA_KEYS) else list(LORA_KEYS)
    for key in keys:
        value = getattr(cfg, key)
        if value is None:
            continue
        env[ENV_BY_KEY[key]] = _format(value)
    if cfg.zmq_namespace:
        env[ENV_BY_KEY["zmq_namespace"]] = str(cfg.zmq_namespace)
    if cfg.force_shm_weight_transfer:
        env[ENV_BY_KEY["force_shm_weight_transfer"]] = "1"
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


# Inline policy specs of decision 6; anything else in ``policy`` is a path to an EMA policy JSON.
INLINE_POLICY_PREFIXES: tuple[str, ...] = ("fixed_threshold:", "fixed_frontier:", "uniform_w4")


def is_policy_path(policy: Optional[str]) -> bool:
    """True when ``policy`` names a policy JSON file rather than an inline spec (or nothing)."""
    if not policy:
        return False
    return not policy.strip().startswith(INLINE_POLICY_PREFIXES)


def read_policy_revision(path: str) -> int:
    """``calibration.policy_revision`` of the policy JSON at ``path`` (raises on a missing or malformed file)."""
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    calibration = payload.get("calibration") or {}
    if "policy_revision" not in calibration:
        raise ValueError(f"{path}: calibration.policy_revision missing")
    return int(calibration["policy_revision"])


def wait_for_policy_revision(path: str, last_revision: int, timeout_s: float, poll_s: float = 0.5) -> int:
    """Block until the policy file's ``calibration.policy_revision`` exceeds ``last_revision``.

    Polls every ``poll_s`` seconds; a missing or half-written file (atomic writers rename into
    place, but be tolerant) counts as "not yet". Raises ``RuntimeError`` naming the path and
    the stale revision once ``timeout_s`` elapsed. Returns the new revision.
    """
    deadline = time.monotonic() + float(timeout_s)
    while True:
        try:
            revision = read_policy_revision(path)
        except (OSError, ValueError, json.JSONDecodeError):
            revision = None
        if revision is not None and revision > last_revision:
            return revision
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"policy revision barrier timed out after {timeout_s}s: {path} still at revision "
                f"{revision if revision is not None else 'unreadable'} (last seen {last_revision}); "
                "the online policy watcher did not advance calibration.policy_revision before the next rollout"
            )
        time.sleep(poll_s)
