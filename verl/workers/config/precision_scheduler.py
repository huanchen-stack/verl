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

from dataclasses import dataclass
from typing import Optional

from verl.base_config import BaseConfig

__all__ = ["PrecisionSchedulerConfig"]


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

    def __post_init__(self) -> None:
        if self.sleep_level is not None and self.sleep_level not in (1, 2):
            raise ValueError(f"precision_scheduler.sleep_level must be null, 1 or 2; got {self.sleep_level!r}")
        if self.enable and self.sleep_level == 2:
            raise ValueError(
                "precision_scheduler.sleep_level=2 is incompatible with precision_scheduler.enable=true: "
                "level 2 discards the INT4 shadow weights that dual precision keeps resident."
            )
