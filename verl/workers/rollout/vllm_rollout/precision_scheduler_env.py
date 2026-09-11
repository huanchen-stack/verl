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
config block has to reach the server actor as environment variables. The implementation
lives next to the dataclass in :mod:`verl.workers.config.precision_scheduler` so that the
driver (``constants_ppo.get_ppo_ray_runtime_env``) can use it without importing the
vLLM rollout package; this module is the public entry point named by decision 9.
The resulting table is documented in ``docs/precision_scheduler/config.md``.
"""

from verl.workers.config.precision_scheduler import (
    ENV_BY_KEY,
    FORWARDED_ENV_KEYS,
    FORWARDED_ENV_PREFIXES,
    LORA_KEYS,
    collect_forwarded_env,
    resolve_sleep_level,
    to_vllm_env,
)

__all__ = [
    "ENV_BY_KEY",
    "FORWARDED_ENV_PREFIXES",
    "FORWARDED_ENV_KEYS",
    "LORA_KEYS",
    "collect_forwarded_env",
    "resolve_sleep_level",
    "to_vllm_env",
]
