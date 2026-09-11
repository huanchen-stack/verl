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
"""Gemma-4 ``gemma4_unified`` checkpoint normalization.

Gemma-4 12B/31B checkpoints were published with the pre-release
``gemma4_unified`` model type (architecture
``Gemma4UnifiedForConditionalGeneration``, nested ``text_config.model_type``
``gemma4_unified_text``).  Transformers does not register that top-level type,
but the nested text config is wire-compatible with ``Gemma4TextConfig``.

For training we only need the language decoder: the checkpoint stores it under
``model.language_model.*`` plus a few embedding-side modality tensors.
Constructing the multimodal top-level model would also allocate absent
vision/audio towers, so we build the exact text decoder instead and remap the
checkpoint keys, mirroring the vLLM text-only registry entry.  E2B/E4B
checkpoints use the plain ``gemma4`` model type and take the regular
``AutoConfig`` path.
"""

import json
import os
from typing import Any

from transformers import AutoConfig, PretrainedConfig

GEMMA4_UNIFIED_MODEL_TYPE = "gemma4_unified"
GEMMA4_LANGUAGE_MODEL_KEY_MAPPING: dict[str, str] = {r"^model\.language_model\.": "model."}


def read_raw_hf_config(local_hf_config_path: str | None) -> dict[str, Any]:
    """Return the parsed ``config.json`` under ``local_hf_config_path`` or ``{}`` if unreadable."""
    if not local_hf_config_path:
        return {}
    config_json = os.path.join(local_hf_config_path, "config.json")
    try:
        with open(config_json, encoding="utf-8") as config_file:
            raw = json.load(config_file)
    except (OSError, TypeError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def is_gemma4_unified(raw_config: dict[str, Any]) -> bool:
    return raw_config.get("model_type") == GEMMA4_UNIFIED_MODEL_TYPE


def build_gemma4_text_config(
    raw_config: dict[str, Any], attn_implementation: str
) -> tuple[PretrainedConfig, dict[str, str]]:
    """Build the ``Gemma4TextConfig`` for a ``gemma4_unified`` checkpoint.

    Returns:
        ``(config, key_mapping)`` where ``key_mapping`` is the regex mapping to
        pass to ``PreTrainedModel.from_pretrained(key_mapping=...)`` so the
        ``model.language_model.*`` checkpoint keys load into the text decoder.
    """
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    text_config = dict(raw_config["text_config"])
    text_config["model_type"] = "gemma4_text"
    text_config["architectures"] = ["Gemma4ForCausalLM"]
    text_config["attn_implementation"] = attn_implementation
    config = Gemma4TextConfig.from_dict(text_config)
    return config, dict(GEMMA4_LANGUAGE_MODEL_KEY_MAPPING)


def load_hf_config_with_model_compat(
    local_hf_config_path: str, trust_remote_code: bool, attn_implementation: str
) -> PretrainedConfig:
    """``AutoConfig.from_pretrained`` with the ``gemma4_unified`` normalization applied.

    Sets ``hf_config._verl_checkpoint_key_mapping`` when the checkpoint keys
    must be remapped at load time; all other model types are untouched.
    """
    raw_config = read_raw_hf_config(local_hf_config_path)
    if is_gemma4_unified(raw_config):
        hf_config, key_mapping = build_gemma4_text_config(raw_config, attn_implementation=attn_implementation)
        hf_config._verl_checkpoint_key_mapping = key_mapping
        return hf_config
    return AutoConfig.from_pretrained(
        local_hf_config_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
    )
