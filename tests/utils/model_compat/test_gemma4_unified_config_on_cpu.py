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
"""gemma4_unified -> Gemma4TextConfig normalization used by HFModelConfig for Gemma-4 12B/31B checkpoints."""

import json
import shutil
from pathlib import Path

import pytest

from verl.utils.model_compat.gemma4 import (
    GEMMA4_LANGUAGE_MODEL_KEY_MAPPING,
    build_gemma4_text_config,
    is_gemma4_unified,
    load_hf_config_with_model_compat,
    read_raw_hf_config,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def unified_dir(tmp_path_factory):
    target = tmp_path_factory.mktemp("gemma4_unified_12b")
    shutil.copy(FIXTURES / "gemma4_unified_12b_config.json", target / "config.json")
    return str(target)


@pytest.fixture
def plain_dir(tmp_path_factory):
    target = tmp_path_factory.mktemp("gemma4_e2b")
    shutil.copy(FIXTURES / "gemma4_e2b_config.json", target / "config.json")
    return str(target)


def test_read_raw_hf_config_tolerates_missing_or_broken_files(tmp_path):
    assert read_raw_hf_config(str(tmp_path)) == {}
    (tmp_path / "config.json").write_text("{not json")
    assert read_raw_hf_config(str(tmp_path)) == {}
    assert read_raw_hf_config(None) == {}


def test_is_gemma4_unified_only_for_the_unified_model_type(unified_dir, plain_dir):
    assert is_gemma4_unified(read_raw_hf_config(unified_dir))
    assert not is_gemma4_unified(read_raw_hf_config(plain_dir))
    assert not is_gemma4_unified({})


def test_build_gemma4_text_config_from_unified_checkpoint(unified_dir):
    raw = json.loads((Path(unified_dir) / "config.json").read_text())

    config, key_mapping = build_gemma4_text_config(raw, attn_implementation="sdpa")

    assert type(config).__name__ == "Gemma4TextConfig"
    assert config.model_type == "gemma4_text"
    assert config.architectures == ["Gemma4ForCausalLM"]
    assert config._attn_implementation == "sdpa"
    # Values come from the nested text_config of the real 12B checkpoint.
    assert config.hidden_size == 3840
    assert config.num_hidden_layers == 48
    assert config.intermediate_size == 15360
    assert config.global_head_dim == 512
    assert config.num_kv_shared_layers == 0
    assert key_mapping == GEMMA4_LANGUAGE_MODEL_KEY_MAPPING == {r"^model\.language_model\.": "model."}


def test_load_hf_config_with_model_compat_normalizes_unified(unified_dir):
    config = load_hf_config_with_model_compat(unified_dir, trust_remote_code=False, attn_implementation="sdpa")

    assert config.model_type == "gemma4_text"
    assert config._verl_checkpoint_key_mapping == GEMMA4_LANGUAGE_MODEL_KEY_MAPPING


def test_load_hf_config_with_model_compat_keeps_autoconfig_path_for_plain_gemma4(plain_dir):
    config = load_hf_config_with_model_compat(plain_dir, trust_remote_code=False, attn_implementation="sdpa")

    assert config.model_type == "gemma4"
    assert config.architectures == ["Gemma4ForConditionalGeneration"]
    assert getattr(config, "_verl_checkpoint_key_mapping", None) is None
