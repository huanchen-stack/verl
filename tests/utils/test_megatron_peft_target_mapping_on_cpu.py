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
"""Megatron -> HF LoRA target-module mapping, including the Qwen3.5 GatedDeltaNet projections."""

import pytest
import torch

from verl.utils.megatron_peft_utils import (
    STACKED_PARAMS,
    add_base_layer_suffix,
    convert_megatron_to_hf_target_modules,
)

QWEN35_MEGATRON_TARGETS = [
    "language_model.decoder.layers.*.self_attention.linear_qkv",
    "language_model.decoder.layers.*.self_attention.linear_proj",
    "language_model.decoder.layers.*.self_attention.in_proj",
    "language_model.decoder.layers.*.self_attention.out_proj",
    "language_model.decoder.layers.*.mlp.linear_fc1",
    "language_model.decoder.layers.*.mlp.linear_fc2",
]


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_qwen35_gated_delta_net_targets_expand_to_hf_names(model_type):
    hf_targets = convert_megatron_to_hf_target_modules(QWEN35_MEGATRON_TARGETS, model_type=model_type)

    assert hf_targets == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_nemotron_h_mamba_projections_keep_identity_names():
    # Nemotron-H Mamba mixers use the HF names in_proj / out_proj verbatim.
    targets = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]

    hf_targets = convert_megatron_to_hf_target_modules(targets, model_type="nemotron_h")

    assert hf_targets == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj",
        "out_proj",
    ]


def test_default_model_type_does_not_expand_in_proj():
    # Without an architecture hint the mapping must keep the vanilla behaviour.
    assert convert_megatron_to_hf_target_modules(["in_proj", "out_proj"]) == ["in_proj", "out_proj"]


def test_dotted_wildcard_targets_fall_back_to_suffix():
    assert convert_megatron_to_hf_target_modules(["decoder.layers.*.mlp.linear_fc1"]) == ["gate_proj", "up_proj"]
    # Unknown suffixes pass through unchanged.
    assert convert_megatron_to_hf_target_modules(["decoder.layers.*.mlp.something"]) == [
        "decoder.layers.*.mlp.something"
    ]


def test_duplicates_are_removed_preserving_order():
    converted = convert_megatron_to_hf_target_modules(["linear_qkv", "linear_q", "linear_qkv"])
    assert converted == ["q_proj", "k_proj", "v_proj"]


def test_stacked_params_include_gated_delta_net_projections():
    expected = (".in_proj_qkv.weight", ".in_proj_z.weight", ".in_proj_b.weight", ".in_proj_a.weight")
    expected += (".out_proj.weight",)
    for suffix in expected:
        assert suffix in STACKED_PARAMS


def test_add_base_layer_suffix_for_gated_delta_net_weights():
    params = [
        ("model.layers.0.linear_attn.in_proj_qkv.weight", torch.zeros(1)),
        ("model.layers.0.linear_attn.out_proj.weight", torch.zeros(1)),
        ("model.layers.0.linear_attn.norm.weight", torch.zeros(1)),
    ]

    names = [name for name, _ in add_base_layer_suffix(iter(params), model_type="qwen3_5")]

    assert names == [
        "model.layers.0.linear_attn.in_proj_qkv.base_layer.weight",
        "model.layers.0.linear_attn.out_proj.base_layer.weight",
        "model.layers.0.linear_attn.norm.weight",
    ]
