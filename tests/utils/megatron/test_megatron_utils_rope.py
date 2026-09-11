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
"""Tests for Hugging Face RoPE configuration and the HybridStack PEFT hook used by Megatron model construction."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("megatron.core")

from verl.utils.megatron_utils import (  # noqa: E402
    enable_peft_recompute_input_grads_for_hybrid_stack,
    get_hf_rope_theta,
    set_hf_rope_theta_if_required,
)


def test_get_hf_rope_theta_from_nested_rope_parameters():
    config = SimpleNamespace(rope_parameters={"full_attention": {"rope_theta": 1_000_000.0}})

    assert get_hf_rope_theta(config) == 1_000_000.0


def test_set_hf_rope_theta_for_rope_model():
    config = SimpleNamespace(rope_parameters={"rope_theta": 10_000.0})
    provider = SimpleNamespace(position_embedding_type="rope")

    set_hf_rope_theta_if_required(config, provider)

    assert config.rope_theta == 10_000.0


def test_skip_hf_rope_theta_for_provider_without_position_embeddings():
    config = SimpleNamespace()
    provider = SimpleNamespace(position_embedding_type="none")

    set_hf_rope_theta_if_required(config, provider)

    assert not hasattr(config, "rope_theta")


def test_missing_rope_theta_still_fails_without_no_position_embedding_provider():
    config = SimpleNamespace()

    with pytest.raises(AttributeError, match="cannot determine RoPE base"):
        set_hf_rope_theta_if_required(config)


def _make_hybrid_stack(recompute_granularity):
    hybrid_block = pytest.importorskip("megatron.core.models.hybrid.hybrid_block")
    stack = object.__new__(hybrid_block.HybridStack)
    nn.Module.__init__(stack)
    stack.config = SimpleNamespace(recompute_granularity=recompute_granularity)
    stack.forward = lambda hidden_states: hidden_states * 2
    root = nn.Module()
    root.stack = stack
    return root, stack


def test_enable_peft_recompute_input_grads_for_hybrid_stack():
    root, stack = _make_hybrid_stack("full")

    enable_peft_recompute_input_grads_for_hybrid_stack(root)
    output = stack(torch.ones(2))

    assert output.requires_grad
    assert stack._verl_peft_recompute_input_grad_patched


def test_hybrid_stack_hook_is_idempotent():
    root, stack = _make_hybrid_stack("full")

    enable_peft_recompute_input_grads_for_hybrid_stack(root)
    first_forward = stack.forward
    enable_peft_recompute_input_grads_for_hybrid_stack(root)

    assert stack.forward is first_forward


def test_hybrid_stack_hook_leaves_non_full_recompute_untouched():
    root, stack = _make_hybrid_stack("selective")
    original_forward = stack.forward

    enable_peft_recompute_input_grads_for_hybrid_stack(root)
    output = stack(torch.ones(2))

    assert stack.forward is original_forward
    assert not output.requires_grad
    assert not getattr(stack, "_verl_peft_recompute_input_grad_patched", False)


def test_hybrid_stack_hook_does_not_detach_inputs_that_already_require_grad():
    root, stack = _make_hybrid_stack("full")
    enable_peft_recompute_input_grads_for_hybrid_stack(root)

    hidden = torch.ones(2, requires_grad=True)
    output = stack(hidden)
    output.sum().backward()

    # The original leaf keeps its autograd edge: no detach happened.
    assert hidden.grad is not None
    assert torch.equal(hidden.grad, torch.full((2,), 2.0))


def test_hook_is_noop_without_hybrid_stack_modules():
    root = nn.Linear(2, 2)

    assert enable_peft_recompute_input_grads_for_hybrid_stack(root) is root
