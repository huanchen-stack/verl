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
"""Fail-closed guards and dispatch gating of the optional Gemma-4 dense FFPA attention path (no ffpa_attn needed)."""

import sys

import pytest
import torch

from verl.models.transformers import gemma4_ffpa, monkey_patch


@pytest.fixture(autouse=True)
def _reset_flag():
    gemma4_ffpa.set_dense_ffpa_enabled(False)
    yield
    gemma4_ffpa.set_dense_ffpa_enabled(False)


@pytest.fixture
def no_ffpa(monkeypatch):
    # Make ``import ffpa_attn`` fail deterministically even if the wheel is installed.
    monkeypatch.setitem(sys.modules, "ffpa_attn", None)
    monkeypatch.setitem(sys.modules, "ffpa_attn.functional", None)


def _qkv(batch=2, seq=6, heads=4, kv_heads=2, head_dim=512, dtype=torch.bfloat16):
    q = torch.randn(batch, seq, heads, head_dim, dtype=dtype)
    k = torch.randn(batch, seq, kv_heads, head_dim, dtype=dtype)
    v = torch.randn(batch, seq, kv_heads, head_dim, dtype=dtype)
    return q, k, v


def _right_padded(batch=2, seq=6):
    mask = torch.ones(batch, seq, dtype=torch.long)
    mask[1, 4:] = 0
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, seq).clone()
    return mask, position_ids


def _run(mask, position_ids, q=None, k=None, v=None, **kwargs):
    if q is None:
        q, k, v = _qkv()
    kwargs.setdefault("softmax_scale", 512**-0.5)
    return gemma4_ffpa.dense_ffpa_forward(q, k, v, mask, position_ids, **kwargs)


def test_valid_inputs_reach_the_lazy_import_with_a_clear_message(no_ffpa):
    mask, position_ids = _right_padded()
    with pytest.raises(ImportError, match="ffpa-attn"):
        _run(mask, position_ids)


def test_left_or_interior_padding_is_rejected():
    mask, position_ids = _right_padded()
    mask[0, 0] = 0  # left padding
    with pytest.raises(RuntimeError, match="right-padded"):
        _run(mask, position_ids)
    mask, position_ids = _right_padded()
    mask[0, 2] = 0  # interior hole
    with pytest.raises(RuntimeError, match="right-padded"):
        _run(mask, position_ids)


def test_packed_position_ids_are_rejected():
    mask, position_ids = _right_padded()
    position_ids[0, 3:] = torch.arange(3)  # reset mid-sequence => packed
    with pytest.raises(RuntimeError, match="packed"):
        _run(mask, position_ids)


def test_reset_position_ids_inside_padding_are_tolerated(no_ffpa):
    mask, position_ids = _right_padded()
    position_ids[1, 4:] = 0  # resets only where the mask is zero
    with pytest.raises(ImportError):
        _run(mask, position_ids)


@pytest.mark.parametrize(
    "bad_kwargs, message",
    [
        ({"softcap": 30.0}, "softcap"),
        ({"sliding_window": 512}, "global-attention"),
        ({"dropout": 0.1}, "dropout"),
        ({"softmax_scale": None}, "attention scale"),
    ],
)
def test_unsupported_attention_options_are_rejected(bad_kwargs, message):
    mask, position_ids = _right_padded()
    with pytest.raises(RuntimeError, match=message):
        _run(mask, position_ids, **bad_kwargs)


def test_non_causal_with_padding_is_rejected():
    mask, position_ids = _right_padded()
    with pytest.raises(RuntimeError, match="causal"):
        _run(mask, position_ids, is_causal=False)


def test_wrong_shapes_and_dtypes_are_rejected():
    mask, position_ids = _right_padded()
    q, k, v = _qkv(head_dim=256)
    with pytest.raises(RuntimeError, match="512"):
        _run(mask, position_ids, q, k, v)
    q, k, v = _qkv(dtype=torch.float32)
    with pytest.raises(RuntimeError, match="fp16/bf16"):
        _run(mask, position_ids, q, k, v)
    with pytest.raises(RuntimeError, match="padding mask"):
        _run(mask.unsqueeze(1), position_ids)
    with pytest.raises(RuntimeError, match="2-D"):
        _run(mask, position_ids.unsqueeze(0))


def test_dispatch_requires_the_knob_and_head_dim_512():
    q512 = torch.empty(1, 2, 1, 512, dtype=torch.bfloat16)
    q256 = torch.empty(1, 2, 1, 256, dtype=torch.bfloat16)
    assert not gemma4_ffpa.should_dispatch(q512)
    gemma4_ffpa.set_dense_ffpa_enabled(True)
    assert gemma4_ffpa.is_dense_ffpa_enabled()
    assert gemma4_ffpa.should_dispatch(q512)
    assert not gemma4_ffpa.should_dispatch(q256)


def test_ulysses_forward_falls_through_to_flash_attention_for_other_head_dims(monkeypatch):
    gemma4_ffpa.set_dense_ffpa_enabled(True)
    called = {}

    def fake_flash(q, k, v, mask, query_length, *args, **kwargs):
        called["shape"] = tuple(q.shape)
        return q

    monkeypatch.setattr(monkey_patch, "_flash_attention_forward", fake_flash)
    q, k, v = _qkv(head_dim=256)
    mask, position_ids = _right_padded()
    out = monkey_patch._ulysses_flash_attention_forward(q, k, v, mask, q.size(1), position_ids=position_ids)
    assert called["shape"] == tuple(q.shape)
    assert out is q


def test_ulysses_forward_dispatches_512_to_dense_ffpa(monkeypatch, no_ffpa):
    gemma4_ffpa.set_dense_ffpa_enabled(True)
    monkeypatch.setattr(monkey_patch, "_flash_attention_forward", lambda *a, **k: pytest.fail("must not run FA2"))
    q, k, v = _qkv(head_dim=512)
    mask, position_ids = _right_padded()
    with pytest.raises(ImportError, match="ffpa-attn"):
        monkey_patch._ulysses_flash_attention_forward(
            q, k, v, mask, q.size(1), position_ids=position_ids, softmax_scale=512**-0.5
        )
