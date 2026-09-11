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
"""index_select_tensor_dict on jagged nested tensors whose ragged dim is not the last (Qwen3.5 M-RoPE position_ids)."""

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu


def _build_batch():
    lengths = [5, 3, 7, 2]
    # Per-sample shape [4, L_i]: the ragged dimension is dim 1 of each sample (ragged_idx=2 in the stack).
    position_ids = [torch.arange(4 * n, dtype=torch.long).reshape(4, n) for n in lengths]
    input_ids = [torch.arange(n, dtype=torch.long) + 100 * i for i, n in enumerate(lengths)]
    batch = TensorDict(
        {
            "position_ids": tu.nested_tensor_from_tensor_list(position_ids, ragged_idx=2),
            "input_ids": tu.nested_tensor_from_tensor_list(input_ids),
            "score": torch.arange(len(lengths), dtype=torch.float32),
        },
        batch_size=len(lengths),
    )
    return batch, position_ids, input_ids


def _assert_selected(selected, indices, position_ids, input_ids):
    assert selected.batch_size == torch.Size([len(indices)])
    pos = selected["position_ids"]
    assert pos.is_nested
    assert pos._ragged_idx == 2
    for out, idx in zip(pos.unbind(), indices, strict=True):
        assert torch.equal(out, position_ids[idx])
    for out, idx in zip(selected["input_ids"].unbind(), indices, strict=True):
        assert torch.equal(out, input_ids[idx])
    assert selected["score"].tolist() == [float(i) for i in indices]


def test_index_select_3d_jagged_matches_list_indexing():
    batch, position_ids, input_ids = _build_batch()
    indices = [2, 0, 3]

    selected = tu.index_select_tensor_dict(batch, indices)

    _assert_selected(selected, indices, position_ids, input_ids)


def test_index_select_forced_padded_fallback_matches_list_indexing(monkeypatch):
    batch, position_ids, input_ids = _build_batch()
    indices = [2, 0, 3]

    original_unbind = torch.Tensor.unbind
    calls = {"failed": 0}

    def failing_unbind(self, *args, **kwargs):
        # Simulate the PyTorch jagged NestedTensor unbind failure on 3D+ tensors.
        if self.is_nested and self.dim() >= 3:
            calls["failed"] += 1
            raise RuntimeError("simulated jagged unbind failure")
        return original_unbind(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "unbind", failing_unbind)
    selected = tu.index_select_tensor_dict(batch, indices)
    monkeypatch.undo()

    assert calls["failed"] >= 1
    _assert_selected(selected, indices, position_ids, input_ids)


def test_index_select_fallback_reraises_non_nested_errors(monkeypatch):
    batch, _, _ = _build_batch()

    def broken_unbind(self, *args, **kwargs):
        raise TypeError("not the error we swallow")

    monkeypatch.setattr(torch.Tensor, "unbind", broken_unbind)
    with pytest.raises(TypeError):
        tu.index_select_tensor_dict(batch, [0])
