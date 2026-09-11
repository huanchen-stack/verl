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
"""Padded-path chunked entropy and singleton zero-copy logits helpers of the FSDP engine."""

import pytest
import torch

import verl.utils.torch_functional as verl_F
from verl.workers.engine.fsdp.transformer_impl import cat_unbound_jagged, entropy_from_padded_logits


@pytest.mark.parametrize("checkpointing", [False, True])
def test_padded_chunked_entropy_matches_unchunked(checkpointing):
    torch.manual_seed(0)
    logits = torch.randn(3, 17, 257, dtype=torch.float32, requires_grad=True)
    reference = verl_F.entropy_from_logits(logits)

    entropy = entropy_from_padded_logits(
        logits,
        entropy_fn=verl_F.entropy_from_logits_with_chunking,
        with_chunking=True,
        chunk_size=8,
        checkpointing=checkpointing,
    )

    assert entropy.shape == (3, 17)
    torch.testing.assert_close(entropy, reference, rtol=1e-5, atol=1e-5)
    # Gradients flow through both the plain and the checkpointed variant.
    entropy.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_padded_unchunked_entropy_uses_plain_helper():
    torch.manual_seed(1)
    logits = torch.randn(2, 5, 33)
    calls = []

    def entropy_fn(x, **kwargs):
        calls.append((tuple(x.shape), kwargs))
        return verl_F.entropy_from_logits(x)

    entropy = entropy_from_padded_logits(logits, entropy_fn=entropy_fn, with_chunking=False, chunk_size=8)

    # Without chunking the [B, S, V] tensor is passed through untouched (no reshape, no chunk_size).
    assert calls == [((2, 5, 33), {})]
    torch.testing.assert_close(entropy, verl_F.entropy_from_logits(logits))


def test_chunked_entropy_receives_flat_tokens_and_chunk_size():
    logits = torch.randn(2, 5, 33)
    calls = []

    def entropy_fn(x, chunk_size):
        calls.append((tuple(x.shape), chunk_size))
        return verl_F.entropy_from_logits_with_chunking(x, chunk_size=chunk_size)

    entropy = entropy_from_padded_logits(logits, entropy_fn=entropy_fn, with_chunking=True, chunk_size=4)

    assert calls == [((10, 33), 4)]
    assert entropy.shape == (2, 5)


def _jagged_logits(seq_lengths, vocab):
    max_len = max(seq_lengths)
    padded = torch.randn(len(seq_lengths), max_len, vocab)
    starts = torch.zeros(len(seq_lengths), dtype=torch.int64)
    jagged = torch.nested.narrow(padded, 1, starts, torch.tensor(seq_lengths), layout=torch.jagged)
    return padded, jagged


def test_singleton_jagged_logits_are_returned_without_copy():
    padded, jagged = _jagged_logits([7], vocab=11)

    unbound = jagged.unbind()
    flat = cat_unbound_jagged(unbound)

    assert flat.shape == (7, 11)
    assert flat.data_ptr() == unbound[0].data_ptr()
    torch.testing.assert_close(flat, padded[0, :7])
    torch.testing.assert_close(flat, torch.cat([unbound[0]]))


def test_multi_sample_jagged_logits_are_concatenated():
    padded, jagged = _jagged_logits([3, 5], vocab=11)

    flat = cat_unbound_jagged(jagged.unbind())

    assert flat.shape == (8, 11)
    torch.testing.assert_close(flat, torch.cat([padded[0, :3], padded[1, :5]]))
