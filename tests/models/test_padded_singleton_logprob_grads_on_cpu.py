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
"""The padded no-padding branch must not use the in-place cross-entropy backward when the singleton
zero-copy logits view is shared with the entropy / sum_pi_squared backward."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def make_engine():
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.engine_config = SimpleNamespace(
        entropy_from_logits_with_chunking=False,
        entropy_from_logits_chunk_size=2048,
        entropy_checkpointing=False,
    )
    engine.compute_entropy_from_logits = verl_F.entropy_from_logits
    return engine


def make_batch(seq_lengths, vocab, calculate_entropy, calculate_sum_pi_squared=False, device="cpu"):
    input_ids = [torch.randint(0, vocab, (n,), device=device) for n in seq_lengths]
    batch = TensorDict({"input_ids": tu.nested_tensor_from_tensor_list(input_ids)}, batch_size=len(seq_lengths))
    tu.assign_non_tensor(
        batch,
        use_remove_padding=False,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        calculate_entropy=calculate_entropy,
        calculate_sum_pi_squared=calculate_sum_pi_squared,
        distillation_use_topk=False,
    )
    # Leaf tensor; callers pass a non-leaf (``logits * 1.0``) to the engine because the
    # engine divides logits by the temperature in place, as with real model outputs.
    logits = torch.randn(len(seq_lengths), max(seq_lengths), vocab, device=device, requires_grad=True)
    rolled = torch.cat([torch.roll(ids, shifts=-1) for ids in input_ids])
    output_args = {"temperature": torch.ones(len(seq_lengths), device=device), "input_ids_rmpad_rolled": rolled}
    return batch, logits, output_args


@pytest.mark.parametrize(
    "calculate_entropy, calculate_sum_pi_squared, expected_inplace",
    [(False, False, True), (True, False, False), (False, True, False), (True, True, False)],
)
def test_inplace_backward_disabled_when_logits_storage_is_shared(
    monkeypatch, calculate_entropy, calculate_sum_pi_squared, expected_inplace
):
    seen = {}

    def fake_logprobs(logits, labels, inplace_backward=True):
        seen["inplace_backward"] = inplace_backward
        return verl_F.logprobs_from_logits_v2(logits, labels)

    monkeypatch.setattr(transformer_impl, "logprobs_from_logits", fake_logprobs)
    batch, logits, output_args = make_batch([7], 33, calculate_entropy, calculate_sum_pi_squared)

    out = make_engine().prepare_model_outputs(SimpleNamespace(logits=logits * 1.0), output_args, batch, None)

    assert seen["inplace_backward"] is expected_inplace
    assert out["log_probs"].is_nested
