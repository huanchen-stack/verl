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
"""GPU smoke: with entropy on, gradients through the singleton zero-copy logits view equal those of the
torch.cat copy path (the flash-attn in-place cross-entropy backward must not touch the shared storage)."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

pytestmark = pytest.mark.gpu_smoke


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


def _grads(monkeypatch, batch, logits, output_args, copy_path):
    if copy_path:
        monkeypatch.setattr(transformer_impl, "cat_unbound_jagged", lambda ts: torch.cat(list(ts)))
    else:
        monkeypatch.undo()
    logits = logits.detach().clone().requires_grad_(True)
    out = make_engine().prepare_model_outputs(SimpleNamespace(logits=logits * 1.0), output_args, batch, None)
    loss = out["log_probs"].values().sum() + out["entropy"].values().sum()
    if "sum_pi_squared" in out:
        loss = loss + out["sum_pi_squared"].values().sum()
    loss.backward()
    return logits.grad.clone()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("calculate_sum_pi_squared", [False, True])
def test_singleton_view_grads_match_copy_path_with_entropy(monkeypatch, calculate_sum_pi_squared):
    pytest.importorskip("flash_attn.ops.triton.cross_entropy", reason="needs flash-attn cross entropy")
    torch.manual_seed(0)
    batch, logits, output_args = make_batch(
        [4096], 32000, calculate_entropy=True, calculate_sum_pi_squared=calculate_sum_pi_squared, device="cuda"
    )
    logits = logits.detach().to(torch.bfloat16).requires_grad_(True)

    reference = _grads(monkeypatch, batch, logits, output_args, copy_path=True)
    view = _grads(monkeypatch, batch, logits, output_args, copy_path=False)

    diff = (reference.float() - view.float()).abs().max().item()
    print({"max_abs_grad_diff": diff, "grad_scale": reference.float().abs().max().item()})
    assert torch.equal(reference, view), diff
