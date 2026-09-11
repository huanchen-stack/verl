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
"""GPU smoke: Gemma-4 dense FFPA (head_dim 512, GQA, causal) against an SDPA reference on a right-padded batch."""

import pytest
import torch
import torch.nn.functional as F

from verl.models.transformers import gemma4_ffpa

pytestmark = pytest.mark.gpu_smoke


def _reference(q, k, v, scale):
    # q/k/v: [B, S, H, D] -> SDPA expects [B, H, S, D]
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        is_causal=True,
        scale=scale,
        enable_gqa=True,
    )
    return out.transpose(1, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_dense_ffpa_matches_sdpa_reference_on_valid_positions():
    pytest.importorskip("ffpa_attn", reason="optional dependency ffpa-attn==0.2.4 not installed")
    torch.manual_seed(0)
    batch, seq, heads, kv_heads, head_dim = 2, 1024, 8, 2, 512
    q = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, seq, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, seq, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
    mask[1, 700:] = 0
    position_ids = torch.arange(seq, device="cuda").unsqueeze(0).expand(batch, seq).clone()
    scale = head_dim**-0.5

    out = gemma4_ffpa.dense_ffpa_forward(q, k, v, mask, position_ids, is_causal=True, softmax_scale=scale)
    ref = _reference(q.detach(), k.detach(), v.detach(), scale)

    valid = mask.bool()
    a = out.float()[valid].flatten()
    b = ref[valid].flatten()
    cosine = F.cosine_similarity(a, b, dim=0).item()
    max_abs = (a - b).abs().max().item()
    assert cosine > 0.999, cosine
    assert max_abs < 5e-2, max_abs

    # The dense path must be differentiable (SM80 varlen backward is what FFPA lacks).
    out.float().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
