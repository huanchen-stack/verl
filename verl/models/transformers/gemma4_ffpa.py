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
"""Optional dense FFPA attention for Gemma-4 global layers (head_dim 512).

Gemma-4 global-attention layers use ``global_head_dim=512``, which FlashAttention-2
cannot execute on A100 (SM80).  FFPA (``ffpa-attn``, Apache-2.0) supports D=512
but its SM80 packed-varlen backward is unusable, so this path is deliberately
*dense* and fail-closed: callers must disable remove-padding and present an
unpacked, right-padded batch.  Right-padded causal batches are safe to run as a
dense tensor without forwarding the padding mask: every valid query can only
attend to positions at or before itself, while all padding is strictly to its
right.  Outputs for padded query positions are discarded by the caller's loss
mask.

Enabled through ``actor_rollout_ref.model.gemma4_dense_ffpa: true`` (default
false).  ``ffpa-attn`` is an optional dependency and is imported lazily; the
rest of verl never imports it.  Sliding-window layers (head_dim 256) still go
through FlashAttention-2.
"""

import logging
from typing import Optional

import torch

from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size

logger = logging.getLogger(__name__)

GEMMA4_FFPA_HEAD_DIM = 512
FFPA_REQUIREMENT = "ffpa-attn==0.2.4"

_ENABLED = False
_BACKEND = None
_LOGGED = False


def set_dense_ffpa_enabled(enabled: bool) -> None:
    """Select the dense FFPA path for head_dim-512 attention (called from apply_monkey_patch)."""
    global _ENABLED
    _ENABLED = bool(enabled)


def is_dense_ffpa_enabled() -> bool:
    return _ENABLED


def should_dispatch(query_states: torch.Tensor) -> bool:
    """True when the knob is on and the query tensor is the Gemma-4 global-attention shape."""
    return _ENABLED and query_states.ndim == 4 and query_states.size(-1) == GEMMA4_FFPA_HEAD_DIM


def _import_ffpa():
    try:
        from ffpa_attn import ffpa_attn_func
        from ffpa_attn.functional import CuTeDSLBackend
    except ImportError as exc:  # pragma: no cover - message is what matters
        raise ImportError(
            "actor_rollout_ref.model.gemma4_dense_ffpa=true requires the optional dependency "
            f"{FFPA_REQUIREMENT} (xlite-dev/ffpa-attn; needs torch>=2.10 and quack-kernels), "
            "which is not installed."
        ) from exc
    return ffpa_attn_func, CuTeDSLBackend


def check_dense_ffpa_inputs(
    query_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    **kwargs,
) -> None:
    """Raise RuntimeError unless the batch can be executed densely without a mask."""
    if get_ulysses_sequence_parallel_world_size() != 1:
        raise RuntimeError("gemma4_dense_ffpa requires sequence parallel size 1")
    if query_states.ndim != 4 or query_states.size(-1) != GEMMA4_FFPA_HEAD_DIM:
        raise RuntimeError(
            f"gemma4_dense_ffpa expected [B,S,H,{GEMMA4_FFPA_HEAD_DIM}], got {tuple(query_states.shape)}"
        )
    if query_states.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"gemma4_dense_ffpa requires fp16/bf16 QKV, got {query_states.dtype}")
    is_causal = bool(kwargs.get("is_causal", True))
    valid_tokens = None
    if attention_mask is not None:
        valid_tokens = attention_mask.bool()
        if valid_tokens.ndim != 2 or valid_tokens.shape != query_states.shape[:2]:
            raise RuntimeError(
                "gemma4_dense_ffpa requires a [B,S] padding mask, got "
                f"{tuple(valid_tokens.shape)} for Q={tuple(query_states.shape)}"
            )
        if not is_causal and not bool(valid_tokens.all().item()):
            raise RuntimeError("gemma4_dense_ffpa only elides right padding for causal attention")
        # A False->True transition means left/interior padding, which would be
        # visible to a later valid causal query if the mask were elided.
        if valid_tokens.size(1) > 1 and bool(((~valid_tokens[:, :-1]) & valid_tokens[:, 1:]).any().item()):
            raise RuntimeError("gemma4_dense_ffpa requires right-padded prefix masks")
    if position_ids is not None:
        if position_ids.ndim != 2 or position_ids.size(0) != query_states.size(0):
            raise RuntimeError("gemma4_dense_ffpa requires ordinary 2-D unpacked position_ids")
        if position_ids.size(-1) > 1:
            non_increasing = position_ids[:, 1:] <= position_ids[:, :-1]
            if valid_tokens is not None:
                non_increasing &= valid_tokens[:, 1:]
            if bool(non_increasing.any().item()):
                raise RuntimeError("gemma4_dense_ffpa detected packed/reset valid position_ids")
    if float(kwargs.get("dropout", 0.0)) != 0.0:
        raise RuntimeError("gemma4_dense_ffpa requires attention dropout 0")
    if kwargs.get("softcap") is not None:
        raise RuntimeError("gemma4_dense_ffpa does not support attention softcap")
    if kwargs.get("sliding_window") is not None:
        raise RuntimeError("gemma4_dense_ffpa must only dispatch global-attention layers")
    if kwargs.get("softmax_scale") is None:
        raise RuntimeError("gemma4_dense_ffpa requires the model's explicit attention scale")


def dense_ffpa_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    """Run Gemma-4 512-d global attention through dense FFPA.

    Args:
        query_states / key_states / value_states: ``[B, S, H(kv), 512]`` fp16/bf16.
        attention_mask: ``[B, S]`` right-padded mask or None.
        position_ids: ``[B, S]`` monotone position ids or None.
        kwargs: transformers flash-attention kwargs (``is_causal``, ``softmax_scale``,
            ``dropout``, ``softcap``, ``sliding_window``).

    Returns:
        ``[B, S, H, 512]`` attention output.
    """
    global _BACKEND, _LOGGED

    check_dense_ffpa_inputs(query_states, attention_mask, position_ids, **kwargs)
    ffpa_attn_func, cute_backend_cls = _import_ffpa()
    if _BACKEND is None:
        _BACKEND = cute_backend_cls()
    output = ffpa_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=bool(kwargs.get("is_causal", True)),
        scale=float(kwargs["softmax_scale"]),
        enable_gqa=True,
        backend=_BACKEND,
    )
    if not _LOGGED:
        logger.info("gemma4_dense_ffpa active: D512 global attention -> dense FFPA")
        _LOGGED = True
    return output.transpose(1, 2).contiguous()
