# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Megatron-Bridge for HF ``Phi3ForCausalLM`` (Phi-3 / Phi-4-mini families).

Megatron-Bridge 0.5.0 ships no Phi3 bridge. Phi3 is Llama-shaped (RMSNorm, GQA, SwiGLU, no biases) with
three differences that this module handles:

* the HF checkpoint stores ``self_attn.qkv_proj`` as one fused ``[q; k; v]`` tensor and ``mlp.gate_up_proj``
  as one fused ``[gate; up]`` tensor. Loading uses :class:`ConcatenatedQKVMapping` (Megatron-Bridge) and a
  new :class:`ConcatenatedGatedMLPMapping`; the LoRA export hook is overridden because Megatron-Bridge's PEFT
  exporter decides how to un-interleave ``linear_out`` by *HF name heuristics* (``q_proj``/``gate_proj``) and
  would otherwise emit ``qkv_proj.lora_B`` in Megatron's GQA-interleaved row order.
* partial rotary (``partial_rotary_factor``) is native to Megatron-Core (``rotary_percent``); the bridge must
  not override it the way the Llama bridge does.
* LongRoPE (``rope_scaling.type == "longrope"``): Megatron-Core has no implementation. HF computes
  ``inv_freq = 1 / (ext_factor * base ** (2i/dim))`` and multiplies cos/sin by an attention factor
  ``sqrt(1 + ln(max_pos / original_max_pos) / ln(original_max_pos))``. A provider pre-wrap hook rescales the
  model's ``inv_freq`` in place, and the attention factor is expressed through Megatron-Core's YaRN
  concentration-factor path (``config.yarn_*``), which the attention layer applies as ``mscale`` to cos/sin
  for every position-embedding type. TE's fused thd RoPE ignores ``mscale``, so ``apply_rope_fusion`` is
  forced off for this architecture. The short/long factor switch (HF picks ``long_factor`` once the sequence
  exceeds ``original_max_position_embeddings``) is resolved once at load from the training sequence length
  (``seq_length``): Phi-4-mini ships identical short and long factors, so the choice does not matter there.

Importing this module registers the bridge; :mod:`verl.models.mcore.bridge` imports it so
``AutoBridge.from_hf_pretrained`` resolves ``Phi3ForCausalLM``.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import Phi3ForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ConcatenatedQKVMapping,
    GatedMLPMapping,
    _module_uses_fsdp,
    split_qkv_weights,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

logger = logging.getLogger(__name__)

_GATE_KEY = "gate"
_UP_KEY = "up"


class ConcatenatedGatedMLPMapping(GatedMLPMapping):
    """Gated-MLP mapping whose HF side is one fused ``[gate; up]`` tensor (Phi3 ``gate_up_proj``).

    Registered as a :class:`GatedMLPMapping` whose ``gate`` and ``up`` HF names are both the fused name, so the
    loader hands ``hf_to_megatron`` ``{"gate": fused, "up": fused}``; the tensor-parallel regrouping
    (``[gate_i; up_i]`` per rank) is inherited. Export re-assembles the single fused tensor.
    """

    def __init__(self, megatron_param: str, hf_param: str):
        super().__init__(megatron_param, gate=hf_param, up=hf_param)
        self.fused_hf_param = hf_param

    def hf_to_megatron(self, hf_weights, megatron_module):  # type: ignore[override]
        fused = hf_weights[_GATE_KEY] if isinstance(hf_weights, dict) else hf_weights
        gate, up = torch.chunk(fused, 2, dim=0)
        return super().hf_to_megatron({_GATE_KEY: gate, _UP_KEY: up}, megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module):  # type: ignore[override]
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=self.fused_hf_param)
        if megatron_weights is None:
            return {}
        megatron_weights = self.maybe_dequantize(megatron_weights)
        if self.tp_size == 1 or _module_uses_fsdp(megatron_module):
            shards = torch.chunk(megatron_weights, self.tp_size, dim=0) if self.tp_size > 1 else [megatron_weights]
        else:
            shards = self.gather_from_tp_ranks(megatron_weights)
        gates, ups = [], []
        for shard in shards:
            g, u = torch.chunk(shard, 2, dim=0)
            gates.append(g)
            ups.append(u)
        return {self.fused_hf_param: torch.cat([torch.cat(gates, dim=0), torch.cat(ups, dim=0)], dim=0)}

    def resolve(self, captures):
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        fused = resolved_hf_param[_GATE_KEY] if isinstance(resolved_hf_param, dict) else resolved_hf_param
        return type(self)(resolved_megatron_param, fused)


def longrope_attention_factor(hf_config) -> float:
    """HF ``_compute_longrope_parameters`` attention factor for a longrope config (1.0 otherwise)."""
    scaling = getattr(hf_config, "rope_scaling", None) or {}
    if scaling.get("type", scaling.get("rope_type")) != "longrope":
        return 1.0
    if scaling.get("attention_factor") is not None:
        return float(scaling["attention_factor"])
    original = getattr(hf_config, "original_max_position_embeddings", None) or scaling.get(
        "original_max_position_embeddings"
    )
    factor = scaling.get("factor")
    if factor is None:
        factor = hf_config.max_position_embeddings / original
    if factor <= 1.0:
        return 1.0
    return math.sqrt(1 + math.log(factor) / math.log(original))


def longrope_ext_factors(hf_config, seq_length: int) -> Optional[List[float]]:
    """The per-frequency LongRoPE factors HF would use for sequences of ``seq_length`` (None if not longrope)."""
    scaling = getattr(hf_config, "rope_scaling", None) or {}
    if scaling.get("type", scaling.get("rope_type")) != "longrope":
        return None
    original = getattr(hf_config, "original_max_position_embeddings", None) or scaling.get(
        "original_max_position_embeddings"
    )
    use_long = original is not None and seq_length > original
    return list(scaling["long_factor"] if use_long else scaling["short_factor"])


def _apply_longrope(model, ext_factors: List[float]) -> None:
    """Divide every RotaryEmbedding's ``inv_freq`` by the LongRoPE factors (in place, idempotent)."""
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    modules = model if isinstance(model, (list, tuple)) else [model]
    for m in modules:
        for name, sub in m.named_modules():
            if isinstance(sub, RotaryEmbedding) and not getattr(sub, "_phi3_longrope_applied", False):
                factors = torch.tensor(ext_factors, dtype=sub.inv_freq.dtype, device=sub.inv_freq.device)
                if factors.numel() != sub.inv_freq.numel():
                    raise ValueError(
                        f"LongRoPE factor count {factors.numel()} != rotary dim/2 {sub.inv_freq.numel()} at {name}"
                    )
                sub.inv_freq = sub.inv_freq / factors
                sub._phi3_longrope_applied = True
                logger.info("[Phi3 bridge] LongRoPE factors applied to %s (%d freqs)", name, factors.numel())


@MegatronModelBridge.register_bridge(source=Phi3ForCausalLM, target=GPTModel, model_type="phi3")
class Phi3Bridge(MegatronModelBridge):
    """HF ``Phi3ForCausalLM`` <-> Megatron-Core ``GPTModel``."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.hidden_dropout = 0.0
        provider.bias_activation_fusion = True
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True
        provider.bias_dropout_fusion = True
        # ConcatenatedQKVMapping reads config.kv_channels directly; Phi3 configs carry no head_dim.
        provider.kv_channels = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // hf_config.num_attention_heads
        )
        # partial_rotary_factor -> rotary_percent is set by CONFIG_MAPPING; keep it (the Llama bridge forces 1.0).

        attention_factor = longrope_attention_factor(hf_config)
        ext_factors = longrope_ext_factors(hf_config, provider.seq_length)
        if ext_factors is not None:
            # mscale is applied by megatron.core Attention through the YaRN concentration-factor helper:
            #   mscale(s, m, m_all) = (0.1*m*ln s + 1) / (0.1*m_all*ln s + 1)
            # With ln s = 10 and m_all = 0 this is exactly m + 1, so m = attention_factor - 1.
            provider.yarn_rotary_scaling_factor = math.e**10
            provider.yarn_mscale = attention_factor - 1.0
            provider.yarn_mscale_all_dim = 0.0
            # TE's fused thd RoPE drops mscale silently.
            provider.apply_rope_fusion = False
            provider.register_pre_wrap_hook(lambda model: (_apply_longrope(model, ext_factors), model)[1])
            logger.info(
                "[Phi3 bridge] LongRoPE: %d factors (%s), attention factor %.6f, seq_length %d",
                len(ext_factors),
                "long" if provider.seq_length > getattr(hf_config, "original_max_position_embeddings", 0) else "short",
                attention_factor,
                provider.seq_length,
            )
        else:
            provider.apply_rope_fusion = True
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }
        mapping_list = [AutoMapping(megatron_param=m, hf_param=h) for m, h in param_mappings.items()]
        mapping_list.extend(
            [
                ConcatenatedQKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    hf_param="model.layers.*.self_attn.qkv_proj.weight",
                ),
                ConcatenatedGatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    hf_param="model.layers.*.mlp.gate_up_proj.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)

    # ---- LoRA export: fused HF modules --------------------------------------------------------------- #

    def _get_fused_adapter_linear_out_slices(
        self,
        megatron_model,
        base_hf_weight_names: List[str],
        linear_out_tensor: torch.Tensor,
        is_expert: bool = False,
    ) -> Optional[Dict[str, torch.Tensor]]:
        names = list(base_hf_weight_names)
        if len(names) == 1 and ".qkv_proj." in names[0]:
            model = megatron_model[0] if isinstance(megatron_model, list) else megatron_model
            feature_dim = linear_out_tensor.shape[-1] if linear_out_tensor.ndim == 2 else None
            q, k, v = split_qkv_weights(model.config, linear_out_tensor, feature_dim=feature_dim)
            return {names[0]: torch.cat((q, k, v), dim=0)}
        if len(names) == 1 and ".gate_up_proj." in names[0]:
            gate, up = self._split_fused_fc1_linear_out_weight(linear_out_tensor, is_expert=is_expert)
            return {names[0]: torch.cat((gate, up), dim=0)}
        return super()._get_fused_adapter_linear_out_slices(
            megatron_model, base_hf_weight_names, linear_out_tensor, is_expert=is_expert
        )

    def _merge_lora_adapter_weights(self, megatron_model, converted_weights_dict, adapter_weights):
        fused = [n for n in converted_weights_dict if ".qkv_proj." in n or ".gate_up_proj." in n]
        if fused:
            raise NotImplementedError(
                "Phi3Bridge: merged LoRA export for fused qkv_proj / gate_up_proj is not implemented; "
                "use model.lora.merge=False (adapter streamed separately)."
            )
        return super()._merge_lora_adapter_weights(megatron_model, converted_weights_dict, adapter_weights)
