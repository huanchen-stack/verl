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
"""GPU parity test for verl/models/mcore/phi3_bridge.py (Phi-4-mini-reasoning, TP=1).

1. HF -> Megatron weight load, then logits parity against the HF model on real prompts.
2. Megatron -> HF export round-trips every tensor (fused qkv_proj / gate_up_proj included).
3. LoRA: Megatron-Bridge PEFT on linear_qkv / linear_fc1 / linear_proj / linear_fc2, adapter export in the
   fused HF layout, and per-layer numerical check that HF base + exported lora_B @ lora_A equals the Megatron
   LoRA layer output (this is what vLLM applies).

Run:  CUDA_VISIBLE_DEVICES=<gpu> pytest -q tests/models/test_phi3_megatron_bridge_on_gpu.py
"""

import os

import pytest
import torch

MODEL = os.environ.get(
    "PHI3_MODEL_PATH",
    "/data/huggingface/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a",
)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isdir(MODEL), reason="needs a GPU and the Phi-4-mini snapshot"
)


@pytest.fixture(scope="module")
def dist():
    from megatron.core import parallel_state

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29655")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(0)
    yield
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


@pytest.fixture(scope="module")
def hf_model():
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    return m


@pytest.fixture(scope="module")
def bridged(dist):
    from verl.models.mcore.bridge import AutoBridge  # registers Phi3Bridge

    bridge = AutoBridge.from_hf_pretrained(MODEL)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.seq_length = 4096
    provider.use_cpu_initialization = False
    # same overrides the verl driver applies (no APEX on this host)
    provider.apply_overrides_and_finalize(
        dtype=torch.bfloat16, overrides={"gradient_accumulation_fusion": False, "attention_backend": "auto"}
    )
    model = provider.provide_distributed_model(wrap_with_ddp=False, bf16=True)
    bridge.load_hf_weights(model)
    for m in model:
        m.eval()
    return bridge, provider, model


def _prompts():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    texts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May? Think step by step.",
        "The quick brown fox jumps over the lazy dog. " * 40,
    ]
    ids = [tok(t, return_tensors="pt").input_ids[0] for t in texts]
    return ids


def _megatron_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    m = model[0]
    ids = input_ids.unsqueeze(0).cuda()
    pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    mask = torch.ones((1, 1, ids.shape[1], ids.shape[1]), device=ids.device, dtype=torch.bool).tril().logical_not()
    with torch.no_grad():
        out = m(input_ids=ids, position_ids=pos, attention_mask=mask)
    return out[0] if out.dim() == 3 else out  # [s, vocab]


def test_logits_parity(hf_model, bridged):
    _, _, model = bridged
    for ids in _prompts():
        with torch.no_grad():
            hf = hf_model(ids.unsqueeze(0).cuda()).logits[0].float()
        mg = _megatron_logits(model, ids).float()
        assert mg.shape == hf.shape, (mg.shape, hf.shape)
        top_hf = hf.argmax(-1)
        top_mg = mg.argmax(-1)
        agree = (top_hf == top_mg).float().mean().item()
        lp_hf = torch.log_softmax(hf, -1)
        lp_mg = torch.log_softmax(mg, -1)
        kl = (lp_hf.exp() * (lp_hf - lp_mg)).sum(-1).mean().item()
        print(f"len={ids.numel()} argmax agreement {agree:.4f} mean KL(hf||mg) {kl:.2e}")
        assert agree > 0.97, f"argmax agreement {agree}"
        # bf16 HF itself sits at KL ~8e-3 from fp32 HF on a 41-token prompt; plain RoPE (no LongRoPE) is at ~2.
        assert kl < 1.5e-2, f"KL {kl}"


def test_weight_round_trip(hf_model, bridged):
    bridge, _, model = bridged
    hf_sd = hf_model.state_dict()
    seen = 0
    for name, tensor in bridge.export_hf_weights(model, cpu=False):
        if name not in hf_sd:
            assert name == "lm_head.weight", name  # tied embeddings: exporter may still emit it
            continue
        ref = hf_sd[name]
        assert tensor.shape == ref.shape, (name, tensor.shape, ref.shape)
        assert torch.equal(tensor.to(ref.device, ref.dtype), ref), f"{name} differs"
        seen += 1
    assert seen >= len(hf_sd) - 1, (seen, len(hf_sd))


def test_lora_export_fused_layout(dist, hf_model, bridged):
    from megatron.bridge.peft.lora import LoRA

    from verl.utils.megatron_peft_utils import build_peft_config_for_vllm

    bridge, provider, model = bridged
    lora = LoRA(target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"], dim=16, alpha=16)
    model = [lora(m, training=True) for m in model]
    # randomise the adapters so B != 0
    torch.manual_seed(0)
    for m in model:
        for n, p in m.named_parameters():
            if "adapter" in n or "lora" in n:
                p.data.normal_(0, 0.02)
    exported = {n: t for n, t in bridge.export_adapter_weights(model, cpu=False)}
    names = sorted(exported)
    print("adapter tensors:", len(names), names[:6])
    layer0 = [n for n in names if ".layers.0." in n]
    assert {n.split(".layers.0.")[1] for n in layer0} == {
        "self_attn.qkv_proj.lora_A.weight",
        "self_attn.qkv_proj.lora_B.weight",
        "self_attn.o_proj.lora_A.weight",
        "self_attn.o_proj.lora_B.weight",
        "mlp.gate_up_proj.lora_A.weight",
        "mlp.gate_up_proj.lora_B.weight",
        "mlp.down_proj.lora_A.weight",
        "mlp.down_proj.lora_B.weight",
    }, layer0
    peft = build_peft_config_for_vllm({"rank": 16, "alpha": 16}, model_type="phi3")
    assert set(peft["target_modules"]) == {"qkv_proj", "o_proj", "gate_up_proj", "down_proj"}, peft

    # numerical check on layer 0: HF fused base + exported B@A  ==  Megatron LoRA layer output
    from megatron.core.utils import unwrap_model

    hf_layer = hf_model.model.layers[0]
    core = unwrap_model(model[0])
    mg_layer = core.decoder.layers[0]
    x = torch.randn(4, 3072, device="cuda", dtype=torch.bfloat16)
    scale = 16 / 16
    for hf_lin, mg_lin, key in (
        (hf_layer.self_attn.qkv_proj, mg_layer.self_attention.linear_qkv, "self_attn.qkv_proj"),
        (hf_layer.mlp.gate_up_proj, mg_layer.mlp.linear_fc1, "mlp.gate_up_proj"),
        (hf_layer.self_attn.o_proj, mg_layer.self_attention.linear_proj, "self_attn.o_proj"),
        (hf_layer.mlp.down_proj, mg_layer.mlp.linear_fc2, "mlp.down_proj"),
    ):
        xin = x if hf_lin.in_features == 3072 else torch.randn(4, hf_lin.in_features, device="cuda", dtype=torch.bfloat16)
        A = exported[f"model.layers.0.{key}.lora_A.weight"].to(xin.dtype)
        B = exported[f"model.layers.0.{key}.lora_B.weight"].to(xin.dtype)
        with torch.no_grad():
            ref = hf_lin(xin) + scale * (xin @ A.t()) @ B.t()
            out = mg_lin(xin)
            out = out[0] if isinstance(out, tuple) else out
        # linear_qkv / linear_fc1 carry the layernorm in TE (layer_norm_weight); compare in HF order via export
        if key in ("self_attn.qkv_proj", "mlp.gate_up_proj"):
            # TE fused layernorm-linear applies RMSNorm to xin first; feed the normed input to the HF side instead
            ln = hf_layer.input_layernorm if key.startswith("self_attn") else hf_layer.post_attention_layernorm
            with torch.no_grad():
                ref = hf_lin(ln(xin)) + scale * (ln(xin) @ A.t()) @ B.t()
            # Megatron output is in interleaved-GQA order for qkv: bring the HF ref into that order for comparison
            if key == "self_attn.qkv_proj":
                # merge_qkv_weights expects [rows, hidden]; reorder the activation columns with a permutation
                # built from an index matrix run through the same helper.
                from megatron.bridge.models.conversion.param_mapping import merge_qkv_weights

                cfg = core.config
                n = (24 + 2 * 8) * 128
                idx = torch.arange(n, device="cuda", dtype=torch.float32).view(n, 1).expand(n, 3072).contiguous()
                q_i, k_i, v_i = idx.split([24 * 128, 8 * 128, 8 * 128], dim=0)
                perm = merge_qkv_weights(cfg, q_i, k_i, v_i)[:, 0].long()
                ref = ref[:, perm]
        err = (out.float() - ref.float()).abs().max().item()
        rel = err / (ref.float().abs().max().item() + 1e-6)
        print(f"layer0 {key}: max abs err {err:.4f} rel {rel:.2e}")
        assert rel < 2e-2, (key, err, rel)
