# Megatron on the RTX PRO 6000 (sm_120): working for Qwen, broken for Phi

Date: 2026-09-17. Host: one RTX PRO 6000 Blackwell Max-Q, 96 GB, compute capability
12.0. Decision 10 (reversed 2026-09-16) makes Megatron TP1 the RL-step trainer; this
records what it took to run it here and one bug that remains.

## The blocker: transformer-engine has no sm_120 kernels

The verl wheelhouse builds TE (and apex, flash-attn) for `TORCH_CUDA_ARCH_LIST
"9.0;10.0"`. That covers the A100/H100/B200 hosts the project has used. This card is
**sm_120**, a different architecture, so TE's own handwritten kernels have no image and
Megatron aborts at the first RMSNorm:

```
RuntimeError: .../rmsnorm_fwd_cuda_kernel.cu:76 in function launch_rmsnorm_fwd_general_:
CUDA Error: no kernel image is available for execution
```

**A cheap import check does not catch this.** `import transformer_engine` succeeds and
`te.Linear` in bf16 succeeds, because that dispatches to cuBLAS, which is
architecture-independent. Only TE's own kernels fail. Any sm_120 verification must call
`te.RMSNorm` or `te.LayerNormLinear`.

NVIDIA's prebuilt `transformer-engine-cu13` on PyPI does not help: it requires a newer
cuBLASLt (`cublasLtGroupedMatrixLayoutInit_internal`) than torch 2.11+cu130 ships, and
neither torch's copy nor CUDA 13.0.3 exports that symbol.

Fix: build TE from source for sm_120, `scripts/precision_scheduler/env/build_transformer_engine_sm120.sh`
(about 10 minutes on 32 cores). CUDA 13 splits out two headers TE needs, so the
`cuda130` env also needs `nvtx-c` and `nccl`.

**This build is fragile by construction.** `pyproject.toml` pins `transformer-engine` to
the wheelhouse index, so any `uv sync` that touches the megatron extra reinstalls the
sm_90/sm_100 wheel and silently reintroduces the failure. Re-run the script after such a
sync and check the RMSNorm probe it ends with.

## Result: Qwen trains, Phi does not

`TRAINER=megatron`, GSM8K, B = 8 x 4 = 32 requests, LoRA 16/16, `merge=false`.

| model | steps | outcome |
|---|---|---|
| Qwen3.5-9B | 3 | **correct**: coherent generations throughout, rollout/actor Pearson 0.9998 every step |
| Phi-4-mini-reasoning | 2-8 | **broken**: step 1 fine, every later step emits only `!!!!!` (token 0) |

Qwen also settles the timing question: step 1 carries 23.8 s of TE/JIT warm-up that no
timer covers (30% of the step), and from step 2 on the phases account for 100% of it.
Warm-up must be excluded, as the mg20 protocol already says.

## The Phi failure is the Phi3 bridge, not the host

Eliminated, each by experiment:

* **Not the optimizer or gradients.** With `ACTOR_LR=0`, so no weight can change, step 2
  is still garbage.
* **Not the adapter values.** Instrumenting the sync shows all 256 exported LoRA tensors
  healthy: A kaiming-initialised, B exactly zero, no NaN. A zero B makes the adapter a
  mathematical no-op, so it cannot corrupt anything by its values.
* **Not the sleep level.** The upstream rule already picks level 1 whenever LoRA is an
  adapter, so weights are never discarded; forcing level 1 explicitly changes nothing.
* **Not sm_120, the TE build, or the sync machinery.** Qwen runs the identical path.
* **Not a stale base.** The sync is adapter-only from step 1 by design
  (`base_sync_done = "dummy" not in rollout.load_format`, and the Phi overlay sets
  `load_format: auto`), and vLLM holds the base it loaded from HF.

What is left: **the adapter's mere presence breaks generation.** vLLM has no adapter at
step 1 and one from step 2, and the delta it applies is provably zero. Two candidates,
both specific to Phi and absent in Qwen:

1. **Tied embeddings.** Phi-4-mini has `tie_word_embeddings=True`, Qwen3.5-9B does not.
   Constant token 0 output is what zeroed logits look like, which points at the LoRA-wrapped
   tied `lm_head` / `embed_tokens`.
2. **The tensor-based LoRA path.** `verl/utils/vllm/utils.py` builds `expected_lora_modules`
   by expanding `packed_modules_mapping`, but passes it only to `from_local_checkpoint`;
   the runtime path uses `from_lora_tensors`, which receives no such expansion. Loading the
   same fused zero adapter from a PEFT **directory** generates fine, so the file path works
   and the tensor path is the difference.

Note vLLM's `phi3.py` self-maps `qkv_proj -> [qkv_proj]`, so the bridge's fused export
layout is correct; the fusion itself is not the bug.

Until this is fixed, Phi-4-mini cannot produce an RL-step number on Megatron. Qwen can.
