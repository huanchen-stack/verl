# Model support (verl side)

Component C9 of the `rollout-precision-scheduler-clean` branch. The vLLM half
(gemma4_unified registry glue, Gemma-4 k_norm fix, Marlin input padding,
model tools) is in `vllm/docs/design/model_support.md`.

## Purpose

The model-motivated fixes that let Qwen3.5, Phi-4-mini-reasoning, Gemma-4 and
Nemotron-H train (FSDP2, single GPU) and roll out under the precision
scheduler, plus one YAML overlay per model replacing the launcher `case`
statements.

## Tiers

| Model | Tier | verl code involved | Overlay |
|---|---|---|---|
| Qwen3.5-9B / 4B | first-class | mm-token guard, jagged index_select fallback, GatedDeltaNet LoRA mapping (Megatron only) | `qwen3_5_9b.yaml`, `qwen3_5_4b.yaml` |
| Phi-4-mini-reasoning | first-class (config only) | none (fused Phi3 target names, `rollout.load_format: auto`) | `phi4_mini_reasoning.yaml` |
| Gemma-4 E2B/E4B (12B/31B `gemma4_unified`) | flagged | `gemma4_dense_ffpa`, `gemma4_unified` normalization + `key_mapping`, padded-path chunked entropy, singleton logits | `gemma4_e2b.yaml` |
| Nemotron-Nano-9B-v2 | flagged | rope-theta skip + HybridStack hook (Megatron only), identity LoRA targets | `nemotron_h.yaml` |
| Qwen3.5-27B, Falcon-H1, DeepSeek distills, SmolLM3, GLM-Z1, Kimi-VL, Granite, MiniCPM | dropped | — | — |

Megatron TP1 (DP=1) is the training driver for every RL-step experiment and
every model, first-class and extensibility tier alike (decision 10, reversed
2026-09-16). FSDP2 is not a fallback for models that lack a Megatron-Bridge
mapping: such a model (Phi3ForCausalLM as of 2026-09-16) gets a bridge under
C9 before it is run. A Megatron e2e one-step GRPO test is required before any
full-step number is reported.

## Mechanisms

### Qwen3.5 mm-token guard — `verl/experimental/agent_loop/agent_loop.py`

Text-only rollouts can generate an `<|image_pad|>` / `<|video_pad|>` token.
`AgentLoopWorker._compute_position_ids` used to label any such token as
multimodal in `mm_token_type_ids`; Qwen3.5's `get_rope_index` then tried to
consume a grid from a `None` iterator and the completed rollout was dropped.
A token is now labelled only when the corresponding `image_grid_thw` /
`video_grid_thw` exists. Generic for every M-RoPE processor.

### Jagged `index_select_tensor_dict` fallback — `verl/utils/tensordict_utils.py`

Some PyTorch versions cannot `unbind()` a jagged NestedTensor whose ragged dim
is not the last sample dim (Qwen3.5 M-RoPE `position_ids`, `[4, seq]` per
sample). On `RuntimeError` the fallback slices the flat `values()` buffer with
`offsets()` along the ragged dim, which is exact. The stash's
`to_padded_tensor` variant was not ported: on torch 2.11 `to_padded_tensor`
returns the wrong shape for that layout (`[4, 4, 4]` for a `[4, j, 17]` stack).

### GatedDeltaNet LoRA target mapping — `verl/utils/megatron_peft_utils.py`

`convert_megatron_to_hf_target_modules(modules, model_type=None)` and
`build_peft_config_for_vllm(lora_config, model_type=None)`:

- for `model_type in {qwen3_5, qwen3_5_moe}` mcore `self_attention.in_proj` ->
  HF `in_proj_qkv, in_proj_z, in_proj_b, in_proj_a` and `out_proj -> out_proj`
  (Megatron-Bridge `qwen35_bridge` naming);
- for every other model type `in_proj`/`out_proj` keep the identity mapping —
  Nemotron-H Mamba mixers use those HF names verbatim, so an unconditional
  expansion would silently drop them;
- dotted wildcard targets (`decoder.layers.*.mlp.linear_fc1`) match on their
  last path component;
- `STACKED_PARAMS_BY_MODEL_TYPE` adds the five projection weights for the
  Qwen3.5 family (and `.in_proj.weight` / `.out_proj.weight` for `nemotron_h`)
  in `add_base_layer_suffix`, so non-merged LoRA weight sync emits `base_layer`
  names, which vLLM's Qwen3.5 loader merges into `in_proj_qkvz` / `in_proj_ba`.
  The vanilla `STACKED_PARAMS` list is unchanged for every other model type.
- `model_type` aliases: `qwen3_5`, `qwen3_5_moe`, `qwen3_5_text`, `qwen3_5_moe_text`.

The Megatron engine passes `hf_config.model_type`.

Vanilla differences (apply to every Megatron+LoRA model, not only the ones
above): the dotted-suffix fallback in `convert_megatron_to_hf_target_modules`
now expands a target such as `decoder.layers.*.mlp.linear_fc1` on its last path
component (`linear_fc1 -> gate_proj, up_proj`); vanilla passed dotted targets
through verbatim. A dotted target whose last component is not a known Megatron
name still passes through unchanged
(`tests/utils/test_megatron_peft_target_mapping_on_cpu.py::test_dotted_suffix_fallback_only_applies_to_the_last_component`).

### Megatron helpers — `verl/utils/megatron_utils.py`

- `set_hf_rope_theta_if_required(hf_config, provider)`: skips the RoPE base
  lookup when the Megatron-Bridge provider has `position_embedding_type ==
  "none"` (Nemotron-H); also removes a circular lazy import from
  `config_converter`.
- `enable_peft_recompute_input_grads_for_hybrid_stack(model)`: registered as a
  pre-wrap hook after the PEFT hook. For a `HybridStack` with
  `recompute_granularity == "full"` and PP=1 the frozen embedding output does
  not require grad, so the checkpoint backward is skipped and every LoRA
  gradient is zero; the hook marks the stack input as requiring grad. Idempotent,
  logs once through the module logger, no-op for `TransformerBlock` models.

### gemma4_unified normalization — `verl/utils/model_compat/gemma4.py`

`load_hf_config_with_model_compat(path, trust_remote_code, attn_implementation)`
is what `HFModelConfig.__post_init__` calls. For `config.json` with
`model_type == "gemma4_unified"` it builds the exact `Gemma4TextConfig`
(`model_type gemma4_text`, `architectures [Gemma4ForCausalLM]`) from the nested
`text_config` and sets `hf_config._verl_checkpoint_key_mapping =
{r"^model\.language_model\.": "model."}`; the FSDP engine passes that to
`from_pretrained(key_mapping=...)` (only when set, so other models and older
transformers are untouched). Plain `gemma4` (E2B/E4B) and every other model take
the `AutoConfig` path.

### Padded-path chunked entropy and singleton logits — `verl/workers/engine/fsdp/transformer_impl.py`

- `entropy_from_padded_logits(logits, entropy_fn, with_chunking, chunk_size, checkpointing)`:
  the `[batch, seq, vocab]` (non-rmpad) branch now honors
  `entropy_from_logits_with_chunking` / `entropy_from_logits_chunk_size` by
  flattening tokens; before, the config was applied only on the rmpad branch.
  Checkpointing goes through the same helper (`use_reentrant=False`).
- `cat_unbound_jagged(tensors)`: with a per-GPU micro-batch of one sample the
  sole jagged view is returned instead of `torch.cat`-ing it (a second
  `[tokens, vocab]` copy of several GiB at 16K x 262K). Because that view
  shares storage with `output.logits`, which the entropy / sum_pi_squared
  backward reads, the padded branch passes
  `inplace_backward = not (calculate_entropy or calculate_sum_pi_squared)` to
  `logprobs_from_logits`, mirroring the rmpad branch; with the in-place
  flash-attn cross-entropy backward the gradient differed by 0.018 (scale 1.0)
  on GPU 5 (`tests/models/test_padded_singleton_logprob_grads_on_gpu.py`).

### Gemma-4 dense FFPA — `verl/models/transformers/gemma4_ffpa.py`

Gemma-4 global-attention layers use `global_head_dim = 512`, which
FlashAttention-2 cannot run on A100. `ffpa-attn` (Apache-2.0) supports D=512
but its SM80 packed-varlen backward is unusable, so the path is dense and
fail-closed: `check_dense_ffpa_inputs` requires SP=1, `[B,S,H,512]`
fp16/bf16, a `[B,S]` right-padded mask (no `False -> True` transition), 2-D
monotone position ids on valid tokens, dropout 0, no softcap, no sliding
window, an explicit `softmax_scale`, and causal attention whenever padding is
present. Right-padded causal batches can run dense without a mask because every
valid query only sees positions at or before itself. Outputs on padded queries
are discarded by the loss mask.

`monkey_patch._ulysses_flash_attention_forward` dispatches to
`gemma4_ffpa.dense_ffpa_forward` only when `should_dispatch(q)` (knob on and
head_dim 512); sliding layers (head_dim 256) still use FA2.
`apply_monkey_patch(..., gemma4_dense_ffpa=...)` sets the module flag and also
patches the attention forward when `use_remove_padding=False`, which the FFPA
path requires. `ffpa_attn` is imported lazily with a message naming the pinned
requirement `ffpa-attn==0.2.4`; it is an optional dependency, not vendored.

## Knobs (all YAML, all default off)

| Key | Default | Meaning |
|---|---|---|
| `actor_rollout_ref.model.gemma4_dense_ffpa` | `false` | route head_dim-512 attention through dense FFPA (needs `use_remove_padding: false`) |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking` / `entropy_from_logits_chunk_size` | existing | now honored on the padded path too |
| `actor_rollout_ref.model.exclude_modules` | existing | Gemma-4: `'^model\.(vision_tower|audio_tower)\..*'` (PEFT cannot wrap `Gemma4ClippableLinear`) |
| `actor_rollout_ref.rollout.load_format` | existing | Phi-4: `auto` |
| `actor_rollout_ref.rollout.precision_scheduler.int4_model` / `int4_modules` | C8 block | per-model INT4 shadow; Gemma-4 uses `mlp_only` |
| `hf_config._verl_checkpoint_key_mapping` | attribute, `None` | set only for `gemma4_unified` |

Removed env vars: `VERL_GEMMA4_DENSE_FFPA` (now the YAML knob),
`VERL_GEMMA4_HEAD512_FALLBACK` (dead: nothing read it).

Note on the all-flags-off padded path (`use_remove_padding: false`): entropy
is now computed through `self.compute_entropy_from_logits` (the
`torch.compile`d function when `use_torch_compile` is on) with
`use_reentrant=False` checkpointing, and the single-sample micro-batch uses the
zero-copy view with the in-place cross-entropy backward disabled whenever
entropy or sum_pi_squared is requested. This is the same math as the dirty
tree and as the rmpad path, but it is not byte-identical to vanilla for
non-rmpad users (vanilla called the eager `verl_F.entropy_from_logits`).

## Per-model overlays — `examples/precision_scheduler/models/`

See the README there. They carry the checkpoint pair (hub ids, archived
snapshot revisions in comments), LoRA targets, `enable_thinking`, and the
model-specific training knobs. Only YAML keys; the vLLM padding flag for
Nemotron is noted as an open item (not yet a `precision_scheduler` key).

## Tests

| Test | Tier | Checks |
|---|---|---|
| `tests/experimental/agent_loop/test_text_only_position_ids_on_cpu.py` | unit | hallucinated tokens stay type 0 without grids; labelled 1/2 with grids; image grid alone labels only image tokens |
| `tests/utils/megatron/test_megatron_utils_rope.py` | unit (importorskip megatron.core) | rope theta lookup/skip/raise; HybridStack hook: patches full-recompute stacks, idempotent, leaves non-full untouched, does not detach inputs that already require grad, no-op without stacks |
| `tests/utils/test_megatron_peft_target_mapping_on_cpu.py` | unit | Qwen3.5 expansion, Nemotron identity, default unchanged, dotted suffix, dedup, STACKED_PARAMS, base_layer suffix |
| `tests/utils/test_tensordict_jagged_index_select_on_cpu.py` | unit | 3-D jagged select equals list indexing on the fast path and with `unbind` forced to fail; non-RuntimeError re-raised |
| `tests/utils/model_compat/test_gemma4_unified_config_on_cpu.py` | unit (real 12B and E2B `config.json` fixtures) | unified -> `Gemma4TextConfig` + key mapping; plain gemma4 -> AutoConfig, no mapping |
| `tests/models/test_padded_chunked_entropy_on_cpu.py` | unit | chunked == unchunked on `[3,17,257]` with/without checkpointing; chunked helper sees `[tokens, vocab]` + chunk_size; singleton view shares storage |
| `tests/models/test_padded_singleton_logprob_grads_on_cpu.py` | unit | `prepare_model_outputs` passes `inplace_backward=False` whenever entropy or sum_pi_squared is on (mocked `logprobs_from_logits`) |
| `tests/models/test_padded_singleton_logprob_grads_on_gpu.py` | gpu-smoke | gradients through the singleton view are identical to the `torch.cat` copy path with entropy (and sum_pi_squared) on |
| `tests/models/test_gemma4_ffpa_guards_on_cpu.py` | unit (no ffpa_attn) | every guard, dispatch gating, FA2 fallthrough for head_dim 256, clear ImportError |
| `tests/models/test_gemma4_ffpa_dense_on_gpu.py` | gpu-smoke (skips without ffpa_attn) | dense FFPA vs SDPA reference, `[2,1024,8/2,512]` right-padded: cosine > 0.999, max_abs < 5e-2, backward runs |
| `tests/examples/test_precision_scheduler_model_overlays_on_cpu.py` | unit | every overlay key exists in the base `ppo_trainer` config (except the three additive dict nodes), precision_scheduler keys by name, Phi-4/Gemma-4 specifics |

## Measured numbers (provenance)

- Dense FFPA vs SDPA, GPU 5, 2026-09-11, random bf16 `[2,1024,8/2,512]`
  right-padded: cosine 0.999998, max_abs 7.87e-3, mean_abs 1.67e-4; backward
  finite. `ffpa_attn` was imported from the archived wheel site
  `/data/huanchen/verl/.codex-report/new-storyline-experiments/eos_hazard_extensibility/tools/ffpa_site`
  (ffpa-attn 0.2.4) via `PYTHONPATH` for the smoke only.
- Archived Gemma-4 training runs that used this path: E2B/E4B/12B BF16 and
  full-W4 20-step baselines completed
  (`.codex-report/new-storyline-experiments/eos_hazard_fullstep_b64_cap16k/runs/gemma4_e2b_qat/gsm8k/bf16/baseline20_dense_ffpa_bf16_chunked_entropy_mb4_retry_20260909`,
  `gemma4_12b_qat/gsm8k/bf16/baseline20_v6_activation_offload_gpu2_20260910`);
  the EMA tail-W4 arms for Gemma-4 mostly failed, which is why the path is
  flagged rather than first-class.
- Nemotron HybridStack hook evidence: stage-0 logs
  `.codex-report/precision-scheduling-validation/stage0/logs/nemotron9b_megatron_tp1_dp7_bs14_n8_cap12288_bf16_hybrid_recompute_gradfix_true12k_{1step,10step_v1}.log`
  (`[PEFT+Recompute] Patched HybridStack.forward ... (1 stack(s))`) versus
  earlier stage-0 logs with `actor/grad_norm 0.0`.
- Phi-4 headline (B64 L16K gsm8k tail 1.412x, math500 1.288x) and Qwen3.5-4B
  bigmath 1.035x: `.codex-report/new-storyline-experiments/eos_hazard_extensibility/reports/ALL_MODEL_DATASET_TIMING_WORK_MASTER.md`.
  The Phi-4 W4 checkpoint is a community quantization flagged screening-only.

## Dropped from the experimental tree and why

- `VERL_GEMMA4_DENSE_FFPA` env read at dispatch time and the `print` logging —
  replaced by the YAML knob and the module logger.
- `VERL_GEMMA4_HEAD512_FALLBACK` — dead flag.
- Inline JSON parsing of `config.json` in `HFModelConfig.__post_init__` — moved
  to the testable helper.
- The stash's `to_padded_tensor` fallback — wrong on torch 2.11 (see above).
- Unconditional `in_proj` expansion — architecture-conditional now.
- Qwen3.5-27B (Megatron TP4) configs and launcher, the extensibility zoo, the
  `ffpa_site` wheel directory (optional dependency instead).
- `prepare_model_pair.py` consolidation of `prepare_gemma4.py` /
  `resolve_and_audit_models.py`: only the Phi-4 tool was in scope; it lives on
  the vLLM side under `tools/precision_scheduler/models/`.

## Open items

- Expose `VLLM_MARLIN_INPUT_PADDING` as a `rollout.precision_scheduler` key
  (C8 block) so the Nemotron overlay needs no environment variable.
- Phi-4's `rollout.load_format: auto` masks a defect in the dummy-load +
  streamed-weights path for Phi3 when an INT4 shadow is attached (embedding /
  norm / lm_head not rebuilt); owned by the weight-sync component.
- No Megatron + HybridStack GPU test exists; regressions there would only show
  as silent zero gradients.
