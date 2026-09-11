# Per-model overlays for the rollout precision scheduler

Each YAML in this directory holds the model-specific settings that used to live
in launcher `case` statements: checkpoint pair (BF16 actor + INT4 shadow), LoRA
target names, rollout load format and the few training-path knobs a model needs.
They contain only YAML keys of the base `ppo_trainer` config (plus the additive
`data.apply_chat_template_kwargs.enable_thinking` and
`rollout.engine_kwargs.vllm.lora_target_modules` entries, which are free-form
dicts in the base config). No environment variables.

| Overlay | Model | Tier | Special handling |
|---|---|---|---|
| `qwen3_5_9b.yaml` | Qwen/Qwen3.5-9B + Intel AutoRound INT4 | first-class | GatedDeltaNet `in_proj_*`/`out_proj` LoRA targets |
| `qwen3_5_4b.yaml` | Qwen/Qwen3.5-4B + Intel AutoRound INT4 | first-class | same as 9B |
| `phi4_mini_reasoning.yaml` | microsoft/Phi-4-mini-reasoning + llm-compressor W4A16 | first-class (config only) | fused Phi3 LoRA names; `rollout.load_format: auto` |
| `gemma4_e2b.yaml` | google/gemma-4-E2B QAT pair | flagged | `gemma4_dense_ffpa`, padded path, `mlp_only`, PEFT exclude regex |
| `nemotron_h.yaml` | nvidia/NVIDIA-Nemotron-Nano-9B-v2 + RedHatAI W4A16 | flagged | identity `in_proj`/`out_proj`; Marlin K padding on the vLLM side |

Usage: merge an overlay on top of the base config. Three nodes are free-form
dicts in the base config and therefore need Hydra's additive (`+key`) semantics:
`data.apply_chat_template_kwargs`, `actor_rollout_ref.model.override_config`
and `actor_rollout_ref.rollout.engine_kwargs.vllm`. From Python:

```python
base = compose(config_name="ppo_trainer")          # hydra
OmegaConf.set_struct(base, False)
cfg = OmegaConf.merge(base, OmegaConf.load("examples/precision_scheduler/models/<model>.yaml"))
```

`tests/examples/test_precision_scheduler_model_overlays_on_cpu.py` checks that
every other key of every overlay exists in the base config.
The `rollout.precision_scheduler.*` keys referenced here are documented in
`docs/precision_scheduler/config.md`; the model-side knobs in
`docs/precision_scheduler/model_support.md`.

Dropped from the experimental zoo (no code, screening only): Qwen3.5-27B and its
TP4 launcher, Falcon-H1, DeepSeek-R1 distills, SmolLM3, GLM-Z1, Kimi-VL, Granite,
MiniCPM.
