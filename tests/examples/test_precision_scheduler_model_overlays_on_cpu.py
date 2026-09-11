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
"""Structural checks for examples/precision_scheduler/models/*.yaml (YAML keys only, no env vars)."""

import copy
import dataclasses
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.workers.config.model import HFModelConfig

REPO = Path(__file__).resolve().parents[2]
OVERLAY_DIR = REPO / "examples" / "precision_scheduler" / "models"
OVERLAYS = sorted(OVERLAY_DIR.glob("*.yaml"))
EXPECTED = {"gemma4_e2b", "nemotron_h", "phi4_mini_reasoning", "qwen3_5_4b", "qwen3_5_9b"}

# Keys of actor_rollout_ref.rollout.precision_scheduler (docs/precision_scheduler/config.md).
PRECISION_SCHEDULER_KEYS = {
    "enable",
    "lora_fast_path",
    "lora_dual_stream",
    "lora_fuse_packed",
    "int4_model",
    "policy",
    "bf16_layers",
    "int4_modules",
    "reprefill",
    "sleep_level",
    "reload_policy_each_rollout",
    "online_observations",
    "validate_shadow",
    "validate_lifecycle",
    "request_trace_dir",
    "request_trace_log_tokens",
}


def test_expected_overlays_exist():
    assert {p.stem for p in OVERLAYS} == EXPECTED


@pytest.mark.parametrize("overlay", OVERLAYS, ids=[p.stem for p in OVERLAYS])
def test_overlay_uses_only_known_keys(overlay):
    cfg = OmegaConf.load(overlay)
    model_fields = {f.name for f in dataclasses.fields(HFModelConfig)}
    model_keys = set(cfg.actor_rollout_ref.model.keys())
    assert model_keys <= model_fields, model_keys - model_fields

    ps = cfg.actor_rollout_ref.rollout.get("precision_scheduler", {})
    assert set(ps.keys()) <= PRECISION_SCHEDULER_KEYS, set(ps.keys()) - PRECISION_SCHEDULER_KEYS
    assert ps.get("int4_model"), "every overlay names its INT4 shadow checkpoint"

    targets = list(cfg.actor_rollout_ref.model.target_modules)
    assert targets and len(set(targets)) == len(targets)
    assert cfg.actor_rollout_ref.rollout.engine_kwargs.vllm.lora_target_modules is not None

    text = overlay.read_text()
    for forbidden in ("VLLM_", "VERL_", "ROLLOUT_QLORA"):
        # Env vars may be mentioned in comments only.
        for line in text.splitlines():
            if forbidden in line:
                assert line.lstrip().startswith("#"), line


# Free-form dict nodes of the base config that overlays extend (Hydra "+key" semantics).
# The launchers used exactly these: +data.apply_chat_template_kwargs.enable_thinking,
# +actor_rollout_ref.model.override_config.attn_implementation and
# +actor_rollout_ref.rollout.engine_kwargs.vllm.lora_target_modules.
ADDITIVE_PREFIXES = (
    "data.apply_chat_template_kwargs.",
    "actor_rollout_ref.model.override_config.",
    "actor_rollout_ref.rollout.engine_kwargs.vllm.",
    "actor_rollout_ref.rollout.precision_scheduler.",
)


def _leaf_paths(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaf_paths(value, f"{prefix}{key}.")
    else:
        yield prefix[:-1]


@pytest.mark.parametrize("overlay", OVERLAYS, ids=[p.stem for p in OVERLAYS])
def test_overlay_merges_into_base_ppo_trainer_config(overlay):
    with initialize_config_dir(config_dir=str(REPO / "verl" / "trainer" / "config"), version_base=None):
        base = compose(config_name="ppo_trainer")
    ov = OmegaConf.load(overlay)
    container = OmegaConf.to_container(ov, resolve=False)

    # Every non-additive key must already exist in the base config (catches typos).
    for path in _leaf_paths(container):
        if path.startswith(ADDITIVE_PREFIXES):
            continue
        assert OmegaConf.select(base, path, throw_on_missing=False) is not None or path in _null_defaults(base, path), (
            path
        )

    # Merge with "+" semantics for the additive nodes, as the launchers did.
    merged = copy.deepcopy(base)
    OmegaConf.set_struct(merged, False)
    merged = OmegaConf.merge(merged, OmegaConf.create(container))
    assert merged.actor_rollout_ref.model.path == ov.actor_rollout_ref.model.path
    assert list(merged.actor_rollout_ref.rollout.engine_kwargs.vllm.lora_target_modules) == list(
        ov.actor_rollout_ref.model.target_modules
    )
    assert merged.actor_rollout_ref.model.lora_rank == 16


def _null_defaults(base, path):
    # OmegaConf.select returns None both for missing keys and for keys whose default is null;
    # distinguish them by checking the parent node.
    parent, _, leaf = path.rpartition(".")
    parent_node = OmegaConf.select(base, parent, throw_on_missing=False)
    return {path} if parent_node is not None and leaf in parent_node else set()


def test_phi4_uses_auto_load_format_and_fused_targets():
    cfg = OmegaConf.load(OVERLAY_DIR / "phi4_mini_reasoning.yaml")
    assert cfg.actor_rollout_ref.rollout.load_format == "auto"
    assert set(cfg.actor_rollout_ref.model.target_modules) == {"qkv_proj", "o_proj", "gate_up_proj", "down_proj"}


def test_gemma4_overlay_selects_dense_ffpa_and_padded_path():
    cfg = OmegaConf.load(OVERLAY_DIR / "gemma4_e2b.yaml")
    assert cfg.actor_rollout_ref.model.gemma4_dense_ffpa is True
    assert cfg.actor_rollout_ref.model.use_remove_padding is False
    assert cfg.actor_rollout_ref.actor.entropy_from_logits_with_chunking is True
    assert cfg.actor_rollout_ref.rollout.precision_scheduler.int4_modules == "mlp_only"
    assert cfg.actor_rollout_ref.model.exclude_modules.startswith("^model")


def test_other_overlays_keep_flash_attention_defaults():
    for stem in ("qwen3_5_9b", "qwen3_5_4b", "phi4_mini_reasoning", "nemotron_h"):
        cfg = OmegaConf.load(OVERLAY_DIR / f"{stem}.yaml")
        assert "gemma4_dense_ffpa" not in cfg.actor_rollout_ref.model
        assert cfg.actor_rollout_ref.model.use_remove_padding is True
