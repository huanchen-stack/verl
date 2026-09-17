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
"""CPU tests for examples/precision_scheduler recipes.

* DRY_RUN=1 of every recipe yields an override list that Hydra composes on top of ``ppo_trainer``
  without error, the dataclasses validate, and the resolved ``precision_scheduler`` block matches
  the policy the recipe was given (decision 6 / 9 contract).
* Recipes set no environment variable for verl / vLLM: the override list is the whole contract.
* ``continuous_ema.sh`` fails closed with a stub watcher and a stub runner.
* Golden: the dry-run contract against an archived ``run_config.json`` (trimmed fixture).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples" / "precision_scheduler"
CONFIG_DIR = REPO / "verl" / "trainer" / "config"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PS = "actor_rollout_ref.rollout.precision_scheduler"

RECIPES = {
    "run_megatron_fullstep": EXAMPLES / "run_megatron_fullstep.sh",
    "run_fsdp_fullstep": EXAMPLES / "run_fsdp_fullstep.sh",
    "rollout_only": EXAMPLES / "recipes" / "rollout_only.sh",
    "full_step": EXAMPLES / "recipes" / "full_step.sh",
    "train_arm": EXAMPLES / "long_run" / "train_arm.sh",
}
POLICIES = {
    "bf16": {"enable": False},
    "full_w4": {"enable": True, "policy": "uniform_w4"},
    "tail_t8": {"enable": True, "policy": "fixed_threshold:8"},
    "fixed_k8000": {"enable": True, "policy": "fixed_frontier:8000"},
}
MODELS = ("qwen3_5_4b", "qwen3_5_9b", "phi4_mini_reasoning", "gemma4_e2b")


def dry_run(script: Path, tmp_path: Path, env_extra: dict[str, str], *args: str) -> tuple[list[str], str]:
    env = {k: v for k, v in os.environ.items() if k not in ("DRY_RUN",)}
    env.update(
        {
            "DRY_RUN": "1",
            "RUN_DIR": str(tmp_path / "run"),
            "PYTHON_BIN": "python",
            "PS_DATA_ROOT": str(tmp_path / "data"),
        }
    )
    env.update(env_extra)
    proc = subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    lines = proc.stdout.splitlines()
    assert lines[-1] == "DRY_RUN_OK", proc.stdout
    return [line.split("\t", 1)[1] for line in lines if line.startswith("OVERRIDE\t")], proc.stdout


def compose_overrides(overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def _policy_file(tmp_path: Path) -> Path:
    path = tmp_path / "policy.json"
    path.write_text('{"schema_version": 6}')
    return path


@pytest.mark.parametrize("recipe", sorted(RECIPES))
@pytest.mark.parametrize("policy", sorted(POLICIES))
@pytest.mark.parametrize("trainer", ("megatron", "fsdp2"))
def test_recipe_dry_run_composes(recipe, policy, trainer, tmp_path):
    if recipe.startswith("run_") and recipe != f"run_{trainer}_fullstep":
        pytest.skip("the wrapper pins its own TRAINER")
    env = {"POLICY": policy, "MODEL_KEY": "qwen3_5_4b"}
    if not recipe.startswith("run_"):
        env["TRAINER"] = trainer
    overrides, _ = dry_run(RECIPES[recipe], tmp_path, env)
    cfg = compose_overrides(overrides)
    rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
    omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    model = omega_conf_to_dataclass(cfg.actor_rollout_ref.model)
    ps = rollout.precision_scheduler
    expected = POLICIES[policy]
    assert ps.enable is expected["enable"]
    if expected["enable"]:
        assert ps.policy == expected["policy"]
        assert ps.lora_fast_path and ps.lora_dual_stream
        assert ps.bf16_layers == "none" and ps.reprefill is False and ps.validate_lifecycle is True
        assert ps.validate_shadow is True, "a dummy-loaded shadow must fail at load, not in generation quality"
        assert ps.online_observations == str(tmp_path / "run" / "switch_observations.jsonl")
        assert ps.int4_model == "Intel/Qwen3.5-4B-int4-AutoRound"
    else:
        assert ps.policy == "" and not ps.lora_fast_path and not ps.lora_dual_stream
    assert ps.request_trace_dir == str(tmp_path / "run" / "traces") and ps.request_trace_log_tokens is True
    assert ps.force_shm_weight_transfer is True and ps.zmq_namespace.startswith("ps_qwen3_5_4b_g")
    assert model.lora_rank == 16 and "in_proj_qkv" in model.target_modules
    assert cfg.actor_rollout_ref.rollout.engine_kwargs.vllm.lora_target_modules == list(model.target_modules)
    assert cfg.data.apply_chat_template_kwargs.enable_thinking is True
    assert cfg.trainer.stable_sample_uid is True and cfg.trainer.ray_master_port_range == "47000:47299"
    assert cfg.trainer.default_local_dir == str(tmp_path / "run" / "checkpoints")
    assert cfg.trainer.rollout_data_dir == str(tmp_path / "run" / "rollouts")
    assert cfg.actor_rollout_ref.actor.strategy == trainer and cfg.trainer.n_gpus_per_node == 1
    if trainer == "megatron":
        # decision 10 (reversed 2026-09-16): Megatron-Bridge PEFT with the mcore target names.
        assert cfg.model_engine == "megatron"
        mg = cfg.actor_rollout_ref.actor.megatron
        assert mg.use_mbridge is True and mg.vanilla_mbridge is False
        assert mg.tensor_model_parallel_size == 1 and mg.pipeline_model_parallel_size == 1
        assert mg.override_transformer_config.recompute_granularity == "full"
        assert mg.override_transformer_config.recompute_method == "uniform"
        assert mg.override_transformer_config.recompute_num_layers == 1
        assert model.lora["rank"] == 16 and model.lora["merge"] is False
        assert "language_model.decoder.layers.*.self_attention.in_proj" in model.lora["target_modules"]
        assert cfg.actor_rollout_ref.ref.megatron.use_mbridge is True
    else:
        assert cfg.actor_rollout_ref.actor.fsdp_config.fsdp_size == 1
    if recipe == "rollout_only":
        assert cfg.trainer.rollout_only is True and cfg.trainer.rollout_only_steps == 1
        assert cfg.actor_rollout_ref.rollout.calculate_log_probs is False
    else:
        assert cfg.trainer.rollout_only is False
        assert cfg.actor_rollout_ref.rollout.calculate_log_probs is True
    if recipe == "train_arm":
        assert cfg.trainer.total_training_steps == 100 and cfg.trainer.save_freq == 10
        assert cfg.trainer.max_actor_ckpt_to_keep == 2


@pytest.mark.parametrize("model_key", MODELS)
def test_every_model_overlay_composes_with_the_driver(model_key, tmp_path):
    # Gemma4 has no Megatron-Bridge mapping on this branch and composes only under fsdp2 (CPU test only).
    trainer = "fsdp2" if model_key == "gemma4_e2b" else "megatron"
    overrides, _ = dry_run(RECIPES["full_step"], tmp_path, {"POLICY": "tail_t8", "MODEL_KEY": model_key, "TRAINER": trainer})
    cfg = compose_overrides(overrides)
    rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
    model = omega_conf_to_dataclass(cfg.actor_rollout_ref.model)
    assert rollout.precision_scheduler.int4_model
    if model_key == "phi4_mini_reasoning":
        assert rollout.load_format == "auto" and "qkv_proj" in model.target_modules
    if model_key == "gemma4_e2b":
        assert model.gemma4_dense_ffpa is True and model.exclude_modules
        assert rollout.precision_scheduler.int4_modules == "mlp_only"


def test_policy_json_and_local_paths(tmp_path):
    policy = _policy_file(tmp_path)
    overrides, _ = dry_run(
        RECIPES["rollout_only"],
        tmp_path,
        {
            "POLICY": str(policy),
            "MODEL_PATH": "/models/qwen",
            "INT4_MODEL_PATH": "/models/qwen-int4",
            "TOTAL_STEPS": "3",
        },
    )
    cfg = compose_overrides(overrides)
    ps = cfg.actor_rollout_ref.rollout.precision_scheduler
    assert ps.enable is True and ps.policy == str(policy) and ps.int4_model == "/models/qwen-int4"
    assert cfg.actor_rollout_ref.model.path == "/models/qwen"
    assert cfg.trainer.rollout_only_steps == 3 and cfg.trainer.total_training_steps == 3


EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
POLICY_NAMES = {
    "bf16": "bf16",
    "full_w4": "uniform_w4",
    "tail_t8": "fixed_threshold_8",
    "fixed_k8000": "fixed_frontier_8000",
}


def _experiment_name(overrides: list[str]) -> str:
    names = [o.split("=", 1)[1] for o in overrides if o.startswith("trainer.experiment_name=")]
    assert len(names) == 1, overrides
    return names[0]


@pytest.mark.parametrize("recipe", sorted(RECIPES))
@pytest.mark.parametrize("policy", sorted(POLICIES) + ["json"])
def test_experiment_name_is_derived_from_the_policy_kind(recipe, policy, tmp_path):
    """Integration defect 2: a JSON policy path put '/' into trainer.experiment_name and the FileLogger
    open() failed seven minutes into every .json launch. The name is the policy *kind* and its parameter,
    never the path, sanitized to [A-Za-z0-9_.-]."""
    if policy == "json":
        path = tmp_path / "nested dir" / "policies" / "dynamic_policy_rev30@x.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"schema_version": 6}')
        env_policy, expected = str(path), "ema_dynamic_policy_rev30_x"
    else:
        env_policy, expected = policy, POLICY_NAMES[policy]
    overrides, _ = dry_run(RECIPES[recipe], tmp_path, {"POLICY": env_policy, "MODEL_KEY": "qwen3_5_4b"})
    name = _experiment_name(overrides)
    assert EXPERIMENT_NAME_RE.match(name), name
    assert name == f"qwen3_5_4b_{expected}"
    cfg = compose_overrides(overrides)
    assert cfg.trainer.experiment_name == name


def test_experiment_name_override_is_kept_but_must_be_a_file_name(tmp_path):
    overrides, _ = dry_run(RECIPES["full_step"], tmp_path, {"POLICY": "bf16", "EXPERIMENT_NAME": "arm-A.v2"})
    assert _experiment_name(overrides) == "arm-A.v2"
    env = dict(os.environ, DRY_RUN="1", RUN_DIR=str(tmp_path), POLICY="bf16", PS_DATA_ROOT=str(tmp_path))
    env["EXPERIMENT_NAME"] = "a/b"
    proc = subprocess.run(["bash", str(RECIPES["full_step"])], env=env, capture_output=True, text=True)
    assert proc.returncode == 2 and "EXPERIMENT_NAME" in proc.stderr


def test_validate_shadow_can_be_switched_off(tmp_path):
    overrides, _ = dry_run(RECIPES["rollout_only"], tmp_path, {"POLICY": "full_w4", "VALIDATE_SHADOW": "0"})
    cfg = compose_overrides(overrides)
    ps = cfg.actor_rollout_ref.rollout.precision_scheduler
    assert ps.enable is True and ps.validate_shadow is False
    overrides, _ = dry_run(RECIPES["rollout_only"], tmp_path, {"POLICY": "bf16", "VALIDATE_SHADOW": "1"})
    assert not any(o.endswith("validate_shadow=true") for o in overrides), "bf16 keeps the vanilla block"


def test_extra_positional_overrides_win(tmp_path):
    overrides, _ = dry_run(
        RECIPES["full_step"], tmp_path, {"POLICY": "bf16"}, "trainer.total_training_steps=7", f"{PS}.enable=true"
    )
    cfg = compose_overrides(overrides)
    assert cfg.trainer.total_training_steps == 7
    assert cfg.actor_rollout_ref.rollout.precision_scheduler.enable is True


def test_unknown_policy_fails(tmp_path):
    env = dict(os.environ, DRY_RUN="1", RUN_DIR=str(tmp_path), POLICY="tail_tx", PS_DATA_ROOT=str(tmp_path))
    proc = subprocess.run(["bash", str(RECIPES["rollout_only"])], env=env, capture_output=True, text=True)
    assert proc.returncode == 2 and "bad POLICY" in proc.stderr


def test_recipes_export_no_project_env_vars():
    """Decision 9: no VLLM_* / VERL_* / ROLLOUT_QLORA exports anywhere in the recipes."""
    scripts = [p for p in EXAMPLES.rglob("*.sh")]
    assert scripts
    for path in scripts:
        for line in path.read_text().splitlines():
            code = line.split("#", 1)[0]
            assert "VLLM_" not in code, f"{path}: {line}"
            assert "VERL_" not in code, f"{path}: {line}"
            assert "ROLLOUT_QLORA" not in code, f"{path}: {line}"


# --- golden: archived run_config.json (trimmed fixture) vs dry-run contract ---------------------


def test_dry_run_matches_archived_run_config(tmp_path):
    """The archived eos_hazard_fullstep qwen35_4b/gsm8k/tail_t8 cell (B16 x 4, cap 16384) as a recipe."""
    archived = json.loads((FIXTURES / "run_config_qwen35_4b_gsm8k_tail_t8.json").read_text())
    env = {
        "POLICY": "tail_t8",
        "MODEL_KEY": "qwen3_5_4b",
        "TRAINER": archived["backend"],  # the archived extensibility cell was an FSDP2 run
        "TRAIN_BATCH_SIZE": str(archived["train_batch_size"]),
        "ROLLOUT_N": str(archived["rollout_n"]),
        "RESPONSE_CAP": str(archived["response_cap"]),
        "TOTAL_STEPS": str(archived["steps"]),
    }
    overrides, _ = dry_run(RECIPES["full_step"], tmp_path, env)
    cfg = compose_overrides(overrides)
    r = cfg.actor_rollout_ref.rollout
    assert cfg.data.train_batch_size == archived["train_batch_size"]
    assert r.n == archived["rollout_n"]
    assert r.max_num_seqs == archived["requests_per_step"]
    assert cfg.data.max_response_length == archived["response_cap"]
    assert r.max_model_len == archived["max_model_len"]
    assert r.temperature == archived["temperature"] and r.top_p == archived["top_p"] and r.top_k == archived["top_k"]
    assert r.precision_scheduler.policy == f"fixed_threshold:{archived['tail_threshold']}"
    assert r.precision_scheduler.reprefill is archived["reprefill"]
    assert r.precision_scheduler.int4_model.split("/")[-1].startswith("Qwen3.5-4B-int4-AutoRound")
    assert r.load_format == archived["rollout_load_format"]
    assert cfg.actor_rollout_ref.model.lora_rank == archived["lora_rank"]
    assert r.enforce_eager is archived["enforce_eager"]
    assert cfg.actor_rollout_ref.actor.strategy == archived["backend"]
    assert cfg.trainer.total_training_steps == archived["steps"]


# --- continuous_ema.sh fail-closed --------------------------------------------------------------

STUB_WATCHER = """\
import json, sys, time
from pathlib import Path
args = sys.argv[1:]
run = Path(args[args.index("--run-dir") + 1]); policy = Path(args[args.index("--policy") + 1])
final = int(args[args.index("--final") + 1]); delay = float(args[args.index("--delay") + 1])
if "--initialize-only" in args:
    policy.write_text('{"schema_version": 6, "policy_revision": 0}'); sys.exit(0)
time.sleep(delay)
(run / "online_ema_state.json").write_text(json.dumps({"completed_steps": final}))
"""
STUB_RUNNER = """\
import os, sys, time
from pathlib import Path
run = Path(os.environ["RUN_DIR"]); (run / "runner_started").write_text(os.environ.get("POLICY", ""))
(run / "runner_args").write_text("\\n".join(sys.argv[1:]))
keys = ("INITIAL_BATCH", "TRAIN_BATCH_SIZE", "ROLLOUT_N")
(run / "runner_batch").write_text(" ".join(os.environ.get(k, "-") for k in keys))
time.sleep(float(sys.argv[1]))
(run / "runner_done").write_text("ok")
"""


def _ema_env(tmp_path: Path, *, final: int, delay: float, runner_sleep: float, steps: int) -> dict[str, str]:
    (tmp_path / "watcher.py").write_text(STUB_WATCHER)
    (tmp_path / "runner.py").write_text(STUB_RUNNER)
    run = tmp_path / "run"
    return dict(
        os.environ,
        RUN_DIR=str(run),
        TOTAL_STEPS=str(steps),
        POLL_SECONDS="0.2",
        WATCHER_CMD=f"python {tmp_path / 'watcher.py'} --run-dir {run} --policy {run / 'policy.json'}"
        f" --final {final} --delay {delay}",
        RUNNER_CMD=f"python {tmp_path / 'runner.py'} {runner_sleep}",
    )


def test_continuous_ema_fails_closed_when_watcher_dies_early(tmp_path):
    env = _ema_env(tmp_path, final=1, delay=0.5, runner_sleep=30, steps=3)
    t0 = time.time()
    proc = subprocess.run(
        ["bash", str(EXAMPLES / "recipes" / "continuous_ema.sh")], env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "watcher exited at revision 1/3" in proc.stderr
    assert time.time() - t0 < 25, "runner was not killed"
    assert not subprocess.run(["pgrep", "-f", str(tmp_path / "runner.py")], capture_output=True).stdout.strip()
    run = tmp_path / "run"
    assert (run / "runner_started").exists() and not (run / "runner_done").exists()
    assert (run / "FAILED").read_text().startswith("watcher_exited_early")
    assert not (run / "CONTINUOUS_EMA_COMPLETE").exists()


def test_continuous_ema_completes_when_watcher_consumes_every_step(tmp_path):
    env = _ema_env(tmp_path, final=3, delay=0.3, runner_sleep=1.5, steps=3)
    proc = subprocess.run(
        ["bash", str(EXAMPLES / "recipes" / "continuous_ema.sh")], env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    run = tmp_path / "run"
    assert (run / "CONTINUOUS_EMA_COMPLETE").exists() and (run / "runner_done").exists()
    assert (run / "runner_started").read_text() == str(run / "policy.json")
    args = (run / "runner_args").read_text().splitlines()
    assert f"{PS}.reload_policy_each_rollout=true" in args and f"{PS}.policy_barrier_timeout_s=600" in args
    # Integration defect 5: the full_step runner reads TRAIN_BATCH_SIZE, not INITIAL_BATCH; the recipe hands
    # the runner the derived TRAIN_BATCH_SIZE so the watcher's --batch and the rollout batch agree.
    assert (run / "runner_batch").read_text() == "64 16 4"


def test_continuous_ema_derives_train_batch_size_from_initial_batch(tmp_path):
    env = _ema_env(tmp_path, final=2, delay=0.3, runner_sleep=1.0, steps=2)
    env.update(INITIAL_BATCH="48", ROLLOUT_N="8")
    proc = subprocess.run(
        ["bash", str(EXAMPLES / "recipes" / "continuous_ema.sh")], env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "run" / "runner_batch").read_text() == "48 6 8"


@pytest.mark.parametrize(
    "env_extra", [{"INITIAL_BATCH": "30", "ROLLOUT_N": "4"}, {"INITIAL_BATCH": "64", "TRAIN_BATCH_SIZE": "8"}]
)
def test_continuous_ema_refuses_an_inconsistent_batch(tmp_path, env_extra):
    env = _ema_env(tmp_path, final=2, delay=0.3, runner_sleep=1.0, steps=2)
    env.update(env_extra)
    proc = subprocess.run(
        ["bash", str(EXAMPLES / "recipes" / "continuous_ema.sh")], env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "INITIAL_BATCH" in proc.stderr
    assert not (tmp_path / "run" / "runner_started").exists()


def test_continuous_ema_dry_run_uses_the_c6_watcher(tmp_path):
    run = tmp_path / "run"
    env = dict(
        os.environ,
        DRY_RUN="1",
        RUN_DIR=str(run),
        TOTAL_STEPS="5",
        INITIAL_BATCH="32",
        RESPONSE_CAP="16384",
        BF_TRACE="/x/bf.jsonl",
        W4_TRACE="/x/w4.jsonl",
        HEATMAP="/x/heatmap.json",
        PYTHON_BIN="python",
        PS_DATA_ROOT=str(tmp_path / "data"),
    )
    proc = subprocess.run(
        ["bash", str(EXAMPLES / "recipes" / "continuous_ema.sh")], env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    watcher = next(line for line in proc.stdout.splitlines() if line.startswith("WATCHER\t")).split("\t", 1)[1]
    assert "verl.experimental.precision_scheduler.cli watch-ema" in watcher
    assert (
        f"--policy {run / 'policy.json'}" in watcher and "--steps 5" in watcher and "--batch 32 --cap 16384" in watcher
    )
    overrides = [line.split("\t", 1)[1] for line in proc.stdout.splitlines() if line.startswith("OVERRIDE\t")]
    cfg = compose_overrides(overrides)
    ps = cfg.actor_rollout_ref.rollout.precision_scheduler
    assert ps.reload_policy_each_rollout is True and ps.policy == str(run / "policy.json")
    assert cfg.trainer.rollout_only is True and cfg.trainer.rollout_only_steps == 5
    assert cfg.data.train_batch_size == 8 and cfg.actor_rollout_ref.rollout.max_num_seqs == 32
    assert ps.policy_barrier_timeout_s == 600
    assert cfg.trainer.experiment_name == "qwen3_5_4b_ema_policy"
    # RUNNER=full_step reads TRAIN_BATCH_SIZE (defect 5): the derived value keeps 32 = 8 x 4 for both.
    proc = subprocess.run(
        ["bash", str(EXAMPLES / "recipes" / "continuous_ema.sh")],
        env=dict(env, RUNNER="full_step"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--batch 32 --cap 16384" in proc.stdout
    overrides = [line.split("\t", 1)[1] for line in proc.stdout.splitlines() if line.startswith("OVERRIDE\t")]
    cfg = compose_overrides(overrides)
    assert cfg.trainer.rollout_only is False
    assert cfg.data.train_batch_size == 8 and cfg.actor_rollout_ref.rollout.max_num_seqs == 32


def test_data_root_is_required(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("PS_DATA_ROOT", "DATA_DIR")}
    env.update(DRY_RUN="1", RUN_DIR=str(tmp_path), POLICY="bf16")
    proc = subprocess.run(["bash", str(RECIPES["rollout_only"])], env=env, capture_output=True, text=True)
    assert proc.returncode == 2 and "PS_DATA_ROOT" in proc.stderr


def test_launch_refuses_without_launcher_gpu(tmp_path):
    """Decision 13: a real launch needs CUDA_VISIBLE_DEVICES from run_gpu.sh (any GPU id is allowed)."""
    base = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "DRY_RUN")}
    base.update(RUN_DIR=str(tmp_path / "r"), POLICY="bf16", PS_DATA_ROOT=str(tmp_path), PYTHON_BIN="python")
    proc = subprocess.run(["bash", str(RECIPES["rollout_only"])], env=base, capture_output=True, text=True)
    assert proc.returncode == 2 and "run_gpu.sh" in proc.stderr
    assert not (tmp_path / "r").exists(), "the run directory is not created before the preflight"
