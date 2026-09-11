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
"""gpu-smoke: the two C10 recipe smokes (Qwen3.5-4B, GSM8K, precision_scheduler.enable=false).

Each test launches its recipe through the decision-13 launcher (``run_gpu.sh``) on the GPU named by
``PS_SMOKE_GPU`` (default 4) with a 40-minute budget, then validates the run directory with
``tools/validate_rollout_run.py``:

* ``rollout_only.sh`` 2 steps, 4 prompts x 4 samples on an 8-row parquet (``PS_SMOKE_DATA_DIR``);
* ``full_step.sh`` 1 step + ``trainer.save_initial_checkpoint=true`` (global_step_0 and global_step_1),
  then ``long_run/evaluate_lora_patch.py`` on the step-0 checkpoint (adapter export from the FSDP shard).

Run:  PS_SMOKE_GPU=4 pytest -m gpu_smoke tests/precision_scheduler/gpu/test_recipes_gpu_smoke.py
Requires the local Qwen3.5-4B snapshot (``PS_SMOKE_MODEL``) and a prepared 8-row GSM8K parquet
(``examples/precision_scheduler/data/prepare_gsm8k.py --train-size 8 --test-size 8``). Recorded passing runs:
docs/precision_scheduler/recipes_and_evaluation.md ("Smoke results").
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
EXAMPLES = REPO / "examples" / "precision_scheduler"
RUN_GPU = REPO / "scripts" / "precision_scheduler" / "env" / "run_gpu.sh"
VALIDATOR = REPO / "tools" / "validate_rollout_run.py"
MODEL = os.environ.get(
    "PS_SMOKE_MODEL",
    "/data/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
)
DATA_DIR = os.environ.get("PS_SMOKE_DATA_DIR", "/data/huanchen/ps_data/gsm8k_smoke_8")
GPU = os.environ.get("PS_SMOKE_GPU", "4")
TIMEOUT = 2400

pytestmark = pytest.mark.gpu_smoke


def _require_inputs():
    if not Path(MODEL).exists():
        pytest.skip(f"model snapshot absent: {MODEL}")
    if not (Path(DATA_DIR) / "train.parquet").exists():
        pytest.skip(f"smoke parquet absent: {DATA_DIR}")


def _launch(recipe: str, run_dir: Path, env_extra: dict[str, str], *overrides: str) -> None:
    env = dict(os.environ)
    env.update(
        {
            "RUN_DIR": str(run_dir),
            "MODEL_KEY": "qwen3_5_4b",
            "MODEL_PATH": MODEL,
            "POLICY": "bf16",
            "DATA_DIR": DATA_DIR,
            "TRAIN_BATCH_SIZE": "4",
            "ROLLOUT_N": "4",
            "PROMPT_CAP": "1024",
            "RUN_TIMEOUT": "35m",
        }
    )
    env.update(env_extra)
    cmd = [str(RUN_GPU), "--gpus", GPU, "--timeout", str(TIMEOUT), "--", "bash", str(EXAMPLES / recipe), *overrides]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=TIMEOUT + 120)
    assert proc.returncode == 0, f"rc={proc.returncode}\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"


def _validate(run_dir: Path, *args: str) -> dict:
    proc = subprocess.run([sys.executable, str(VALIDATOR), str(run_dir), *args], capture_output=True, text=True)
    summary = json.loads(proc.stdout)
    assert proc.returncode == 0 and summary["valid"], summary["errors"]
    return summary


def test_rollout_only_two_steps(tmp_path):
    _require_inputs()
    run_dir = tmp_path / "rollout_only"
    _launch("recipes/rollout_only.sh", run_dir, {"TOTAL_STEPS": "2", "RESPONSE_CAP": "2048"})
    summary = _validate(run_dir, "--expected-requests", "16", "--steps", "2", "--require-complete")
    assert summary["request_count"] == 32 and len(summary["rollout_seconds"]) == 2


def test_full_step_initial_checkpoint_and_evaluator(tmp_path):
    _require_inputs()
    run_dir = tmp_path / "full_step"
    _launch(
        "recipes/full_step.sh",
        run_dir,
        {"TOTAL_STEPS": "1", "SAVE_FREQ": "1", "RESPONSE_CAP": "1024", "PPO_MAX_TOKEN_LEN": "4096"},
        "trainer.save_initial_checkpoint=true",
    )
    _validate(run_dir, "--expected-requests", "16", "--steps", "1", "--mode", "full_step", "--require-complete")
    log = (run_dir / "driver.log").read_text(errors="replace")
    assert "VERL_INITIAL_CHECKPOINT_COMPLETE step=0" in log
    for step in (0, 1):
        assert (run_dir / "checkpoints" / f"global_step_{step}" / "actor" / "lora_train_meta.json").exists()
    out = tmp_path / "eval" / "bf16_step0.jsonl"
    cmd = [
        str(RUN_GPU),
        "--gpus",
        GPU,
        "--timeout",
        "1500",
        "--",
        sys.executable,
        str(EXAMPLES / "long_run" / "evaluate_lora_patch.py"),
        "--base-model",
        MODEL,
        "--data",
        f"{DATA_DIR}/test.parquet",
        "--adapter",
        str(run_dir / "checkpoints" / "global_step_0"),
        "--checkpoint-step",
        "0",
        "--configuration",
        "bf16_smoke",
        "--max-response-tokens",
        "256",
        "--max-prompt-tokens",
        "1024",
        "--limit",
        "8",
        "--gpu-memory-utilization",
        "0.5",
        "--enforce-eager",
        "--output",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1700)
    assert proc.returncode == 0, proc.stderr[-4000:]
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["requests"] == 8 and 0.0 <= summary["wilson_95_low"] <= summary["wilson_95_high"] <= 1.0
    adapter = run_dir / "checkpoints" / "global_step_0" / "actor" / "lora_adapter"
    assert (adapter / "adapter_config.json").exists() and (adapter / "adapter_model.safetensors").exists()
