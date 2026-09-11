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
"""Rewards (unit + golden re-scoring of archived dumps) and the long-run summarizer golden.

Golden tiers skip when the archive under /data/huanchen/verl/.codex-report is absent.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples" / "precision_scheduler"
ARCHIVE = Path("/data/huanchen/verl/.codex-report/new-storyline-experiments")
NO_REPREFILL = ARCHIVE / "no_reprefill_rl_100step_20260814"
HARDMATH = ARCHIVE / "hardmath_lora_accuracy_100step_20260825"
EOS_FULLSTEP = ARCHIVE / "eos_hazard_fullstep_b64_cap16k"
LEARNABILITY = ARCHIVE / "bf16_learnability_search_20260827"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rewards():
    return _load("ps_rewards", EXAMPLES / "rewards.py")


# --- unit ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output,gold,expected",
    [
        ("The answer is 1,250.", "1250", 1.0),
        ("So the total is $250.", "250", 1.0),
        ("#### 250", "250", 1.0),
        ("Answer: 250.", "250", 1.0),
        ("I think 251", "250", 0.0),
        ("no digits here", "250", 0.0),
        ("x" * 3000 + " 7 " + "y" * 2000, "7", 0.0),  # clipped to the last 2000 characters
        ("" + " 7 " + "y" * 1990, "7", 1.0),
    ],
)
def test_gsm8k_last_number(rewards, output, gold, expected):
    result = rewards.compute_score("openai/gsm8k", output, gold)
    assert result == {"score": expected, "accuracy": expected}


@pytest.mark.parametrize(
    "output,gold,expected",
    [
        ("<think>x</think><answer>16</answer>", "16", 1.0),
        ("<think>x</think><answer>17</answer>", "16", 0.0),
        (r"Thus \boxed{\frac{1}{2}}.", "1/2", 1.0),
    ],
)
def test_bigmath_math_verify_cases(rewards, output, gold, expected):
    """The archived test_bigmath_accuracy_reward.py cases."""
    result = rewards.compute_score("bigmath_math_verify", output, gold)
    assert result["score"] == expected and result["accuracy"] == expected


def test_search_format_bonus(rewards):
    with_tags = rewards.compute_score("bigmath_qerl_search", "<think>x</think><answer>16</answer>", "16")
    assert with_tags == {"score": 1.1, "accuracy": 1.0, "format": 0.1}
    no_think = rewards.compute_score("bigmath_qerl_search", "<answer>16</answer>", "16")
    assert no_think == {"score": 1.0, "accuracy": 1.0, "format": 0.0}


def test_eos_hazard_substring_and_unknown(rewards):
    assert rewards.compute_score("eos_hazard/mmlu_pro", "... the answer is (B) ...", "b") == {
        "score": 1.0,
        "accuracy": 1.0,
    }
    assert rewards.compute_score("eos_hazard/hotpotqa", "nothing", "Paris")["score"] == 0.0
    with pytest.raises(ValueError):
        rewards.compute_score("unknown/source", "x", "y")


# --- golden: re-score archived rollout dumps ------------------------------------------------

DUMPS = [
    ("openai/gsm8k", NO_REPREFILL / "runs" / "b64" / "tail_t8" / "rollouts" / "1.jsonl", 256),
    ("bigmath_math_verify", HARDMATH / "runs" / "train_pure_w4_exact1_auto_gate" / "rollouts" / "1.jsonl", 24),
    ("bigmath_math_verify", EOS_FULLSTEP / "runs/phi4_mini_reasoning/math500/bf16/main/rollouts/1.jsonl", 24),
    ("bigmath_qerl_search", LEARNABILITY / "runs" / "full_lr3_n8_nokl" / "rollouts" / "1.jsonl", 24),
]


@pytest.mark.parametrize("data_source,path,limit", DUMPS, ids=[str(p.relative_to(ARCHIVE)) for _, p, _ in DUMPS])
def test_rescoring_archived_dumps(rewards, data_source, path, limit):
    if not path.exists():
        pytest.skip(f"archive absent: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][:limit]
    assert rows
    for row in rows:
        result = rewards.compute_score(data_source, row["output"], row["gts"])
        assert math.isclose(result["score"], float(row["score"]), abs_tol=1e-6), (row["uid"], result, row["score"])


# --- golden: summarizer vs archived efficiency summaries ------------------------------------


def _run_summarizer(tmp_path: Path, phase: str, required: str) -> dict:
    out = tmp_path / f"{phase}.json"
    cmd = [
        sys.executable,
        str(EXAMPLES / "long_run" / "summarize_runs.py"),
        "--runs",
        str(NO_REPREFILL / "runs"),
        "--glob",
        "*/*",
        "--config-id",
        "b{batch_size}/{policy}",
        "--timed-steps",
        "2:11",
        "--required-steps",
        required,
        "--phase",
        phase,
        "--output",
        str(out),
        "--allow-incomplete",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(out.read_text())


@pytest.mark.parametrize(
    "phase,archived,required",
    [
        ("efficiency", "efficiency_summary.json", "*=11"),
        ("final", "efficiency_accuracy_summary.json", "b128/bf16=100,b128/full_w4=100,b128/tail_t8=100,*=11"),
    ],
)
def test_summarizer_reproduces_archived_summary(tmp_path, phase, archived, required):
    golden_path = NO_REPREFILL / archived
    if not golden_path.exists():
        pytest.skip(f"archive absent: {golden_path}")
    golden = json.loads(golden_path.read_text())
    ours = _run_summarizer(tmp_path, phase, required)
    assert ours["found_configurations"] == golden["found_configurations"] == 15
    mine = {c["config_id"]: c for c in ours["configurations"]}
    for theirs in golden["configurations"]:
        c = mine[theirs["config_id"]]
        assert c["required_steps"] == theirs["required_steps"]
        assert c["efficiency_window"]["timed_steps"] == theirs["efficiency_window"]["timed_steps"]
        assert c["efficiency_window"]["aggregates"] == theirs["efficiency_window"]["aggregates"], theirs["config_id"]
        # The archived summaries were generated while the three B128 runs were still training; the
        # per-step curve must agree on the archived prefix and the summary whenever the step count does.
        n = len(theirs["training_accuracy_reward"]["per_step"])
        assert c["training_accuracy_reward"]["per_step"][:n] == theirs["training_accuracy_reward"]["per_step"]
        if c["steps_available"] == theirs["steps_available"]:
            assert c["training_accuracy_reward"]["summary"] == theirs["training_accuracy_reward"]["summary"]
        else:
            assert c["steps_available"] > theirs["steps_available"]
