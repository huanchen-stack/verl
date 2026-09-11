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
"""One reward entry point for every precision-scheduler dataset, dispatching on ``data_source``.

Merges the four archived reward files (``gsm8k_flexible_reward.py``, ``universal_reward.py``,
``bigmath_accuracy_reward.py``, ``qerl_style_reward.py``); the scoring rules are byte-for-byte the
archived ones so re-scoring an archived rollout dump reproduces the stored rewards:

* ``openai/gsm8k``            last number of the final 2000 characters, Decimal-equal to the gold
                              (commas stripped);
* ``bigmath_math_verify``     last ``<answer>...</answer>`` re-boxed as ``\\boxed{}`` (or the raw text)
                              through ``verl.utils.reward_score.math_verify``; binary. The archived
                              ``universal_reward.py`` (eos math500 / bigmath_hard) passed the raw string
                              without re-boxing; both give the same score on every archived dump;
* ``bigmath_qerl_search`` /   same accuracy plus a 0.1 format bonus when an answer tag and
  ``clean_bigmath_learnability`` ``</think>`` are present (the BF16 learnability searches; the archived
                              20260904 search crashed on its own ``clean_bigmath_learnability`` source);
* ``eos_hazard/<workload>``   case-insensitive substring match of the gold in the last 2000 chars
                              (the non-math extensibility workloads).

Every branch returns ``{"score", "accuracy"}`` (the search branch adds ``"format"``). Use it as
``reward.custom_reward_function.path=examples/precision_scheduler/rewards.py`` with ``name=compute_score``.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

NUMBER_RE = re.compile(r"-?(?:\d[\d,]*\.?\d*|\.\d+)")
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
TAIL_CHARS = 2000
FORMAT_BONUS = 0.1
MATH_VERIFY_TIMEOUT = 30.0

GSM8K = "openai/gsm8k"
BIGMATH = "bigmath_math_verify"
BIGMATH_SEARCH = ("bigmath_qerl_search", "clean_bigmath_learnability")
EOS_HAZARD_PREFIX = "eos_hazard/"


def _number(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None


def gsm8k_accuracy(solution_str: str, ground_truth: str) -> float:
    """Score the last numeric answer, independent of an artificial format tag."""
    matches = NUMBER_RE.findall(solution_str[-TAIL_CHARS:])
    predicted = _number(matches[-1]) if matches else None
    expected = _number(str(ground_truth))
    return float(predicted is not None and expected is not None and predicted == expected)


def bigmath_verifier_input(solution_str: str) -> tuple[str, bool]:
    """Math-Verify does not treat ``<answer>`` tags as an extraction target; re-box the last one."""
    answers = ANSWER_RE.findall(solution_str)
    if answers:
        return rf"\boxed{{{answers[-1].strip()}}}", True
    return solution_str, False


def bigmath_accuracy(solution_str: str, ground_truth: str) -> tuple[float, bool]:
    from verl.utils.reward_score.math_verify import compute_score as verify_math

    verifier_input, has_answer_tag = bigmath_verifier_input(solution_str)
    accuracy = float(verify_math(verifier_input, str(ground_truth), timeout_score=0, timeout=MATH_VERIFY_TIMEOUT))
    return accuracy, has_answer_tag


def substring_accuracy(solution_str: str, ground_truth: str) -> float:
    return float(str(ground_truth).strip().lower() in solution_str[-TAIL_CHARS:].lower())


def compute_score(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None
) -> dict[str, float]:
    del extra_info
    if data_source == GSM8K:
        score = gsm8k_accuracy(solution_str, ground_truth)
        return {"score": score, "accuracy": score}
    if data_source == BIGMATH:
        accuracy, _ = bigmath_accuracy(solution_str, ground_truth)
        return {"score": accuracy, "accuracy": accuracy}
    if data_source in BIGMATH_SEARCH:
        accuracy, has_answer_tag = bigmath_accuracy(solution_str, ground_truth)
        format_reward = FORMAT_BONUS if has_answer_tag and "</think>" in solution_str.lower() else 0.0
        return {"score": accuracy + format_reward, "accuracy": accuracy, "format": format_reward}
    if data_source.startswith(EOS_HAZARD_PREFIX):
        score = substring_accuracy(solution_str, ground_truth)
        return {"score": score, "accuracy": score}
    raise ValueError(f"Unsupported data source: {data_source}")
