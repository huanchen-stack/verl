#!/usr/bin/env python3
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
"""Summarize a set of long runs: timing aggregates over a warm-up-excluded window, per-step reward
and response-length curves, completeness report.

Generalized from the archived ``summarize_no_reprefill_rl.py``: the study constants (config id
format, required steps per configuration, timed window, controls) are arguments. Output schema is
the archived one (``schema_version 1``) so the archived ``efficiency_summary.json`` is a golden.

    summarize_runs.py --runs <study>/runs --glob '*/*' --config-id 'b{batch_size}/{policy}' \
        --timed-steps 2:11 --required-steps 'b128/bf16=100,b128/full_w4=100,b128/tail_t8=100,*=11' \
        --output efficiency_summary.json [--allow-incomplete]

Each run directory holds ``run_config.json`` and FileLogger rows under ``metrics/**/*.jsonl``
(``{"step": k, "data": {...}}``); a later row for the same step wins (resumed runs).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

TIMING_KEYS = (
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "timing_s/step",
    "perf/throughput",
    "perf/total_num_tokens",
)
REWARD_KEYS = ("critic/score/mean", "critic/rewards/mean")


def finite(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def aggregates(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def load_steps(run_dir: Path) -> tuple[dict[int, dict], list[str]]:
    steps: dict[int, dict] = {}
    files = sorted(run_dir.glob("metrics/**/*.jsonl"))
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                step = int(row["step"])
                data = dict(row["data"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            steps[step] = data
    return steps, [str(path) for path in files]


def parse_range(text: str) -> list[int]:
    lo, hi = (int(x) for x in text.split(":"))
    return list(range(lo, hi + 1))


def parse_required(text: str) -> list[tuple[str, int]]:
    """``'b128/bf16=100,*=11'`` -> ordered (glob, steps) pairs; first match wins."""
    pairs = []
    for item in text.split(","):
        if not item.strip():
            continue
        pattern, steps = item.split("=")
        pairs.append((pattern.strip(), int(steps)))
    return pairs


def required_for(config_id: str, rules: list[tuple[str, int]], default: int) -> int:
    for pattern, steps in rules:
        if fnmatch.fnmatchcase(config_id, pattern):
            return steps
    return default


def summarize_run(
    run_dir: Path, config: dict, config_id: str, required: int, timed_steps: list[int], warmup_steps: list[int]
) -> tuple[dict, dict | None]:
    steps, metric_files = load_steps(run_dir)
    missing = [step for step in range(1, required + 1) if step not in steps]
    incomplete = {"config_id": config_id, "required": required, "max_step": max(steps, default=0)} if missing else None

    timed = [(step, steps[step]) for step in timed_steps if step in steps]
    timing_summary: dict[str, dict] = {}
    for key in TIMING_KEYS:
        values = [value for _, data in timed if (value := finite(data.get(key))) is not None]
        if values:
            timing_summary[key] = aggregates(values)
    adjusted = []
    for _, data in timed:
        step_time = finite(data.get("timing_s/step"))
        save_time = finite(data.get("timing_s/save_checkpoint")) or 0.0
        if step_time is not None:
            adjusted.append(step_time - save_time)
    if adjusted:
        timing_summary["timing_s/step_excluding_checkpoint"] = aggregates(adjusted)

    reward_steps = []
    for step in sorted(steps):
        data = steps[step]
        reward = None
        for key in REWARD_KEYS:
            reward = finite(data.get(key))
            if reward is not None:
                break
        reward_steps.append(
            {
                "step": step,
                "accuracy_reward": reward,
                "response_length_mean": finite(data.get("response_length/mean")),
                "response_length_max": finite(data.get("response_length/max")),
                "response_length_clip_ratio": finite(data.get("response_length/clip_ratio")),
                "total_tokens": finite(data.get("perf/total_num_tokens")),
                "step_seconds": finite(data.get("timing_s/step")),
            }
        )
    rewards = [row["accuracy_reward"] for row in reward_steps if row["accuracy_reward"] is not None]
    reward_summary = None
    if rewards:
        reward_summary = {
            **aggregates(rewards),
            "last_10_mean": statistics.fmean(rewards[-10:]),
            "best": max(rewards),
            "area_under_step_curve": sum(rewards),
        }
    payload = {
        "config_id": config_id,
        "config": config,
        "required_steps": required,
        "steps_available": len(steps),
        "max_step": max(steps, default=0),
        "efficiency_window": {
            "warmup_step": warmup_steps[0] if len(warmup_steps) == 1 else warmup_steps,
            "timed_steps": timed_steps,
            "available_timed_steps": [step for step, _ in timed],
            "aggregates": timing_summary,
        },
        "training_accuracy_reward": {
            "semantics": "rule-based accuracy reward on online training rollouts; not held-out accuracy",
            "summary": reward_summary,
            "per_step": reward_steps,
        },
        "metric_files": metric_files,
        "complete_marker": (run_dir / "COMPLETE").exists(),
    }
    return payload, incomplete


def summarize(args: argparse.Namespace) -> dict:
    runs_root = Path(args.runs)
    run_dirs = sorted(path.parent for path in runs_root.glob(f"{args.glob}/run_config.json"))
    rules = parse_required(args.required_steps)
    timed_steps = parse_range(args.timed_steps)
    warmup_steps = parse_range(args.warmup_steps)
    configs, incomplete = [], []
    for run_dir in run_dirs:
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        try:
            config_id = args.config_id.format(**config)
        except KeyError:
            config_id = str(run_dir.relative_to(runs_root))
        required = required_for(config_id, rules, args.default_required_steps)
        payload, missing = summarize_run(run_dir, config, config_id, required, timed_steps, warmup_steps)
        configs.append(payload)
        if missing:
            incomplete.append(missing)
    expected = args.expected_configurations or len(run_dirs)
    complete = len(run_dirs) == expected and not incomplete
    return {
        "schema_version": 1,
        "study": args.study,
        "phase": args.phase,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "expected_configurations": expected,
        "found_configurations": len(run_dirs),
        "incomplete": incomplete,
        "controls": {
            **(json.loads(args.controls) if args.controls else {}),
            "efficiency_warmup_steps": warmup_steps,
            "efficiency_timed_steps": timed_steps,
        },
        "configurations": configs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True, help="runs root directory")
    parser.add_argument(
        "--glob", default="*", help="glob (relative to --runs) of run directories holding run_config.json"
    )
    parser.add_argument("--config-id", default="{policy}", help="format string over run_config.json keys")
    parser.add_argument("--warmup-steps", default="1:1", help="lo:hi steps excluded from the timing window")
    parser.add_argument("--timed-steps", default="2:11", help="lo:hi steps of the timing window")
    parser.add_argument("--required-steps", default="", help="comma list of <config-id glob>=<steps>; first match wins")
    parser.add_argument("--default-required-steps", type=int, default=11)
    parser.add_argument("--expected-configurations", type=int, default=0, help="0 = number of runs found")
    parser.add_argument("--study", default="precision scheduler long runs")
    parser.add_argument("--phase", default="efficiency")
    parser.add_argument("--controls", default="", help="JSON object recorded under controls")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = summarize(args)
    if not payload["complete"] and not args.allow_incomplete:
        raise SystemExit(f"summary incomplete: {payload['incomplete']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps({"output": str(args.output), "complete": payload["complete"], "incomplete": payload["incomplete"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
