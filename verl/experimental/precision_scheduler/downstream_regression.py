# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Downstream (non-rollout) step time is linear in sampled tokens: one fit, three archived variants.

* :func:`fit_points` - ``np.polyfit`` degree 1 with r2 / MAPE / RMSE (the replay-regression fit).
* :func:`fit_split` - inference and training seconds against sampled response tokens
  (the 120-run temporal-guard regression).
* :func:`fit_from_metrics` - ``timing_s/step - timing_s/gen`` against ``perf/total_num_tokens`` over
  pure-precision runs, solved with ``np.linalg.lstsq`` (the validation-report fit, intercept 12.4814 s,
  slope 0.00087839 s/token on Qwen3.5-9B Megatron).
* :func:`slope_from_metrics` - the per-model slope the EMA lane passes as ``--downstream-slope``.

Every variant returns plain dictionaries so the numbers can be recorded in the policy JSON.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot else 1.0


def fit_points(x, y) -> dict[str, Any]:
    """Degree-1 polyfit with residual statistics (``pred`` is the fitted vector)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("fit_points needs two equally long 1-D arrays with at least two points")
    coef = np.polyfit(x, y, 1)
    pred = np.polyval(coef, x)
    resid = y - pred
    return {
        "slope_s_per_token": float(coef[0]),
        "intercept_s": float(coef[1]),
        "r2": _r2(y, pred),
        "mape_pct": float(np.mean(np.abs(resid) / np.maximum(y, 1e-9)) * 100),
        "rmse_s": float(np.sqrt(np.mean(resid**2))),
        "pred": pred,
    }


def fit_split(points: list[dict[str, Any]]) -> dict[str, Any]:
    """120-run regression: inference (old-logprob + ref) and training (update) seconds vs sampled tokens."""
    x = np.asarray([p["sampled_response_tokens"] for p in points], dtype=float)
    result: dict[str, Any] = {"n": len(points), "predictor": "sampled_response_tokens"}
    for name, field in (
        ("inference", "inference_seconds_measured_fit_input"),
        ("training", "training_seconds_measured_fit_input"),
    ):
        y = np.asarray([p[field] for p in points], dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        result[name] = {
            "slope_seconds_per_sampled_token": float(slope),
            "intercept_seconds": float(intercept),
            "r2": _r2(y, pred),
        }
    return result


def read_points_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in csv.DictReader(Path(path).open(newline="")):
        converted = {}
        for key, value in row.items():
            try:
                converted[key] = int(value)
            except ValueError:
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
        rows.append(converted)
    return rows


def fit_replay_points(
    rows: list[dict[str, Any]], *, min_points: int = 8
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Megatron replay regression over measured (non-warmup) points; returns ``(summary, annotated rows)``."""
    rows = [dict(r) for r in rows if not r.get("warmup")]
    if len(rows) < min_points:
        raise RuntimeError(f"Need at least {min_points} measured points, found {len(rows)}")
    x = np.array([r["total_tokens"] for r in rows], dtype=float)
    inference = fit_points(x, np.array([r["inference_s"] for r in rows]))
    training = fit_points(x, np.array([r["training_s"] for r in rows]))
    for i, row in enumerate(rows):
        row["pred_inference_s"] = float(inference["pred"][i])
        row["pred_training_s"] = float(training["pred"][i])
        row["inference_residual_pct"] = 100 * (row["inference_s"] / row["pred_inference_s"] - 1)
        row["training_residual_pct"] = 100 * (row["training_s"] / row["pred_training_s"] - 1)
    batch_sizes = sorted({int(r["batch_size"]) for r in rows})
    summary = {
        "n": len(rows),
        "token_range": [int(x.min()), int(x.max())],
        "batch_sizes": batch_sizes,
        "inference": {k: v for k, v in inference.items() if k != "pred"},
        "training": {k: v for k, v in training.items() if k != "pred"},
        "by_batch": {
            str(b): {
                "n": len(subset),
                "inference_mape_pct": float(np.mean([abs(r["inference_residual_pct"]) for r in subset])),
                "training_mape_pct": float(np.mean([abs(r["training_residual_pct"]) for r in subset])),
            }
            for b in batch_sizes
            for subset in [[r for r in rows if int(r["batch_size"]) == b]]
        },
        "definition": (
            "inference=actor old-logprob with entropy + reference logprob; training=PPO actor update; "
            "predictor=total prompt+response tokens"
        ),
    }
    return summary, rows


def load_replay_runs(paths: list[Path]) -> list[dict[str, Any]]:
    """Rows of the replay worker's per-GPU JSON outputs, tagged with the source file stem."""
    import json

    rows = []
    for path in paths:
        for row in json.loads(Path(path).read_text()):
            row = dict(row)
            row["gpu"] = int(path.stem[-1]) if path.stem[-1].isdigit() else path.stem
            rows.append(row)
    return rows


def downstream_xy(metrics: list[dict[str, Any]], *, steps: set[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(perf/total_num_tokens, timing_s/step - timing_s/gen)`` for metric rows, optionally filtered by step."""
    xs, ys = [], []
    for row in metrics:
        data = row.get("data", row)
        if steps is not None and int(row.get("step", -1)) not in steps:
            continue
        try:
            xs.append(float(data["perf/total_num_tokens"]))
            ys.append(float(data["timing_s/step"]) - float(data["timing_s/gen"]))
        except (KeyError, TypeError, ValueError):
            continue
    return np.asarray(xs), np.asarray(ys)


def fit_from_metrics(metrics: list[dict[str, Any]], *, steps: set[int] | None = None) -> dict[str, float]:
    """Validation-report fit (``lstsq`` with an intercept column) of downstream seconds vs total tokens."""
    x, y = downstream_xy(metrics, steps=steps)
    if len(x) < 3:
        raise ValueError("fit_from_metrics needs at least three metric rows")
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    return {"intercept_seconds": float(coef[0]), "seconds_per_token": float(coef[1]), "r2": _r2(y, pred), "n": len(x)}


def slope_from_metrics(metrics: list[dict[str, Any]]) -> tuple[float, int]:
    """Per-model downstream slope as computed inline by the EMA lane (``polyfit``, clipped at zero)."""
    x, y = downstream_xy(metrics)
    if len(x) < 3:
        return 0.0, len(x)
    return max(0.0, float(np.polyfit(x, y, 1)[0])), len(x)
