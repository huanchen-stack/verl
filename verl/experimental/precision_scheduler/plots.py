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
"""Generic figures for the toolkit (matplotlib is imported lazily; the package never requires it)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .hazard import HazardTable, survival
from .policy_builder import decisions_array


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_speedup_heatmap(payload: dict[str, Any], output: Path, *, title: str = "BF16 / INT4 speedup") -> None:
    """RdYlGn speedup matrix centred at 1.0 (missing cells grey)."""
    plt = _plt()
    from matplotlib.colors import TwoSlopeNorm

    array = np.array([[np.nan if v is None else v for v in row] for row in payload["speedup_bf16_over_int4"]], float)
    finite = array[np.isfinite(array)]
    cmap = plt.get_cmap("RdYlGn").with_extremes(bad="#d9d9d9")
    norm = None
    if finite.size:
        norm = TwoSlopeNorm(vmin=min(float(finite.min()), 0.95), vcenter=1.0, vmax=max(float(finite.max()), 1.05))
    figure, axis = plt.subplots(
        figsize=(max(7.0, len(payload["seq_lens"]) * 1.05), max(4.5, len(payload["batch_sizes"]) * 0.7))
    )
    image = axis.imshow(np.ma.masked_invalid(array), aspect="auto", origin="lower", cmap=cmap, norm=norm)
    axis.set_xticks(range(len(payload["seq_lens"])), labels=[str(v) for v in payload["seq_lens"]])
    axis.set_yticks(range(len(payload["batch_sizes"])), labels=[str(v) for v in payload["batch_sizes"]])
    axis.set_xlabel("Context length (tokens)")
    axis.set_ylabel("Batch size")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="speedup")
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            axis.text(
                j,
                i,
                "N/A" if not np.isfinite(array[i, j]) else f"{array[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_policy_surface(policy: dict[str, Any], output: Path, *, prompt_bucket: int = 0) -> None:
    """Committed switch frontier as a (live batch x observed frontier) surface for one prompt bucket."""
    plt = _plt()
    decisions = decisions_array(policy)[:, prompt_bucket, :]  # [frontier, live]
    table = policy["lookup_table"]
    frontiers = table["frontier_start"] + table["frontier_step"] * np.arange(table["frontier_count"])
    figure, axis = plt.subplots(figsize=(9, 4.5))
    masked = np.ma.masked_equal(decisions.T, 0)
    image = axis.imshow(
        masked,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=(frontiers[0], frontiers[-1], 1, table["live_batch_count"]),
    )
    axis.set_xlabel("Observed response frontier (tokens)")
    axis.set_ylabel("Live requests")
    axis.set_title(f"Committed switch frontier (prompt bucket {prompt_bucket}); blank = stay BF16")
    figure.colorbar(image, ax=axis, label="switch frontier (tokens)")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_survival(tables: dict[str, HazardTable], frontiers: np.ndarray, output: Path, *, start_index: int = 0) -> None:
    """Bin-start survival curves from ``start_index`` for several hazard tables (e.g. bf16 vs w4)."""
    plt = _plt()
    figure, axis = plt.subplots(figsize=(7, 4))
    for label, table in tables.items():
        axis.plot(frontiers[start_index:], survival(table, start_index), label=label)
    axis.set_xlabel("Response tokens")
    axis.set_ylabel("P(alive at bin start)")
    axis.set_ylim(0, 1.02)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_regression(
    x, y, fit: dict[str, Any], output: Path, *, xlabel: str = "tokens", ylabel: str = "seconds"
) -> None:
    plt = _plt()
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(x, y, s=12)
    line = np.linspace(x.min(), x.max(), 50)
    slope = fit.get("slope_s_per_token", fit.get("seconds_per_token", fit.get("slope_seconds_per_sampled_token")))
    intercept = fit.get("intercept_s", fit.get("intercept_seconds"))
    axis.plot(
        line,
        slope * line + intercept,
        color="C1",
        label=f"{slope:.3e} s/token + {intercept:.2f} s (r2 {fit['r2']:.4f})",
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
