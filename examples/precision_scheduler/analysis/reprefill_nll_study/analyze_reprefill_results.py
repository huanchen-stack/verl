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
"""Analyze a reuse-vs-re-prefill run: recompute the bootstrap summary from the per-window
metrics, decide the gate, and (optionally) draw the two figures.

numpy-only (matplotlib only with plots enabled), so the summary step doubles as the golden
oracle: ``summarize()`` with the archived seed reproduces the archived ``summary.json`` of
``qwen35_9b_w4qdq_longtail16`` exactly from its ``window_metrics.jsonl``.

    python analyze_reprefill_results.py --run-dir RUN --out-dir OUT [--no-plots]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SUMMARY_KEYS = [
    "delta_nll_reuse_minus_reprefill",
    "delta_kl_reuse_minus_reprefill",
    "kl_bf16_to_reuse",
    "kl_bf16_to_reprefill",
    "js_reuse_reprefill",
    "reuse_top1_agreement_bf16",
    "reprefill_top1_agreement_bf16",
]
DEFAULT_OFFSETS = [0, 512, 1024, 2048, 4096]
DEFAULT_SEED = 20260812
BOOTSTRAP_RESAMPLES = 10000


def summarize(rows: list[dict], offsets: list[int], seed: int, wall_seconds: float | None = None) -> dict:
    """Per-offset mean / median / bootstrap CI95 / positive fraction of every summary key.

    The resampling order (one ``default_rng(seed)`` shared across offsets and keys, 10000
    resamples each, in ``offsets`` x ``SUMMARY_KEYS`` order) is the archived one; do not
    reorder it or the archived CI95 values stop reproducing.
    """
    rng = np.random.default_rng(seed)
    summary: dict = {"num_traces": len({r["trace_index"] for r in rows}), "windows": {}}
    if wall_seconds is not None:
        summary["wall_seconds"] = wall_seconds
    for offset in offsets:
        group = [r for r in rows if r["offset"] == offset]
        metrics = {}
        for key in SUMMARY_KEYS:
            vals = np.array([r[key] for r in group], dtype=np.float64)
            boots = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(BOOTSTRAP_RESAMPLES)])
            metrics[key] = {
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                "positive_fraction": float((vals > 0).mean()),
            }
        summary["windows"][str(offset)] = metrics
    return summary


def decide(summary: dict, offsets: list[int]) -> dict:
    """The study's gate: re-prefill is only worth a verifier experiment when it lowers the
    continuation NLL (delta = NLL(reuse) - NLL(re-prefill) > 0 with CI95 above zero)."""
    deltas = {str(o): summary["windows"][str(o)]["delta_nll_reuse_minus_reprefill"] for o in offsets}
    reuse_better = all(d["mean"] < 0 and d["ci95"][1] < 0 for d in deltas.values())
    reprefill_better = all(d["mean"] > 0 and d["ci95"][0] > 0 for d in deltas.values())
    if reprefill_better:
        gate = "GO: re-prefill lowered NLL at every offset; run the verifier experiment"
    else:
        gate = "STOP: re-prefill did not improve NLL; verifier experiment not run"
    last = str(max(offsets))
    return {
        "gate": gate,
        "reuse_better_at_every_offset": reuse_better,
        "reprefill_better_at_every_offset": reprefill_better,
        "delta_nll_by_offset": {o: d["mean"] for o, d in deltas.items()},
        "primary_offset": last,
        "primary_delta_nll": summary["windows"][last]["delta_nll_reuse_minus_reprefill"],
        "primary_delta_kl": summary["windows"][last]["delta_kl_reuse_minus_reprefill"],
    }


def _summaries_match(ours: dict, archived: dict, offsets: list[int], tol: float = 1e-9) -> bool:
    for offset in offsets:
        for key in SUMMARY_KEYS:
            a = ours["windows"][str(offset)][key]
            b = archived["windows"][str(offset)][key]
            if abs(a["mean"] - b["mean"]) > tol or abs(a["median"] - b["median"]) > tol:
                return False
            if a["positive_fraction"] != b["positive_fraction"]:
                return False
            if any(abs(x - y) > tol for x, y in zip(a["ci95"], b["ci95"], strict=True)):
                return False
    return True


def _dataset_of(row: dict) -> str:
    return row.get("dataset") or Path(row["source"]).parts[-4]


def _labels(offsets: list[int]) -> list[str]:
    return [str(o) if o < 1000 else f"{o / 1000:g}K" for o in offsets]


def plot(summary: dict, rows: list[dict], offsets: list[int], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = _labels(offsets)

    def stat(key: str):
        means, lo, hi = [], [], []
        for o in offsets:
            x = summary["windows"][str(o)][key]
            means.append(x["mean"])
            lo.append(x["ci95"][0])
            hi.append(x["ci95"][1])
        return np.array(means), np.array(lo), np.array(hi)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    m, lo, hi = stat("delta_nll_reuse_minus_reprefill")
    axes[0].errorbar(offsets, m, yerr=[m - lo, hi - m], color="#D85837", marker="o", lw=2, capsize=3)
    axes[0].axhline(0, color="black", lw=1, ls="--")
    axes[0].set_xticks(offsets, labels)
    axes[0].set_xlabel("Start of scored window after switch")
    axes[0].set_ylabel("NLL(reuse) - NLL(re-prefill), nats/token")
    axes[0].set_title("Signed continuation loss")
    axes[0].grid(axis="y", alpha=0.2)
    for key, color, marker, label in [
        ("kl_bf16_to_reuse", "#2457A6", "o", "Reuse BF16 state"),
        ("kl_bf16_to_reprefill", "#8E5AA9", "s", "W4 re-prefill"),
    ]:
        m, lo, hi = stat(key)
        axes[1].errorbar(offsets, m, yerr=[m - lo, hi - m], color=color, marker=marker, lw=2, capsize=3, label=label)
    axes[1].set_xticks(offsets, labels)
    axes[1].set_xlabel("Start of scored window after switch")
    axes[1].set_ylabel("KL(BF16 || branch), nats/token")
    axes[1].set_title("Distance from BF16 distribution")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "main_result.pdf", bbox_inches="tight")
    fig.savefig(out / "main_result.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 3.8))
    data = [[r["delta_nll_reuse_minus_reprefill"] for r in rows if r["offset"] == o] for o in offsets]
    parts = ax.violinplot(data, positions=np.arange(len(offsets)), showmeans=False, showmedians=True, widths=0.75)
    for body in parts["bodies"]:
        body.set_facecolor("#D85837")
        body.set_edgecolor("#D85837")
        body.set_alpha(0.35)
    parts["cmedians"].set_color("#A63A2B")
    rng = np.random.default_rng(7)
    for i, vals in enumerate(data):
        ax.scatter(i + rng.uniform(-0.09, 0.09, len(vals)), vals, s=13, color="#7E3527", alpha=0.7)
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_xticks(range(len(offsets)), labels)
    ax.set_xlabel("Start of scored window after switch")
    ax.set_ylabel("NLL(reuse) - NLL(re-prefill), nats/token")
    ax.set_title(f"Per-trace effects (n={summary['num_traces']} long-tail survivors)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "per_trace.pdf", bbox_inches="tight")
    fig.savefig(out / "per_trace.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="directory with window_metrics.jsonl (and summary.json)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--offsets", nargs="*", type=int, default=None, help="default: from manifest.json, else 0..4096")
    p.add_argument("--seed", type=int, default=None, help="bootstrap seed; default: from manifest.json, else 20260812")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    rows = [json.loads(x) for x in (a.run_dir / "window_metrics.jsonl").read_text().splitlines() if x.strip()]
    manifest_path = a.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    offsets = a.offsets or manifest.get("offsets") or DEFAULT_OFFSETS
    seed = a.seed if a.seed is not None else manifest.get("seed", DEFAULT_SEED)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(rows, offsets, seed)
    (a.out_dir / "summary.recomputed.json").write_text(json.dumps(summary, indent=2))
    archived_path = a.run_dir / "summary.json"
    matches = None
    if archived_path.exists():
        matches = _summaries_match(summary, json.loads(archived_path.read_text()), offsets)

    sources: dict[str, int] = {}
    for r in rows:
        if r["offset"] == offsets[0]:
            name = _dataset_of(r)
            sources[name] = sources.get(name, 0) + 1
    window = rows[0]["scored_tokens"] if rows else None
    analysis = {
        "run_dir": str(a.run_dir),
        "num_rows": len(rows),
        "sources": sources,
        "offsets": offsets,
        "window": window,
        "seed": seed,
        "summary_matches_archived": matches,
        **decide(summary, offsets),
    }
    (a.out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    if not a.no_plots:
        plot(summary, rows, offsets, a.out_dir)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
