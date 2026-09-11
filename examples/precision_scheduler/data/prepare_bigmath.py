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
"""Big-Math-RL-Verified-Processed difficulty-band splits with a sha256 manifest.

Two archived builders merged, both reproduced byte-for-byte (the golden tests compare parquet
sha256 against the archived manifests):

* ``splits`` (hardmath ``prepare_bigmath_splits.py``): disjoint train / calibration / validation /
  monitor from a ``llama8b_solve_rate`` band, optional source allowlist and domain exclusion,
  ``test.parquet`` alias of validation and a ``pilot_data/`` directory holding only the calibration
  split (``data_source = bigmath_math_verify``).
* ``learnability`` (bf16_learnability ``prepare_datasets.py``): ``big_math`` source only, complete-prompt
  filter, one band in 1/64 units, train + validation (``data_source = clean_bigmath_learnability``).

    prepare_bigmath.py splits --source <level_3_4_5 parquet> --output-dir out --seed 20260825 \
        --min-solve-rate 0.015625 --max-solve-rate 0.015625 --sources olympiads,aops_forum,amc_aime,harp,omnimath
    prepare_bigmath.py learnability --source <parquet> --output-dir out --band 2 10 --seed 20260904

``--source`` is the local snapshot parquet (archived runs: snapshot ``c79efbb6``,
``level_3_4_5/train-00000-of-00001.parquet``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SYSTEM_PROMPT = (
    "Solve the following math problem step by step. Enclose the reasoning in "
    "<think>...</think> and the final answer in <answer>...</answer>."
)
DEFAULT_SOURCE = (
    "/data/huggingface/hub/datasets--open-r1--Big-Math-RL-Verified-Processed/snapshots/"
    "c79efbb6d3b75e3a2bcc27a5c569119918132345/level_3_4_5/train-00000-of-00001.parquet"
)
# Applied to the end of the prompt so that an earlier worked example cannot make an incomplete
# trailing problem pass the completeness filter (learnability builder).
TRAILING_TASK = re.compile(
    r"(?is)(\?|find|determine|calculate|prove|show|what|how many|evaluate|"
    r"compute|express|maximum|minimum|solve|which|answer|value|range|convert|"
    r"rewrite|simplify|factor|____|\\_\\_)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialize_domain(value) -> list[str]:
    import numpy as np

    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def is_complete_prompt(prompt: str) -> bool:
    prompt = str(prompt).strip()
    return 80 <= len(prompt) <= 2048 and bool(TRAILING_TASK.search(prompt[-320:]))


def to_verl_rows(frame, split: str, *, data_source: str, system_prompt: str, label: str | None = None):
    import pandas as pd

    rows = []
    for row in frame.itertuples(index=False):
        extra = {"split": split}
        if label is not None:
            extra["search_dataset"] = label
        extra.update(
            {
                "source": str(row.source),
                "source_index": int(row.source_index),
                "difficulty_proxy": float(row.llama8b_solve_rate),
                "domain": serialize_domain(row.domain),
            }
        )
        if label is None:  # archived hardmath key order: source, source_index, split, difficulty_proxy, domain
            extra = {
                "source": extra["source"],
                "source_index": extra["source_index"],
                "split": split,
                "difficulty_proxy": extra["difficulty_proxy"],
                "domain": extra["domain"],
            }
        rows.append(
            {
                "data_source": data_source,
                "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": str(row.prompt)}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(row.solution)},
                "extra_info": extra,
            }
        )
    return pd.DataFrame(rows)


def prepare(
    *,
    source: Path,
    output_dir: Path,
    seed: int = 20260825,
    min_solve_rate: float = 0.03125,
    max_solve_rate: float = 0.25,
    sources: list[str] | None = None,
    exclude_domain_substrings: list[str] | None = None,
    train_size: int = 4096,
    calibration_size: int = 256,
    validation_size: int = 1024,
    monitor_size: int = 256,
    system_prompt: str = SYSTEM_PROMPT,
    data_source: str = "bigmath_math_verify",
) -> dict:
    """The hardmath split builder; returns the manifest (also written to output_dir/manifest.json)."""
    import pandas as pd

    exclude_domain_substrings = list(exclude_domain_substrings or [])
    frame = pd.read_parquet(source).reset_index(names="source_index")
    eligible = frame[frame["llama8b_solve_rate"].between(min_solve_rate, max_solve_rate, inclusive="both")].copy()
    source_allowlist = None
    if sources:
        source_allowlist = tuple(item.strip() for item in sources if item.strip())
        eligible = eligible[eligible["source"].isin(source_allowlist)].copy()
    for substring in exclude_domain_substrings:
        needle = substring.lower()
        mask = eligible["domain"].map(
            lambda value, needle=needle: needle in " | ".join(serialize_domain(value)).lower()
        )
        eligible = eligible[~mask].copy()
    eligible = eligible.drop_duplicates(subset="prompt", keep="first")
    required = train_size + calibration_size + validation_size
    if len(eligible) < required:
        raise RuntimeError(f"Need {required} unique prompts, found {len(eligible)}")

    selected = eligible.sample(n=required, random_state=seed, replace=False)
    train = selected.iloc[:train_size].copy()
    calibration = selected.iloc[train_size : train_size + calibration_size].copy()
    validation = selected.iloc[train_size + calibration_size :].copy()
    monitor = validation.iloc[:monitor_size].copy()
    split_frames = {"train": train, "calibration": calibration, "validation": validation, "monitor": monitor}
    prompt_sets = {name: set(f["prompt"]) for name, f in split_frames.items()}
    if prompt_sets["train"] & prompt_sets["calibration"]:
        raise RuntimeError("train/calibration prompt leakage")
    if prompt_sets["train"] & prompt_sets["validation"]:
        raise RuntimeError("train/validation prompt leakage")
    if prompt_sets["calibration"] & prompt_sets["validation"]:
        raise RuntimeError("calibration/validation prompt leakage")
    if not prompt_sets["monitor"].issubset(prompt_sets["validation"]):
        raise RuntimeError("monitor must be a validation subset")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for split, split_frame in split_frames.items():
        path = output_dir / f"{split}.parquet"
        to_verl_rows(split_frame, split, data_source=data_source, system_prompt=system_prompt).to_parquet(
            path, index=False
        )
        artifacts[split] = {
            "path": str(path.resolve()),
            "rows": len(split_frame),
            "unique_prompts": int(split_frame["prompt"].nunique()),
            "solve_rate_mean": float(split_frame["llama8b_solve_rate"].mean()),
            "solve_rate_median": float(split_frame["llama8b_solve_rate"].median()),
            "solve_rate_min": float(split_frame["llama8b_solve_rate"].min()),
            "solve_rate_max": float(split_frame["llama8b_solve_rate"].max()),
            "sha256": sha256(path),
        }
    # The drivers expect train.parquet and test.parquet: test is an alias of validation, and
    # pilot_data/ exposes only the disjoint calibration split to rollout-only calibration runs.
    test_path = output_dir / "test.parquet"
    to_verl_rows(validation, "validation", data_source=data_source, system_prompt=system_prompt).to_parquet(
        test_path, index=False
    )
    pilot_dir = output_dir / "pilot_data"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    pilot = to_verl_rows(calibration, "calibration", data_source=data_source, system_prompt=system_prompt)
    pilot.to_parquet(pilot_dir / "train.parquet", index=False)
    pilot.to_parquet(pilot_dir / "test.parquet", index=False)
    artifacts["test_alias"] = {
        "path": str(test_path.resolve()),
        "rows": len(validation),
        "canonical_split": "validation",
        "sha256": sha256(test_path),
    }
    artifacts["pilot_data"] = {
        "path": str(pilot_dir.resolve()),
        "rows": len(calibration),
        "canonical_split": "calibration",
        "train_sha256": sha256(pilot_dir / "train.parquet"),
        "test_sha256": sha256(pilot_dir / "test.parquet"),
    }
    manifest = {
        "source": str(source),
        "source_sha256": sha256(Path(source).resolve()),
        "seed": seed,
        "difficulty_filter": {
            "field": "llama8b_solve_rate",
            "minimum_inclusive": min_solve_rate,
            "maximum_inclusive": max_solve_rate,
            "source_allowlist": list(source_allowlist) if source_allowlist else None,
            "excluded_domain_substrings": exclude_domain_substrings,
            "eligible_unique_prompts": len(eligible),
        },
        "ordering": "deterministic pandas sample; train then calibration then validation",
        "monitor_relation": "first monitor_size rows of held-out validation",
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def learnability_source_frame(source: Path):
    import pandas as pd

    frame = pd.read_parquet(source).reset_index(names="source_index")
    frame = frame[frame["source"].eq("big_math")].copy()
    frame = frame[frame["prompt"].map(is_complete_prompt)].copy()
    return frame[frame["solution"].notna() & frame["solution"].astype(str).str.strip().ne("")]


def prepare_learnability_band(
    frame,
    *,
    source: Path,
    output_dir: Path,
    label: str,
    lo_count: int,
    hi_count: int,
    seed: int = 20260904,
    train_size: int = 4096,
    validation_size: int = 512,
    system_prompt: str = SYSTEM_PROMPT,
    data_source: str = "clean_bigmath_learnability",
) -> dict:
    lo, hi = lo_count / 64, hi_count / 64
    eligible = frame[frame["llama8b_solve_rate"].between(lo, hi, inclusive="both")].copy()
    eligible = eligible.drop_duplicates("prompt", keep="first")
    needed = train_size + validation_size
    if len(eligible) < needed:
        raise RuntimeError(f"{label}: need {needed} prompts, found {len(eligible)}")
    chosen = eligible.sample(n=needed, random_state=seed, replace=False)
    train = chosen.iloc[:train_size].copy()
    validation = chosen.iloc[train_size:].copy()
    if set(train.prompt) & set(validation.prompt):
        raise RuntimeError(f"{label}: train/validation overlap")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = output_dir / "train.parquet", output_dir / "test.parquet"
    to_verl_rows(train, "train", data_source=data_source, system_prompt=system_prompt, label=label).to_parquet(
        train_path, index=False
    )
    to_verl_rows(
        validation, "validation", data_source=data_source, system_prompt=system_prompt, label=label
    ).to_parquet(test_path, index=False)
    manifest = {
        "label": label,
        "source_path": str(source),
        "source_filter": "big_math",
        "prompt_filter": "80-2048 chars and trailing task marker in final 320 chars",
        "seed": seed,
        "difficulty_proxy": {
            "field": "llama8b_solve_rate",
            "minimum_inclusive": lo,
            "maximum_inclusive": hi,
            "eligible_unique_prompts": int(len(eligible)),
        },
        "train": {
            "rows": len(train),
            "mean_solve_rate": float(train.llama8b_solve_rate.mean()),
            "sha256": sha256(train_path),
        },
        "validation": {
            "rows": len(validation),
            "mean_solve_rate": float(validation.llama8b_solve_rate.mean()),
            "sha256": sha256(test_path),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_learnability(*, source: Path, output_dir: Path, manifest: dict) -> dict:
    """Rebuild one archived learnability band from its manifest (golden helper)."""
    proxy = manifest["difficulty_proxy"]
    return prepare_learnability_band(
        learnability_source_frame(source),
        source=source,
        output_dir=output_dir,
        label=manifest["label"],
        lo_count=round(proxy["minimum_inclusive"] * 64),
        hi_count=round(proxy["maximum_inclusive"] * 64),
        seed=manifest["seed"],
        train_size=manifest["train"]["rows"],
        validation_size=manifest["validation"]["rows"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("splits", help="hardmath train/calibration/validation/monitor splits")
    sp.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE))
    sp.add_argument("--output-dir", type=Path, required=True)
    sp.add_argument("--seed", type=int, default=20260825)
    sp.add_argument("--min-solve-rate", type=float, default=0.03125)
    sp.add_argument("--max-solve-rate", type=float, default=0.25)
    sp.add_argument("--sources", help="comma-separated source allowlist")
    sp.add_argument("--exclude-domain-substring", action="append", default=[])
    sp.add_argument("--train-size", type=int, default=4096)
    sp.add_argument("--calibration-size", type=int, default=256)
    sp.add_argument("--validation-size", type=int, default=1024)
    sp.add_argument("--monitor-size", type=int, default=256)
    sp.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    lp = sub.add_parser("learnability", help="one solve-rate band (1/64 units), big_math source, complete prompts")
    lp.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE))
    lp.add_argument("--output-dir", type=Path, required=True)
    lp.add_argument("--label", default=None)
    lp.add_argument("--band", type=int, nargs=2, metavar=("LO", "HI"), required=True)
    lp.add_argument("--seed", type=int, default=20260904)
    lp.add_argument("--train-size", type=int, default=4096)
    lp.add_argument("--validation-size", type=int, default=512)
    lp.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "splits":
        manifest = prepare(
            source=args.source,
            output_dir=args.output_dir,
            seed=args.seed,
            min_solve_rate=args.min_solve_rate,
            max_solve_rate=args.max_solve_rate,
            sources=args.sources.split(",") if args.sources else None,
            exclude_domain_substrings=args.exclude_domain_substring,
            train_size=args.train_size,
            calibration_size=args.calibration_size,
            validation_size=args.validation_size,
            monitor_size=args.monitor_size,
            system_prompt=args.system_prompt,
        )
    else:
        lo, hi = args.band
        manifest = prepare_learnability_band(
            learnability_source_frame(args.source),
            source=args.source,
            output_dir=args.output_dir,
            label=args.label or f"bigmath_band_{lo}_{hi}",
            lo_count=lo,
            hi_count=hi,
            seed=args.seed,
            train_size=args.train_size,
            validation_size=args.validation_size,
            system_prompt=args.system_prompt,
        )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
