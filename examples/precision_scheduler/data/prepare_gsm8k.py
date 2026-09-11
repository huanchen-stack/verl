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
"""GSM8K parquet builder for the precision-scheduler recipes.

Merges the archived ``prepare_gsm8k_thinking_parquet.py`` (one train/test slice) and
``prepare_gsm8k_temporal_guard_sets.py`` (N disjoint seeded sets) into one CLI.

Input is either the QeRL JSONL export (``--input``; rows carry a rendered ``<|im_start|>user``
prompt and ``answer``) or ``--hf openai/gsm8k`` (train split, ``shuffle(seed)[:limit]``, the
selection recorded in the export). Rows are stored as chat-message lists (``prompt_format
messages``) so the v1 agent loop renders them with the model tokenizer; ``rendered`` keeps the
legacy pre-rendered string.

    prepare_gsm8k.py --input qerl_gsm8k_train_2048_rollout.jsonl --output-dir out --train-size 2048 --test-size 32
    prepare_gsm8k.py --input ... --output-dir out --disjoint-sets 8x32 --selection-seed 20260811

Every output directory gets a ``summary.json`` (row counts, prompt token stats with the given
tokenizer) and, for disjoint sets, a ``manifest.json`` with the ``source_indices`` per set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

USER_RE = re.compile(r"<\|im_start\|>user\n(?P<content>.*?)<\|im_end\|>", re.DOTALL)
GSM_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
DATA_SOURCE = "openai/gsm8k"


def extract_user_content(prompt: str) -> str:
    match = USER_RE.search(prompt)
    if not match:
        raise ValueError(f"Could not parse user content from prompt prefix: {prompt[:200]!r}")
    return match.group("content")


def gsm_answer(value: str) -> str:
    """``#### 250`` -> ``250`` (the QeRL export already stores the bare answer)."""
    match = GSM_ANSWER_RE.search(value)
    return match.group(1).strip().replace(",", "") if match else value.strip()


def load_qerl_export(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "index": idx,
                    "question": extract_user_content(obj["prompt"]),
                    "answer": str(obj.get("answer", "")),
                    "dataset": obj.get("dataset", "QeRL/GSM8K"),
                    "source_dataset": obj.get("source_dataset"),
                    "source_split": obj.get("source_split"),
                    "source_file": str(path),
                }
            )
    return rows


def load_hf_gsm8k(name: str, shuffle_seed: int, limit: int) -> list[dict[str, Any]]:
    """Reproduce the QeRL export selection: ``dataset.shuffle(seed)[:limit]`` on the train split."""
    from datasets import load_dataset

    ds = load_dataset(name, "main", split="train").shuffle(seed=shuffle_seed)
    rows = []
    for idx, row in enumerate(ds):
        if idx >= limit:
            break
        rows.append(
            {
                "index": idx,
                "question": row["question"],
                "answer": gsm_answer(row["answer"]),
                "dataset": "QeRL/GSM8K",
                "source_dataset": name,
                "source_split": "train",
                "source_file": f"hf:{name}:train:shuffle(seed={shuffle_seed})[:{limit}]",
            }
        )
    return rows


def render_thinking_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def to_verl_row(item: dict[str, Any], *, split: str, prompt_format: str, tokenizer, extra: dict | None = None) -> dict:
    messages = [{"role": "user", "content": item["question"]}]
    prompt: Any = messages if prompt_format == "messages" else render_thinking_prompt(tokenizer, messages)
    info = {
        "index": item["index"],
        "split": split,
        "dataset": item["dataset"],
        "source_dataset": item["source_dataset"],
        "source_split": item["source_split"],
        "thinking_template": True,
        "source_file": item["source_file"],
    }
    if extra:
        info.update(extra)
    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": item["answer"]},
        "extra_info": info,
    }


def prompt_token_lengths(rows: list[dict], tokenizer) -> list[int]:
    lengths = []
    for row in rows:
        prompt = row["prompt"]
        if isinstance(prompt, list):
            prompt = render_thinking_prompt(tokenizer, prompt)
        lengths.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    return lengths


def token_stats(lengths: list[int]) -> dict[str, float]:
    return {
        "prompt_token_min": min(lengths),
        "prompt_token_max": max(lengths),
        "prompt_token_mean": sum(lengths) / len(lengths),
    }


def write_parquet(rows: list[dict], path: Path) -> None:
    import pandas as pd

    pd.DataFrame(rows).to_parquet(path, index=False)


def build_slice(
    items: list[dict],
    *,
    train_size: int,
    test_size: int,
    start_index: int,
    shuffle_seed: int | None,
    prompt_format: str,
    tokenizer,
) -> tuple[list[dict], list[dict], dict]:
    """Archived single-slice builder: rows[start:] -> first ``train_size`` train rows, first ``test_size`` test."""
    selected = items[start_index : start_index + max(train_size, test_size)]
    rows = [to_verl_row(item, split="train", prompt_format=prompt_format, tokenizer=tokenizer) for item in selected]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
        for shuffled_idx, row in enumerate(rows):
            row["extra_info"] = dict(row["extra_info"], shuffle_seed=shuffle_seed, shuffle_index=shuffled_idx)
    if len(rows) < train_size:
        raise RuntimeError(f"Only prepared {len(rows)} rows; need {train_size}")
    train_rows = [dict(row) for row in rows[:train_size]]
    test_rows = []
    for row in rows[: min(test_size, len(rows))]:
        copied = dict(row)
        copied["extra_info"] = dict(row["extra_info"], split="test")
        test_rows.append(copied)
    summary = {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "start_index": start_index,
        "shuffle_seed": shuffle_seed,
        "prompt_format": prompt_format,
        **token_stats(prompt_token_lengths(train_rows, tokenizer)),
        "first_prompt": train_rows[0]["prompt"],
    }
    return train_rows, test_rows, summary


def build_disjoint_sets(
    items: list[dict], *, num_sets: int, set_size: int, selection_seed: int, prompt_format: str, tokenizer
) -> tuple[list[list[dict]], dict]:
    """The archived temporal-guard builder: ``shuffle(seed)[:num_sets*set_size]`` split into disjoint sets."""
    needed = num_sets * set_size
    if len(items) < needed:
        raise RuntimeError(f"Need {needed} rows, found {len(items)}")
    indices = list(range(len(items)))
    random.Random(selection_seed).shuffle(indices)
    selected = indices[:needed]
    manifest: dict[str, Any] = {
        "source_rows": len(items),
        "selection": f"shuffle(seed={selection_seed})[:{needed}]",
        "selection_seed": selection_seed,
        "num_sets": num_sets,
        "set_size": set_size,
        "sets": [],
    }
    sets = []
    seen: set[int] = set()
    for set_index in range(num_sets):
        set_indices = selected[set_index * set_size : (set_index + 1) * set_size]
        if seen.intersection(set_indices):
            raise RuntimeError("Prompt sets are not disjoint")
        seen.update(set_indices)
        rows = [
            to_verl_row(
                items[source_index],
                split="train",
                prompt_format=prompt_format,
                tokenizer=tokenizer,
                extra={"position_in_set": position, "set_index": set_index, "selection_seed": selection_seed},
            )
            for position, source_index in enumerate(set_indices)
        ]
        lengths = prompt_token_lengths(rows, tokenizer)
        manifest["sets"].append(
            {"set_index": set_index, "source_indices": set_indices, "rows": len(rows), **token_stats(lengths)}
        )
        sets.append(rows)
    return sets, manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="QeRL GSM8K JSONL export")
    src.add_argument("--hf", metavar="NAME", help="HF dataset id (openai/gsm8k): train split, shuffle(seed)[:limit]")
    parser.add_argument("--hf-shuffle-seed", type=int, default=0)
    parser.add_argument("--hf-limit", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="HF id or local snapshot used for token statistics")
    parser.add_argument("--prompt-format", choices=("messages", "rendered"), default="messages")
    parser.add_argument("--train-size", type=int, default=32)
    parser.add_argument("--test-size", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--shuffle-seed", type=int, default=None, help="shuffle the slice rows (archived --shuffle-seed)"
    )
    parser.add_argument(
        "--disjoint-sets", metavar="NxM", help="write N disjoint sets of M rows (set_00/, set_01/, ...)"
    )
    parser.add_argument("--selection-seed", type=int, default=20260811)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    items = load_qerl_export(args.input) if args.input else load_hf_gsm8k(args.hf, args.hf_shuffle_seed, args.hf_limit)
    source = str(args.input) if args.input else items[0]["source_file"]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.disjoint_sets:
        num_sets, set_size = (int(x) for x in args.disjoint_sets.lower().split("x"))
        sets, manifest = build_disjoint_sets(
            items,
            num_sets=num_sets,
            set_size=set_size,
            selection_seed=args.selection_seed,
            prompt_format=args.prompt_format,
            tokenizer=tokenizer,
        )
        manifest = {"source": source, **manifest}
        for set_index, rows in enumerate(sets):
            set_dir = out / f"set_{set_index:02d}"
            set_dir.mkdir(parents=True, exist_ok=True)
            write_parquet(rows, set_dir / "train.parquet")
            test_rows = [dict(row, extra_info=dict(row["extra_info"], split="test")) for row in rows]
            write_parquet(test_rows, set_dir / "test.parquet")
            (set_dir / "summary.json").write_text(json.dumps(manifest["sets"][set_index], indent=2) + "\n")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in manifest.items() if k != "sets"}, indent=2))
        return

    train_rows, test_rows, summary = build_slice(
        items,
        train_size=args.train_size,
        test_size=args.test_size,
        start_index=args.start_index,
        shuffle_seed=args.shuffle_seed,
        prompt_format=args.prompt_format,
        tokenizer=tokenizer,
    )
    write_parquet(train_rows, out / "train.parquet")
    write_parquet(test_rows, out / "test.parquet")
    summary = {"input": source, "output_dir": str(out), **summary}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
