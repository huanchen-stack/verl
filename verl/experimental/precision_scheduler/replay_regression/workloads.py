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
"""Build token workloads for the replay regression from archived rollout JSONL dumps.

Each workload ``point_NN.json`` is a batch of tokenized (prompt, response) pairs whose response is
truncated at a cap; the manifest records ``total_tokens`` so :mod:`downstream_regression` can fit
inference / training seconds against tokens without running a rollout engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

DEFAULT_BATCHES = (32, 64, 128)
DEFAULT_ANCHOR_CAPS = {
    32: (1024, 4096, 8192, 16384, 24000),
    64: (1024, 4096, 8192, 12000, 20000),
    128: (1024, 3072, 6144, 10000, 16000),
}


def workload_specs(
    points: int,
    seed: int,
    *,
    batches: tuple[int, ...] = DEFAULT_BATCHES,
    anchor_caps: dict[int, tuple[int, ...]] | None = None,
    anchored_points: int = 15,
    min_random_cap: int = 768,
) -> list[tuple[int, int]]:
    """``(batch_size, cap)`` per point: anchors cycle the caps per batch, then random caps.

    Spanning both batch and token count makes a shared token slope identifiable rather than
    confounded with batch size.
    """
    anchor_caps = anchor_caps or DEFAULT_ANCHOR_CAPS
    rng = random.Random(seed)
    specs: list[tuple[int, int]] = []
    while len(specs) < points:
        batch = batches[len(specs) % len(batches)]
        caps = anchor_caps[batch]
        if len(specs) < anchored_points:
            cap = caps[(len(specs) // len(batches)) % len(caps)]
        else:
            cap = rng.randint(min_random_cap, caps[-1])
        specs.append((batch, cap))
    return specs


def collect_corpus(
    root: Path,
    tokenizer,
    *,
    seed: int,
    corpus_size: int = 512,
    prompt_max: int = 2048,
    response_max: int = 24576,
    glob: str = "**/rollouts/*.jsonl",
) -> list[dict[str, Any]]:
    """Distinct tokenized trajectories from rollout dumps (``{"input": ..., "output": ...}`` rows)."""
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    paths = sorted(root.glob(glob))
    random.Random(seed).shuffle(paths)
    for path in paths:
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt_text, response_text = item.get("input", ""), item.get("output", "")
            key = hashlib.sha1((prompt_text + "\0" + response_text).encode()).hexdigest()
            if not prompt_text or not response_text or key in seen:
                continue
            seen.add(key)
            prompt = tokenizer(prompt_text, add_special_tokens=False).input_ids
            response = tokenizer(response_text, add_special_tokens=False).input_ids
            if tokenizer.eos_token_id is not None and (not response or response[-1] != tokenizer.eos_token_id):
                response.append(tokenizer.eos_token_id)
            if prompt and response:
                rows.append(
                    {
                        "id": key,
                        "prompt": prompt[-prompt_max:],
                        "response": response[:response_max],
                        "source": str(path),
                    }
                )
        # Token ids are Python integers; thousands of 24K-token trajectories cost GiBs.
        if len(rows) >= corpus_size:
            break
    return rows


def write_workloads(rows: list[dict[str, Any]], specs: list[tuple[int, int]], output: Path, seed: int) -> dict:
    rng = random.Random(seed)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (batch_size, cap) in enumerate(specs):
        chosen = rng.sample(rows, batch_size)
        samples = [{**row, "response": row["response"][:cap]} for row in chosen]
        obj = {
            "point": index,
            "batch_size": batch_size,
            "cap": cap,
            "total_tokens": sum(len(s["prompt"]) + len(s["response"]) for s in samples),
            "response_tokens": sum(len(s["response"]) for s in samples),
            "samples": samples,
        }
        path = output / f"point_{index:02d}.json"
        path.write_text(json.dumps(obj))
        manifest.append(
            {k: obj[k] for k in ("point", "batch_size", "cap", "total_tokens", "response_tokens")} | {"path": str(path)}
        )
    summary = {"corpus_size": len(rows), "points": manifest}
    (output / "manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory searched for **/rollouts/*.jsonl")
    parser.add_argument("--model", required=True, help="tokenizer path or HF id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--corpus-size", type=int, default=512)
    parser.add_argument("--min-corpus", type=int, default=128)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = collect_corpus(args.root, tokenizer, seed=args.seed, corpus_size=args.corpus_size)
    if len(rows) < args.min_corpus:
        raise RuntimeError(f"Only {len(rows)} usable distinct trajectories")
    summary = write_workloads(rows, workload_specs(args.points, args.seed), args.output, args.seed)
    tokens = [p["total_tokens"] for p in summary["points"]]
    print(
        json.dumps(
            {"corpus_size": len(rows), "points": len(tokens), "min_tokens": min(tokens), "max_tokens": max(tokens)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
