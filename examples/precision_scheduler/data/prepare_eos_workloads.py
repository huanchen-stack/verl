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
"""Multi-workload (EOS-hazard extensibility) dataset pipeline: source -> render -> materialize.

Merges the archived ``eos_hazard_extensibility/{prepare_datasets.py,render_prompts.py}`` and
``eos_hazard_fullstep_b64_cap16k/prepare_data.py`` (plus the ``--rows a:b`` variant slicing of
``b32_16k_sensitivity/prepare_fullstep_data.py``). The HF loaders are kept verbatim; every stage writes a
sha256 manifest so the archived manifests are goldens.

  source      HF hub -> <out>/<dataset>.jsonl: ``shuffle(seed)`` then the first ``needed`` usable rows,
              three 32-row blocks, ``messages`` = [system INSTRUCTION, user question].
  render      per-model chat-template rendering of a source file (sha256 + prompt_token_max in the manifest).
  materialize rendered + source -> verl parquet with message prompts; every row's template is
              re-rendered and must token-match the archived rendered prompt; ``--rows a:b`` selects the
              row range (default 0:64), ``--test-rows`` the size of test.parquet.

    prepare_eos_workloads.py source --out datasets --datasets gsm8k math500 [--seed 20260902 --needed 96]
    prepare_eos_workloads.py render --source datasets/gsm8k.jsonl --out rendered/qwen3_5_4b/gsm8k.jsonl \
        --tokenizer Qwen/Qwen3.5-4B --enable-thinking
    prepare_eos_workloads.py materialize --source datasets/gsm8k.jsonl --rendered rendered/qwen3_5_4b/gsm8k.jsonl \
        --out data/qwen3_5_4b/gsm8k --tokenizer Qwen/Qwen3.5-4B --enable-thinking --rows 0:64 --dataset gsm8k
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SEED = 20260902
NEEDED = 96
BLOCK = 32
INSTRUCTION = (
    "Solve the task carefully. Show your reasoning, then end with a clearly "
    "marked final answer. Do not continue after the final answer."
)
WORKLOADS = ("gsm8k", "math500", "bigmath_hard", "gpqa_diamond", "mmlu_pro", "livecodebench", "ifeval", "hotpotqa")
MATH = {"gsm8k", "math500", "bigmath_hard"}
GSM_ANSWER_RE = re.compile(r"####\s*([^\n]+)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- source (HF loaders verbatim from the archived prepare_datasets.py) ---------------------------


def choices_text(choices: Any) -> str:
    if isinstance(choices, dict):
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        return "\n".join(f"{a}. {b}" for a, b in zip(labels, texts, strict=False))
    if isinstance(choices, list):
        return "\n".join(
            f"{chr(65 + i)}. {x.get('text', x) if isinstance(x, dict) else x}" for i, x in enumerate(choices)
        )
    return str(choices or "")


def take(ds, converter, *, seed: int, needed: int, predicate=None) -> list[dict[str, Any]]:
    ds = ds.shuffle(seed=seed)
    rows: list[dict[str, Any]] = []
    for source_index, row in enumerate(ds):
        if predicate is not None and not predicate(row):
            continue
        item = converter(row)
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        item["question"] = question
        item["source_index_after_shuffle"] = source_index
        rows.append(item)
        if len(rows) == needed:
            return rows
    raise RuntimeError(f"only found {len(rows)} usable rows; need {needed}")


def load_bigmath(hub_root: Path):
    from datasets import Dataset

    candidates = sorted(
        hub_root.glob("datasets--open-r1--Big-Math-RL-Verified-Processed/snapshots/*/level_3_4_5/train-*.parquet")
    )
    if not candidates:
        raise FileNotFoundError("local Big-Math-RL parquet is missing")
    return Dataset.from_parquet(str(candidates[0]))


def source_rows(name: str, *, seed: int, needed: int, hub_root: Path) -> list[dict[str, Any]]:
    from datasets import Dataset, load_dataset

    kw = {"seed": seed, "needed": needed}
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        return take(ds, lambda r: {"question": r["question"], "answer": r["answer"]}, **kw)
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return take(
            ds,
            lambda r: {
                "question": r["problem"],
                "answer": r["answer"],
                "subject": r.get("subject"),
                "level": r.get("level"),
            },
            **kw,
        )
    if name == "bigmath_hard":
        ds = load_bigmath(hub_root)
        return take(
            ds,
            lambda r: {
                "question": r["prompt"],
                "answer": r["solution"],
                "source": r.get("source"),
                "solve_rate": r.get("llama8b_solve_rate"),
            },
            predicate=lambda r: 1 / 64 <= float(r.get("llama8b_solve_rate", 1)) <= 2 / 64,
            **kw,
        )
    if name == "gpqa_diamond":
        try:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        except Exception:
            # Public mirror of the original 198-row Diamond split (same task/schema).
            ds = load_dataset("bdytx5/gpqa_gpqa_diamond", split="train")

        def convert_gpqa(r):
            answer = r.get("Correct Answer")
            options = [answer, r.get("Incorrect Answer 1"), r.get("Incorrect Answer 2"), r.get("Incorrect Answer 3")]
            record = str(r.get("Record ID", r.get("question_id", r.get("Question", ""))))
            options = sorted(options, key=lambda x: hashlib.sha256(f"{record}|{x}".encode()).digest())
            return {
                "question": r.get("Question", r.get("question", "")) + "\n" + choices_text(options),
                "answer": answer,
                "actual_source": "GPQA Diamond (public mirror: bdytx5)",
                "record_id": record,
                "subdomain": r.get("Subdomain"),
            }

        return take(ds, convert_gpqa, **kw)
    if name == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        return take(
            ds,
            lambda r: {
                "question": r["question"] + "\n" + choices_text(r.get("options")),
                "answer": r.get("answer"),
                "category": r.get("category"),
            },
            **kw,
        )
    if name == "livecodebench":
        ds = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True)
        return take(
            ds,
            lambda r: {
                "question": r.get("question_title", "") + "\n" + r.get("question_content", r.get("prompt", "")),
                "answer": r.get("starter_code", ""),
                "platform": r.get("platform"),
            },
            **kw,
        )
    if name == "ifeval":
        try:
            ds = load_dataset("google/IFEval", split="train")
            actual_source = "google/IFEval"
        except Exception:
            blobs = hub_root / "datasets--allenai--IFBench_test" / "blobs"
            candidates = [p for p in blobs.glob("*") if p.stat().st_size > 100_000]
            if not candidates:
                raise
            ds = Dataset.from_parquet(str(max(candidates, key=lambda p: p.stat().st_size)))
            actual_source = "allenai/IFBench_test fallback"
        return take(
            ds,
            lambda r: {
                "question": r.get("prompt", ""),
                "answer": None,
                "instruction_id_list": r.get("instruction_id_list"),
                "kwargs": r.get("kwargs"),
                "actual_source": actual_source,
            },
            **kw,
        )
    if name == "hotpotqa":
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        return take(
            ds,
            lambda r: {
                "question": r["question"],
                "answer": r.get("answer"),
                "supporting_facts": r.get("supporting_facts"),
            },
            **kw,
        )
    raise KeyError(name)


def source_dataset(
    name: str, out: Path, *, seed: int = SEED, needed: int = NEEDED, hub_root: Path = Path("/data/huggingface/hub")
) -> dict:
    rows = source_rows(name, seed=seed, needed=needed, hub_root=hub_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            row.update(
                {
                    "dataset": name,
                    "row_id": f"{name}-{i:04d}",
                    "block": i // BLOCK,
                    "messages": [
                        {"role": "system", "content": INSTRUCTION},
                        {"role": "user", "content": row["question"]},
                    ],
                }
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "path": str(out),
        "rows": len(rows),
        "sha256": sha256(out),
        "actual_source": rows[0].get("actual_source", name),
        "blocks": [{"block": b, "offset": b * BLOCK, "rows": BLOCK} for b in range(needed // BLOCK)],
    }


# --- render ------------------------------------------------------------------------------------------


def render_kwargs(enable_thinking: bool) -> dict:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking:
        kwargs["enable_thinking"] = True
    return kwargs


def render_dataset(source: Path, out: Path, *, tokenizer_path: str, enable_thinking: bool) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    rows = read_jsonl(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    prompt_token_max = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt = tokenizer.apply_chat_template(row["messages"], **render_kwargs(enable_thinking))
            tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            prompt_token_max = max(prompt_token_max, tokens)
            payload = {
                "row_id": row["row_id"],
                "block": row["block"],
                "prompt": prompt,
                "prompt_tokens": tokens,
                "answer": row.get("answer"),
                "metadata": {k: v for k, v in row.items() if k not in {"messages", "question", "answer"}},
            }
            encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
            digest.update(encoded)
            handle.write(encoded.decode())
    return {"path": str(out), "rows": len(rows), "sha256": digest.hexdigest(), "prompt_token_max": prompt_token_max}


# --- materialize --------------------------------------------------------------------------------------


def gsm_answer(value: str) -> str:
    match = GSM_ANSWER_RE.search(value)
    return match.group(1).strip() if match else value.strip()


def parse_rows(text: str) -> tuple[int, int]:
    lo, hi = (int(x) for x in text.split(":"))
    return lo, hi


def materialize(
    *,
    source: Path,
    rendered: Path,
    output_dir: Path,
    tokenizer_path: str,
    enable_thinking: bool,
    dataset: str,
    rows: str = "0:64",
    test_rows: int = 16,
    prompts_per_step: int = 16,
    responses_per_prompt: int = 4,
) -> dict:
    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    rendered_rows = read_jsonl(rendered)
    source_by_id = {row["row_id"]: row for row in read_jsonl(source)}
    lo, hi = parse_rows(rows)
    if len(rendered_rows) < hi:
        raise RuntimeError(f"{rendered} has only {len(rendered_rows)} rows; need {hi}")
    out_rows = []
    for index, row in enumerate(rendered_rows[lo:hi]):
        original = source_by_id[row["row_id"]]
        regenerated = tokenizer.apply_chat_template(original["messages"], **render_kwargs(enable_thinking))
        if tokenizer.encode(regenerated, add_special_tokens=False) != tokenizer.encode(
            row["prompt"], add_special_tokens=False
        ):
            raise RuntimeError(f"template mismatch for {dataset}/{row['row_id']}")
        answer = str(original.get("answer", row.get("answer", "")))
        if dataset == "gsm8k":
            data_source, ground_truth = "openai/gsm8k", gsm_answer(answer)
        elif dataset in MATH:
            data_source, ground_truth = "bigmath_math_verify", answer
        else:
            data_source, ground_truth = f"eos_hazard/{dataset}", answer
        out_rows.append(
            {
                "data_source": data_source,
                # The v1 agent loop expects chat messages and renders them itself; the rendered string
                # above is used only for the exact-token validation.
                "prompt": original["messages"],
                "ability": "math" if dataset in MATH else "reasoning",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {"index": index, "row_id": row["row_id"], "dataset": dataset, "rendered_prompt": True},
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(out_rows)
    frame.to_parquet(output_dir / "train.parquet", index=False)
    frame.iloc[:test_rows].to_parquet(output_dir / "test.parquet", index=False)
    manifest = {
        "dataset": dataset,
        "rows": len(out_rows),
        "source_rows": rows,
        "steps": len(out_rows) // prompts_per_step,
        "prompts_per_step": prompts_per_step,
        "responses_per_prompt": responses_per_prompt,
        "requests_per_step": prompts_per_step * responses_per_prompt,
        "rendered_prompt_source": str(rendered),
        "prompt_storage": "messages; rendered by the model tokenizer in verl",
        "train_sha256": sha256(output_dir / "train.parquet"),
        "row_ids": [row["row_id"] for row in rendered_rows[lo:hi]],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# --- CLI ------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("source")
    s.add_argument("--out", type=Path, required=True, help="output directory (<out>/<dataset>.jsonl + manifest.json)")
    s.add_argument("--datasets", nargs="+", default=list(WORKLOADS), choices=WORKLOADS)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--needed", type=int, default=NEEDED)
    s.add_argument("--hub-root", type=Path, default=Path("/data/huggingface/hub"))
    r = sub.add_parser("render")
    r.add_argument("--source", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--enable-thinking", action="store_true")
    m = sub.add_parser("materialize")
    m.add_argument("--source", type=Path, required=True)
    m.add_argument("--rendered", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--tokenizer", required=True)
    m.add_argument("--enable-thinking", action="store_true")
    m.add_argument("--dataset", required=True)
    m.add_argument("--rows", default="0:64")
    m.add_argument("--test-rows", type=int, default=16)
    m.add_argument("--prompts-per-step", type=int, default=16)
    m.add_argument("--responses-per-prompt", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "source":
        manifest = {"selection_seed": args.seed, "datasets": {}}
        for name in args.datasets:
            manifest["datasets"][name] = source_dataset(
                name, args.out / f"{name}.jsonl", seed=args.seed, needed=args.needed, hub_root=args.hub_root
            )
            print(f"prepared {name}: {manifest['datasets'][name]['rows']} rows")
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        return 0
    if args.command == "render":
        info = render_dataset(
            args.source, args.out, tokenizer_path=args.tokenizer, enable_thinking=args.enable_thinking
        )
    else:
        info = materialize(
            source=args.source,
            rendered=args.rendered,
            output_dir=args.out,
            tokenizer_path=args.tokenizer,
            enable_thinking=args.enable_thinking,
            dataset=args.dataset,
            rows=args.rows,
            test_rows=args.test_rows,
            prompts_per_step=args.prompts_per_step,
            responses_per_prompt=args.responses_per_prompt,
        )
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
