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
"""Deterministic held-out evaluator of a LoRA patch on the common BF16 base model (100-step protocol).

From the archived hardmath ``evaluate_bf16_patch.py``: greedy decoding (temperature 0, fixed seed) of a
held-out parquet with vLLM + the adapter, scored with ``examples/precision_scheduler/rewards.py``
(dispatch on the parquet ``data_source``), one JSONL row per prompt and a ``.summary.json`` with the
accuracy and its Wilson 95% interval. Parameterised: base model, seed, thinking, row limit.

``--adapter`` accepts a PEFT adapter directory (``adapter_config.json`` + ``adapter_model.safetensors``)
or a verl FSDP checkpoint (``global_step_<k>`` or its ``actor/`` directory), in which case the LoRA
weights are exported first with ``verl.model_merger`` into ``<checkpoint>/actor/lora_adapter`` (also
available standalone as ``export-adapter``). Megatron checkpoints already hold
``actor/model/huggingface/adapter``.

    evaluate_lora_patch.py --base-model Qwen/Qwen3.5-9B --data validation.parquet --adapter <ckpt>/global_step_10 \
        --checkpoint-step 10 --configuration full_w4 --output eval/full_w4_step10.jsonl
    evaluate_lora_patch.py export-adapter <ckpt>/global_step_0 [--out <dir>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

DEFAULT_SEED = 20260825
HERE = Path(__file__).resolve().parent


def wilson(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count == 0:
        return float("nan"), float("nan")
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return center - radius, center + radius


def load_rewards():
    spec = importlib.util.spec_from_file_location("ps_rewards", HERE.parent / "rewards.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def is_peft_dir(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (path / "adapter_model.safetensors").is_file()


def export_fsdp_adapter(checkpoint: Path, out_dir: Path | None = None) -> Path:
    """Write the PEFT adapter of a verl FSDP LoRA checkpoint (``global_step_k`` or ``actor``)."""
    actor = checkpoint / "actor" if (checkpoint / "actor").is_dir() else checkpoint
    if not (actor / "fsdp_config.json").is_file():
        raise FileNotFoundError(f"not a verl FSDP checkpoint: {actor}")
    out_dir = out_dir or actor / "lora_adapter"
    if is_peft_dir(out_dir):
        return out_dir
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    merger = FSDPModelMerger(
        ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            local_dir=str(actor),
            target_dir=str(out_dir.parent),
            hf_model_config_path=str(actor / "huggingface"),
            trust_remote_code=True,
        )
    )
    world_size = merger._get_world_size()
    rank_zero = merger._load_rank_zero_state_dict(world_size)
    mesh, names = merger._extract_device_mesh_info(rank_zero, world_size)
    total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, names)
    merged = merger._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, names)
    saved = merger.save_lora_adapter(merged)
    if saved is None:
        raise RuntimeError(f"no LoRA parameters in {actor}")
    saved_path = Path(saved)
    if saved_path != out_dir:
        saved_path.rename(out_dir)
    return out_dir


def resolve_adapter(path: Path | None) -> Path | None:
    if path is None:
        return None
    if is_peft_dir(path):
        return path
    megatron = path / "actor" / "model" / "huggingface" / "adapter"
    if is_peft_dir(megatron):
        return megatron
    return export_fsdp_adapter(path)


def render_prompts(tokenizer, frame, enable_thinking: bool) -> list[str]:
    prompts = []
    for messages in frame["prompt"]:
        messages = messages.tolist() if hasattr(messages, "tolist") else messages
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking:
            kwargs["enable_thinking"] = True
        prompts.append(tokenizer.apply_chat_template(list(messages), **kwargs))
    return prompts


def evaluate(args: argparse.Namespace) -> dict:
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    adapter = resolve_adapter(args.adapter)
    rewards = load_rewards()
    frame = pd.read_parquet(args.data)
    if args.limit:
        frame = frame.iloc[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.base_model, trust_remote_code=True)
    prompts = render_prompts(tokenizer, frame, not args.no_thinking)
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_response_tokens, seed=args.seed)
    started = time.time()
    llm = LLM(
        model=str(args.base_model),
        dtype="bfloat16",
        max_model_len=args.max_response_tokens + args.max_prompt_tokens,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=adapter is not None,
        max_lora_rank=args.max_lora_rank,
        enable_prefix_caching=False,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
    )
    request = None if adapter is None else LoRARequest("evaluated_patch", 1, str(adapter))
    generations = llm.generate(prompts, params, lora_request=request)
    responses = [item.outputs[0].text for item in generations]
    golds = [item["ground_truth"] for item in frame["reward_model"]]
    sources = [args.data_source or str(source) for source in frame["data_source"]]
    scored = [rewards.compute_score(src, out, gold) for src, out, gold in zip(sources, responses, golds, strict=True)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for row_index, (info, output, reward) in enumerate(zip(frame["extra_info"], generations, scored, strict=True)):
            info = dict(info)
            stream.write(
                json.dumps(
                    {
                        "configuration": args.configuration,
                        "checkpoint_step": args.checkpoint_step,
                        "row_index": row_index,
                        "source_index": info.get("source_index", info.get("index")),
                        "reward": float(reward["accuracy"]),
                        "response_tokens": len(output.outputs[0].token_ids),
                        "finish_reason": output.outputs[0].finish_reason,
                        "response": responses[row_index],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    successes = int(sum(float(item["accuracy"]) for item in scored))
    low, high = wilson(successes, len(scored))
    summary = {
        "configuration": args.configuration,
        "checkpoint_step": args.checkpoint_step,
        "base_model": str(args.base_model),
        "base_dtype": "bfloat16",
        "adapter": str(adapter) if adapter else None,
        "data": str(args.data),
        "requests": len(scored),
        "decoding": {"temperature": 0.0, "max_response_tokens": args.max_response_tokens, "seed": args.seed},
        "accuracy": successes / len(scored),
        "successes": successes,
        "wilson_95_low": low,
        "wilson_95_high": high,
        "mean_response_tokens": sum(len(item.outputs[0].token_ids) for item in generations) / len(generations),
        "wall_seconds": time.time() - started,
        "results": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    ex = sub.add_parser("export-adapter", help="export the PEFT adapter of a verl FSDP LoRA checkpoint")
    ex.add_argument("checkpoint", type=Path)
    ex.add_argument("--out", type=Path, default=None)
    ev = sub.add_parser("evaluate", help="greedy held-out evaluation (default command)")
    ev.add_argument("--base-model", required=True)
    ev.add_argument("--tokenizer", default=None)
    ev.add_argument("--data", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--adapter", type=Path, default=None)
    ev.add_argument("--checkpoint-step", type=int, required=True)
    ev.add_argument("--configuration", required=True)
    ev.add_argument("--data-source", default=None, help="override the parquet data_source for scoring")
    ev.add_argument("--max-response-tokens", type=int, default=8192)
    ev.add_argument("--max-prompt-tokens", type=int, default=2048)
    ev.add_argument("--max-num-seqs", type=int, default=128)
    ev.add_argument("--max-lora-rank", type=int, default=16)
    ev.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ev.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ev.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows (smoke)")
    ev.add_argument("--no-thinking", action="store_true")
    ev.add_argument("--enforce-eager", action="store_true")
    ev.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("export-adapter", "evaluate", "-h", "--help"):
        argv = ["evaluate", *argv]
    args = build_parser().parse_args(argv)
    if args.command == "export-adapter":
        print(export_fsdp_adapter(args.checkpoint, args.out))
        return 0
    evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
