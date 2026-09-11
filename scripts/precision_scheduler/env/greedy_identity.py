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
"""Greedy-decode a fixed prompt set with vLLM offline `LLM(...)` and dump the token ids.

Run once per environment (vanilla, dirty with all precision-scheduler flags off, clean), always
under `activate.sh <env>` and `run_gpu.sh`; `tests/precision_scheduler/gpu/test_greedy_identity.py`
compares the three dumps token for token. Identity holds because all three trees share the same
precompiled kernels and weights and decode greedily; it establishes that "vanilla" equals "dirty
with flags off" before any refactor (C0 acceptance).

    greedy_identity.py --env clean --out /tmp/clean.json [--model PATH] [--prompts FILE]
                       [--max-tokens 64] [--gpu-memory-utilization 0.5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Phi-4-mini-reasoning: a standard transformer, so vLLM's batch-invariant mode applies (it refuses
# Qwen3.5's GDN linear attention). Qwen3.5-4B lives at
# /data/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
DEFAULT_MODEL = (
    "/data/huggingface/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a"
)
DEFAULT_PROMPTS = (
    Path(__file__).resolve().parents[3] / "tests" / "precision_scheduler" / "gpu" / "fixtures" / "gsm8k_16_prompts.json"
)
# Every precision-scheduler knob must be unset so the dirty tree behaves like vanilla.
FLAG_PREFIXES = ("VLLM_DUAL_PRECISION", "ROLLOUT_QLORA", "VLLM_LORA_ENABLE_DUAL_STREAM", "VLLM_REPREFILL")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, choices=("clean", "dirty", "vanilla"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("PS_IDENTITY_MODEL", DEFAULT_MODEL))
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="1 (default) decodes prompts one at a time: batched bf16 decoding is not batch-invariant, so a "
        "larger value makes even two runs of the same tree diverge (measured: 4/16 prompts at 16)",
    )
    ap.add_argument("--batch-invariant", action="store_true", help="set VLLM_BATCH_INVARIANT=1 before importing vllm")
    ap.add_argument("--enforce-eager", action="store_true", help="diagnostic: skip torch.compile and CUDA graphs")
    args = ap.parse_args(argv)
    if args.batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"

    leaked = sorted(k for k in os.environ if k.startswith(FLAG_PREFIXES))
    if leaked:
        print(f"greedy_identity: refusing to run with feature flags set: {leaked}", file=sys.stderr)
        return 2
    if os.environ.get("PS_ENV") != args.env:
        print(f"greedy_identity: PS_ENV={os.environ.get('PS_ENV')!r} != --env {args.env}", file=sys.stderr)
        return 2

    import vllm
    from vllm import LLM, SamplingParams

    vllm_root = os.environ.get("VLLM_ROOT", "")
    if not vllm.__file__.startswith(vllm_root.rstrip("/") + "/"):
        print(f"greedy_identity: vllm imported from {vllm.__file__}, not {vllm_root}", file=sys.stderr)
        return 2

    spec = json.loads(Path(args.prompts).read_text())
    messages = [p["messages"] for p in spec["prompts"]]

    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        seed=0,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)
    outs = llm.generate(prompts, sp)
    results = []
    for i, o in enumerate(outs):
        c = o.outputs[0]
        results.append(
            {
                "index": i,
                "prompt_token_ids": list(o.prompt_token_ids),
                "token_ids": list(c.token_ids),
                "text": c.text,
                "finish_reason": c.finish_reason,
            }
        )
    dump = {
        "env": args.env,
        "vllm_file": vllm.__file__,
        "vllm_version": vllm.__version__,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "max_num_seqs": args.max_num_seqs,
        "batch_invariant": args.batch_invariant,
        "enforce_eager": args.enforce_eager,
        "elapsed_s": round(time.time() - t0, 2),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(dump, indent=1))
    n = sum(len(r["token_ids"]) for r in results)
    print(f"greedy_identity[{args.env}]: {len(results)} prompts, {n} tokens, vllm={vllm.__file__} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
