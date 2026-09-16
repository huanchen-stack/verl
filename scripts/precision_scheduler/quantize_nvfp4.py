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
"""Produce an NVFP4 checkpoint with NVIDIA ModelOpt, for models the Hub has none of.

The precision-scheduler baselines need a BF16 / W4A16 / NVFP4 triple per model.
Qwen3.5-9B has all three on the Hub; Phi-4-mini-reasoning has no NVFP4, so this
builds one from its BF16 weights. Calibration prompts come from the run's own
GSM8K parquet, so the static activation scales are fit on the distribution the
rollouts actually see rather than on generic web text.

    quantize_nvfp4.py --model microsoft/Phi-4-mini-reasoning \
        --calib-parquet ~/ps_data/gsm8k_messages_2048/train.parquet \
        --output-dir ~/models/Phi-4-mini-reasoning-NVFP4
"""

from __future__ import annotations

import argparse
from pathlib import Path


def calibration_texts(parquet: Path, tokenizer, limit: int) -> list[str]:
    """Render the parquet's chat messages with the target model's own template."""
    import pandas as pd

    frame = pd.read_parquet(parquet)
    column = "prompt" if "prompt" in frame.columns else frame.columns[0]
    texts: list[str] = []
    for value in frame[column].tolist()[:limit]:
        messages = list(value) if not isinstance(value, str) else None
        if messages is None:
            texts.append(value)
            continue
        texts.append(
            tokenizer.apply_chat_template(
                [dict(m) for m in messages], tokenize=False, add_generation_prompt=True
            )
        )
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="BF16 checkpoint (HF id or path)")
    parser.add_argument("--calib-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-seq-len", type=int, default=1024)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    texts = calibration_texts(args.calib_parquet, tokenizer, args.calib_samples)
    print(f"calibrating on {len(texts)} prompts from {args.calib_parquet}", flush=True)

    def forward_loop(module) -> None:
        for index, text in enumerate(texts):
            batch = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.calib_seq_len,
            ).to("cuda")
            with torch.no_grad():
                module(**batch)
            if (index + 1) % 32 == 0:
                print(f"  calibrated {index + 1}/{len(texts)}", flush=True)

    # NVFP4_DEFAULT_CFG is W4A4: 4-bit float weights in groups of 16 plus quantized
    # activations, which is what the Hub's NVFP4 Qwen checkpoint also carries, so the
    # two models' NVFP4 arms stay comparable.
    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)
    mtq.print_quant_summary(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_hf_checkpoint(model, export_dir=str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"NVFP4 checkpoint written to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
