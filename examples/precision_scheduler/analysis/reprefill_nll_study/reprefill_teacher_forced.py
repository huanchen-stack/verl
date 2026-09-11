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
"""Teacher-forced reuse-vs-re-prefill mechanism study (single GPU, HF transformers).

For each long-tail BF16 rollout trace, the response is cut at ``--switch-output-token``.
Three continuations of the same canonical tokens are scored:

* ``bf16``: the BF16 model continues from its own state (the reference);
* ``reuse``: the fake-INT4 model continues from the BF16 KV/state (what the dual-precision
  runtime does by default: no re-prefill);
* ``reprefill``: the fake-INT4 model rebuilds the whole prefix itself (what
  ``VLLM_DUAL_PRECISION_REPREFILL=1`` does).

Fake INT4 is a symmetric round-to-nearest, per-row group-128 quantize-dequantize of every
``nn.Linear`` in the language model applied in place after the BF16 phase (the BF16 states and
reference logits are staged to disk first), so both INT4 branches share one identical W4 model.
It is a mechanism study, not the GPTQ/Marlin production path.

Traces are selected by an explicit ``(dataset, request_id)`` list (``--trace-ids``; the
committed ``archived_trace_ids.json`` is the archived run's list), never by path-derived sort
order. Outputs in ``--output``: ``manifest.json``, ``identity.json``, ``quantization.json``,
``window_metrics.jsonl`` (one row per trace x offset), ``summary.json`` (bootstrap CI95 via
``analyze_reprefill_results.summarize``), ``COMPLETED``.

Archived run (Qwen3.5-9B, 16 traces, switch 4096, offsets 0/512/1024/2048/4096, window 64):

    run_gpu.sh --gpus N -- python reprefill_teacher_forced.py \\
        --model /data/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \\
        --inputs eurus=.../eurus/.../responses.jsonl bigmath=.../responses.jsonl gsm8k=.../responses.jsonl \\
        --output runs/qwen35_9b_w4qdq_longtail16
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_reprefill_results import summarize  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_TRACE_IDS = HERE / "archived_trace_ids.json"
#: ``<|im_end|>`` of the Qwen3.5 tokenizer, the id the archived run scored as EOS.
ARCHIVED_EOS_TOKEN_ID = 248046


@dataclass
class Trace:
    dataset: str
    source: str
    request_id: str
    prompt: str
    output_ids: list[int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="HF snapshot directory (local files only)")
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="DATASET=PATH",
        help="BF16 rollout dumps (responses.jsonl with request_id, prompt, output_token_ids), one per dataset label",
    )
    p.add_argument("--output", type=Path, required=True, help="run directory (must not exist)")
    p.add_argument(
        "--trace-ids",
        type=Path,
        default=DEFAULT_TRACE_IDS,
        help="JSON with 'traces': [{dataset, request_id}, ...] in scoring order (default: the archived 16)",
    )
    p.add_argument("--limit", type=int, default=None, help="score only the first N ids of the list")
    p.add_argument("--switch-output-token", type=int, default=4096)
    p.add_argument("--offsets", nargs="*", type=int, default=[0, 512, 1024, 2048, 4096])
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument(
        "--eos-token-id",
        type=int,
        default=ARCHIVED_EOS_TOKEN_ID,
        help="token whose log-probability is reported per window (archived: Qwen3.5 <|im_end|>)",
    )
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_inputs(specs: list[str]) -> dict[str, Path]:
    inputs: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--inputs entries are DATASET=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        if label in inputs:
            raise ValueError(f"duplicate dataset label {label!r}")
        inputs[label] = Path(path)
    return inputs


def load_traces(inputs: dict[str, Path], wanted: list[dict], minimum_output: int) -> list[Trace]:
    """Return the traces named by ``wanted`` (``{dataset, request_id}`` entries), in list order.

    Every id must exist in its dataset's dump and have at least ``minimum_output`` response
    tokens (switch + max offset + window + 1), otherwise the selection is an error rather than a
    silent substitution.
    """
    by_key: dict[tuple[str, str], Trace] = {}
    needed = {(str(e["dataset"]), str(e["request_id"])) for e in wanted}
    for entry in wanted:
        if entry["dataset"] not in inputs:
            raise ValueError(f"unknown dataset {entry['dataset']!r} in trace list; --inputs labels: {sorted(inputs)}")
    for dataset, path in inputs.items():
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (dataset, str(row["request_id"]))
                if key not in needed or key in by_key:
                    continue
                by_key[key] = Trace(
                    dataset=dataset,
                    source=str(path),
                    request_id=str(row["request_id"]),
                    prompt=str(row["prompt"]),
                    output_ids=[int(x) for x in row["output_token_ids"]],
                )
    traces: list[Trace] = []
    for entry in wanted:
        key = (str(entry["dataset"]), str(entry["request_id"]))
        trace = by_key.get(key)
        if trace is None:
            raise ValueError(f"trace {key} not found in {inputs[key[0]]}")
        if len(trace.output_ids) < minimum_output:
            raise ValueError(f"trace {key} has {len(trace.output_ids)} output tokens, shorter than {minimum_output}")
        traces.append(trace)
    return traces


def move_cache(cache: Any, device: torch.device) -> Any:
    moved = copy.deepcopy(cache)
    for layer in moved.layers:
        for name, value in list(vars(layer).items()):
            if isinstance(value, torch.Tensor):
                setattr(layer, name, value.to(device=device))
        if hasattr(layer, "device"):
            layer.device = device
    return moved


def prefill(model: nn.Module, token_ids: list[int], device: torch.device, chunk: int) -> tuple[Any, torch.Tensor]:
    cache = None
    last = None
    for start in range(0, len(token_ids), chunk):
        x = torch.tensor(token_ids[start : start + chunk], device=device).unsqueeze(0)
        out = model(input_ids=x, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        last = out.logits[0, -1].detach()
        del out, x
    assert cache is not None and last is not None
    return cache, last


def score_continuation(
    model: nn.Module,
    cache: Any,
    last_prefix_logits: torch.Tensor,
    continuation: list[int],
    offsets: list[int],
    window: int,
    device: torch.device,
    chunk: int,
) -> dict[int, torch.Tensor]:
    """Logits (fp16, CPU) at every continuation position inside an offset window."""
    wanted = {target_pos for offset in offsets for target_pos in range(offset, offset + window)}
    captured: dict[int, torch.Tensor] = {}
    if 0 in wanted:
        captured[0] = last_prefix_logits.detach().to("cpu", dtype=torch.float16)
    # Logits from continuation input position p predict target position p+1.
    for start in range(0, len(continuation) - 1, chunk):
        stop = min(len(continuation) - 1, start + chunk)
        x = torch.tensor(continuation[start:stop], device=device).unsqueeze(0)
        out = model(input_ids=x, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        for local, input_pos in enumerate(range(start, stop)):
            target_pos = input_pos + 1
            if target_pos in wanted:
                captured[target_pos] = out.logits[0, local].detach().to("cpu", dtype=torch.float16)
        del out, x
    missing = wanted.difference(captured)
    if missing:
        raise RuntimeError(f"Missing scored positions: {sorted(missing)[:8]}")
    return captured


def qdq_int4_language_model(model: nn.Module, group_size: int) -> dict[str, float | int | str]:
    """In-place symmetric RTN INT4 quantize-dequantize of every language-model ``nn.Linear``
    (embeddings, norms, vision tower and LM head excluded)."""
    container = getattr(model, "model", model)
    root = getattr(container, "language_model", container)
    modules = [(name, mod) for name, mod in root.named_modules() if isinstance(mod, nn.Linear)]
    total = 0
    sqerr = 0.0
    sqnorm = 0.0
    maxerr = 0.0
    started = time.time()
    w = None
    with torch.no_grad():
        for index, (_, mod) in enumerate(modules):
            w = mod.weight.data
            rows, cols = w.shape
            total += w.numel()
            for row0 in range(0, rows, 256):
                original = w[row0 : row0 + 256].float()
                pad = (-cols) % group_size
                grouped = torch.nn.functional.pad(original, (0, pad)).reshape(original.shape[0], -1, group_size)
                scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-8) / 7.0
                quant = torch.round(grouped / scale).clamp_(-8, 7)
                restored = (quant * scale).reshape(original.shape[0], -1)[:, :cols]
                error = restored - original
                sqerr += float(error.square().sum().item())
                sqnorm += float(original.square().sum().item())
                maxerr = max(maxerr, float(error.abs().max().item()))
                w[row0 : row0 + 256].copy_(restored.to(dtype=w.dtype))
                del original, grouped, scale, quant, restored, error
            if (index + 1) % 32 == 0:
                print(f"quantized {index + 1}/{len(modules)} linear modules", flush=True)
    if w is not None and w.is_cuda:
        torch.cuda.synchronize(w.device)
    return {
        "modules": len(modules),
        "parameters": total,
        "relative_weight_rmse": math.sqrt(sqerr / max(sqnorm, 1e-30)),
        "max_absolute_weight_error": maxerr,
        "seconds": time.time() - started,
        "recipe": f"symmetric RTN INT4, per-row groups of {group_size}, dequantized to BF16",
        "scope": (
            "all nn.Linear weights inside language_model.model; embeddings, norms, vision tower, and LM head excluded"
        ),
    }


def distribution_metrics(
    bf_logits: torch.Tensor,
    reuse_logits: torch.Tensor,
    replay_logits: torch.Tensor,
    targets: torch.Tensor,
    eos_token_id: int,
) -> dict[str, float]:
    """Per-window NLL / KL / JS / top-1 agreement of the two INT4 branches against BF16."""
    bf = bf_logits.float()
    reuse = reuse_logits.float()
    replay = replay_logits.float()
    bf_lp = torch.log_softmax(bf, dim=-1)
    reuse_lp = torch.log_softmax(reuse, dim=-1)
    replay_lp = torch.log_softmax(replay, dim=-1)
    bf_p = bf_lp.exp()
    reuse_p = reuse_lp.exp()
    replay_p = replay_lp.exp()
    idx = targets.long().unsqueeze(-1)
    nll_bf = -bf_lp.gather(-1, idx).mean()
    nll_reuse = -reuse_lp.gather(-1, idx).mean()
    nll_replay = -replay_lp.gather(-1, idx).mean()
    kl_bf_reuse = (bf_p * (bf_lp - reuse_lp)).sum(-1).mean()
    kl_bf_replay = (bf_p * (bf_lp - replay_lp)).sum(-1).mean()
    midpoint = 0.5 * (reuse_p + replay_p)
    midpoint_lp = midpoint.clamp_min(1e-30).log()
    js = 0.5 * ((reuse_p * (reuse_lp - midpoint_lp)).sum(-1) + (replay_p * (replay_lp - midpoint_lp)).sum(-1)).mean()
    if eos_token_id >= bf.shape[-1]:
        raise ValueError(f"--eos-token-id {eos_token_id} is outside the vocabulary ({bf.shape[-1]} entries)")
    eos = eos_token_id
    return {
        "bf16_nll": float(nll_bf),
        "reuse_nll": float(nll_reuse),
        "reprefill_nll": float(nll_replay),
        "delta_nll_reuse_minus_reprefill": float(nll_reuse - nll_replay),
        "kl_bf16_to_reuse": float(kl_bf_reuse),
        "kl_bf16_to_reprefill": float(kl_bf_replay),
        "delta_kl_reuse_minus_reprefill": float(kl_bf_reuse - kl_bf_replay),
        "js_reuse_reprefill": float(js),
        "reuse_top1_agreement_bf16": float((reuse.argmax(-1) == bf.argmax(-1)).float().mean()),
        "reprefill_top1_agreement_bf16": float((replay.argmax(-1) == bf.argmax(-1)).float().mean()),
        "reuse_eos_logprob": float(reuse_lp[:, eos].mean()),
        "reprefill_eos_logprob": float(replay_lp[:, eos].mean()),
    }


def identity_check(model: nn.Module, device: torch.device) -> dict[str, float]:
    """A copied cache must continue exactly like the native one (guards ``move_cache``)."""
    ids = list(range(11, 11 + 96))
    cache, _ = prefill(model, ids[:64], device, 32)
    copied = move_cache(cache, device)
    x = torch.tensor(ids[64:80], device=device).unsqueeze(0)
    with torch.inference_mode():
        reuse = model(input_ids=x, past_key_values=copied, use_cache=True).logits.float().cpu()
        native_cache, _ = prefill(model, ids[:64], device, 32)
        native = model(input_ids=x, past_key_values=native_cache, use_cache=True).logits.float().cpu()
    diff = (reuse - native).abs()
    return {"max_abs_logit_error": float(diff.max()), "mean_abs_logit_error": float(diff.mean())}


def main() -> None:
    a = parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU (run under run_gpu.sh --gpus N)")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    inputs = parse_inputs(a.inputs)
    wanted = json.loads(a.trace_ids.read_text())["traces"]
    if a.limit is not None:
        wanted = wanted[: a.limit]
    a.output.mkdir(parents=True, exist_ok=False)
    state_dir = a.output / "bf16_intermediates"
    state_dir.mkdir()
    minimum = a.switch_output_token + max(a.offsets) + a.window + 1
    traces = load_traces(inputs, wanted, minimum)
    manifest = {
        "created_unix": time.time(),
        "hostname": platform.node(),
        "command": " ".join(sys.argv),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "model": str(a.model),
        "model_config_sha256": sha256(a.model / "config.json"),
        "inputs": [{"dataset": label, "path": str(p), "sha256": sha256(p)} for label, p in inputs.items()],
        "trace_ids": {"path": str(a.trace_ids), "sha256": sha256(a.trace_ids)},
        "num_traces": len(traces),
        "switch_output_token": a.switch_output_token,
        "offsets": a.offsets,
        "window": a.window,
        "chunk": a.chunk,
        "group_size": a.group_size,
        "seed": a.seed,
        "eos_token_id": a.eos_token_id,
        "execution": "single GPU; BF16 state/logits staged to CPU disk before in-place W4 QDQ",
    }
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"selected {len(traces)} traces", flush=True)

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map=str(device), local_files_only=True
    ).eval()
    with torch.inference_mode():
        identity = identity_check(model, device)
    (a.output / "identity.json").write_text(json.dumps(identity, indent=2))
    print("identity", identity, flush=True)
    if identity["max_abs_logit_error"] > 0.05:
        raise RuntimeError(f"Identity failure: {identity}")

    # Phase 1: BF16 reference logits and the exact switch state, staged to disk.
    with torch.inference_mode():
        for i, trace in enumerate(traces):
            prompt_ids = tokenizer.encode(trace.prompt, add_special_tokens=False)
            switch = len(prompt_ids) + a.switch_output_token
            all_ids = prompt_ids + trace.output_ids
            prefix = all_ids[:switch]
            continuation = all_ids[switch : switch + max(a.offsets) + a.window + 1]
            # At a decode boundary the sampled final prefix token has not yet been consumed
            # into state: both branches consume prefix[-1] under their active model; reuse
            # starts from the BF16 state through prefix[-2], re-prefill rebuilds everything.
            cache, _ = prefill(model, prefix[:-1], device, a.chunk)
            torch.save(move_cache(cache, torch.device("cpu")), state_dir / f"cache_{i:03d}.pt")
            last = torch.tensor([prefix[-1]], device=device).unsqueeze(0)
            out = model(input_ids=last, past_key_values=cache, use_cache=True)
            caps = score_continuation(
                model, out.past_key_values, out.logits[0, -1], continuation, a.offsets, a.window, device, a.chunk
            )
            torch.save(caps, state_dir / f"logits_{i:03d}.pt")
            meta = {
                "dataset": trace.dataset,
                "request_id": trace.request_id,
                "source": trace.source,
                "prompt_ids": prompt_ids,
                "prefix": prefix,
                "continuation": continuation,
                "output_tokens": len(trace.output_ids),
            }
            torch.save(meta, state_dir / f"meta_{i:03d}.pt")
            del cache, out, caps
            torch.cuda.empty_cache()
            print(f"BF16 phase {i + 1}/{len(traces)}", flush=True)

    quant = qdq_int4_language_model(model, a.group_size)
    (a.output / "quantization.json").write_text(json.dumps(quant, indent=2))
    print("quantization", quant, flush=True)

    # Phase 2: reuse (BF16 state + W4 model) vs re-prefill (W4 state + W4 model).
    rows: list[dict[str, Any]] = []
    started = time.time()
    metrics_path = a.output / "window_metrics.jsonl"
    with torch.inference_mode():
        for i, trace in enumerate(traces):
            meta = torch.load(state_dir / f"meta_{i:03d}.pt", weights_only=False)
            bf_caps = torch.load(state_dir / f"logits_{i:03d}.pt", weights_only=False)
            cache_cpu = torch.load(state_dir / f"cache_{i:03d}.pt", weights_only=False)
            reuse_cache = move_cache(cache_cpu, device)
            prefix = meta["prefix"]
            continuation = meta["continuation"]

            last = torch.tensor([prefix[-1]], device=device).unsqueeze(0)
            out = model(input_ids=last, past_key_values=reuse_cache, use_cache=True)
            reuse_caps = score_continuation(
                model, out.past_key_values, out.logits[0, -1], continuation, a.offsets, a.window, device, a.chunk
            )
            del cache_cpu, reuse_cache, out
            replay_cache, replay_last = prefill(model, prefix, device, a.chunk)
            replay_caps = score_continuation(
                model, replay_cache, replay_last, continuation, a.offsets, a.window, device, a.chunk
            )
            del replay_cache

            for offset in a.offsets:
                positions = list(range(offset, offset + a.window))
                bf = torch.stack([bf_caps[p] for p in positions])
                reuse = torch.stack([reuse_caps[p] for p in positions])
                replay = torch.stack([replay_caps[p] for p in positions])
                targets = torch.tensor([continuation[p] for p in positions])
                row = {
                    "trace_index": i,
                    "dataset": trace.dataset,
                    "request_id": trace.request_id,
                    "source": trace.source,
                    "prompt_tokens": len(meta["prompt_ids"]),
                    "output_tokens": meta["output_tokens"],
                    "switch_absolute_token": len(prefix),
                    "offset": offset,
                    "scored_tokens": len(positions),
                    **distribution_metrics(bf, reuse, replay, targets, a.eos_token_id),
                }
                rows.append(row)
                with metrics_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            del bf_caps, reuse_caps, replay_caps, meta
            torch.cuda.empty_cache()
            print(f"W4 phase {i + 1}/{len(traces)}", flush=True)

    summary = summarize(rows, a.offsets, a.seed, wall_seconds=time.time() - started)
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (a.output / "COMPLETED").write_text("ok\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
