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
"""Tail-W4 calibration: continue BF16 prefixes under W4 from population-quantile cut points.

The legacy second calibration path decoded every request from token zero under uniform W4, which
describes a population the scheduler never creates (a switched request has a BF16 prefix). This
path measures what the scheduler actually does: at each *cut frontier* ``F_k`` every request of the
BF16 calibration that was still generating at ``F_k`` is re-issued with prompt = original prompt +
its own first ``F_k`` generated tokens and decoded under W4 with ``max_tokens = cap - F_k``. One
request appears once per cut it survives, so a cap-runner is in every group -- the population
alive at ``F_k`` is exactly the calibration requests with length >= ``F_k``. The cuts default to
population quantiles of the BF16 lengths (``--cut-quantiles 0.667,0.75,0.8,0.9``), so the second
path costs ``sum(1 - q)`` continuations per BF16 request (0.88 N for the default four) in a single
batch, and re-targets itself per model. ``--cut-tokens 0`` reproduces the uniform-W4 path.

Output: a request-lifetime trace in the standard format plus two start-row keys, ``prefix_tokens``
and ``cut_frontier`` (both the cut, in generated tokens), and a ``calibration_manifest.json``.
Request ids are ``<trace_request_id>@<cut>`` so the same prompt at different cuts stays distinct.
:func:`~.calibration.grouped_traces` consumes the trace (one W4 group per cut).

The engine is only touched by :class:`VllmEngine` (lazy import); planning, id handling and trace
writing are pure and unit-tested with a fake engine. Nothing here runs on import.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .traces import read_jsonl

DEFAULT_QUANTILES = (2.0 / 3.0, 0.75, 0.8, 0.9)


@dataclass(frozen=True)
class ContinuationRequest:
    request_id: str  # "<trace_request_id>@<cut>"
    trace_request_id: str
    cut: int  # BF16 generated tokens kept as prefix
    prompt_ids: tuple[int, ...]
    prefix_ids: tuple[int, ...]
    max_tokens: int
    bf16_length: int  # the request's full BF16 generation length (capped)

    @property
    def input_ids(self) -> list[int]:
        return list(self.prompt_ids) + list(self.prefix_ids)


@dataclass
class ContinuationResult:
    request_id: str
    generation_tokens: int
    finish_reason: str
    token_ids: list[int] | None = None


def cut_frontiers(
    lengths: Sequence[int],
    *,
    quantiles: Sequence[float] | None = None,
    tokens: Sequence[int] | None = None,
    step: int = 250,
    cap: int,
) -> list[int]:
    """Cut points on the frontier grid: population quantiles of ``lengths`` (default) or explicit tokens.

    Quantile cuts are rounded *down* to the grid, deduplicated and sorted; a cut of 0 (uniform W4)
    is allowed explicitly via ``tokens``. Cuts at or beyond the cap are rejected.
    """
    if (quantiles is None) == (tokens is None):
        raise ValueError("give exactly one of quantiles or tokens")
    if tokens is not None:
        cuts = sorted({int(t) for t in tokens})
    else:
        arr = np.asarray(lengths, dtype=np.int64)
        if not len(arr):
            raise ValueError("no BF16 lengths to take quantiles of")
        cuts = sorted({int(np.floor(np.percentile(arr, 100.0 * q) / step) * step) for q in quantiles})
    for cut in cuts:
        if cut < 0 or cut % step or cut >= cap:
            raise ValueError(f"cut {cut} is not on the {step}-token grid below cap {cap}")
    return cuts


def bf16_requests(trace: Path, *, cap: int, limit: int | None = None) -> list[dict[str, Any]]:
    """Completed BF16 calibration requests with their generated token ids.

    Returns rows ``{"trace_request_id", "request_id", "prompt_tokens", "token_ids", "length"}`` in
    start order (at most ``limit``); the trace must have been written with token logging on.
    """
    starts: list[dict[str, Any]] = []
    finishes: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(trace):
        if row.get("event") == "start" and (limit is None or len(starts) < limit):
            starts.append(row)
        elif row.get("event") == "finish":
            finishes[str(row.get("request_id"))] = row
    out = []
    for start in starts:
        rid = str(start["request_id"])
        fin = finishes.get(rid)
        if fin is None:
            raise ValueError(f"request {rid} has no finish row in {trace}")
        ids = fin.get("token_ids")
        if ids is None:
            raise ValueError(f"{trace} has no token_ids: rerun the BF16 calibration with request_trace_log_tokens=true")
        ids = [int(t) for t in ids][:cap]
        out.append(
            {
                "trace_request_id": str(start.get("trace_request_id", rid)),
                "request_id": rid,
                "prompt_tokens": int(start["prompt_tokens"]),
                "token_ids": ids,
                "length": min(int(fin["generation_tokens"]), cap),
            }
        )
    return out


def plan_continuations(
    rows: Iterable[dict[str, Any]],
    cuts: Sequence[int],
    *,
    cap: int,
    prompt_ids_for: Callable[[str], Sequence[int]],
) -> list[ContinuationRequest]:
    """One continuation per (request, cut) with request length >= cut > 0, or every request for cut 0.

    ``prompt_ids_for(trace_request_id)`` must return the exact prompt token ids used by the BF16
    rollout; its length is checked against the trace's ``prompt_tokens``.
    """
    plan: list[ContinuationRequest] = []
    for row in rows:
        tid = row["trace_request_id"]
        prompt = tuple(int(t) for t in prompt_ids_for(tid))
        if len(prompt) != row["prompt_tokens"]:
            raise ValueError(
                f"{tid}: rebuilt prompt has {len(prompt)} tokens, trace recorded {row['prompt_tokens']} "
                "(chat template / tokenizer mismatch)"
            )
        ids = row["token_ids"]
        for cut in cuts:
            if cut > 0 and row["length"] < cut:
                continue
            if len(ids) < cut:
                raise ValueError(f"{tid}: {len(ids)} logged token ids but cut {cut}")
            plan.append(
                ContinuationRequest(
                    request_id=f"{tid}@{cut}",
                    trace_request_id=tid,
                    cut=int(cut),
                    prompt_ids=prompt,
                    prefix_ids=tuple(ids[:cut]),
                    max_tokens=int(cap - cut),
                    bf16_length=int(row["length"]),
                )
            )
    return plan


def split_request_id(request_id: str) -> tuple[str, int]:
    """``"<trace_request_id>@<cut>"`` -> ``(trace_request_id, cut)``."""
    base, sep, cut = request_id.rpartition("@")
    if not sep:
        raise ValueError(f"not a continuation request id: {request_id!r}")
    return base, int(cut)


def manifest(cuts: Sequence[int], plan: Sequence[ContinuationRequest], *, quantiles: Sequence[float] | None, cap: int, bf16_requests_count: int) -> dict[str, Any]:
    per_cut = {int(c): sum(1 for r in plan if r.cut == c) for c in cuts}
    return {
        "kind": "tail-w4 continuation calibration",
        "cap": int(cap),
        "cuts": [int(c) for c in cuts],
        "cut_quantiles": [float(q) for q in quantiles] if quantiles is not None else None,
        "bf16_requests": int(bf16_requests_count),
        "continuations": int(len(plan)),
        "continuations_per_cut": per_cut,
        "max_continuation_tokens": int(sum(r.max_tokens for r in plan)),
        "bf16_remaining_tokens": int(sum(r.bf16_length - r.cut for r in plan)),
        "prefix_tokens_to_prefill": int(sum(r.cut for r in plan)),
    }


class ContinuationTraceWriter:
    """Writes the continuation trace in the request-lifetime format (plus ``prefix_tokens`` / ``cut_frontier``)."""

    def __init__(self, path: Path, *, log_tokens: bool = False, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log_tokens = log_tokens
        self._clock = clock
        self._handle = self.path.open("w")

    def start(self, req: ContinuationRequest) -> None:
        self._write(
            {
                "event": "start",
                "timestamp": self._clock(),
                "request_id": req.request_id,
                "trace_request_id": req.trace_request_id,
                "prompt_tokens": len(req.prompt_ids),
                "prefix_tokens": req.cut,
                "cut_frontier": req.cut,
                "bf16_length": req.bf16_length,
            }
        )

    def finish(self, res: ContinuationResult) -> None:
        row: dict[str, Any] = {
            "event": "finish",
            "timestamp": self._clock(),
            "request_id": res.request_id,
            "trace_request_id": split_request_id(res.request_id)[0],
            "generation_tokens": int(res.generation_tokens),
            "finish_reason": res.finish_reason,
        }
        if self.log_tokens and res.token_ids is not None:
            row["token_ids"] = list(res.token_ids)
        self._write(row)

    def _write(self, row: dict[str, Any]) -> None:
        self._handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def run_continuations(engine: Any, plan: Sequence[ContinuationRequest], writer: ContinuationTraceWriter) -> list[ContinuationResult]:
    """Record starts, run ``engine.generate(plan)`` (all requests in one batch), record finishes."""
    for req in plan:
        writer.start(req)
    results: list[ContinuationResult] = list(engine.generate(plan))
    got = {r.request_id for r in results}
    missing = [r.request_id for r in plan if r.request_id not in got]
    if missing:
        raise RuntimeError(f"engine returned no result for {len(missing)} requests, e.g. {missing[:3]}")
    for res in results:
        writer.finish(res)
    return results


def prompt_ids_from_parquet(
    parquet: Path,
    tokenizer: Any,
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
    prompt_key: str = "prompt",
) -> Callable[[str], list[int]]:
    """``trace_request_id -> prompt token ids`` for the verl dataset the BF16 rollout used.

    Trace ids are ``idx-<extra_info.index>_<session>``; the prompt is tokenized the way the agent
    loop does it (``apply_chat_template(..., add_generation_prompt=True, tokenize=True, **kwargs)``).
    """
    import pandas as pd

    frame = pd.read_parquet(parquet)
    by_index: dict[int, Any] = {}
    for pos in range(len(frame)):
        extra = frame.iloc[pos].get("extra_info") if "extra_info" in frame.columns else None
        index = int(extra["index"]) if isinstance(extra, dict) and "index" in extra else pos
        by_index[index] = frame.iloc[pos][prompt_key]
    kwargs = dict(chat_template_kwargs or {})
    cache: dict[int, list[int]] = {}

    def lookup(trace_request_id: str) -> list[int]:
        head = trace_request_id.split("_")[0]
        if not head.startswith("idx-"):
            raise ValueError(f"unexpected trace id {trace_request_id!r}")
        index = int(head[len("idx-") :])
        if index not in cache:
            messages = [dict(m) for m in by_index[index]]
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kwargs)
            if not isinstance(ids, list):  # newer transformers return a BatchEncoding
                ids = ids["input_ids"]
            cache[index] = [int(t) for t in ids]
        return cache[index]

    return lookup


class VllmEngine:
    """In-process vLLM engine under the dual-precision runtime with the uniform-W4 policy.

    Built from the same config block the recipes use (``PrecisionSchedulerConfig`` ->
    ``to_vllm_env``), so the shadow, LoRA fast path and validation knobs match the RL runs.
    """

    def __init__(
        self,
        *,
        model: str,
        int4_model: str,
        cap: int,
        prompt_cap: int,
        seed: int = 42,
        gpu_memory_utilization: float = 0.5,
        max_num_seqs: int = 64,
        lora_adapter: str | None = None,
        lora_fast_path: bool = True,
        lora_dual_stream: bool = True,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        from verl.workers.config.precision_scheduler import PrecisionSchedulerConfig, to_vllm_env

        cfg = PrecisionSchedulerConfig(
            enable=True,
            int4_model=int4_model,
            policy="uniform_w4",
            lora_fast_path=lora_fast_path,
            lora_dual_stream=lora_dual_stream,
        )
        for key, value in {**to_vllm_env(cfg), **(extra_env or {})}.items():
            os.environ[key] = value
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM

        self.cap = int(cap)
        self.seed = int(seed)
        self.lora_adapter = lora_adapter
        self.llm = LLM(
            model=model,
            trust_remote_code=True,
            enable_lora=lora_adapter is not None,
            max_lora_rank=16,
            max_loras=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=int(prompt_cap + cap),
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=False,
            dtype="bfloat16",
            seed=seed,
            disable_log_stats=True,
        )

    def generate(self, plan: Sequence[ContinuationRequest]) -> list[ContinuationResult]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        lora = None
        if self.lora_adapter is not None:
            from vllm.lora.request import LoRARequest

            lora = LoRARequest("calib_adapter", 1, self.lora_adapter)
        prompts = [TokensPrompt(prompt_token_ids=req.input_ids) for req in plan]
        params = [SamplingParams(max_tokens=req.max_tokens, temperature=1.0, top_p=1.0, seed=self.seed) for req in plan]
        outputs = self.llm.generate(prompts, params, lora_request=lora, use_tqdm=False)
        results = []
        for req, out in zip(plan, outputs):
            comp = out.outputs[0]
            results.append(
                ContinuationResult(
                    request_id=req.request_id,
                    generation_tokens=len(comp.token_ids),
                    finish_reason=str(comp.finish_reason),
                    token_ids=list(comp.token_ids),
                )
            )
        return results


def plan_summary(plan: Sequence[ContinuationRequest]) -> dict[str, Any]:
    return {k: v for k, v in manifest(sorted({r.cut for r in plan}), plan, quantiles=None, cap=0, bf16_requests_count=0).items() if k in ("continuations", "continuations_per_cut", "max_continuation_tokens", "bf16_remaining_tokens", "prefix_tokens_to_prefill")}


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = [
    "DEFAULT_QUANTILES",
    "ContinuationRequest",
    "ContinuationResult",
    "ContinuationTraceWriter",
    "VllmEngine",
    "bf16_requests",
    "cut_frontiers",
    "manifest",
    "plan_continuations",
    "plan_summary",
    "prompt_ids_from_parquet",
    "run_continuations",
    "split_request_id",
    "write_manifest",
]
