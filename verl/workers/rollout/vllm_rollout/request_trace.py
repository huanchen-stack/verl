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
"""Opt-in per-request lifetime trace written by the vLLM server actor.

One JSONL file per (replica, node) with a ``start`` row when a request enters
``vLLMHttpServer.generate`` and a ``finish`` row when it leaves.  File name and keys are
frozen: the policy builders and the online-EMA loop parse
``request_lifetimes_replica000_node000.jsonl`` by name and ``generation_tokens`` by key.

Schema (``json.dumps(sort_keys=True)``, one row per line)::

    start:  {"event": "start", "prompt_tokens": int, "request_id": str, "timestamp": float
             [, "trace_request_id": str]}
    finish: {"event": "finish", "finish_reason": str, "generation_tokens": int, "request_id": str,
             "timestamp": float [, "token_ids": list[int]] [, "trace_request_id": str]}

``request_id`` is the engine id (a fresh uuid per turn); ``trace_request_id`` is the stable
agent-loop id (``<uid>_<session_id>``) when the caller supplies one, so paired runs can be
joined without touching the engine id.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from typing import Any, Optional

__all__ = ["RequestLifetimeTracer", "trace_file_name"]


def trace_file_name(replica_rank: int, node_rank: int) -> str:
    return f"request_lifetimes_replica{int(replica_rank):03d}_node{int(node_rank):03d}.jsonl"


class RequestLifetimeTracer:
    """Append-only JSONL writer with one handle per server (the archived code reopened per row)."""

    def __init__(
        self,
        trace_dir: str | os.PathLike[str],
        replica_rank: int,
        node_rank: int,
        *,
        log_tokens: bool = False,
        clock: Callable[[], float] = time.time,
    ):
        self.trace_dir = os.fspath(trace_dir)
        self.log_tokens = bool(log_tokens)
        self._clock = clock
        os.makedirs(self.trace_dir, exist_ok=True)
        self.path = os.path.join(self.trace_dir, trace_file_name(replica_rank, node_rank))
        self._handle = None  # opened on first row so idle (node_rank > 0) servers create no file

    @classmethod
    def from_config(
        cls,
        config: Any,
        replica_rank: int,
        node_rank: int,
        environ: Optional[dict[str, str]] = None,
    ) -> Optional[RequestLifetimeTracer]:
        """Build a tracer from ``rollout.precision_scheduler`` or return None when tracing is off.

        ``request_trace_dir`` / ``request_trace_log_tokens`` from the config block win; the
        ``VERL_REQUEST_TRACE_DIR`` / ``VERL_REQUEST_TRACE_LOG_TOKENS`` environment variables
        (the wire format emitted from the same block) are the fallback.
        """
        environ = os.environ if environ is None else environ
        trace_dir = getattr(config, "request_trace_dir", None)
        if trace_dir:
            log_tokens = bool(getattr(config, "request_trace_log_tokens", False))
            return cls(trace_dir, replica_rank, node_rank, log_tokens=log_tokens)
        trace_dir = environ.get("VERL_REQUEST_TRACE_DIR")
        if trace_dir:
            log_tokens = environ.get("VERL_REQUEST_TRACE_LOG_TOKENS", "0") == "1"
            return cls(trace_dir, replica_rank, node_rank, log_tokens=log_tokens)
        return None

    def _write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            self._handle = open(self.path, "a", encoding="utf-8")
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()

    def record_start(self, request_id: str, prompt_tokens: int, trace_request_id: Optional[str] = None) -> None:
        record: dict[str, Any] = {
            "event": "start",
            "timestamp": self._clock(),
            "request_id": request_id,
            "prompt_tokens": int(prompt_tokens),
        }
        if trace_request_id is not None:
            record["trace_request_id"] = trace_request_id
        self._write(record)

    def record_finish(
        self,
        request_id: str,
        generation_tokens: int,
        finish_reason: Optional[str],
        token_ids: Optional[Iterable[int]] = None,
        trace_request_id: Optional[str] = None,
    ) -> None:
        record: dict[str, Any] = {
            "event": "finish",
            "timestamp": self._clock(),
            "request_id": request_id,
            "generation_tokens": int(generation_tokens),
            "finish_reason": finish_reason,
        }
        if self.log_tokens and token_ids is not None:
            record["token_ids"] = [int(t) for t in token_ids]
        if trace_request_id is not None:
            record["trace_request_id"] = trace_request_id
        self._write(record)

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass
