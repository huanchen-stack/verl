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
"""Request lifetime tracer: golden schema vs the archived trace, opt-in behavior through generate(),
forced sleep level, and the stable trace id format."""

from __future__ import annotations

import asyncio
import itertools
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from verl.workers.config import PrecisionSchedulerConfig
from verl.workers.rollout.vllm_rollout.request_trace import RequestLifetimeTracer, trace_file_name

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ARCHIVED_TRACE = Path(
    "/data/huanchen/verl/.codex-report/rl-workflow/raw/e2_24k_fullstep_traces/e2_24k_full_r3/bf16/"
    "request_lifetimes_replica000_node000.jsonl"
)
ARCHIVED_TOKEN_TRACE = Path(
    "/data/huanchen/verl/.codex-report/new-storyline-experiments/best_t8_no_reprefill_gpu7/attempts/bf16/"
    "set_02/seed_42/traces/request_lifetimes_replica000_node000.jsonl"
)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _key_sets(rows):
    starts = {tuple(sorted(r)) for r in rows if r["event"] == "start"}
    finishes = {tuple(sorted(r)) for r in rows if r["event"] == "finish"}
    return starts, finishes


@pytest.mark.parametrize(
    ("fixture", "log_tokens"),
    [
        (FIXTURES / "request_lifetimes_replica000_node000.jsonl", False),
        pytest.param(ARCHIVED_TRACE, False, id="archive-e2_24k"),
        pytest.param(ARCHIVED_TOKEN_TRACE, True, id="archive-best_t8-token_ids"),
    ],
)
def test_trace_schema_matches_archived_trace(tmp_path, fixture, log_tokens):
    if not fixture.exists():
        pytest.skip(f"archived trace not found: {fixture}")
    archived = _rows(fixture)
    clock = itertools.count(1786432300.0, 0.5)
    tracer = RequestLifetimeTracer(tmp_path, 0, 0, log_tokens=log_tokens, clock=lambda: next(clock))
    n = len({r["request_id"] for r in archived})
    for i in range(n):
        tracer.record_start(f"{i:032x}", prompt_tokens=40 + i)
    for i in range(n):
        tracer.record_finish(f"{i:032x}", generation_tokens=i * 3, finish_reason="stop", token_ids=range(i * 3))
    tracer.close()

    assert tracer.path == str(tmp_path / "request_lifetimes_replica000_node000.jsonl")
    assert trace_file_name(3, 12) == "request_lifetimes_replica003_node012.jsonl"
    ours = _rows(Path(tracer.path))
    # Cardinality (the collector's check) and the exact key sets per event.
    assert len(ours) == len(archived)
    assert _key_sets(ours) == _key_sets(archived)
    # Rows are json.dumps(sort_keys=True): the archived line and ours have keys in the same order.
    archived_first = json.loads(fixture.read_text(encoding="utf-8").splitlines()[0])
    assert list(archived_first) == sorted(archived_first)
    first_line = Path(tracer.path).read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith('{"event": "start", "prompt_tokens"')
    starts = {r["request_id"] for r in ours if r["event"] == "start"}
    finishes = {r["request_id"] for r in ours if r["event"] == "finish"}
    assert starts == finishes and len(starts) == n
    if log_tokens:
        row = next(r for r in ours if r["event"] == "finish" and r["generation_tokens"] == 3)
        assert row["token_ids"] == [0, 1, 2]


def test_from_config_precedence(tmp_path):
    assert RequestLifetimeTracer.from_config(PrecisionSchedulerConfig(), 0, 0, environ={}) is None
    assert RequestLifetimeTracer.from_config(None, 0, 0, environ={}) is None
    cfg = PrecisionSchedulerConfig(request_trace_dir=str(tmp_path / "cfg"), request_trace_log_tokens=True)
    tracer = RequestLifetimeTracer.from_config(cfg, 1, 0, environ={"VERL_REQUEST_TRACE_DIR": "/ignored"})
    assert tracer.trace_dir == str(tmp_path / "cfg") and tracer.log_tokens is True
    # Idle servers create no file until the first row.
    assert not Path(tracer.path).exists()
    env = {"VERL_REQUEST_TRACE_DIR": str(tmp_path / "env"), "VERL_REQUEST_TRACE_LOG_TOKENS": "1"}
    tracer = RequestLifetimeTracer.from_config(PrecisionSchedulerConfig(), 0, 0, environ=env)
    assert tracer.trace_dir == str(tmp_path / "env") and tracer.log_tokens is True


def test_trace_request_id_is_echoed_in_both_rows(tmp_path):
    tracer = RequestLifetimeTracer(tmp_path, 0, 0)
    tracer.record_start("abc", 10, trace_request_id="idx-31_1")
    tracer.record_finish("abc", 5, "length", trace_request_id="idx-31_1")
    tracer.record_finish("def", 0, "aborted")
    rows = _rows(Path(tracer.path))
    assert rows[0]["trace_request_id"] == rows[1]["trace_request_id"] == "idx-31_1"
    assert "trace_request_id" not in rows[2] and rows[2]["finish_reason"] == "aborted"
    assert "token_ids" not in rows[1]


# --------------------------------------------------------------------------------------------
# generate() integration on a bare vLLMHttpServer with a fake engine.
# --------------------------------------------------------------------------------------------


class _FakeOutput:
    def __init__(self, token_ids, finish_reason="stop"):
        self.token_ids = list(token_ids)
        self.finish_reason = finish_reason
        self.logprobs = None


class _FakeRequestOutput:
    def __init__(self, outputs):
        self.outputs = outputs
        self.metrics = None


class _FakeEngine:
    def __init__(self, token_ids=(1, 2, 3), aborted=False):
        self.token_ids = token_ids
        self.aborted = aborted
        self.calls = []

    def generate(self, prompt, sampling_params, request_id, **kwargs):
        self.calls.append(request_id)

        async def gen():
            yield _FakeRequestOutput([] if self.aborted else [_FakeOutput(self.token_ids)])

        return gen()


def _bare_server(tmp_path, ps_cfg, engine, monkeypatch):
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout import vllm_async_server

    server = object.__new__(vllm_async_server.vLLMHttpServer)

    class _Config(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

    server.config = _Config(
        max_model_len=4096,
        prompt_length=512,
        response_length=64,
        full_determinism=False,
        skip_tokenizer_init=True,
        enable_rollout_routing_replay=False,
        mtp=None,
        precision_scheduler=ps_cfg,
    )
    server.model_config = SimpleNamespace(lora_rank=0, lora={}, processor=None)
    server.engine = engine
    server.replica_rank = 0
    server.node_rank = 0
    server.global_steps = 7
    server.request_tracer = RequestLifetimeTracer.from_config(ps_cfg, 0, 0, environ={})
    monkeypatch.setattr(vllm_async_server, "extract_prompt_logprobs", lambda **kw: None)
    return server, vllm_async_server


@pytest.mark.parametrize("log_tokens", [False, True])
def test_generate_records_start_and_finish(tmp_path, monkeypatch, log_tokens):
    ps_cfg = PrecisionSchedulerConfig(request_trace_dir=str(tmp_path), request_trace_log_tokens=log_tokens)
    engine = _FakeEngine(token_ids=(5, 6, 7, 8))
    server, _ = _bare_server(tmp_path, ps_cfg, engine, monkeypatch)
    out = asyncio.run(
        server.generate(
            prompt_ids=[1, 2, 3], sampling_params={"max_tokens": 8}, request_id="req-1", trace_request_id="idx-3_0"
        )
    )
    assert out.token_ids == [5, 6, 7, 8] and out.stop_reason == "completed"
    assert engine.calls == ["req-1"]  # engine id untouched by the trace id
    rows = _rows(tmp_path / "request_lifetimes_replica000_node000.jsonl")
    assert [r["event"] for r in rows] == ["start", "finish"]
    assert rows[0]["prompt_tokens"] == 3 and rows[0]["trace_request_id"] == "idx-3_0"
    assert rows[1]["generation_tokens"] == 4 and rows[1]["finish_reason"] == "stop"
    assert rows[1]["trace_request_id"] == "idx-3_0"
    assert ("token_ids" in rows[1]) is log_tokens
    if log_tokens:
        assert rows[1]["token_ids"] == [5, 6, 7, 8]


def test_generate_records_aborted_finish(tmp_path, monkeypatch):
    ps_cfg = PrecisionSchedulerConfig(request_trace_dir=str(tmp_path))
    server, _ = _bare_server(tmp_path, ps_cfg, _FakeEngine(aborted=True), monkeypatch)
    out = asyncio.run(server.generate(prompt_ids=[1], sampling_params={"max_tokens": 8}, request_id="req-2"))
    assert out.stop_reason == "aborted"
    rows = _rows(tmp_path / "request_lifetimes_replica000_node000.jsonl")
    assert rows[1] == {
        "event": "finish",
        "finish_reason": "aborted",
        "generation_tokens": 0,
        "request_id": "req-2",
        "timestamp": rows[1]["timestamp"],
    }


def test_generate_without_trace_dir_writes_nothing(tmp_path, monkeypatch):
    server, _ = _bare_server(tmp_path, PrecisionSchedulerConfig(), _FakeEngine(), monkeypatch)
    assert server.request_tracer is None
    asyncio.run(server.generate(prompt_ids=[1], sampling_params={"max_tokens": 8}, request_id="req-3"))
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------------------------
# Forced sleep level.
# --------------------------------------------------------------------------------------------


def _sleep_server(ps_cfg, monkeypatch, mode):
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout import vllm_async_server

    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(free_cache_engine=True, mtp=None, precision_scheduler=ps_cfg)
    server.node_rank = 0
    server.rollout_mode = mode
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    server.engine = SimpleNamespace(sleep=AsyncMock(), reset_encoder_cache=AsyncMock())
    return server, vllm_async_server


@pytest.mark.parametrize(
    ("ps_cfg", "expected"),
    [
        (PrecisionSchedulerConfig(), 1),
        (PrecisionSchedulerConfig(sleep_level=2), 2),
        (PrecisionSchedulerConfig(enable=True), 1),
        (None, 1),
    ],
)
def test_colocated_sleep_level(monkeypatch, ps_cfg, expected):
    from verl.workers.rollout.replica import RolloutMode

    server, _ = _sleep_server(ps_cfg, monkeypatch, RolloutMode.COLOCATED)
    asyncio.run(server.sleep())
    server.engine.sleep.assert_awaited_once_with(level=expected)


@pytest.mark.parametrize(
    ("ps_cfg", "lora_as_adapter", "expected"),
    [
        (PrecisionSchedulerConfig(), False, 2),  # upstream rule: full weights -> level 2
        (PrecisionSchedulerConfig(), True, 1),  # upstream rule: adapters -> level 1
        (PrecisionSchedulerConfig(sleep_level=1), False, 1),  # override wins over the rule
        (PrecisionSchedulerConfig(sleep_level=2), True, 2),
        (PrecisionSchedulerConfig(enable=True), False, 1),  # dual precision forces 1
    ],
)
def test_hybrid_sleep_level(monkeypatch, ps_cfg, lora_as_adapter, expected):
    from verl.workers.rollout.replica import RolloutMode

    server, _ = _sleep_server(ps_cfg, monkeypatch, RolloutMode.HYBRID)
    server.model_config = SimpleNamespace(lora_rank=16 if lora_as_adapter else 0, lora={})
    asyncio.run(server._sleep_hybrid())
    server.engine.sleep.assert_awaited_once_with(level=expected)


# --------------------------------------------------------------------------------------------
# Stable trace id: agent loop -> LLMServerClient -> server.generate(trace_request_id=...).
# --------------------------------------------------------------------------------------------

STABLE_ID_RE = re.compile(r"^idx-\d+_\d+$")


def test_stable_trace_id_matches_july_format():
    rows = _rows(FIXTURES / "request_steps_stable_trace_id_july.jsonl")
    assert rows and all(STABLE_ID_RE.match(r["trace_request_id"]) for r in rows)
    assert all(re.fullmatch(r"[0-9a-f]{32}", r["request_id"]) for r in rows)
    assert STABLE_ID_RE.match(f"{'idx-31'}_{1}")


def test_single_turn_agent_loop_passes_trace_request_id(monkeypatch):
    pytest.importorskip("vllm")
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from verl.workers.rollout.replica import TokenOutput

    loop = object.__new__(SingleTurnAgentLoop)
    loop.rollout_config = SimpleNamespace(full_determinism=False)
    loop.enable_continuous_token = False
    loop.response_length = 16
    loop.process_multi_modal_info = AsyncMock(return_value={})
    loop._get_mm_processor_kwargs = MagicMock(return_value=None)
    loop.apply_chat_template = AsyncMock(return_value=[1, 2, 3])
    generate = AsyncMock(return_value=TokenOutput(token_ids=[4, 5], log_probs=None, extra_fields={}))
    loop.server_manager = SimpleNamespace(generate=generate)

    asyncio.run(
        loop.run(
            sampling_params={}, priority=0, raw_prompt=[{"role": "user", "content": "hi"}], uid="idx-31", session_id=1
        )
    )
    kwargs = generate.call_args.kwargs
    assert kwargs["trace_request_id"] == "idx-31_1"
    assert re.fullmatch(r"[0-9a-f]{32}", kwargs["request_id"])

    generate.reset_mock()
    asyncio.run(loop.run(sampling_params={}, priority=0, raw_prompt=[{"role": "user", "content": "hi"}]))
    assert "trace_request_id" not in generate.call_args.kwargs


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_llm_server_client_gates_trace_id_per_backend(backend):
    from verl.workers.rollout.llm_server import LLMServerClient
    from verl.workers.rollout.replica import TokenOutput

    client = object.__new__(LLMServerClient)
    client.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(name=backend)))
    output = TokenOutput(token_ids=[1], log_probs=None, extra_fields={})
    server = SimpleNamespace(generate=SimpleNamespace(remote=AsyncMock(return_value=output)))
    client._acquire_server = AsyncMock(return_value=("s0", server))
    client._release_server = MagicMock()

    asyncio.run(
        client.generate("idx-31_1", prompt_ids=[1, 2], sampling_params={}, trace_request_id="idx-31_1", priority=0)
    )
    kwargs = server.generate.remote.call_args.kwargs
    assert re.fullmatch(r"[0-9a-f]{32}", kwargs["request_id"])  # engine id stays a fresh uuid
    assert ("trace_request_id" in kwargs) is (backend == "vllm")
    if backend == "vllm":
        assert kwargs["trace_request_id"] == "idx-31_1"
