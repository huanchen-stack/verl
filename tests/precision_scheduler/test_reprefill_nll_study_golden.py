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
"""CPU tests for examples/precision_scheduler/analysis/reprefill_nll_study (component C7).

Golden tier: the analyze step reproduces the archived ``summary.json`` of
``qwen35_9b_w4qdq_longtail16`` from the archived per-window metrics (committed fixture; the
archive copy, when present, must be byte-identical). Unit tier: trace selection by explicit
id list, the metric function on synthetic logits, the analyze CLI.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "examples" / "precision_scheduler" / "analysis" / "reprefill_nll_study"
GOLDEN = Path(__file__).resolve().parent / "golden" / "reprefill_nll_study"
ARCHIVE = Path("/data/huanchen/verl/.codex-report/reprefill-study/runs/qwen35_9b_w4qdq_longtail16")
OFFSETS = [0, 512, 1024, 2048, 4096]
SEED = 20260812


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, STUDY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scorer():
    pytest.importorskip("torch")
    return _load("reprefill_teacher_forced")


@pytest.fixture(scope="module")
def analyzer():
    return _load("analyze_reprefill_results")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# golden
# ---------------------------------------------------------------------------


def test_fixture_matches_the_archive_when_present():
    if not ARCHIVE.exists():
        pytest.skip(f"archive absent: {ARCHIVE}")
    for name in ("window_metrics.jsonl", "summary.json", "manifest.json", "identity.json", "quantization.json"):
        ours = hashlib.sha256((GOLDEN / name).read_bytes()).hexdigest()
        theirs = hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest()
        assert ours == theirs, name


def test_summary_is_reproduced_from_the_archived_window_metrics(analyzer):
    rows = _rows(GOLDEN / "window_metrics.jsonl")
    assert len(rows) == 80
    archived = json.loads((GOLDEN / "summary.json").read_text())
    summary = analyzer.summarize(rows, OFFSETS, SEED)
    assert summary["num_traces"] == archived["num_traces"] == 16
    assert set(summary["windows"]) == set(archived["windows"]) == {str(o) for o in OFFSETS}
    for offset, metrics in archived["windows"].items():
        for key, stats in metrics.items():
            ours = summary["windows"][offset][key]
            assert ours["mean"] == pytest.approx(stats["mean"], abs=1e-9)
            assert ours["median"] == pytest.approx(stats["median"], abs=1e-9)
            assert ours["positive_fraction"] == stats["positive_fraction"]
            # Same seed, same resampling order: the bootstrap CI is exact.
            assert ours["ci95"] == pytest.approx(stats["ci95"], abs=1e-9)


def test_headline_negative_result_in_the_fixture(analyzer):
    """The numbers the README states: reuse beats re-prefill at every offset."""
    summary = json.loads((GOLDEN / "summary.json").read_text())
    for offset in OFFSETS:
        delta = summary["windows"][str(offset)]["delta_nll_reuse_minus_reprefill"]
        assert delta["mean"] < 0
        assert delta["ci95"][1] < 0
    d0 = summary["windows"]["0"]
    assert d0["delta_nll_reuse_minus_reprefill"]["mean"] == pytest.approx(-0.0643, abs=5e-4)
    assert d0["kl_bf16_to_reuse"]["mean"] == pytest.approx(0.0405, abs=5e-4)
    assert d0["kl_bf16_to_reprefill"]["mean"] == pytest.approx(0.0969, abs=5e-4)
    decision = analyzer.decide(summary, OFFSETS)
    assert decision["gate"].startswith("STOP")
    assert decision["reuse_better_at_every_offset"] is True


def test_archived_trace_ids_match_the_fixture_order():
    ids = json.loads((STUDY / "archived_trace_ids.json").read_text())["traces"]
    rows = _rows(GOLDEN / "window_metrics.jsonl")
    seen: list[dict] = []
    for row in rows:
        entry = {"dataset": row["source"].split("/results/")[1].split("/")[0], "request_id": row["request_id"]}
        if entry not in seen:
            seen.append(entry)
    assert ids == seen
    assert [row["trace_index"] for row in rows if row["offset"] == 0] == list(range(16))


def test_analyze_cli_writes_analysis_json(analyzer, tmp_path):
    out = tmp_path / "measured"
    subprocess.run(
        [
            sys.executable,
            str(STUDY / "analyze_reprefill_results.py"),
            "--run-dir",
            str(GOLDEN),
            "--out-dir",
            str(out),
            "--no-plots",
        ],
        check=True,
        cwd=tmp_path,
    )
    analysis = json.loads((out / "analysis.json").read_text())
    assert analysis["sources"] == {"eurus": 10, "bigmath": 4, "gsm8k": 2}
    assert analysis["summary_matches_archived"] is True
    assert analysis["gate"].startswith("STOP")
    assert (out / "summary.recomputed.json").exists()


# ---------------------------------------------------------------------------
# unit: trace selection and metrics
# ---------------------------------------------------------------------------


def _write_responses(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w") as stream:
        for request_id, prompt, length in rows:
            stream.write(
                json.dumps({"request_id": request_id, "prompt": prompt, "output_token_ids": list(range(length))}) + "\n"
            )


def test_traces_are_selected_by_explicit_id_list_in_list_order(scorer, tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_responses(a, [("req-1", "p1", 100), ("req-2", "p2", 10), ("req-3", "p3", 100)])
    _write_responses(b, [("req-1", "q1", 100), ("req-9", "q9", 100)])
    wanted = [
        {"dataset": "b", "request_id": "req-9"},
        {"dataset": "a", "request_id": "req-1"},
        {"dataset": "b", "request_id": "req-1"},
    ]
    traces = scorer.load_traces({"a": a, "b": b}, wanted, minimum_output=50)
    assert [(t.dataset, t.request_id) for t in traces] == [("b", "req-9"), ("a", "req-1"), ("b", "req-1")]
    assert traces[1].prompt == "p1" and traces[2].prompt == "q1"
    with pytest.raises(ValueError, match="shorter than"):
        scorer.load_traces({"a": a}, [{"dataset": "a", "request_id": "req-2"}], minimum_output=50)
    with pytest.raises(ValueError, match="not found"):
        scorer.load_traces({"a": a}, [{"dataset": "a", "request_id": "req-7"}], minimum_output=50)
    with pytest.raises(ValueError, match="unknown dataset"):
        scorer.load_traces({"a": a}, [{"dataset": "zzz", "request_id": "req-1"}], minimum_output=50)


def test_distribution_metrics_identity_and_ordering(scorer):
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(0)
    bf = torch.randn(8, 32, generator=gen)
    targets = bf.argmax(-1)
    same = scorer.distribution_metrics(bf, bf, bf, targets, eos_token_id=3)
    assert same["delta_nll_reuse_minus_reprefill"] == pytest.approx(0.0)
    assert same["kl_bf16_to_reuse"] == pytest.approx(0.0, abs=1e-6)
    assert same["js_reuse_reprefill"] == pytest.approx(0.0, abs=1e-6)
    assert same["reuse_top1_agreement_bf16"] == 1.0
    noisy = bf + 0.5 * torch.randn(8, 32, generator=gen)
    closer = scorer.distribution_metrics(bf, bf, noisy, targets, eos_token_id=3)
    assert closer["delta_nll_reuse_minus_reprefill"] < 0
    assert closer["kl_bf16_to_reuse"] < closer["kl_bf16_to_reprefill"]
    assert closer["reprefill_top1_agreement_bf16"] <= 1.0
