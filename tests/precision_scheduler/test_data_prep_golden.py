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
"""Golden tests of the dataset prep scripts against the archived artifacts.

* GSM8K: a committed 50-row slice of the QeRL export must reproduce the first 50 rows of the archived
  training parquet (prompts, ground truth, index) and their token statistics; with the full archived
  export the 2048-row parquet, its summary.json token stats and the temporal-guard disjoint sets are
  reproduced (skipped when the archive is absent).
* BigMath: the hardmath ``olympiad_exact1_full`` split manifest (sha256 of every parquet) and the
  bf16-learnability band manifests (skipped when the HF snapshot / archive is absent).
* EOS workloads: the archived per-dataset JSONL sha256, the rendered-prompt sha256 for three models and
  the materialized row ids (skipped when absent; ``source`` needs the HF hub and is marked slow).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "examples" / "precision_scheduler" / "data"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
ARCHIVE = Path("/data/huanchen/verl/.codex-report/new-storyline-experiments")
QERL_EXPORT = Path("/data/huanchen/vllm/.codex-reports/rollout/datasets/qerl_gsm8k_train_2048_rollout.jsonl")
NO_REPREFILL_DATA = ARCHIVE / "no_reprefill_rl_100step_20260814" / "data"
TEMPORAL_GUARD = ARCHIVE / "temporal_guard_120" / "datasets" / "manifest.json"
QWEN35_4B = Path("/data/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
HARDMATH = ARCHIVE / "hardmath_lora_accuracy_100step_20260825"
LEARNABILITY = ARCHIVE / "bf16_learnability_search_20260904"
EOS = ARCHIVE / "eos_hazard_extensibility"
EOS_FULLSTEP = ARCHIVE / "eos_hazard_fullstep_b64_cap16k"
BIGMATH_SNAPSHOT = Path(
    "/data/huggingface/hub/datasets--open-r1--Big-Math-RL-Verified-Processed/snapshots/"
    "c79efbb6d3b75e3a2bcc27a5c569119918132345/level_3_4_5/train-00000-of-00001.parquet"
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, DATA / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tokenizer():
    if not QWEN35_4B.exists():
        pytest.skip("Qwen3.5-4B snapshot absent")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(QWEN35_4B), trust_remote_code=True)


# --- GSM8K --------------------------------------------------------------------------------------


def test_prepare_gsm8k_slice_matches_committed_archive_rows(tmp_path):
    prep = _load("prepare_gsm8k")
    tokenizer = _tokenizer()
    items = prep.load_qerl_export(FIXTURES / "gsm8k_qerl_export_50.jsonl")
    train, test, summary = prep.build_slice(
        items,
        train_size=50,
        test_size=8,
        start_index=0,
        shuffle_seed=None,
        prompt_format="messages",
        tokenizer=tokenizer,
    )
    golden = json.loads((FIXTURES / "gsm8k_archived_train_first50.json").read_text())
    assert len(train) == 50 and len(test) == 8
    for row, expected in zip(train, golden["rows"], strict=True):
        assert row["prompt"] == expected["prompt"]
        assert row["reward_model"]["ground_truth"] == expected["ground_truth"]
        assert row["extra_info"]["index"] == expected["index"]
        assert row["data_source"] == "openai/gsm8k"
    assert [t["extra_info"]["split"] for t in test] == ["test"] * 8
    assert summary["prompt_token_min"] == golden["prompt_token_min"]
    assert summary["prompt_token_max"] == golden["prompt_token_max"]
    assert summary["prompt_token_mean"] == pytest.approx(golden["prompt_token_mean"])
    # CLI end to end (parquet + summary.json)
    import subprocess
    import sys

    out = tmp_path / "gsm8k"
    subprocess.run(
        [
            sys.executable,
            str(DATA / "prepare_gsm8k.py"),
            "--input",
            str(FIXTURES / "gsm8k_qerl_export_50.jsonl"),
            "--output-dir",
            str(out),
            "--tokenizer",
            str(QWEN35_4B),
            "--train-size",
            "50",
            "--test-size",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    written = json.loads((out / "summary.json").read_text())
    assert written["train_rows"] == 50 and written["prompt_token_max"] == golden["prompt_token_max"]
    assert (out / "train.parquet").exists() and (out / "test.parquet").exists()


def test_prepare_gsm8k_reproduces_full_archive(tmp_path):
    if not (QERL_EXPORT.exists() and (NO_REPREFILL_DATA / "train.parquet").exists()):
        pytest.skip("QeRL export / archived parquet absent")
    import pandas as pd

    prep = _load("prepare_gsm8k")
    tokenizer = _tokenizer()
    items = prep.load_qerl_export(QERL_EXPORT)
    train, _, summary = prep.build_slice(
        items,
        train_size=2048,
        test_size=32,
        start_index=0,
        shuffle_seed=None,
        prompt_format="messages",
        tokenizer=tokenizer,
    )
    archived = pd.read_parquet(NO_REPREFILL_DATA / "train.parquet")
    archived_summary = json.loads((NO_REPREFILL_DATA / "summary.json").read_text())
    assert len(train) == len(archived) == 2048
    assert all(list(a["prompt"]) == [dict(m) for m in b] for a, b in zip(train, archived["prompt"], strict=True))
    assert all(
        a["reward_model"]["ground_truth"] == b["ground_truth"]
        for a, b in zip(train, archived["reward_model"], strict=True)
    )
    assert all(a["extra_info"]["index"] == b["index"] for a, b in zip(train, archived["extra_info"], strict=True))
    for key in ("prompt_token_min", "prompt_token_max"):
        assert summary[key] == archived_summary[key]
    assert summary["prompt_token_mean"] == pytest.approx(archived_summary["prompt_token_mean"])


def test_prepare_gsm8k_disjoint_sets_reproduce_temporal_guard_manifest():
    if not (QERL_EXPORT.exists() and TEMPORAL_GUARD.exists()):
        pytest.skip("QeRL export / temporal guard manifest absent")
    prep = _load("prepare_gsm8k")
    golden = json.loads(TEMPORAL_GUARD.read_text())
    tokenizer = _tokenizer()
    items = prep.load_qerl_export(QERL_EXPORT)
    sets, manifest = prep.build_disjoint_sets(
        items,
        num_sets=golden["num_sets"],
        set_size=golden["set_size"],
        selection_seed=golden["selection_seed"],
        prompt_format="messages",
        tokenizer=tokenizer,
    )
    assert manifest["selection"] == golden["selection"]
    for mine, theirs in zip(manifest["sets"], golden["sets"], strict=True):
        assert mine["source_indices"] == theirs["source_indices"]
        assert mine["rows"] == theirs["rows"]
    assert len(sets) == golden["num_sets"] and all(len(s) == golden["set_size"] for s in sets)


# --- BigMath ------------------------------------------------------------------------------------


def _bigmath_case(manifest_path: Path):
    if not (BIGMATH_SNAPSHOT.exists() and manifest_path.exists()):
        pytest.skip(f"BigMath snapshot or manifest absent: {manifest_path}")
    return json.loads(manifest_path.read_text())


def test_prepare_bigmath_reproduces_hardmath_manifest(tmp_path):
    manifest = _bigmath_case(HARDMATH / "candidate_data" / "olympiad_exact1_full" / "manifest.json")
    prep = _load("prepare_bigmath")
    filt = manifest["difficulty_filter"]
    out = tmp_path / "bigmath"
    result = prep.prepare(
        source=BIGMATH_SNAPSHOT,
        output_dir=out,
        seed=manifest["seed"],
        min_solve_rate=filt["minimum_inclusive"],
        max_solve_rate=filt["maximum_inclusive"],
        sources=filt["source_allowlist"],
        exclude_domain_substrings=filt["excluded_domain_substrings"],
        train_size=manifest["artifacts"]["train"]["rows"],
        calibration_size=manifest["artifacts"]["calibration"]["rows"],
        validation_size=manifest["artifacts"]["validation"]["rows"],
        monitor_size=manifest["artifacts"]["monitor"]["rows"],
    )
    assert result["source_sha256"] == manifest["source_sha256"]
    assert result["difficulty_filter"]["eligible_unique_prompts"] == filt["eligible_unique_prompts"]
    for split in ("train", "calibration", "validation", "monitor"):
        assert result["artifacts"][split]["sha256"] == manifest["artifacts"][split]["sha256"], split
    assert result["artifacts"]["test_alias"]["sha256"] == manifest["artifacts"]["test_alias"]["sha256"]
    assert result["artifacts"]["pilot_data"]["train_sha256"] == manifest["artifacts"]["pilot_data"]["train_sha256"]


@pytest.mark.parametrize(
    "band", sorted(p.name for p in LEARNABILITY.glob("datasets/*")) if LEARNABILITY.exists() else []
)
def test_prepare_bigmath_reproduces_learnability_manifests(tmp_path, band):
    manifest = _bigmath_case(LEARNABILITY / "datasets" / band / "manifest.json")
    prep = _load("prepare_bigmath")
    result = prep.prepare_learnability(source=BIGMATH_SNAPSHOT, output_dir=tmp_path / band, manifest=manifest)
    for split in ("train", "validation"):
        assert result[split]["sha256"] == manifest[split]["sha256"], (band, split)
        assert result[split]["rows"] == manifest[split]["rows"]
    assert result["difficulty_proxy"] == manifest["difficulty_proxy"]


# --- EOS workloads --------------------------------------------------------------------------------

EOS_MODELS = {"qwen35_9b": "qwen3_5_9b", "qwen35_4b": "qwen3_5_4b", "phi4_mini_reasoning": "phi4_mini_reasoning"}


def _eos_manifest(name: str) -> dict:
    path = EOS / "manifests" / name
    if not path.exists():
        pytest.skip(f"archive absent: {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("dataset", ("gsm8k", "math500", "bigmath_hard"))
def test_eos_render_reproduces_archived_prompt_sha(tmp_path, dataset):
    rendered = _eos_manifest("rendered_prompts.json")["rendered"]
    models = json.loads((EOS / "manifests" / "resolved_models.json").read_text())["models"]
    prep = _load("prepare_eos_workloads")
    source = EOS / "datasets" / f"{dataset}.jsonl"
    if not source.exists():
        pytest.skip("archived source jsonl absent")
    for archived_key in ("qwen35_4b", "phi4_mini_reasoning"):
        model = models[archived_key]
        if not Path(model["bf16_path"]).exists():
            pytest.skip(f"snapshot absent for {archived_key}")
        out = tmp_path / archived_key / f"{dataset}.jsonl"
        info = prep.render_dataset(
            source, out, tokenizer_path=model["bf16_path"], enable_thinking=bool(model.get("enable_thinking"))
        )
        assert info["sha256"] == rendered[archived_key][dataset]["sha256"], (archived_key, dataset)
        assert info["prompt_token_max"] == rendered[archived_key][dataset]["prompt_token_max"]


def test_eos_materialize_reproduces_archived_row_ids(tmp_path):
    manifest_path = EOS_FULLSTEP / "data" / "qwen35_4b" / "gsm8k" / "manifest.json"
    if not manifest_path.exists() or not (EOS / "datasets" / "gsm8k.jsonl").exists():
        pytest.skip("archive absent")
    models = json.loads((EOS / "manifests" / "resolved_models.json").read_text())["models"]
    model = models["qwen35_4b"]
    if not Path(model["bf16_path"]).exists():
        pytest.skip("snapshot absent")
    golden = json.loads(manifest_path.read_text())
    prep = _load("prepare_eos_workloads")
    result = prep.materialize(
        source=EOS / "datasets" / "gsm8k.jsonl",
        rendered=Path(golden["rendered_prompt_source"]),
        output_dir=tmp_path / "gsm8k",
        tokenizer_path=model["bf16_path"],
        enable_thinking=bool(model.get("enable_thinking")),
        rows="0:64",
        test_rows=16,
        dataset="gsm8k",
    )
    assert result["row_ids"] == golden["row_ids"]
    assert result["rows"] == golden["rows"] == 64
    import pandas as pd

    frame = pd.read_parquet(tmp_path / "gsm8k" / "train.parquet")
    assert list(frame["data_source"].unique()) == ["openai/gsm8k"]
    assert frame.iloc[0]["reward_model"]["ground_truth"].isdigit()


@pytest.mark.skipif(os.environ.get("PS_RUN_SLOW") != "1", reason="needs the HF hub; PS_RUN_SLOW=1")
def test_eos_source_reproduces_archived_dataset_sha(tmp_path):
    datasets = _eos_manifest("datasets.json")["datasets"]
    prep = _load("prepare_eos_workloads")
    for name in ("gsm8k", "math500"):
        info = prep.source_dataset(name, tmp_path / f"{name}.jsonl", seed=20260902, needed=96)
        assert info["sha256"] == datasets[name]["sha256"], name
