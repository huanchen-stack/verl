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
"""CPU parts of the replay-regression tool: workload specs/manifest and batch assembly."""

import json
from pathlib import Path

import pytest

from verl.experimental.precision_scheduler.replay_regression.workloads import (
    collect_corpus,
    workload_specs,
    write_workloads,
)

ARCHIVE = Path("/data/huanchen/verl/.codex-report/new-storyline-experiments/megatron_replay_regression/workloads")


class FakeTokenizer:
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        class R:
            input_ids = [ord(c) % 50 + 3 for c in text]

        return R()


def test_workload_specs_are_deterministic_and_anchor_first():
    specs = workload_specs(32, 20260814)
    assert specs[:3] == [(32, 1024), (64, 1024), (128, 1024)]
    assert specs[3:6] == [(32, 4096), (64, 4096), (128, 3072)]
    assert specs == workload_specs(32, 20260814)
    assert all(768 <= cap <= 24000 for _, cap in specs[15:])
    if (ARCHIVE / "manifest.json").exists():
        manifest = json.loads((ARCHIVE / "manifest.json").read_text())
        assert [(p["batch_size"], p["cap"]) for p in manifest["points"]] == specs


def test_collect_corpus_and_write_workloads(tmp_path):
    root = tmp_path / "runs"
    (root / "a" / "rollouts").mkdir(parents=True)
    rows = [{"input": f"prompt {i}", "output": "answer " * (i + 1)} for i in range(6)] + [
        {"input": "prompt 0", "output": "answer "}
    ]
    (root / "a" / "rollouts" / "step1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    corpus = collect_corpus(root, FakeTokenizer(), seed=1)
    assert len(corpus) == 6  # duplicate dropped
    assert all(r["response"][-1] == 2 for r in corpus)
    summary = write_workloads(corpus, [(2, 5), (3, 100)], tmp_path / "out", seed=3)
    assert [p["batch_size"] for p in summary["points"]] == [2, 3]
    point = json.loads((tmp_path / "out" / "point_00.json").read_text())
    assert all(len(s["response"]) <= 5 for s in point["samples"])
    assert point["total_tokens"] == sum(len(s["prompt"]) + len(s["response"]) for s in point["samples"])


def test_build_batch_padding():
    torch = pytest.importorskip("torch")
    from verl.experimental.precision_scheduler.replay_regression.worker import build_batch

    batch = build_batch([{"prompt": [5, 6], "response": [7]}, {"prompt": [8], "response": [9, 10, 11]}], pad_id=0)
    assert batch.batch["input_ids"].tolist() == [[5, 6, 7, 0, 0], [0, 8, 9, 10, 11]]
    assert batch.batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [0, 1, 1, 1, 1]]
    assert batch.batch["response_mask"].tolist() == [[1, 0, 0], [1, 1, 1]]
    assert batch.batch["prompts"].tolist() == [[5, 6], [0, 8]]
    assert batch.meta_info["temperature"] == 1.0
    assert isinstance(batch.batch["position_ids"], torch.Tensor)
