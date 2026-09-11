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
"""GPU smoke: vanilla vs dirty-with-flags-off vs clean greedy token identity on 16 GSM8K prompts.

Each environment runs `scripts/precision_scheduler/env/greedy_identity.py` in its own subprocess,
activated with `activate.sh <env>` and wrapped in `run_gpu.sh` (decision 13). Vanilla runs twice
first: if two runs of the same tree already differ, the cross-tree comparison is meaningless, so
the test fails with that diagnosis instead. Greedy decoding is only reproducible run-to-run with
VLLM_BATCH_INVARIANT=1 *and* enforce_eager=True (measured 2026-09-11: with torch.compile/CUDA
graphs two identical Phi-4-mini runs differ on 2/16 prompts; Qwen3.5-4B differs on 2-4/16 even
with max_num_seqs=1 and batch-invariant mode refuses its GDN attention). Hence the defaults:
Phi-4-mini-reasoning (first-class in the plan), --batch-invariant, --enforce-eager. Run it as

    check_env.py --expect clean --pick-gpus 1   # -> e.g. 0
    run_gpu.sh --gpus 0 --timeout 3600 -- $PYTHON_BIN -m pytest -p no:cacheprovider -q \\
        -m gpu_smoke tests/precision_scheduler/gpu/test_greedy_identity.py

The GPU is taken from CUDA_VISIBLE_DEVICES (set by the outer run_gpu.sh); with no such variable
the test picks one with `check_env.py --pick-gpus 1`. Set PS_IDENTITY_OUT_DIR to keep the token
dumps (pytest's basetemp is pruned by other pytest sessions), PS_IDENTITY_MODEL to change the model.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

VERL_ROOT = Path(__file__).resolve().parents[3]
ENV_DIR = VERL_ROOT / "scripts" / "precision_scheduler" / "env"
SCRIPT = ENV_DIR / "greedy_identity.py"
RUNS = (("vanilla", "vanilla_a"), ("vanilla", "vanilla_b"), ("dirty", "dirty"), ("clean", "clean"))
FLAG_PREFIXES = ("VLLM_DUAL_PRECISION", "ROLLOUT_QLORA", "VLLM_LORA_ENABLE_DUAL_STREAM", "VLLM_REPREFILL")

pytestmark = pytest.mark.gpu_smoke


def _gpu_id() -> str:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if cvd:
        return cvd
    out = subprocess.run(
        [os.environ.get("PYTHON_BIN", "python"), str(ENV_DIR / "check_env.py"), "--pick-gpus", "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        pytest.skip(f"no free GPU: {out.stderr.strip()}")
    return out.stdout.strip()


def _model_present() -> None:
    model = os.environ.get("PS_IDENTITY_MODEL")
    if model is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("ps_greedy_identity", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.DEFAULT_MODEL
    if not Path(model).exists():
        pytest.skip(f"model checkpoint not found: {model}")


def _run_env(kind: str, gpu: str, out: Path, timeout: int = 1500) -> subprocess.CompletedProcess:
    script = (
        f"source {ENV_DIR / 'activate.sh'} {kind} && "
        f"exec {ENV_DIR / 'run_gpu.sh'} --gpus {gpu} --timeout {timeout} -- "
        f'"$PYTHON_BIN" {SCRIPT} --env {kind} --batch-invariant --enforce-eager --out {out}'
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "PS_ENV", "VLLM_ROOT", "VERL_ROOT", "CUDA_VISIBLE_DEVICES")
        and not k.startswith(FLAG_PREFIXES)
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, cwd="/", check=False)


def _first_divergence(a: dict, b: dict) -> list[int | None]:
    out = []
    for r, g in zip(a["results"], b["results"], strict=True):
        assert g["prompt_token_ids"] == r["prompt_token_ids"], f"prompt {r['index']} tokenized differently"
        t, u = r["token_ids"], g["token_ids"]
        first = next((i for i, (x, y) in enumerate(zip(t, u, strict=False)) if x != y), None)
        if first is None and len(t) != len(u):
            first = min(len(t), len(u))
        out.append(first)
    return out


def test_greedy_token_identity_across_trees(tmp_path):
    _model_present()
    for kind in ("vanilla", "clean"):
        cmd = ["bash", "-c", f"source {ENV_DIR / 'activate.sh'} {kind} && echo $VLLM_ROOT"]
        root = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
        if not (Path(root) / "vllm" / "_C.abi3.so").exists():
            pytest.skip(f"host lacks the {kind} vLLM worktree at {root!r}")
    gpu = _gpu_id()
    out_dir = Path(os.environ.get("PS_IDENTITY_OUT_DIR") or tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    dumps = {}
    for kind, name in RUNS:
        out = out_dir / f"{name}.json"
        proc = _run_env(kind, gpu, out)
        assert proc.returncode == 0, f"{name} failed rc={proc.returncode}\n--- stderr tail ---\n{proc.stderr[-4000:]}"
        dumps[name] = json.loads(out.read_text())
        assert dumps[name]["env"] == kind
        assert len(dumps[name]["results"]) == 16

    # run-to-run determinism of the oracle itself
    rerun = _first_divergence(dumps["vanilla_a"], dumps["vanilla_b"])
    assert all(d is None for d in rerun), (
        f"vanilla is not reproducible run-to-run (first divergence per prompt: {rerun}); "
        "the cross-tree comparison would be meaningless with this model/mode"
    )

    roots = {n: d["vllm_file"] for n, d in dumps.items() if n != "vanilla_b"}
    assert len(set(roots.values())) == 3, f"the runs did not use three different trees: {roots}"

    ref = dumps["vanilla_a"]
    for name in ("dirty", "clean"):
        div = _first_divergence(ref, dumps[name])
        bad = [(i, d) for i, d in enumerate(div) if d is not None]
        assert not bad, (
            f"{name} diverges from vanilla on prompts (index, first differing token): {bad}\n"
            f"vanilla text[{bad[0][0]}]: {ref['results'][bad[0][0]]['text']!r}\n"
            f"{name} text[{bad[0][0]}]: {dumps[name]['results'][bad[0][0]]['text']!r}"
        )
