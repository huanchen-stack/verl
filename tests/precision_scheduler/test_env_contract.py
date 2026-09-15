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
"""CPU unit tests for scripts/precision_scheduler/env/check_env.py (the environment contract).

The check functions are exercised with fakes and monkeypatched paths so the tests are host
independent; the last two tests run the real script as a subprocess under `activate.sh` and are
skipped when this host lacks the clean / vanilla worktrees.
"""

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ENV_DIR = Path(__file__).resolve().parents[2] / "scripts" / "precision_scheduler" / "env"
CHECK_ENV = ENV_DIR / "check_env.py"
ACTIVATE = ENV_DIR / "activate.sh"


@pytest.fixture(scope="module")
def ce():
    spec = importlib.util.spec_from_file_location("ps_check_env", CHECK_ENV)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve postponed annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _fake_pkg(root: Path, name: str, version: str | None = None) -> types.SimpleNamespace:
    (root / name).mkdir(parents=True, exist_ok=True)
    init = root / name / "__init__.py"
    init.write_text("")
    ns = types.SimpleNamespace(__file__=str(init))
    if version is not None:
        ns.__version__ = version
    return ns


@pytest.fixture
def good_layout(tmp_path, ce):
    """A fake but complete clean layout: honest vllm + verl trees, payload, TE patch, LD path."""
    vllm_root = tmp_path / "vllm-clean"
    verl_root = tmp_path / "verl-clean"
    purelib = tmp_path / "site-packages"
    vllm = _fake_pkg(vllm_root, "vllm", ce.FALLBACK_VLLM_VERSION)
    verl = _fake_pkg(verl_root, "verl")
    for p in ce.PRECOMPILED_PAYLOAD:
        target = vllm_root / "vllm" / p
        if p.endswith((".so", ".py", "vllm-rs")):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("")
        else:
            target.mkdir(parents=True, exist_ok=True)
    te_utils = purelib / ce.TE_PATCH_REL
    te_utils.parent.mkdir(parents=True)
    te_utils.write_text(f"    {ce.TE_PATCH_LINE}\n")
    env = {
        "PS_ENV": "clean",
        "VLLM_ROOT": str(vllm_root),
        "VERL_ROOT": str(verl_root),
        "PYTHON_BIN": sys.executable,
        "LD_LIBRARY_PATH": ce.cu13_lib_dir(str(purelib)) + ":/usr/lib",
    }
    return dict(
        env=env,
        vllm=vllm,
        verl=verl,
        vllm_root=vllm_root,
        verl_root=verl_root,
        te_utils=str(te_utils),
        purelib=str(purelib),
    )


def _run(ce, layout, expect="clean", **over):
    kw = dict(
        env=layout["env"],
        executable=sys.executable,
        vllm_module=layout["vllm"],
        verl_module=layout["verl"],
        metadata_version=layout["vllm"].__version__,
        te_utils=layout["te_utils"],
        purelib=layout["purelib"],
    )
    kw.update(over)
    return ce.check_expect(expect, **kw)


# ----------------------------------------------------------------------------- --expect (unit)
def test_good_layout_passes(ce, good_layout):
    rep = _run(ce, good_layout)
    assert rep.ok, rep.failures
    assert rep.facts["payload_missing"] == []
    assert rep.facts["vllm_version"] == ce.FALLBACK_VLLM_VERSION


def test_ps_env_mismatch_is_refused(ce, good_layout):
    rep = _run(ce, good_layout, expect="vanilla")
    assert any("PS_ENV='clean' but --expect vanilla" in f for f in rep.failures)


def test_wrong_executable(ce, good_layout):
    rep = _run(ce, good_layout, executable="/usr/bin/python3")
    assert any("sys.executable" in f and "PYTHON_BIN" in f for f in rep.failures)


def test_vllm_from_wrong_tree(ce, good_layout, tmp_path):
    other = _fake_pkg(tmp_path / "vllm-dirty", "vllm", ce.FALLBACK_VLLM_VERSION)
    rep = _run(ce, good_layout, vllm_module=other)
    assert any("vllm imported from" in f and "not under VLLM_ROOT" in f for f in rep.failures)


def test_verl_from_wrong_tree(ce, good_layout, tmp_path):
    other = _fake_pkg(tmp_path / "verl-dirty", "verl")
    rep = _run(ce, good_layout, verl_module=other)
    assert any("verl imported from" in f and "not under VERL_ROOT" in f for f in rep.failures)


def test_dishonest_version_fails_clean_but_not_dirty(ce, good_layout):
    good_layout["vllm"].__version__ = "0.1.dev17121+g34e66a379.d20260902"
    rep = _run(ce, good_layout, metadata_version="0.18.0")
    reasons = "\n".join(rep.failures)
    assert "does not parse >= 0.16" in reasons
    assert "!= vllm.__version__" in reasons
    # the dirty reference tree is allowed to keep its legacy version file + fake metadata
    good_layout["env"]["PS_ENV"] = "dirty"
    rep = _run(ce, good_layout, expect="dirty", metadata_version="0.18.0")
    assert rep.ok, rep.failures
    rep = _run(ce, good_layout, expect="dirty", metadata_version="0.7.0")
    assert any(">= 0.8.5" in f for f in rep.failures)


def test_metadata_mismatch(ce, good_layout):
    rep = _run(ce, good_layout, metadata_version="0.18.0")
    assert any("importlib.metadata.version('vllm') '0.18.0' != vllm.__version__" in f for f in rep.failures)


def test_te_patch_reverted(ce, good_layout):
    Path(good_layout["te_utils"]).write_text('    max_version = PkgVersion("2.7.4.post1")\n')
    rep = _run(ce, good_layout)
    assert any("TE flash-attn gate patch missing" in f for f in rep.failures)


def test_te_missing(ce, good_layout):
    rep = _run(ce, good_layout, te_utils=good_layout["te_utils"] + ".nope")
    assert any("transformer_engine not found" in f for f in rep.failures)


def test_payload_incomplete(ce, good_layout):
    (good_layout["vllm_root"] / "vllm" / "_moe_C.abi3.so").unlink()
    (good_layout["vllm_root"] / "vllm" / "_version.py").unlink()
    rep = _run(ce, good_layout)
    bad = [f for f in rep.failures if "precompiled payload incomplete" in f]
    assert bad and "_moe_C.abi3.so" in bad[0] and "_version.py" in bad[0]
    assert rep.facts["payload_missing"] == ["_moe_C.abi3.so", "_version.py"]


def test_ld_library_path_rule(ce, good_layout):
    good_layout["env"]["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.0/lib64:" + good_layout["env"]["LD_LIBRARY_PATH"]
    rep = _run(ce, good_layout)
    assert any("LD_LIBRARY_PATH must start with" in f for f in rep.failures)
    good_layout["env"]["LD_LIBRARY_PATH"] = ""
    rep = _run(ce, good_layout)
    assert any("LD_LIBRARY_PATH must start with" in f for f in rep.failures)


def test_vanilla_requires_exact_base_commit(ce, good_layout, monkeypatch):
    good_layout["env"]["PS_ENV"] = "vanilla"
    calls = {}

    def fake_git(root, *args):
        calls[args] = True
        if args[:2] == ("rev-parse", "HEAD"):
            return "deadbeef" * 5
        if args[:2] == ("status", "--porcelain"):
            return ""
        return None

    monkeypatch.setattr(ce, "_run_git", fake_git)
    rep = _run(ce, good_layout, expect="vanilla")
    assert any("HEAD" in f and "!= 6bdabbad5b" in f for f in rep.failures)


def test_unknown_expect(ce, good_layout):
    rep = ce.check_expect("weird", env=good_layout["env"])
    assert rep.failures == ["unknown --expect 'weird'; choose from ('clean', 'dirty', 'vanilla')"]


# ----------------------------------------------------------------------------- --pick-gpus
def test_free_gpus_filters_foreign_and_busy(ce):
    used = {0: 0, 1: 0, 2: 3000, 3: 10, 4: 2048, 5: 0, 6: 0, 7: 0}
    procs = {0: [], 1: [], 2: [], 3: [111], 4: [], 5: [222, 333], 6: [], 7: [444]}
    owners = {111: "me", 222: "me", 333: "other", 444: "other"}
    free = ce.free_gpus(used, procs, me="me", owner_of=lambda p: owners.get(p))
    # 2 too much memory; 5 and 7 have foreign processes; 3 (own process, 10 MiB) and 4 (== limit) stay;
    # 1 is an ordinary GPU on this host
    assert free == [0, 1, 3, 4, 6]


def test_free_gpus_ignores_absent_indices(ce):
    free = ce.free_gpus({0: 0, 2: 0}, {0: [], 2: []}, me="me", owner_of=lambda p: "me")
    assert free == [0, 2]


def test_pick_gpus_cli_exit_3_when_short(ce, monkeypatch, capsys):
    monkeypatch.setattr(ce, "pick_gpus", lambda n: [0])
    assert ce.main(["--pick-gpus", "2"]) == 3
    assert capsys.readouterr().out == ""
    assert ce.main(["--pick-gpus", "1"]) == 0
    assert capsys.readouterr().out.strip() == "0"


def test_pick_gpus_composes_with_expect(ce, monkeypatch, capsys):
    monkeypatch.setattr(ce, "pick_gpus", lambda n: [0, 2, 3][:n])
    monkeypatch.setattr(ce, "check_expect", lambda *a, **k: ce.Report(failures=["boom"]))
    assert ce.main(["--expect", "clean", "--pick-gpus", "2"]) == 1
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(ce, "check_expect", lambda *a, **k: ce.Report())
    assert ce.main(["--expect", "clean", "--pick-gpus", "2"]) == 0
    assert capsys.readouterr().out.strip() == "0,2"


# ----------------------------------------------------------------------------- --write-version-file
def test_version_from_describe(ce):
    assert ce.version_from_describe("v0.22.1rc0-23-g6bdabbad5b") == "0.22.1rc1.dev23+g6bdabbad5b.precompiled"
    assert ce.version_from_describe("v0.22.1rc0-0-g6bdabbad5b") == "0.22.1rc0+g6bdabbad5b.precompiled"
    assert ce.version_from_describe("v0.22.0-5-gabcdef1234") == "0.22.1.dev5+gabcdef1234.precompiled"
    assert ce.version_from_describe("garbage") is None


def test_render_version_file_is_importable(ce, tmp_path):
    text = ce.render_version_file(ce.FALLBACK_VLLM_VERSION)
    ns = {}
    exec(compile(text, "_version.py", "exec"), ns)
    assert ns["__version__"] == ce.FALLBACK_VLLM_VERSION
    assert ns["__version_tuple__"] == (0, 22, 1, "rc1", "dev23", "g6bdabbad5b.precompiled")
    assert ns["__commit_id__"] == "g6bdabbad5b"
    assert ce.parse_version_tuple(ns["__version__"]) >= ce.MIN_VLLM_VERSION


def test_write_version_file_falls_back_without_git(ce, tmp_path):
    root = tmp_path / "vllm-x"
    (root / "vllm").mkdir(parents=True)
    version, source = ce.write_version_file(str(root))
    assert version == ce.FALLBACK_VLLM_VERSION
    assert source == "fallback constant"
    assert ce.FALLBACK_VLLM_VERSION in (root / "vllm" / "_version.py").read_text()


def test_write_version_file_derives_from_git(ce, tmp_path, monkeypatch):
    root = tmp_path / "vllm-y"
    (root / "vllm").mkdir(parents=True)
    monkeypatch.setattr(ce, "_run_git", lambda r, *a: "v0.22.1rc0-23-g6bdabbad5b" if a[0] == "describe" else None)
    version, source = ce.write_version_file(str(root))
    assert version == ce.FALLBACK_VLLM_VERSION
    assert source.startswith("git describe")


# ----------------------------------------------------------------------------- subprocess (this host)
def _activated(kind: str, *cmd: str) -> subprocess.CompletedProcess:
    script = f'source {ACTIVATE} {kind} && exec "$PYTHON_BIN" ' + " ".join(cmd)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PS_ENV", "VLLM_ROOT", "VERL_ROOT")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, cwd="/", check=False)


def _needs_host_layout():
    for kind in ("clean", "vanilla"):
        cmd = ["bash", "-c", f"source {ACTIVATE} {kind} && echo $VLLM_ROOT"]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        root = r.stdout.strip()
        if not root or not (Path(root) / "vllm" / "_C.abi3.so").exists():
            pytest.skip(f"host lacks the {kind} vLLM worktree at {root!r}")


def test_check_env_refuses_wrong_tree_subprocess():
    _needs_host_layout()
    wrong = _activated("clean", str(CHECK_ENV), "--expect", "vanilla")
    assert wrong.returncode == 1, wrong.stderr
    assert "PS_ENV='clean' but --expect vanilla" in wrong.stderr
    assert "!= 6bdabbad5b" in wrong.stderr or "HEAD" in wrong.stderr


def test_check_env_passes_matching_tree_subprocess():
    _needs_host_layout()
    for kind in ("clean", "vanilla"):
        ok = _activated(kind, str(CHECK_ENV), "--expect", kind, "--json")
        assert ok.returncode == 0, ok.stderr
        assert "check_env: OK" in ok.stderr
        facts = __import__("json").loads(ok.stdout)
        assert facts["vllm_file"].startswith(facts["vllm_root"])
        assert facts["verl_file"].startswith(facts["verl_root"])
        assert facts["metadata_version"] == facts["vllm_version"]
        assert facts["payload_missing"] == []


def test_clean_requires_base_ancestry(ce, good_layout, monkeypatch):
    monkeypatch.setattr(ce, "_run_git", lambda r, *a: "cafebabe" * 5 if a[:2] == ("rev-parse", "HEAD") else None)
    monkeypatch.setattr(ce, "_is_ancestor", lambda r, c: False)
    rep = _run(ce, good_layout)
    reasons = "\n".join(rep.failures)
    assert "does not descend from 6bdabbad5b" in reasons
    assert "does not descend from 2390a3f5cf" in reasons


# ----------------------------------------------------------------------------- run_gpu.sh contract
def _launcher_code_lines() -> list[str]:
    text = (ENV_DIR / "run_gpu.sh").read_text()
    return [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def test_run_gpu_never_calls_ray_stop():
    # `ray stop --force` kills every Ray process of this user host-wide (all agents share one account);
    # the process-group kill already covers Ray started under the private RAY_TMPDIR.
    assert not any("ray stop" in ln for ln in _launcher_code_lines())
    assert "never ray stop --force here" in (ENV_DIR / "run_gpu.sh").read_text()


def test_run_gpu_watchdog_and_leftover_scoping():
    code = "\n".join(_launcher_code_lines())
    assert 'setsid "$@" &' in code, "child must run in its own session/process group"
    assert "setsid bash -c 'sleep" in code, "watchdog must be its own process group (no orphaned sleep holding pipes)"
    assert 'kill -- -"$WATCH"' in code
    assert "ps -o sid=" in code, "leftovers are identified by session id, not by user (shared account)"


def test_run_gpu_accepts_gpu_1_and_refuses_missing_gpus():
    # GPU 1 is an ordinary GPU on this host (the original measurement host's exclusion was dropped
    # 2026-09-14): the launcher must not reject it by id. It may still refuse it as busy (rc 3).
    r = subprocess.run(
        ["bash", str(ENV_DIR / "run_gpu.sh"), "--gpus", "1", "--", "true"], capture_output=True, text=True
    )
    assert r.returncode != 2 and "never allowed" not in r.stderr, r.stderr
    r = subprocess.run(["bash", str(ENV_DIR / "run_gpu.sh"), "--", "true"], capture_output=True, text=True)
    assert r.returncode == 2 and "--gpus required" in r.stderr
