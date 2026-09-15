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
"""Environment contract check and GPU preflight for the precision-scheduler project.

Three independent modes, composable on one command line:

  --expect clean|dirty|vanilla   verify that the process is running under the matching
                                 ``activate.sh`` environment (see ``ENVIRONMENT.md``); exit 1 with
                                 one line per failed check.
  --pick-gpus N                  print N free GPU ids (from the allowed set {0,...,7}) comma
                                 separated on stdout; exit 3 if fewer than N are free. A GPU is not
                                 free when it hosts a compute process owned by another user or has
                                 more than 2048 MiB in use.
  --write-version-file ROOT      write an honest, gitignored ``ROOT/vllm/_version.py`` naming the
                                 commit the precompiled payload was built for.

Every launcher and GPU test calls ``--expect`` first. The checks are host independent: every
expected path comes from the environment variables ``activate.sh`` exports (PS_ENV, VLLM_ROOT,
VERL_ROOT, PYTHON_BIN, PYTHONPATH, LD_LIBRARY_PATH) and from sysconfig.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path

# The vanilla vLLM base every branch of this project starts from; the precompiled cu130 payload was
# fetched from wheels.vllm.ai for exactly this commit.
VLLM_BASE_COMMIT = "6bdabbad5bce747865fd3a249658518a4269cc22"
VLLM_BASE_SHORT = "6bdabbad5b"
# The verl measurement base of the clean branch.
VERL_BASE_COMMIT = "2390a3f5cff9d80d4844f8b4616ef8c2616276de"
# Fallback when `git describe` is unavailable (e.g. a tarball checkout).
FALLBACK_VLLM_VERSION = "0.22.1rc1.dev23+g6bdabbad5b.precompiled"
MIN_VLLM_VERSION = (0, 16)

# Gitignored artifacts that `populate_vllm_worktree.sh` copies (16) plus the generated version file.
PRECOMPILED_PAYLOAD = (
    "_C.abi3.so",
    "_C_stable_libtorch.abi3.so",
    "_flashmla_C.abi3.so",
    "_flashmla_extension_C.abi3.so",
    "_moe_C.abi3.so",
    "cumem_allocator.abi3.so",
    "spinloop.abi3.so",
    "vllm-rs",
    "third_party/deep_gemm",
    "third_party/flashmla/flash_mla_interface.py",
    "third_party/triton_kernels",
    "vllm_flash_attn/_vllm_fa2_C.abi3.so",
    "vllm_flash_attn/_vllm_fa3_C.abi3.so",
    "vllm_flash_attn/cute",
    "vllm_flash_attn/layers",
    "vllm_flash_attn/ops",
    "_version.py",
)
assert len(PRECOMPILED_PAYLOAD) == 17

# TransformerEngine carries a hand patch that keeps FlashAttention enabled with flash-attn 2.8.3.post1.
TE_PATCH_REL = "transformer_engine/pytorch/attention/dot_product_attention/utils.py"
TE_PATCH_LINE = 'max_version = PkgVersion("2.8.3.post1")'

ALLOWED_GPUS = (0, 1, 2, 3, 4, 5, 6, 7)
GPU_BUSY_MIB = 2048
ENV_KINDS = ("clean", "dirty", "vanilla")


@dataclass
class Report:
    """Collected facts and failures of one `--expect` run."""

    facts: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.failures.append(reason)

    @property
    def ok(self) -> bool:
        return not self.failures


# --------------------------------------------------------------------------------------- helpers
def parse_version_tuple(version: str) -> tuple[int, ...]:
    """Leading numeric release segment of a PEP 440 version ("0.22.1rc1.dev23+g..." -> (0, 22, 1))."""
    m = re.match(r"^\s*v?(\d+(?:\.\d+)*)", version or "")
    if not m:
        return ()
    return tuple(int(x) for x in m.group(1).split("."))


def _run_git(root: str | os.PathLike, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _is_ancestor(root: str | os.PathLike, commit: str) -> bool:
    """True when `commit` is an ancestor of HEAD in `root` (git prints nothing; rc 0 means yes)."""
    return _run_git(root, "merge-base", "--is-ancestor", commit, "HEAD") is not None


def _is_under(path: str | None, root: str | None) -> bool:
    if not path or not root:
        return False
    p = Path(path).resolve()
    r = Path(root).resolve()
    return p == r or r in p.parents


def cu13_lib_dir(purelib: str | None = None) -> str:
    purelib = purelib or sysconfig.get_paths()["purelib"]
    return os.path.join(purelib, "nvidia", "cu13", "lib")


def te_utils_path() -> str | None:
    spec = importlib.util.find_spec("transformer_engine")
    if spec is None or not spec.submodule_search_locations:
        return None
    pkg_dir = Path(list(spec.submodule_search_locations)[0]).parent
    return str(pkg_dir / TE_PATCH_REL)


def missing_payload(vllm_root: str) -> list[str]:
    base = Path(vllm_root) / "vllm"
    return [p for p in PRECOMPILED_PAYLOAD if not (base / p).exists()]


def version_from_describe(describe: str) -> str | None:
    """setuptools-scm style version from `git describe --tags --long` output.

    "v0.22.1rc0-23-g6bdabbad5b" -> "0.22.1rc1.dev23+g6bdabbad5b.precompiled" (the rc bump mirrors
    what vLLM's setup.py produced for the editable install; distance 0 keeps the tag as is).
    """
    m = re.match(r"^v?(.+)-(\d+)-g([0-9a-f]+)$", describe.strip())
    if not m:
        return None
    tag, distance, sha = m.group(1), int(m.group(2)), m.group(3)
    if distance == 0:
        return f"{tag}+g{sha}.precompiled"
    rc = re.match(r"^(\d+\.\d+\.\d+)rc(\d+)$", tag)
    if rc:
        next_tag = f"{rc.group(1)}rc{int(rc.group(2)) + 1}"
    else:
        parts = tag.split(".")
        parts[-1] = str(int(re.sub(r"\D.*$", "", parts[-1]) or 0) + 1)
        next_tag = ".".join(parts)
    return f"{next_tag}.dev{distance}+g{sha}.precompiled"


def honest_vllm_version(vllm_root: str, base_commit: str = VLLM_BASE_COMMIT) -> tuple[str, str]:
    """(version, source) for the precompiled payload commit; falls back to the fixed string."""
    describe = _run_git(vllm_root, "describe", "--tags", "--long", "--match", "v*", base_commit)
    if describe:
        v = version_from_describe(describe)
        if v:
            return v, f"git describe {base_commit[:10]} -> {describe}"
    return FALLBACK_VLLM_VERSION, "fallback constant"


def render_version_file(version: str) -> str:
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)(rc\d+)?(?:\.(dev\d+))?(?:\+(.+))?$", version)
    if not m:
        raise ValueError(f"cannot render version tuple for {version!r}")
    tup: list[str] = [m.group(1), m.group(2), m.group(3)]
    for g in (m.group(4), m.group(5), m.group(6)):
        if g:
            tup.append(repr(g))
    commit = None
    if m.group(6):
        commit = m.group(6).split(".")[0]
    return (
        "# file generated by vcs-versioning\n"
        "# don't change, don't track in version control\n"
        "from __future__ import annotations\n\n"
        '__all__ = ["__version__", "__version_tuple__", "version", "version_tuple", '
        '"__commit_id__", "commit_id"]\n\n'
        "version: str\n__version__: str\n__version_tuple__: tuple[int | str, ...]\n"
        "version_tuple: tuple[int | str, ...]\ncommit_id: str | None\n__commit_id__: str | None\n\n"
        f"__version__ = version = {version!r}\n"
        f"__version_tuple__ = version_tuple = ({', '.join(tup)})\n\n"
        f"__commit_id__ = commit_id = {commit!r}\n"
    )


def write_version_file(vllm_root: str) -> tuple[str, str]:
    version, source = honest_vllm_version(vllm_root)
    target = Path(vllm_root) / "vllm" / "_version.py"
    target.write_text(render_version_file(version))
    return version, source


# --------------------------------------------------------------------------------------- --expect
def check_expect(
    expect: str,
    env: dict | None = None,
    *,
    executable: str | None = None,
    vllm_module=None,
    verl_module=None,
    metadata_version: str | None = None,
    te_utils: str | None = None,
    purelib: str | None = None,
    import_te: bool = False,
) -> Report:
    """Run every contract check for `expect`; every parameter defaults to the live process state."""
    env = os.environ if env is None else env
    rep = Report()
    if expect not in ENV_KINDS:
        rep.fail(f"unknown --expect {expect!r}; choose from {ENV_KINDS}")
        return rep

    ps_env = env.get("PS_ENV")
    vllm_root = env.get("VLLM_ROOT")
    verl_root = env.get("VERL_ROOT")
    python_bin = env.get("PYTHON_BIN")
    rep.facts.update(expect=expect, ps_env=ps_env, vllm_root=vllm_root, verl_root=verl_root, python_bin=python_bin)

    if ps_env != expect:
        rep.fail(f"PS_ENV={ps_env!r} but --expect {expect}: source activate.sh {expect} first")
    if not vllm_root:
        rep.fail("VLLM_ROOT is not set (source activate.sh)")
    if not verl_root:
        rep.fail("VERL_ROOT is not set (source activate.sh)")

    # interpreter
    executable = executable or sys.executable
    rep.facts["executable"] = executable
    if python_bin and Path(executable).resolve() != Path(python_bin).resolve():
        rep.fail(f"sys.executable {executable} != PYTHON_BIN {python_bin}")

    # vllm tree + version
    try:
        vllm_module = vllm_module or importlib.import_module("vllm")
    except Exception as e:  # noqa: BLE001
        rep.fail(f"import vllm failed: {type(e).__name__}: {e}")
        vllm_module = None
    if vllm_module is not None:
        vllm_file = getattr(vllm_module, "__file__", None)
        vllm_version = getattr(vllm_module, "__version__", None)
        rep.facts.update(vllm_file=vllm_file, vllm_version=vllm_version)
        if vllm_root and not _is_under(vllm_file, vllm_root):
            rep.fail(f"vllm imported from {vllm_file}, not under VLLM_ROOT {vllm_root}")
        if metadata_version is None:
            try:
                metadata_version = importlib.metadata.version("vllm")
            except importlib.metadata.PackageNotFoundError:
                metadata_version = None
        rep.facts["metadata_version"] = metadata_version
        if expect == "dirty":
            # The dirty reference tree keeps its legacy layout on purpose (generated 0.1.dev version file,
            # fake 0.18.0 metadata shim, parser try/except in verl); only verl's import gate must hold.
            if parse_version_tuple(metadata_version or "") < (0, 8, 5):
                rep.fail(f"dirty metadata version {metadata_version!r} fails verl's >= 0.8.5 gate")
        else:
            if parse_version_tuple(vllm_version or "") < MIN_VLLM_VERSION:
                rep.fail(f"vllm.__version__ {vllm_version!r} does not parse >= 0.16 (dishonest _version.py?)")
            if metadata_version != vllm_version:
                rep.fail(
                    f"importlib.metadata.version('vllm') {metadata_version!r} != vllm.__version__ "
                    f"{vllm_version!r} (metadata shim missing or stale)"
                )

    # verl tree
    try:
        verl_module = verl_module or importlib.import_module("verl")
    except Exception as e:  # noqa: BLE001
        rep.fail(f"import verl failed: {type(e).__name__}: {e}")
        verl_module = None
    if verl_module is not None:
        verl_file = getattr(verl_module, "__file__", None)
        rep.facts["verl_file"] = verl_file
        if verl_root and not _is_under(verl_file, verl_root):
            rep.fail(f"verl imported from {verl_file}, not under VERL_ROOT {verl_root}")

    # TE patch
    te_utils = te_utils or te_utils_path()
    rep.facts["te_utils"] = te_utils
    if not te_utils or not os.path.exists(te_utils):
        rep.fail("transformer_engine not found (utils.py missing)")
    else:
        text = Path(te_utils).read_text()
        if TE_PATCH_LINE not in text:
            rep.fail(f"TE flash-attn gate patch missing: {te_utils} lacks `{TE_PATCH_LINE}` (TE reinstalled?)")
    if import_te and rep.ok:
        try:
            te_utils_mod = importlib.import_module("transformer_engine.pytorch.attention.dot_product_attention.utils")
            fa = te_utils_mod.FlashAttentionUtils
            rep.facts["te_flash_attn"] = bool(fa.is_installed)
            if not fa.is_installed:
                rep.fail("transformer_engine FlashAttentionUtils.is_installed is False")
        except Exception as e:  # noqa: BLE001
            rep.fail(f"import transformer_engine.pytorch failed: {type(e).__name__}: {e}")

    # precompiled payload
    if vllm_root:
        missing = missing_payload(vllm_root)
        rep.facts["payload_missing"] = missing
        if missing:
            rep.fail(f"precompiled payload incomplete in {vllm_root}/vllm: missing {', '.join(missing)}")

    # LD_LIBRARY_PATH
    want = cu13_lib_dir(purelib)
    ld = env.get("LD_LIBRARY_PATH", "")
    rep.facts["ld_library_path_head"] = ld.split(":", 1)[0] if ld else ""
    if ld.split(":", 1)[0] != want:
        rep.fail(f"LD_LIBRARY_PATH must start with {want} (got {ld.split(':', 1)[0]!r}); TE needs cublasLt 13.6")

    # tree identity per kind (skipped for non-git checkouts, e.g. tarballs; vanilla must be git)
    if vllm_root and os.path.isdir(vllm_root):
        head = _run_git(vllm_root, "rev-parse", "HEAD")
        rep.facts["vllm_head"] = head
        if expect == "vanilla":
            if head != VLLM_BASE_COMMIT:
                rep.fail(f"vanilla VLLM_ROOT {vllm_root} HEAD {head} != {VLLM_BASE_SHORT}")
            dirty = _run_git(vllm_root, "status", "--porcelain", "--", "vllm")
            if dirty:
                rep.fail(f"vanilla VLLM_ROOT {vllm_root} has local changes under vllm/: {dirty.splitlines()[0]}")
        elif expect == "clean" and head and not _is_ancestor(vllm_root, VLLM_BASE_COMMIT):
            rep.fail(f"clean VLLM_ROOT {vllm_root} HEAD {head} does not descend from {VLLM_BASE_SHORT}")
    if verl_root and os.path.isdir(verl_root) and expect != "dirty":
        head = _run_git(verl_root, "rev-parse", "HEAD")
        rep.facts["verl_head"] = head
        if head and not _is_ancestor(verl_root, VERL_BASE_COMMIT):
            rep.fail(f"VERL_ROOT {verl_root} HEAD {head} does not descend from {VERL_BASE_COMMIT[:10]}")
    return rep


# ------------------------------------------------------------------------------------ --pick-gpus
def query_gpus(nvidia_smi: str = "nvidia-smi") -> tuple[dict[int, int], dict[int, list[int]]]:
    """(memory_used_mib per gpu index, compute pids per gpu index)."""
    mem_out = subprocess.run(
        [nvidia_smi, "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    used: dict[int, int] = {}
    for line in mem_out.splitlines():
        if not line.strip():
            continue
        idx, mem = (x.strip() for x in line.split(","))
        used[int(idx)] = int(mem)
    procs: dict[int, list[int]] = {g: [] for g in used}
    for g in used:
        out = subprocess.run(
            [nvidia_smi, "--query-compute-apps=pid", "--format=csv,noheader", "-i", str(g)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        procs[g] = [int(p.strip()) for p in out.splitlines() if p.strip().isdigit()]
    return used, procs


def pid_owner(pid: int) -> str | None:
    out = subprocess.run(["ps", "-o", "user=", "-p", str(pid)], capture_output=True, text=True, check=False)
    return out.stdout.strip() or None


def free_gpus(
    used: dict[int, int],
    procs: dict[int, list[int]],
    *,
    me: str,
    owner_of=pid_owner,
    allowed=ALLOWED_GPUS,
    busy_mib: int = GPU_BUSY_MIB,
) -> list[int]:
    """Allowed GPUs with no foreign compute process and <= busy_mib MiB in use, ascending."""
    out = []
    for g in allowed:
        if g not in used:
            continue
        if used[g] > busy_mib:
            continue
        if any(owner_of(p) != me for p in procs.get(g, [])):
            continue
        out.append(g)
    return out


def pick_gpus(n: int) -> list[int]:
    used, procs = query_gpus()
    import getpass

    return free_gpus(used, procs, me=getpass.getuser())[:n]


# ------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expect", choices=ENV_KINDS, help="verify the activated environment kind")
    ap.add_argument("--import-te", action="store_true", help="also import transformer_engine.pytorch (slow)")
    ap.add_argument("--pick-gpus", type=int, metavar="N", help="print N free allowed GPU ids to stdout")
    ap.add_argument("--write-version-file", metavar="VLLM_ROOT", help="write an honest vllm/_version.py")
    ap.add_argument("--json", action="store_true", help="print the fact dictionary as JSON on stdout")
    args = ap.parse_args(argv)
    if not (args.expect or args.pick_gpus or args.write_version_file):
        ap.error("nothing to do: give --expect, --pick-gpus and/or --write-version-file")

    rc = 0
    if args.write_version_file:
        version, source = write_version_file(args.write_version_file)
        print(f"check_env: wrote {args.write_version_file}/vllm/_version.py = {version} ({source})", file=sys.stderr)

    if args.expect:
        rep = check_expect(args.expect, import_te=args.import_te)
        for reason in rep.failures:
            print(f"check_env: FAIL {reason}", file=sys.stderr)
        if rep.ok:
            print(
                f"check_env: OK {args.expect}: python={rep.facts.get('executable')} "
                f"vllm={rep.facts.get('vllm_file')} ({rep.facts.get('vllm_version')}) "
                f"verl={rep.facts.get('verl_file')}",
                file=sys.stderr,
            )
        if args.json and not args.pick_gpus:
            print(json.dumps(rep.facts, indent=2, default=str))
        if not rep.ok:
            return 1

    if args.pick_gpus:
        if args.pick_gpus < 1:
            print("check_env: --pick-gpus N must be >= 1", file=sys.stderr)
            return 2
        chosen = pick_gpus(args.pick_gpus)
        if len(chosen) < args.pick_gpus:
            print(
                f"check_env: only {len(chosen)} free GPU(s) among {list(ALLOWED_GPUS)}, need {args.pick_gpus}: "
                f"{chosen}",
                file=sys.stderr,
            )
            return 3
        print(",".join(str(g) for g in chosen))
    return rc


if __name__ == "__main__":
    sys.exit(main())
