# Environment: rationale, invariants and rebuild triggers

This file explains *why* the environment looks the way it does. `SETUP.md` is the from-scratch
recipe; this one is the reference you read when something breaks or when you are about to
reinstall a package. Every fact below was verified on the measurement host (2026-09-11 audit, C0
inventory) unless marked otherwise.

## 1. One conda env, three vLLM trees, selected by `PYTHONPATH`

There is exactly one Python environment (`envs/vllm`: Python 3.12, torch 2.11.0+cu130, TE 2.13.0,
megatron-core 0.18.0, ray 2.56.0, transformers 5.9.0). vLLM is an *editable* install whose native
kernels are the precompiled cu130 wheel for upstream commit `6bdabbad5b`. Three vLLM checkouts
exist and all three are importable from the same env:

| kind      | vLLM tree                              | verl tree               | purpose                                         |
|-----------|----------------------------------------|-------------------------|-------------------------------------------------|
| `vanilla` | detached worktree at `6bdabbad5b`      | `verl-clean`            | oracle for "nothing changed"                    |
| `clean`   | branch `rollout-precision-scheduler-clean` | `verl-clean` (same branch) | the re-implementation                       |
| `dirty`   | the original experimental checkout     | the original verl checkout | read-only reference for equivalence runs    |

`activate.sh <kind>` selects one by putting its root first on `PYTHONPATH`. This works because a
package found on `PYTHONPATH` wins over the editable-install finder that lives in `site-packages`
(verified: `vllm.__file__` and `vllm._C.__file__` both resolve into the `PYTHONPATH` tree). Nothing
is ever reinstalled to switch trees, and the dirty tree is never modified.

Consequence: a shell that did *not* source `activate.sh` silently imports the editable (dirty)
tree. That is why every launcher and every GPU test runs `check_env.py --expect <kind>` first and
why `check_env.py` verifies `vllm.__file__` against `$VLLM_ROOT` rather than trusting `PS_ENV`.

verl is never installed; it is imported from `$VERL_ROOT` through `PYTHONPATH` exactly as the
original launchers did.

## 2. The precompiled payload and the "no native change" rule

`pip install -e` with `VLLM_USE_PRECOMPILED=1` downloads the wheel that vLLM's CI built for the
merge-base of the checked-out branch with upstream `main`, extracts the compiled artifacts into
the source tree, and leaves them there gitignored. For this project that commit is
`6bdabbad5bce747865fd3a249658518a4269cc22`, CUDA variant `cu130`. The extracted files are:

```
vllm/_C.abi3.so                    vllm/_C_stable_libtorch.abi3.so   vllm/_flashmla_C.abi3.so
vllm/_flashmla_extension_C.abi3.so vllm/_moe_C.abi3.so               vllm/cumem_allocator.abi3.so
vllm/spinloop.abi3.so              vllm/vllm-rs                       vllm/third_party/deep_gemm/
vllm/third_party/flashmla/flash_mla_interface.py                     vllm/third_party/triton_kernels/*
vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so   vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so
vllm/vllm_flash_attn/{cute,layers,ops}/
```

plus the generated `vllm/_version.py`. `populate_vllm_worktree.sh` copies the first sixteen into a
new worktree; `check_env.py` refuses to run if any of the seventeen is missing.

The same payload serves the vanilla, clean and dirty trees because **none of them changes
`csrc/`, `cmake/` or `CMakeLists.txt`** (`git diff --stat 6bdabbad5b..HEAD -- csrc cmake
CMakeLists.txt` is empty for every branch). This is a hard rule of the project: the day a commit
touches native code, the precompiled payload is invalid for that tree and a full CUDA build of
vLLM (tens of minutes, needs the full toolchain) becomes necessary. Keep everything in Python,
Triton and torch.

## 3. Honest version, honest metadata

`vllm.__version__` comes only from the gitignored, generated `vllm/_version.py`. The original
editable install generated `0.1.dev17121+g42d148122.precompiled` (setuptools-scm could not see a
tag at install time) and a later `pip install -e` into the side `.venv` rewrote it to
`0.1.dev17121+g34e66a379.d20260902`. Both are lies with real consequences: verl branches on
`vllm.__version__` in `vllm_async_server.py` (`>0.11`, `==0.12`, `>=0.13`, `>=0.16`) and on
`importlib.metadata.version("vllm")` in `verl/third_party/vllm/__init__.py` (`>=0.7.0` or
`ValueError`, `>=0.8.5` for sleep level 2) and `vllm_rollout.py` (`>=0.11.0`). With the 0.1.dev
string pristine verl `2390a3f5` takes the pre-0.11 branch and dies with
`ImportError: cannot import name 'FlexibleArgumentParser' from 'vllm.utils'`; the experimental tree
papered over that with a try/except that the clean branch does not carry.

The honest version is what setuptools-scm computes for the payload commit now that the tag is
present locally: `git describe --tags --long 6bdabbad5b` gives `v0.22.1rc0-23-g6bdabbad5b`, hence
`0.22.1rc1.dev23+g6bdabbad5b.precompiled`. `check_env.py --write-version-file <root>` writes that
file (deriving it from `git describe` of the payload commit, falling back to the constant), and a
minimal dist-info with the same `Version:` lives *outside* both repos at
`<ENVSHIMS>/vllm_metadata/vllm-0.22.1rc1.dev23+g6bdabbad5b.precompiled.dist-info/{METADATA,WHEEL}`.
It goes on `PYTHONPATH` so `importlib.metadata` finds it before the editable dist-info in
`site-packages`. `check_env.py` asserts `vllm.__version__ == importlib.metadata.version("vllm")`
and that it parses `>= 0.16` for the clean and vanilla kinds.

Why a shim and not `pip install`: rewriting pip's own dist-info would tie the honest version to
one checkout, while the shim applies to whichever tree `PYTHONPATH` selects. Only one vllm
dist-info may be on `PYTHONPATH`; never combine the shim with the old `fake_vllm_metadata`.

The `dirty` kind deliberately keeps its legacy layout (0.1.dev version file, fake `0.18.0`
metadata from the report directory, the parser try/except) so that archived launches stay
bit-for-bit reproducible. `check_env.py --expect dirty` therefore only checks that the metadata
version passes verl's `>= 0.8.5` gate.

## 4. `LD_LIBRARY_PATH` rule (TransformerEngine and cuBLAS)

`import transformer_engine.pytorch` dlopens `libcublasLt.so.13` and needs the symbol
`cublasLtGroupedMatrixLayoutInit_internal`, present in the pip package `nvidia-cublas==13.6.0.2`
(`site-packages/nvidia/cu13/lib`) but not in the system toolkit's `libcublasLt.so.13` 13.1.0.3
under `/usr/local/cuda-13.0/lib64`. The measurement host's `~/.bashrc` prepends the system
directory, which makes TE fail to import in every interactive shell. The rule:

```
LD_LIBRARY_PATH=<site-packages>/nvidia/cu13/lib:<site-packages>/nvidia/cudnn/lib:<site-packages>/nvidia/nccl/lib:$LD_LIBRARY_PATH
```

`activate.sh` exports exactly this; `check_env.py` fails unless the first entry is the pip cu13
lib dir. Never rely on the login shell.

Related trap: installing anything that depends on `nvidia-cublas` (flash-attn does) downgrades it
to 13.1.0.3. Always re-pin afterwards:

```bash
python -m pip install --no-deps --force-reinstall nvidia-cublas==13.6.0.2
```

## 5. The TransformerEngine flash-attn gate patch

TE 2.13.0 accepts flash-attn `>=2.1.1,<=2.8.3`; `packaging` orders `2.8.3.post1 > 2.8.3`, so with
flash-attn 2.8.3.post1 installed TE silently disabled FlashAttention, fell back to unfused
attention and OOMed on long-sequence log-prob (this was the root cause behind the original
"long sequence OOM", see the archived `env_fix_notes.md`). The fix is a one-line edit *inside
site-packages*:

```diff
--- <site-packages>/transformer_engine/pytorch/attention/dot_product_attention/utils.py
+++ <site-packages>/transformer_engine/pytorch/attention/dot_product_attention/utils.py
@@ class FlashAttentionUtils:
-    max_version = PkgVersion("2.8.3")
+    max_version = PkgVersion("2.8.3.post1")
```

Verification (needs the LD_LIBRARY_PATH rule; CPU is enough for the import):

```bash
python - <<'PY'
from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils as F
print(F.version, F.is_installed, F.max_version, F.v2_7_0_plus)   # 2.8.3.post1 True 2.8.3.post1 True
PY
```

RECORD-hash check. Because the edit is a hand patch, pip's `RECORD` still lists the pristine
hash, which is exactly how a silent revert is detected:

```
RECORD:  transformer_engine/pytorch/attention/dot_product_attention/utils.py,sha256=UZrIc7YaNRD-sSjpIDmXbWwdT9k5BdISx2LjIazSIeA,98289
on disk: sha256=kdcgc-YLKp0X_E8yEw-KwIECc-e4t1lby1aZPr5XPQ0        (patched)
```

If the on-disk hash ever equals the RECORD hash, TE was reinstalled and the patch is gone.
`check_env.py` checks the cheaper invariant on every run (the literal line
`max_version = PkgVersion("2.8.3.post1")` must be present); `--import-te` additionally imports TE
and asserts `FlashAttentionUtils.is_installed`. The audit confirmed this is the only modified
file in `transformer_engine`, and that `megatron_core` and `mbridge` are pristine.

## 6. Wheel provenance and build commands

| package | version | how it got here |
|---|---|---|
| torch / torchvision / torchaudio | 2.11.0+cu130 / 0.26.0+cu130 / 2.11.0+cu130 | pip, PyTorch cu130 index |
| vllm | editable, payload for `6bdabbad5b` cu130 | `VLLM_USE_PRECOMPILED=1 pip install -e` (wheels.vllm.ai) |
| flash-attn | 2.8.3.post1 (`cp312-cp312-linux_x86_64`, built locally) | `MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=8.0 pip install flash-attn==2.8.3.post1 --no-build-isolation --no-cache-dir` (about 30 min; then re-pin cublas) |
| transformer-engine (+ `transformer_engine_torch`) | 2.13.0 | pip; then the section 5 patch |
| megatron-core / megatron-bridge / mbridge | 0.18.0 / 0.5.0 / 0.15.1 | pip |
| flashinfer-python / flashinfer-cubin | 0.6.11.post2 | pinned by vLLM `requirements/cuda.txt` |
| nvidia-cublas | 13.6.0.2 | pip, force re-pinned after flash-attn |
| nvidia-cudnn-cu13 / nvidia-nccl-cu13 | 9.19.0.56 / 2.28.9 | pip (torch deps) |
| ray, transformers, peft, compressed-tensors, tensordict | 2.56.0, 5.9.0, 0.19.1, 0.15.0.1, 0.10.0 | pip |
| causal_conv1d / mamba_ssm | 1.6.2.post1 / 2.3.2.post1 (`cp312`, built against torch 2.11+cu13) | built from source into a scratch env, then copied to `<ENVSHIMS>/pydeps` (no wheel cache survived; rebuild from source if torch changes) |
| fla (flash-linear-attention, fla-core) | 0.5.1 | pure Python; copied to `<ENVSHIMS>/pydeps` |

`nvcc` 13.0.88 and driver 580.105.08 were in use; A100-SXM4-80GB (sm80), hence
`TORCH_CUDA_ARCH_LIST=8.0`.

## 7. The `envshims` directory: what it holds and what was dropped

`<ENVSHIMS>` (on the measurement host `/data/huanchen/envshims`) lives outside both repositories:

```
<ENVSHIMS>/pydeps/          fla/, fla_core-0.5.1.dist-info, flash_linear_attention-0.5.1.dist-info,
                            causal_conv1d/, causal_conv1d-1.6.2.post1.dist-info, causal_conv1d_cuda.*.so,
                            mamba_ssm/, mamba_ssm-2.3.2.post1.dist-info, selective_scan_cuda.*.so
<ENVSHIMS>/vllm_metadata/   vllm-0.22.1rc1.dev23+g6bdabbad5b.precompiled.dist-info/{METADATA,WHEEL}
```

Why these three packages: Megatron's `gated_delta_net.py` imports `fla` and its utils import
`causal_conv1d`; both are needed for Qwen3.5 linear attention. The conda env's own `fla` is an
incomplete tree (no `__init__.py`, its dist-info a dangling symlink into a deleted-in-spirit old
env), and `causal_conv1d` / `mamba_ssm` are absent from it altogether.

Dropped from the experimental `vllm_env_extra_deps` shim directory (584 MB in the report dir):

- `mbridge`, `modelopt`, `pulp`, `scipy` symlinks: identical packages are installed natively in the
  env (`diff -rq` clean); the symlinks only shadowed them.
- the `megatron` symlink into the old stripped env: its only difference from the native
  megatron-core 0.18.0 was a 4-line `MEGATRON_DISABLE_JIT_FUSER` gate in `core/jit.py`. The clean
  branch is FSDP2-first and single-GPU (decisions 10), no recipe sets that variable, so the gate is
  not carried. If a Megatron recipe ever needs it, apply the 4-line patch to the native copy and
  document it like the TE patch (namespace packages make a single-file overlay impossible).
- `fake_vllm_metadata/vllm-0.18.0.dist-info`: replaced by the honest dist-info (section 3).
- the stale `flash_attn-2.8.1+cu12torch2.8` wheel, the zero-byte `=1.24.0` / `=1.4.3` files: junk.
- the `.venv` inside the vLLM checkout: has torch but no TE/megatron/verl; not used by any
  clean-branch tooling (it remains useful only for `pre-commit`/`ruff`).

## 8. Rebuild and re-check triggers

| event | consequence | action |
|---|---|---|
| `pip install/upgrade transformer-engine`, env recreation | section 5 patch silently reverted; TE falls back to unfused attention | re-apply the patch; `check_env.py --expect clean --import-te` |
| installing anything that pulls `nvidia-cublas` (flash-attn, some torch extras) | cublas 13.1.0.3; TE import fails with `cublasLtGroupedMatrixLayoutInit_internal` | `pip install --no-deps --force-reinstall nvidia-cublas==13.6.0.2` |
| any `pip install -e` of vLLM (either env) | regenerates `vllm/_version.py` with a dishonest string | `check_env.py --write-version-file <root>` for every tree |
| a commit under `csrc/`, `cmake/`, `CMakeLists.txt` | precompiled payload invalid for that tree | do not; otherwise full vLLM CUDA build |
| torch upgrade | `causal_conv1d_cuda` / `selective_scan_cuda` / flash-attn ABI break | rebuild all three from source |
| new vLLM worktree | no payload, no version file | `populate_vllm_worktree.sh` + `--write-version-file` |
| `ray start` as a daemon from another shell | workers import whatever that shell had on `PYTHONPATH` | never; Ray is started in-process so workers inherit the driver's env |

## 9. GPU discipline (decision 13)

`check_env.py --pick-gpus N` selects from all GPUs `{0,...,7}` (the GPU 1 exclusion of the original
measurement host was dropped on 2026-09-14: this host has no reserved GPU), excludes any GPU
with a compute process owned by another user or more than 2048 MiB in use, prints N ids and exits
3 when fewer are free. `run_gpu.sh --gpus <ids> [--timeout s] -- <cmd>` runs the command in its
own process group with `CUDA_VISIBLE_DEVICES` set, uses a private `RAY_TMPDIR`, kills the group
and stops Ray on exit/timeout/Ctrl-C, and exits 4 if a process of ours is still on the GPUs
afterwards. Tests that spawn vLLM in-process go through the same launcher.

## 10. Greedy token identity across trees: measured nondeterminism

The C0 acceptance smoke test (`tests/precision_scheduler/gpu/test_greedy_identity.py`) decodes 16
GSM8K prompts greedily (64 tokens, temperature 0, `enforce_eager=False`, gpu_memory_utilization
0.5) in the vanilla, dirty-with-flags-off and clean trees and asserts token identity. Measured on
one A100 on 2026-09-11 (token dumps under the session scratchpad `c0/run2..run6`):

| model | mode | vanilla vs vanilla (rerun) | vanilla vs dirty | vanilla vs clean |
|---|---|---|---|---|
| Qwen3.5-4B | 16 prompts batched | 4/16 prompts differ | 6/16 | 5/16 |
| Qwen3.5-4B | `max_num_seqs=1` | 2/16 | not run | not run |
| Qwen3.5-4B | `VLLM_BATCH_INVARIANT=1` | refused: "batch_invariant mode is not supported for GDN_ATTN" | | |
| Phi-4-mini-reasoning | `max_num_seqs=1` + batch-invariant | 2/16 (cold-cache run vs warm) | run b == dirty exactly (0/16) | 2/16 vs run b |

| Phi-4-mini-reasoning | `max_num_seqs=1` + batch-invariant + `enforce_eager` | **0/16** | **0/16** | **0/16** |

With torch.compile and CUDA graphs on, two runs of the *same* vanilla tree already differ, so any
cross-tree difference in that mode is within run-to-run noise (the divergences start late in the
response, token 8-62, the signature of a bf16 numerics flip, not of a different code path; the
residual source is the compiled/graph path, most likely inductor or Triton autotuning with cold vs
warm caches). In eager batch-invariant mode decoding is reproducible and **vanilla, dirty with all
flags off, and clean produce identical token ids on all 16 prompts (1024 tokens each)**. That is
the mode the test uses; it deviates from the card's `enforce_eager=False` for exactly this reason.
The script keeps `--max-num-seqs`, `--batch-invariant` and `--enforce-eager` switches, and the test
asserts vanilla-vs-vanilla reproducibility first so that a regression fails with the right diagnosis.
