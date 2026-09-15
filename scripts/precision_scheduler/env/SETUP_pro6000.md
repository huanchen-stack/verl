# verl + vllm development environment

Set up 2026-09-14 on this box (1x RTX PRO 6000 Blackwell 96 GB, sm_120, driver
595 / CUDA 13.2, 32 CPUs, 251 GB RAM, home dir on NFS, no sudo).

## Layout

| Path | What |
|---|---|
| `~/verl` | verl fork (upstream main, 2026-09-11). Installed editable. |
| `~/vllm` | vllm fork (upstream main commit `6bdabbad5`, v0.22.1rc0-23). Installed editable with the matching precompiled wheel. |
| `~/verl/.venv` | The one Python env everything runs from (uv-managed CPython 3.12.14). |
| `~/miniforge3/envs/cuda130` | CUDA 13.0.3 toolkit (nvcc + headers). Used only for JIT / source builds. |
| `~/rl_env.sh` | `source` this in every shell before running anything. |
| `~/run_grpo_gsm8k_1gpu.sh` | Single-GPU GRPO template (FSDP actor + vllm rollout, GSM8K). |
| `~/data/gsm8k/{train,test}.parquet` | Preprocessed by `examples/data_preprocess/gsm8k.py`. |

Both repos got an `upstream` remote (vllm-project/vllm, verl-project/verl) and
upstream tags were fetched, which is what makes vllm's version resolve to
`0.22.1rc1.dev23+g6bdabbad5`.

## Daily use

```bash
source ~/rl_env.sh          # GPU 0 only, venv, nvcc, VERL_USE_UV=0
bash ~/run_grpo_gsm8k_1gpu.sh                       # 1-step smoke test
TOTAL_STEPS=100 MODEL_PATH=Qwen/Qwen3-4B bash ~/run_grpo_gsm8k_1gpu.sh
```

Any verl `examples/*.sh` script also works after sourcing `rl_env.sh`
(they honour `VERL_USE_UV=0` and use the activated interpreter). Override
`NGPUS_PER_NODE=1 ROLLOUT_TP=1` for those, since they default to 8 GPUs.

Python-only edits in `~/vllm` or `~/verl` take effect immediately (editable
installs). For C++/CUDA edits in `~/vllm`:

```bash
source ~/rl_env.sh
cd ~/vllm && uv pip install -e . --no-build-isolation   # full build, ~30-60 min, sm_120 only
```

## GPU visibility

`CUDA_VISIBLE_DEVICES=0` is exported by `~/.bashrc`, `~/.profile` and
`~/rl_env.sh`, so every shell and every Ray worker sees exactly one GPU.
Change it per-run with `CUDA_VISIBLE_DEVICES=... bash ...` if that ever changes.

## Key versions

torch 2.11.0+cu130, vllm 0.22.1rc1.dev23 (editable), verl 0.10.0.dev (editable),
flash-attn 2.8.3, flashinfer 0.6.11.post2 + flashinfer-jit-cache, transformers
5.9.0, ray 2.55.1, tensordict 0.10.0, liger-kernel 0.8.2, peft 0.19.1.

## Things that bit during setup (and the fix baked in)

- **No CUDA toolkit on the host** and no sudo. vllm's sampler uses flashinfer
  kernels that are JIT-compiled with nvcc, so the engine died at startup.
  Fix: `cuda-toolkit=13.0.3` in a conda env, `CUDA_HOME` pointed at it, headers
  symlinked from `targets/x86_64-linux/include` into `include/`, plus the
  prebuilt `flashinfer-jit-cache` wheel so common kernels need no compile.
- **System python3.12 has no dev headers** (`pyconfig.h` missing), so any
  torch/flashinfer JIT build failed. Fix: venv built on uv-managed Python 3.12.
- **`uv sync` / `uv run` revert vllm to PyPI 0.24.0** (verl's lock pins it).
  Fix: run with `VERL_USE_UV=0` (set by `rl_env.sh`). If you ever re-sync,
  reinstall afterwards:
  `VLLM_USE_PRECOMPILED=1 uv pip install -e ~/vllm` and the jit-cache wheel.
- verl's lock pins torch 2.11.0; vllm main at this commit also wants 2.11.0,
  so the two coexist. Bumping the vllm fork to a newer upstream main may pull a
  newer torch; check `requirements/cuda.txt` first.

## Rebuild from scratch

```bash
cd ~/verl
uv sync --python "$(uv python find --managed-python 3.12)" --all-packages --extra vllm --extra fsdp
VLLM_USE_PRECOMPILED=1 uv pip install -e ~/vllm
uv pip install "flashinfer-jit-cache==$(.venv/bin/python -c 'import flashinfer;print(flashinfer.__version__)')" \
    --index https://flashinfer.ai/whl/cu130/
mamba create -n cuda130 -c nvidia -c conda-forge cuda-toolkit=13.0.3   # if the conda env is gone
```

## Precision-scheduler worktrees

The environment above is the shared venv. The precision-scheduler work itself
lives in two worktrees on the `-pro6000-adapt` branches, reached by putting both
roots first on `PYTHONPATH` (which beats the editable install in site-packages,
as `ENVIRONMENT.md` describes for the original host):

| worktree | branch | purpose |
|---|---|---|
| `~/vllm-ps` | `rollout-precision-scheduler-clean-pro6000-adapt` | the runtime being changed |
| `~/verl-ps` | `rollout-precision-scheduler-clean-pro6000-adapt` | harness, policy toolkit, docs |
| `~/vllm-ps-run` | detached | pinned copy for long measurement runs |

A fresh vLLM worktree needs the gitignored precompiled payload before it will
import:

```bash
bash ~/verl-ps/scripts/precision_scheduler/env/populate_vllm_worktree.sh ~/vllm ~/vllm-ps
python ~/verl-ps/scripts/precision_scheduler/env/check_env.py --write-version-file ~/vllm-ps
```

`~/vllm-ps-run` exists so a multi-hour TPOT grid cannot pick up edits made to
`~/vllm-ps` while it runs; each precision row is a fresh child process that
re-imports vLLM, so editing the tree under a live run is not safe.

Tests that need checkpoints take `DUAL_PRECISION_HF_HUB`; on this host that is
`/mnt/home/huanchen/.cache/huggingface/hub`. Without it they skip, since they
default to the original host's `/data/huggingface/hub`.

Note that `activate_pro6000.sh` sets up the shared venv only. It does **not** put
the worktrees on `PYTHONPATH`; the runner scripts and test invocations do that
themselves, so a bare activated shell imports vLLM from `~/vllm` on `main`.
