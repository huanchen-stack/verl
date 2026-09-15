#!/usr/bin/env bash
# Activate the verl + vllm development environment.
#
#   source ~/rl_env.sh
#
# What it does:
#   * exposes only GPU 0 to CUDA (CUDA_VISIBLE_DEVICES=0)
#   * puts the CUDA 13.0 toolkit (conda env cuda130) on PATH / CUDA_HOME
#   * activates ~/verl/.venv  (uv-managed Python 3.12 with dev headers, torch 2.11+cu130, verl editable,
#     vllm editable from ~/vllm, flash-attn, ray, ...)
#   * tells verl's example scripts to use this activated interpreter instead of
#     re-syncing through `uv run` (which would replace the editable ~/vllm with
#     the PyPI vllm==0.24.0 pinned in uv.lock)
#   * keeps HF / vllm caches in the default ~/.cache locations

export CUDA_VISIBLE_DEVICES=0

VERL_ROOT="$HOME/verl"
VLLM_ROOT="$HOME/vllm"

# CUDA 13.0 toolkit (nvcc + headers) from conda env `cuda130`. Needed for
# flashinfer's JIT kernels (used by vllm sampling / attention on this Blackwell
# sm_120 GPU) and for building vllm's C++/CUDA extensions from source.
# Deliberately NOT added to LD_LIBRARY_PATH: torch ships its own CUDA runtime
# libs via the nvidia-*-cu13 pip wheels and must keep using those.
export CUDA_HOME="$HOME/miniforge3/envs/cuda130"
case ":$PATH:" in *":$CUDA_HOME/bin:"*) ;; *) export PATH="$CUDA_HOME/bin:$PATH" ;; esac
# Only compile kernels for the GPU actually present (RTX PRO 6000 Blackwell).
export TORCH_CUDA_ARCH_LIST="12.0"

if [ ! -f "$VERL_ROOT/.venv/bin/activate" ]; then
    echo "rl_env.sh: $VERL_ROOT/.venv not found. Rebuild with:" >&2
    echo "  cd $VERL_ROOT && uv sync --python \"\$(uv python find --managed-python 3.12)\" --all-packages --extra vllm --extra fsdp" >&2
    echo "  VLLM_USE_PRECOMPILED=1 uv pip install -e $VLLM_ROOT" >&2
    echo "  uv pip install flashinfer-jit-cache==\$(python -c 'import flashinfer;print(flashinfer.__version__)') --index https://flashinfer.ai/whl/cu130/" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$VERL_ROOT/.venv/bin/activate"

# verl's examples/*.sh honour these: run with the ambient (this) python and
# never let `uv sync` / `uv run` clobber the editable vllm checkout.
export VERL_USE_UV=0
export VERL_UV_NO_INSTALL=vllm

# Ray >= 2.47 tries to re-wrap workers in `uv run`; verl disables it too, but
# be explicit so ad-hoc ray usage behaves the same way.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# Precompiled-wheel mode for any future `pip install -e ~/vllm` re-install
# (python-only changes to vllm need no rebuild at all: it is editable).
export VLLM_USE_PRECOMPILED=1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "verl+vllm env active: python=$(command -v python)  GPU(s)=$CUDA_VISIBLE_DEVICES"
