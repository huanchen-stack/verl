#!/usr/bin/env bash
# Build transformer-engine from source with sm_120 kernels (RTX PRO 6000 Blackwell).
#
#   bash build_transformer_engine_sm120.sh
#
# WHY: the verl wheelhouse builds TE for TORCH_CUDA_ARCH_LIST "9.0;10.0" (Hopper and
# datacenter Blackwell). This host is sm_120, a different architecture, so TE's own
# handwritten kernels have no image here and Megatron dies at the first RMSNorm with
# "CUDA Error: no kernel image is available for execution". cuBLAS-backed ops such as
# te.Linear still work, so a smoke test that only exercises those passes and hides this.
#
# NOT FIXABLE WITH A PREBUILT WHEEL: NVIDIA's transformer-engine-cu13 on PyPI needs a
# newer cuBLASLt (cublasLtGroupedMatrixLayoutInit_internal) than torch 2.11+cu130 ships,
# and neither torch's copy nor CUDA 13.0.3 has that symbol.
#
# IMPORTANT: pyproject pins transformer-engine to the verl wheelhouse index, so any
# `uv sync` reinstalls the sm_90/sm_100 wheel and silently reintroduces the failure.
# Re-run this script after every sync that touches the megatron extra, and verify with
# the RMSNorm check at the end.
#
# Prerequisites installed into the cuda130 conda env by this script's notes:
#   mamba install -n cuda130 -c nvidia -c conda-forge nvtx-c nccl
# (TE needs nvtx3/nvToolsExt.h and nccl.h, which CUDA 13 splits out.)
set -euo pipefail
source /mnt/home/huanchen/rl_env.sh >/dev/null 2>&1
SP=/mnt/home/huanchen/verl/.venv/lib/python3.12/site-packages
export CUDA_HOME=/mnt/home/huanchen/miniforge3/envs/cuda130
export PATH="$CUDA_HOME/bin:$PATH"
export CUDNN_PATH="$SP/nvidia/cudnn"
export CPLUS_INCLUDE_PATH="$SP/nvidia/cudnn/include:$SP/nvidia/nccl/include:$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$CUDA_HOME/include:${C_INCLUDE_PATH:-}"
export NVTE_BUILD_THREADS_PER_JOB=4
export LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/nccl/lib:$SP/nvidia/cu13/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/nccl/lib:$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS="120"
export MAX_JOBS=28
export CC=gcc CXX=g++
cd /mnt/home/huanchen/build_te
uv pip install --python /mnt/home/huanchen/verl/.venv/bin/python --no-build-isolation --no-deps -v ./TransformerEngine
echo "BUILD_EXIT=$?"

# Verify the kernel that fails with the wheelhouse build.
"$SP/../../../bin/python" - <<'PY'
import torch, transformer_engine, transformer_engine.pytorch as te
x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
te.RMSNorm(512, params_dtype=torch.bfloat16).cuda()(x)
torch.cuda.synchronize()
print(f"OK transformer-engine {transformer_engine.__version__} RMSNorm runs on "
      f"{torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
PY
