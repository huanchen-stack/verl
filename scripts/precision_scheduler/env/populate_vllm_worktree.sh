#!/usr/bin/env bash
# populate_vllm_worktree.sh -- copy the gitignored precompiled cu130 payload into a fresh vLLM worktree.
#
# Usage:  populate_vllm_worktree.sh <source vllm checkout> <target vllm worktree>
#
# The payload (7 top-level .so, vllm-rs, vllm_flash_attn kernels + cute/layers/ops, deep_gemm,
# triton_kernels, flash_mla_interface.py) is the wheels.vllm.ai wheel built for csrc @ 6bdabbad5b; it is
# ABI-identical for vanilla, dirty and clean trees because none of them change csrc/, cmake/ or
# CMakeLists.txt. All paths are gitignored, so `git status` stays clean afterwards. Write the honest
# vllm/_version.py separately with `check_env.py --write-version-file <target>`.
set -euo pipefail
SRC=${1:?source vllm checkout}; DST=${2:?target vllm worktree}
for p in _C.abi3.so _C_stable_libtorch.abi3.so _flashmla_C.abi3.so _flashmla_extension_C.abi3.so \
         _moe_C.abi3.so cumem_allocator.abi3.so spinloop.abi3.so vllm-rs \
         third_party/deep_gemm third_party/flashmla/flash_mla_interface.py third_party/triton_kernels \
         vllm_flash_attn/_vllm_fa2_C.abi3.so vllm_flash_attn/_vllm_fa3_C.abi3.so \
         vllm_flash_attn/cute vllm_flash_attn/layers vllm_flash_attn/ops; do
  mkdir -p "$(dirname "$DST/vllm/$p")"
  cp -a "$SRC/vllm/$p" "$DST/vllm/$p"
done
echo "populated $DST"
