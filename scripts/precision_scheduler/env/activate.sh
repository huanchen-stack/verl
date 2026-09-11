#!/usr/bin/env bash
# activate.sh -- select which vLLM / verl checkout `import vllm` and `import verl` resolve to.
#
# Usage:  source scripts/precision_scheduler/env/activate.sh clean|dirty|vanilla [VLLM_ROOT_OVERRIDE]
#
#   clean    vLLM /data/huanchen/vllm-clean   + verl $VERL_ROOT (default /data/huanchen/verl-clean)
#   vanilla  vLLM /data/huanchen/vllm-vanilla + verl $VERL_ROOT (default /data/huanchen/verl-clean)
#   dirty    vLLM /data/huanchen/vllm         + verl /data/huanchen/verl, using the report-dir shims
#            (vllm_env_extra_deps, fake_vllm_metadata) verbatim so old launches stay reproducible.
#
# Export VERL_ROOT before sourcing to point verl at a per-component worktree. The second positional
# argument overrides VLLM_ROOT (per-component vLLM worktrees). The mechanism: PYTHONPATH precedes the
# editable-install finder in site-packages, so the first `vllm/` package on PYTHONPATH wins (verified in
# the C0 inventory). LD_LIBRARY_PATH is forced to start with the pip nvidia cu13 lib dir because
# transformer_engine dlopens libcublasLt.so.13 and needs 13.6.0.2, not the /usr/local/cuda-13.0 copy
# that ~/.bashrc puts first (see ENVIRONMENT.md). Nothing here touches the conda env itself.
#
# Contract consumed by check_env.py: PS_ENV, VLLM_ROOT, VERL_ROOT, PYTHON_BIN, PYTHONPATH, LD_LIBRARY_PATH.
PS_ENV=${1:-clean}
_SP=/data/huanchen/miniforge3/envs/vllm/lib/python3.12/site-packages
case "$PS_ENV" in
  clean)   VLLM_ROOT=${2:-/data/huanchen/vllm-clean};   VERL_ROOT=${VERL_ROOT:-/data/huanchen/verl-clean} ;;
  vanilla) VLLM_ROOT=${2:-/data/huanchen/vllm-vanilla}; VERL_ROOT=${VERL_ROOT:-/data/huanchen/verl-clean} ;;
  dirty)   VLLM_ROOT=${2:-/data/huanchen/vllm};         VERL_ROOT=${VERL_ROOT:-/data/huanchen/verl} ;;
  *) echo "activate.sh: unknown env '$PS_ENV'" >&2; return 1 2>/dev/null || exit 1 ;;
esac
export PS_ENV VLLM_ROOT VERL_ROOT
export PYTHON_BIN=/data/huanchen/miniforge3/envs/vllm/bin/python
export LD_LIBRARY_PATH="$_SP/nvidia/cu13/lib:$_SP/nvidia/cudnn/lib:$_SP/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
if [ "$PS_ENV" = dirty ]; then
  _RD=/data/huanchen/verl/.codex-report/rl-workflow
  export PYTHONPATH="$_RD/vllm_env_extra_deps:$_RD/fake_vllm_metadata:$VERL_ROOT"
else
  export PYTHONPATH="$VLLM_ROOT:/data/huanchen/envshims/pydeps:/data/huanchen/envshims/vllm_metadata:$VERL_ROOT"
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1 TOKENIZERS_PARALLELISM=false
export PATH="/data/huanchen/miniforge3/envs/vllm/bin:$PATH"
