# SETUP: reproducing the environment on a new machine

Exact commands, in order, for a Linux x86_64 host with NVIDIA A100 (sm80) GPUs, driver >= 580 and
the CUDA 13.0 toolkit at `/usr/local/cuda-13.0` (needed only to *build* flash-attn, causal-conv1d
and mamba-ssm). Placeholders you choose once:

| placeholder | meaning | value on the measurement host |
|---|---|---|
| `<CONDA>` | miniforge/conda root | `/data/huanchen/miniforge3` |
| `<WORK>` | directory holding all checkouts | `/data/huanchen` |
| `<ENVSHIMS>` | shim root outside every repo | `/data/huanchen/envshims` |
| `<HF_HOME>` | Hugging Face cache with the model snapshots | `/data/huggingface` |

`ENVIRONMENT.md` explains every non-obvious step. Steps 1-5 build packages (about 45 minutes of
compile time in total); steps 6-11 are seconds each.

## 1. Conda env with pinned versions

```bash
<CONDA>/bin/conda create -n vllm -c conda-forge python=3.12 uv -y
export PY=<CONDA>/envs/vllm/bin/python

$PY -m pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
$PY -m pip install \
    transformers==5.9.0 ray[default]==2.56.0 flashinfer-python==0.6.11.post2 flashinfer-cubin==0.6.11.post2 \
    peft==0.19.1 compressed-tensors==0.15.0.1 tensordict==0.10.0 accelerate==1.14.0 datasets==5.0.0 \
    hydra-core==1.3.3 omegaconf==2.3.1 wandb==0.28.0 pandas==3.0.3 pyarrow==24.0.0 numpy==2.3.5 \
    codetiming dill pylatexenc pybind11 liger-kernel==0.8.0 latex2sympy2_extended math_verify \
    tensorboard TransferQueue==0.1.8 torchdata pytest==9.1.1 pre-commit einops==0.8.2
$PY -m pip install megatron-core==0.18.0 megatron-bridge==0.5.0 mbridge==0.15.1 nvidia-modelopt==0.44.0
$PY -m pip install transformer-engine[pytorch]==2.13.0
```

Check the CUDA runtime libraries that came with torch; TE needs cublas 13.6.0.2 (re-pinned again
after step 2):

```bash
$PY -m pip list | grep -E 'nvidia-(cublas|cudnn-cu13|nccl-cu13) '
# nvidia-cublas 13.6.0.2, nvidia-cudnn-cu13 9.19.0.56, nvidia-nccl-cu13 2.28.9
```

## 2. flash-attn build, then the cublas re-pin

```bash
export CUDA_HOME=/usr/local/cuda-13.0
MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=8.0 \
$PY -m pip install flash-attn==2.8.3.post1 --no-build-isolation --no-cache-dir     # ~30 min

# flash-attn's resolver downgrades nvidia-cublas to 13.1.0.3, which breaks TE. Undo that:
$PY -m pip install --no-deps --force-reinstall nvidia-cublas==13.6.0.2
```

## 3. TransformerEngine version-gate patch

```bash
SP=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
F=$SP/transformer_engine/pytorch/attention/dot_product_attention/utils.py
grep -n 'max_version = PkgVersion("2.8.3")' $F          # must print exactly one line
sed -i 's/max_version = PkgVersion("2.8.3")/max_version = PkgVersion("2.8.3.post1")/' $F
```

The diff you just applied:

```diff
--- transformer_engine/pytorch/attention/dot_product_attention/utils.py
+++ transformer_engine/pytorch/attention/dot_product_attention/utils.py
@@ class FlashAttentionUtils:
-    max_version = PkgVersion("2.8.3")
+    max_version = PkgVersion("2.8.3.post1")
```

Verify (this needs the `LD_LIBRARY_PATH` rule; on a GPU-less box the import still works):

```bash
LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:$SP/nvidia/cudnn/lib:$SP/nvidia/nccl/lib $PY - <<'PY'
from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils as F
assert (str(F.version), F.is_installed, str(F.max_version), F.v2_7_0_plus) == ("2.8.3.post1", True, "2.8.3.post1", True), (F.version, F.is_installed, F.max_version)
print("TE flash-attn gate OK")
PY
```

## 4. vLLM: clone at the pinned base, editable install with the precompiled cu130 wheel

```bash
cd <WORK>
git clone https://github.com/vllm-project/vllm.git vllm-src
cd vllm-src
git fetch --tags origin
git checkout 6bdabbad5bce747865fd3a249658518a4269cc22

VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=6bdabbad5bce747865fd3a249658518a4269cc22 \
VLLM_MAIN_CUDA_VERSION=13.0 \
$PY -m pip install -e . --no-build-isolation --no-deps
```

This fetches `https://wheels.vllm.ai/6bdabbad5b.../cu130/vllm/` and extracts the compiled
payload into `vllm/` (gitignored). Confirm that torch was not replaced (`$PY -c 'import
torch;print(torch.__version__)'` -> `2.11.0+cu130`) and that the payload is complete:

```bash
git status --porcelain            # empty: everything extracted is gitignored
ls vllm/_C.abi3.so vllm/_moe_C.abi3.so vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so vllm/third_party/deep_gemm
```

vLLM's own `requirements/cuda.txt` pins `flashinfer-python==0.6.11.post2`, `torch==2.11.0` and
`torchvision==0.26.0`, which is why step 1 used those numbers; `--no-deps` keeps pip from touching
them again.

## 5. Extra dependencies: fla, causal-conv1d, mamba-ssm

`fla` is pure Python; the other two are CUDA extensions built against this exact torch. Build
them into a scratch target so they never pollute the env, then place them in `<ENVSHIMS>/pydeps`:

```bash
mkdir -p <ENVSHIMS>/pydeps
$PY -m pip install --no-deps --target <ENVSHIMS>/pydeps flash-linear-attention==0.5.1 fla-core==0.5.1

export CUDA_HOME=/usr/local/cuda-13.0
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=8.0 \
$PY -m pip install --no-deps --no-build-isolation --no-cache-dir --target <ENVSHIMS>/pydeps causal-conv1d==1.6.2.post1
MAMBA_FORCE_BUILD=TRUE MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=8.0 \
$PY -m pip install --no-deps --no-build-isolation --no-cache-dir --target <ENVSHIMS>/pydeps mamba-ssm==2.3.2.post1
```

Expected layout afterwards (the `.so` names carry the `cpython-312-x86_64-linux-gnu` tag):

```
<ENVSHIMS>/pydeps/{fla, fla_core-0.5.1.dist-info, flash_linear_attention-0.5.1.dist-info,
                   causal_conv1d, causal_conv1d-1.6.2.post1.dist-info, causal_conv1d_cuda.*.so,
                   mamba_ssm, mamba_ssm-2.3.2.post1.dist-info, selective_scan_cuda.*.so}
```

If a build pulls `nvidia-cublas` again, repeat the re-pin from step 2.

## 6. verl clone, branch, worktree layout

```bash
cd <WORK>
git clone https://github.com/huanchen-stack/verl.git verl-src          # fork; upstream verl-project/verl
git -C verl-src remote add upstream https://github.com/verl-project/verl.git
git -C verl-src fetch origin rollout-precision-scheduler-clean
git -C verl-src worktree add <WORK>/verl-clean origin/rollout-precision-scheduler-clean
# (the branch is based on verl 2390a3f5; check_env.py verifies that ancestry)

cd <WORK>/vllm-src
git fetch origin rollout-precision-scheduler-clean                 # fork remote if the branch lives there
git worktree add <WORK>/vllm-clean origin/rollout-precision-scheduler-clean
git worktree add --detach <WORK>/vllm-vanilla 6bdabbad5bce747865fd3a249658518a4269cc22
```

Layout after this step:

```
<WORK>/vllm-src        editable install target, base 6bdabbad5b (payload lives here)
<WORK>/vllm-clean      branch rollout-precision-scheduler-clean          <- PS_ENV=clean
<WORK>/vllm-vanilla    detached 6bdabbad5b, never edited                 <- PS_ENV=vanilla
<WORK>/verl-src        fork clone, main
<WORK>/verl-clean      branch rollout-precision-scheduler-clean (verl)   <- VERL_ROOT for clean and vanilla
<ENVSHIMS>/pydeps      step 5
<ENVSHIMS>/vllm_metadata   step 8
```

## 7. Populate the worktrees with the precompiled payload

```bash
ENV=<WORK>/verl-clean/scripts/precision_scheduler/env
bash $ENV/populate_vllm_worktree.sh <WORK>/vllm-src <WORK>/vllm-clean
bash $ENV/populate_vllm_worktree.sh <WORK>/vllm-src <WORK>/vllm-vanilla
git -C <WORK>/vllm-clean status --porcelain      # must stay empty
git -C <WORK>/vllm-vanilla status --porcelain    # must stay empty
```

## 8. Honest version file and metadata shim

```bash
for r in <WORK>/vllm-src <WORK>/vllm-clean <WORK>/vllm-vanilla; do $PY $ENV/check_env.py --write-version-file $r; done
# each prints: wrote .../vllm/_version.py = 0.22.1rc1.dev23+g6bdabbad5b.precompiled (git describe ...)

V=0.22.1rc1.dev23+g6bdabbad5b.precompiled
D=<ENVSHIMS>/vllm_metadata/vllm-$V.dist-info
mkdir -p $D
printf 'Metadata-Version: 2.1\nName: vllm\nVersion: %s\n' "$V" > $D/METADATA
printf 'Wheel-Version: 1.0\nGenerator: precision_scheduler-envshim\nRoot-Is-Purelib: false\nTag: cp312-cp312-linux_x86_64\n' > $D/WHEEL
```

## 9. Host paths in `activate.sh`

`activate.sh` carries the measurement host's paths as defaults. On a new machine edit the four
assignments at the top of the script once (`_SP`, the three `VLLM_ROOT`/`VERL_ROOT` defaults, the
`PYTHON_BIN` line, the `envshims` entries in `PYTHONPATH`) to your `<CONDA>`, `<WORK>` and
`<ENVSHIMS>`. The `dirty` kind only exists on the measurement host; leave it or delete the case.

## 10. Acceptance

```bash
cd /tmp                                     # never run from inside a repo
source $ENV/activate.sh clean
$PYTHON_BIN $ENV/check_env.py --expect clean --import-te --json
# stderr: check_env: OK clean: python=<CONDA>/envs/vllm/bin/python vllm=<WORK>/vllm-clean/vllm/__init__.py (0.22.1rc1.dev23+g6bdabbad5b.precompiled) verl=<WORK>/verl-clean/verl/__init__.py
# exit 0

source $ENV/activate.sh vanilla && $PYTHON_BIN $ENV/check_env.py --expect vanilla     # exit 0
source $ENV/activate.sh clean   && $PYTHON_BIN $ENV/check_env.py --expect vanilla     # exit 1 (wrong tree refused)

export PYTHONDONTWRITEBYTECODE=1
$PYTHON_BIN -m pytest -p no:cacheprovider -q <WORK>/verl-clean/tests/precision_scheduler/test_env_contract.py
```

## 11. Models and the GPU identity smoke test

Download the checkpoints into `<HF_HOME>` (`HF_HOME=<HF_HOME> huggingface-cli download Qwen/Qwen3.5-4B`,
likewise `Qwen/Qwen3.5-9B` and `microsoft/Phi-4-mini-reasoning` for the recipes). Then, with a free
GPU:

```bash
G=$($PYTHON_BIN $ENV/check_env.py --expect clean --pick-gpus 1)
PS_IDENTITY_MODEL=<HF_HOME>/hub/models--Qwen--Qwen3.5-4B/snapshots/<sha> \
$ENV/run_gpu.sh --gpus $G --timeout 3600 -- $PYTHON_BIN -m pytest -p no:cacheprovider -q -m gpu_smoke \
    <WORK>/verl-clean/tests/precision_scheduler/gpu/test_greedy_identity.py
nvidia-smi --query-compute-apps=pid,used_memory --format=csv -i $G     # must be empty afterwards
```

On a host without the `dirty` reference checkout the test still needs all three kinds; edit
`ENVS` in the test or point the `dirty` case of `activate.sh` at a third copy of `vllm-vanilla`.
