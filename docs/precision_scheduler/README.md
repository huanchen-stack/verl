# Rollout precision scheduler: documentation index

Clean re-implementation of the Tail-W4 / precision-scheduling work (dual-precision rollout with
scheduler-side BF16 to INT4 switching for long-tail responses) as tested, default-off branches in
two repositories.

| repo | branch | base | role |
|---|---|---|---|
| vLLM (fork of vllm-project/vllm) | `rollout-precision-scheduler-clean` | vanilla upstream `6bdabbad5b` (2026-05-31), precompiled cu130 kernels, no native code changes | runtime: LoRA fast path, dual-precision residency, precision-keyed CUDA graphs, scheduler-side switching, re-prefill, torch-free policy module |
| verl (fork of verl-project/verl) | `rollout-precision-scheduler-clean` | `2390a3f5` (2026-06-26, the measurement base) | training harness, config-to-env translation, profiling and policy toolkit, recipes, datasets, long-run evaluation |

Every component ships with a CPU unit test, a golden or GPU equivalence test where an oracle
exists, and a design description. vLLM-side designs live under `docs/design/` in the vLLM repo;
verl-side designs live here.

## Start here

- [`PLAN.md`](PLAN.md): the master plan, verbatim. Goal, constraints, ground rules, the thirteen
  numbered decisions with their outcomes, one card per component (C0-C10) and the execution
  schedule. The cards are the specification each component was built against.
- [`../../scripts/precision_scheduler/env/SETUP.md`](../../scripts/precision_scheduler/env/SETUP.md):
  from-scratch environment setup on a new machine, exact commands in order, ending with the
  `check_env.py --expect clean` acceptance step.
- [`../../scripts/precision_scheduler/env/ENVIRONMENT.md`](../../scripts/precision_scheduler/env/ENVIRONMENT.md):
  why the environment is shaped the way it is: one conda env with three vLLM trees selected by
  `PYTHONPATH`, the precompiled payload and the no-native-change rule, the honest version and
  metadata shim, the `LD_LIBRARY_PATH` rule, the TransformerEngine patch with its RECORD-hash
  check, wheel provenance and rebuild triggers.

## Environment tooling (`scripts/precision_scheduler/env/`)

| file | purpose |
|---|---|
| `activate.sh clean\|dirty\|vanilla [VLLM_ROOT]` | select the vLLM/verl trees and pin `LD_LIBRARY_PATH`; source it before any python |
| `check_env.py --expect KIND [--pick-gpus N] [--write-version-file ROOT]` | contract check every launcher runs first; GPU preflight (decision 13); honest `vllm/_version.py` writer |
| `run_gpu.sh --gpus IDS [--timeout S] -- CMD` | the GPU launcher: own process group, private Ray temp dir, kill on exit, fails if a process of ours remains |
| `populate_vllm_worktree.sh SRC DST` | copy the gitignored precompiled payload into a new vLLM worktree |
| `greedy_identity.py --env KIND --out FILE` | greedy token dump used by the vanilla / dirty / clean identity smoke test |

Tests: `tests/precision_scheduler/test_env_contract.py` (CPU) and
`tests/precision_scheduler/gpu/test_greedy_identity.py` (`-m gpu_smoke`).

## Component designs (verl side)

Added by each component as it lands; the vLLM-side counterparts are under `docs/design/` in the
vLLM repository.

| component | document |
|---|---|
| C0 environment, worktrees, honest version | `scripts/precision_scheduler/env/ENVIRONMENT.md` |
| C5 switching policy module (torch-free, vLLM) | vLLM `docs/design/` |
| C6 profiling and policy-building toolkit | `docs/precision_scheduler/` (added by C6) |
| C8 verl telemetry and experiment harness | [`telemetry_and_harness.md`](telemetry_and_harness.md); config-to-env wire format in [`config.md`](config.md); pending vLLM patch in `pending_vllm_patches/` |
| C10 datasets, recipes, long-run evaluation | `docs/precision_scheduler/` (added by C10) |
