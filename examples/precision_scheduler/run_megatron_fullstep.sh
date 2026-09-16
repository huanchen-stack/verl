#!/usr/bin/env bash
# run_megatron_fullstep.sh -- the Megatron TP1/DP1 GRPO driver (decision 10, reversed 2026-09-16):
# run_fullstep.sh with TRAINER pinned to megatron. This is the reporting trainer.
set -euo pipefail
export TRAINER=megatron
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_fullstep.sh" "$@"
