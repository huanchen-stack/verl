#!/usr/bin/env bash
# run_fsdp_fullstep.sh -- run_fullstep.sh with TRAINER pinned to fsdp2. CPU compose tests only;
# never for a GPU run (decision 10, reversed 2026-09-16: Megatron TP1 is the reporting trainer).
set -euo pipefail
export TRAINER=fsdp2
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_fullstep.sh" "$@"
