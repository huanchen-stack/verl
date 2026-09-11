#!/usr/bin/env bash
# run_gpu.sh -- the decision-13 GPU launcher.
#
# Usage:  run_gpu.sh --gpus 2[,3] [--timeout SECONDS] -- <command...>
#
# Runs <command> in its own process group with CUDA_VISIBLE_DEVICES set to the given GPUs, kills the
# whole group on exit / timeout / Ctrl-C, stops any Ray it started (private RAY_TMPDIR), and fails
# with rc 4 if a process of ours is still on those GPUs afterwards. Refuses GPU 1 (rc 2) and refuses
# to start on a GPU that already has a compute process or more than 2048 MiB in use (rc 3). Otherwise
# the exit code is the command's. Pick free GPUs first with `check_env.py --pick-gpus N`.
set -uo pipefail
GPUS=""; TIMEOUT=0
while [ $# -gt 0 ]; do case "$1" in
  --gpus) GPUS=$2; shift 2;; --timeout) TIMEOUT=$2; shift 2;; --) shift; break;; *) echo "bad arg $1" >&2; exit 2;; esac; done
[ -n "$GPUS" ] || { echo "run_gpu.sh: --gpus required" >&2; exit 2; }
case ",$GPUS," in *,1,*) echo "run_gpu.sh: GPU 1 is never allowed" >&2; exit 2;; esac
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" | wc -l)
  if [ "$used" -gt 2048 ] || [ "$procs" -gt 0 ]; then echo "run_gpu.sh: GPU $g busy (${used}MiB, $procs procs)" >&2; exit 3; fi
done
export CUDA_VISIBLE_DEVICES=$GPUS
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_$$}; mkdir -p "$RAY_TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
setsid "$@" & CHILD=$!
cleanup() {
  kill -TERM -- -"$CHILD" 2>/dev/null; sleep 3; kill -KILL -- -"$CHILD" 2>/dev/null
  ray stop --force >/dev/null 2>&1 || true
  rm -rf "$RAY_TMPDIR"
  sleep 2
  for g in ${GPUS//,/ }; do
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g"); do
      if [ "$(ps -o user= -p "$p" 2>/dev/null)" = "$(id -un)" ]; then
        echo "run_gpu.sh: leftover pid $p on GPU $g, killing" >&2; kill -KILL "$p" 2>/dev/null; LEFT=1
      fi
    done
  done
}
trap 'cleanup; exit 130' INT TERM
if [ "$TIMEOUT" -gt 0 ]; then ( sleep "$TIMEOUT"; echo "run_gpu.sh: timeout after ${TIMEOUT}s" >&2; kill -TERM -- -"$CHILD" 2>/dev/null ) & WATCH=$!; fi
wait "$CHILD"; RC=$?
[ -n "${WATCH:-}" ] && kill "$WATCH" 2>/dev/null
LEFT=0; cleanup
[ "$LEFT" = 1 ] && { echo "run_gpu.sh: FAILED cleanup contract" >&2; exit 4; }
exit $RC
