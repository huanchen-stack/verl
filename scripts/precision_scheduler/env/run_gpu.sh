#!/usr/bin/env bash
# run_gpu.sh -- the decision-13 GPU launcher.
#
# Usage:  run_gpu.sh --gpus 2[,3] [--timeout SECONDS] -- <command...>
#
# Runs <command> in its own process group with CUDA_VISIBLE_DEVICES set to the given GPUs, kills the
# whole group on exit / timeout / Ctrl-C (Ray started under the private RAY_TMPDIR dies with it), and fails
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
  # never ray stop --force here: it kills every Ray process of this user host-wide
  # (the process-group kill above already covers Ray started under the private RAY_TMPDIR)
  rm -rf "$RAY_TMPDIR"
  # "Ours" means: in the session that setsid created for the child (all descendants inherit it),
  # since every agent on this host shares one user account. Give the CUDA context a grace period
  # to be released before declaring a leftover.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 2; ours=""
    for g in ${GPUS//,/ }; do
      for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g"); do
        [ "$(ps -o sid= -p "$p" 2>/dev/null | tr -d ' ')" = "$CHILD" ] && ours="$ours $p"
      done
    done
    [ -z "$ours" ] && break
  done
  for p in $ours; do
    echo "run_gpu.sh: leftover pid $p of our session on GPUs $GPUS, killing" >&2; kill -KILL "$p" 2>/dev/null; LEFT=1
  done
}
trap 'cleanup; exit 130' INT TERM
# The watchdog runs in its own process group so that killing it also kills its `sleep`; an orphaned
# sleep would keep the caller's stdout/stderr pipe open until the full timeout elapsed.
if [ "$TIMEOUT" -gt 0 ]; then
  setsid bash -c 'sleep "$1"; echo "run_gpu.sh: timeout after $1s" >&2; kill -TERM -- -"$2" 2>/dev/null' _ "$TIMEOUT" "$CHILD" </dev/null & WATCH=$!
fi
wait "$CHILD"; RC=$?
[ -n "${WATCH:-}" ] && { kill -- -"$WATCH" 2>/dev/null; kill "$WATCH" 2>/dev/null; }
LEFT=0; cleanup
[ "$LEFT" = 1 ] && { echo "run_gpu.sh: FAILED cleanup contract" >&2; exit 4; }
exit $RC
