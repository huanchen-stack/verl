"""Halt rule for an online-EMA run: stop it if, over the first N measured steps, it is slower
than the paired BF16 baseline.

    python ema_halt_monitor.py --run-dir RUN --experiment NAME --project continuous_ema \
        --baseline docs/.../qwen3_5_9b_bf16.jsonl --decide-at 10 --pid <continuous_ema.sh pid>

Both runs use data.shuffle=False and the same train batch, so step k draws the same prompts in
both; the comparison is paired per step. Step 1 is warm-up in both and is excluded. The decision
is taken once, at step --decide-at, on the mean wall time per step over steps 2..decide-at. A
per-token figure is reported alongside for context but does not drive the halt: the question is
whether the scheduled rollout makes the step faster.

Writes RUN/halt_status.json after every step and RUN/HALTED (then kills the process tree) when
the rule fires. Exits 0 when the run finishes or the rule did not fire; 3 when it halted.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def read_metrics(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # partial last line
        rows[int(rec["step"])] = rec.get("data", {})
    return rows


def descendants(pid: int) -> list[int]:
    out = []
    try:
        kids = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    except Exception:
        kids = []
    for k in kids:
        out.extend(descendants(int(k)))
    out.append(pid)
    return out


def kill_tree(pid: int) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        for p in descendants(pid):
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        for _ in range(30):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(1)
    for p in descendants(pid):
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass


def summarize(ema: dict[int, dict], base: dict[int, dict], upto: int) -> dict:
    steps = [s for s in sorted(ema) if 2 <= s <= upto and s in base]
    if not steps:
        return {"steps_compared": 0}
    e_step = [ema[s]["timing_s/step"] for s in steps]
    b_step = [base[s]["timing_s/step"] for s in steps]
    e_tok = [ema[s]["perf/total_num_tokens"] for s in steps]
    b_tok = [base[s]["perf/total_num_tokens"] for s in steps]
    e_gen = [ema[s]["timing_s/gen"] for s in steps]
    b_gen = [base[s]["timing_s/gen"] for s in steps]
    return {
        "steps_compared": len(steps),
        "last_step": steps[-1],
        "ema_mean_step_s": sum(e_step) / len(steps),
        "bf16_mean_step_s": sum(b_step) / len(steps),
        "wall_time_speedup_vs_bf16": (sum(b_step) / len(steps)) / (sum(e_step) / len(steps)),
        "gen_speedup_vs_bf16": (sum(b_gen) / len(steps)) / (sum(e_gen) / len(steps)),
        "per_token_step_speedup_vs_bf16": (sum(e_tok) / sum(e_step)) / (sum(b_tok) / sum(b_step)),
        "ema_mean_tokens": sum(e_tok) / len(steps),
        "bf16_mean_tokens": sum(b_tok) / len(steps),
        "per_step": [
            {"step": s, "ema_s": ema[s]["timing_s/step"], "bf16_s": base[s]["timing_s/step"],
             "ema_tokens": ema[s]["perf/total_num_tokens"], "bf16_tokens": base[s]["perf/total_num_tokens"],
             "ema_reward": ema[s].get("critic/rewards/mean"), "bf16_reward": base[s].get("critic/rewards/mean")}
            for s in steps
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--project", default="continuous_ema")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--baseline", type=Path, required=True, help="bf16 baseline FileLogger jsonl")
    ap.add_argument("--decide-at", type=int, default=10)
    ap.add_argument("--pid", type=int, required=True, help="continuous_ema.sh pid to kill on halt")
    ap.add_argument("--poll", type=float, default=15.0)
    args = ap.parse_args()

    base = read_metrics(args.baseline)
    metrics = args.run_dir / "metrics" / args.project / f"{args.experiment}.jsonl"
    status = args.run_dir / "halt_status.json"
    decided = False
    seen = -1
    while True:
        alive = True
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            alive = False
        ema = read_metrics(metrics)
        last = max(ema) if ema else 0
        if last != seen:
            seen = last
            summ = summarize(ema, base, upto=max(last, 2))
            summ["decided"] = decided
            summ["decide_at"] = args.decide_at
            status.write_text(json.dumps(summ, indent=2) + "\n")
            if summ.get("steps_compared"):
                print(f"[halt-monitor] step {last}: ema {summ['ema_mean_step_s']:.1f}s/step vs bf16 "
                      f"{summ['bf16_mean_step_s']:.1f}s/step -> wall-time {summ['wall_time_speedup_vs_bf16']:.3f}x, "
                      f"per-token {summ['per_token_step_speedup_vs_bf16']:.3f}x", flush=True)
            if not decided and last >= args.decide_at:
                decided = True
                summ = summarize(ema, base, upto=args.decide_at)
                summ["decided"] = True
                summ["decide_at"] = args.decide_at
                slower = summ["wall_time_speedup_vs_bf16"] < 1.0
                summ["halted"] = slower
                status.write_text(json.dumps(summ, indent=2) + "\n")
                if slower:
                    msg = (f"HALT: EMA is slower than BF16 over steps 2..{args.decide_at}: "
                           f"{summ['ema_mean_step_s']:.1f}s vs {summ['bf16_mean_step_s']:.1f}s per step "
                           f"({summ['wall_time_speedup_vs_bf16']:.3f}x)")
                    print(f"[halt-monitor] {msg}", flush=True)
                    (args.run_dir / "HALTED").write_text(msg + "\n")
                    kill_tree(args.pid)
                    return 3
                print(f"[halt-monitor] step {args.decide_at} check passed: {summ['wall_time_speedup_vs_bf16']:.3f}x; letting the run finish", flush=True)
        if not alive:
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
