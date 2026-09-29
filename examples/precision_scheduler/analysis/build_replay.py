#!/usr/bin/env python
"""build_replay.py -- extract recorded responses from a request lifetime trace into a replay file.

    python build_replay.py --trace <run>/traces/request_lifetimes_replica000_node000.jsonl --out replay.json
        [--steps 1-24 --requests-per-step 32]

The trace is the rollout.precision_scheduler.request_trace_dir file of a run with
trainer.stable_sample_uid=true and request_trace_log_tokens=true: every ``finish`` event carries the
response ``token_ids`` and the stable ``trace_request_id`` ``idx-<dataset index>_<n>``. The output
maps that id to the token ids and is read by ReplayAgentLoop (verl.experimental.agent_loop.replay_agent_loop).
A Tail-W4 run writes one finish event per precision segment; only single-segment (BF16 / uniform)
traces are accepted, so the replayed sequence is the sequence the recorded run trained on.
"""
import argparse
import json
import sys
from collections import Counter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-index", type=int, default=0, help="keep dataset indices >= this")
    ap.add_argument("--max-index", type=int, default=None, help="keep dataset indices <= this")
    args = ap.parse_args()

    table: dict[str, list[int]] = {}
    seen: Counter = Counter()
    with open(args.trace) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") != "finish" or "trace_request_id" not in rec or "token_ids" not in rec:
                continue
            key = rec["trace_request_id"]
            seen[key] += 1
            idx = int(key.split("_")[0].split("-")[1])
            if idx < args.min_index or (args.max_index is not None and idx > args.max_index):
                continue
            table[key] = rec["token_ids"]
    dup = [k for k, v in seen.items() if v > 1]
    if dup:
        print(f"refusing: {len(dup)} ids have several finish events (multi-segment trace), e.g. {dup[:3]}", file=sys.stderr)
        return 2
    if not table:
        print("no finish events with token ids in the trace", file=sys.stderr)
        return 2
    with open(args.out, "w") as f:
        json.dump(table, f)
    lens = [len(v) for v in table.values()]
    idxs = sorted({int(k.split("_")[0].split("-")[1]) for k in table})
    print(f"{len(table)} responses, dataset index {idxs[0]}..{idxs[-1]}, response tokens mean {sum(lens)/len(lens):.0f} max {max(lens)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
