# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``python -m verl.experimental.precision_scheduler.cli <command>``.

Commands: ``build-policy`` (one offline build), ``watch-ema`` (online watcher next to a rollout),
``fit-downstream`` (the downstream-time regressions), ``grid`` (heatmap.json / legacy CSV conversions
with the validity guard) and ``build-fixed-frontier`` (heuristic baseline JSON).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import downstream_regression as regression
from .calibration import grouped_traces, paired_traces
from .cost_model import PolicyGrid
from .online_ema import DEFAULT_COHORTS, DEFAULT_TRACE, OnlineEmaWatcher
from .policy_builder import build_policy, fixed_frontier_policy, validate_policy, write_policy_atomic
from .tail_w4_calibration import DEFAULT_QUANTILES
from .tpot_grid import TpotGrid, legacy_grid_rows, matrix_payload, validate_heatmap, write_legacy_grid_csv
from .traces import read_jsonl, read_metrics


def _add_grid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=32, help="initial rollout batch (live-batch axis)")
    parser.add_argument("--cap", type=int, default=16384, help="response cap (tokens)")
    parser.add_argument("--step", type=int, default=250, help="frontier scan interval (tokens)")
    parser.add_argument("--prompt-step", type=int, default=128)
    parser.add_argument("--prompt-max", type=int, default=2048)


def _grid(args: argparse.Namespace) -> PolicyGrid:
    return PolicyGrid(
        step=args.step, cap=args.cap, batch=args.batch, prompt_step=args.prompt_step, prompt_max=args.prompt_max
    )


def _add_calibration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bf-trace", type=Path, required=True, help="pure-BF16 baseline request-lifetime trace")
    parser.add_argument("--w4-trace", type=Path, help="pure-W4 baseline request-lifetime trace (legacy second path)")
    parser.add_argument(
        "--w4-cont-trace",
        type=Path,
        help="tail-W4 continuation trace from calib-tail-w4 (one W4 group per cut); with --w4-trace too, the pure-W4 "
        "baseline becomes the cut-0 group",
    )
    parser.add_argument("--calibration-requests", type=int, default=128)
    parser.add_argument("--heatmap", type=Path, required=True, help="profiler heatmap.json")
    parser.add_argument("--alpha", type=float, default=0.2, help="fixed cohort weight for --update ema")
    parser.add_argument("--update", choices=("weighted", "ema", "clock"), default="weighted", help="online update rule")
    parser.add_argument("--rho", type=float, default=0.9, help="clock: discount of the accumulated events/exposure per rollout")
    parser.add_argument("--prior-weight", type=float, default=32.0, help="weighted: calibration worth this many requests")
    parser.add_argument("--alpha-min", type=float, default=0.05, help="weighted: floor of the cohort weight (drift)")
    parser.add_argument("--max-step-tokens", type=int, default=2000, help="hysteresis: max frontier move per revision (0 = off)")
    parser.add_argument("--bf16-online", action="store_true", help="re-estimate the BF16 line from the run (see online_ema.py caveat)")
    parser.add_argument("--bf16-prior-weight", type=float, default=None, help="BF16 online: calibration worth this many requests (default: --prior-weight)")
    parser.add_argument("--bf16-probe-every", type=int, default=0, help="every K-th rollout never switches (uncensored BF16 tail evidence); 0 = off")
    parser.add_argument("--downstream-slope", type=float, default=0.0, help="downstream seconds per sampled token")
    parser.add_argument(
        "--w4-token-penalty",
        type=float,
        default=0.0,
        help="quality price in seconds per expected INT4-decoded token, charged on the W4 segment (0 = cost only)",
    )
    parser.add_argument("--skip-heatmap-guard", action="store_true", help="accept a heatmap with median speedup ~1.0")


def _calibration(args: argparse.Namespace, grid: PolicyGrid):
    if args.w4_cont_trace is not None:
        return grouped_traces(
            args.bf_trace, args.w4_cont_trace, grid, requests=args.calibration_requests, include_uniform_w4=args.w4_trace
        )
    if args.w4_trace is None:
        raise SystemExit("give --w4-trace (legacy pure-W4 baseline) and/or --w4-cont-trace (calib-tail-w4 output)")
    return paired_traces(args.bf_trace, args.w4_trace, grid, requests=args.calibration_requests)


def _load_heatmap(path: Path, skip_guard: bool) -> TpotGrid:
    payload = json.loads(path.read_text())
    if not skip_guard:
        validate_heatmap(payload)
    return TpotGrid.from_heatmap_json(payload)


def cmd_build_policy(args: argparse.Namespace) -> int:
    grid = _grid(args)
    calibration = _calibration(args, grid)
    policy, switch_states = build_policy(
        calibration.bf16,
        calibration.w4_groups,
        _load_heatmap(args.heatmap, args.skip_heatmap_guard),
        grid,
        args.downstream_slope,
        args.revision,
        args.alpha,
        calibration_kind=calibration.metadata["kind"],
        extra_calibration={"base_source": calibration.metadata},
        w4_token_penalty=args.w4_token_penalty,
    )
    write_policy_atomic(args.output, policy)
    print(
        json.dumps(
            {
                "policy": str(args.output),
                "switch_states": switch_states,
                "states": len(policy["lookup_table"]["committed_frontiers"]),
            }
        )
    )
    return 0


def cmd_watch_ema(args: argparse.Namespace) -> int:
    grid = _grid(args)
    calibration = _calibration(args, grid)
    watcher = OnlineEmaWatcher(
        run_dir=args.run_dir,
        policy_path=args.policy,
        calibration=calibration,
        tpot=_load_heatmap(args.heatmap, args.skip_heatmap_guard),
        grid=grid,
        alpha=args.alpha,
        slope=args.downstream_slope,
        w4_token_penalty=args.w4_token_penalty,
        steps=args.steps,
        trace_name=args.trace_name,
        cohort_name=args.cohort_name,
        gate_cohorts_by_completed_steps=not args.no_cohort_gate,
        update=args.update,
        prior_weight=args.prior_weight,
        alpha_min=args.alpha_min,
        max_step_tokens=args.max_step_tokens,
        bf16_online=args.bf16_online,
        bf16_prior_weight=args.bf16_prior_weight,
        bf16_probe_every=args.bf16_probe_every,
        rho=args.rho,
    )
    watcher.run(initialize_only=args.initialize_only, poll_interval=args.poll_interval)
    return 0


def cmd_calib_tail_w4(args: argparse.Namespace) -> int:
    """Plan (and unless --dry-run, run) the tail-W4 continuation calibration."""
    from . import tail_w4_calibration as tw

    rows = tw.bf16_requests(args.bf_trace, cap=args.cap, limit=args.calibration_requests)
    lengths = [r["length"] for r in rows]
    if args.cut_tokens is not None:
        cuts = tw.cut_frontiers(lengths, tokens=args.cut_tokens, step=args.step, cap=args.cap)
        quantiles = None
    else:
        quantiles = args.cut_quantiles
        cuts = tw.cut_frontiers(lengths, quantiles=quantiles, step=args.step, cap=args.cap)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    lookup = tw.prompt_ids_from_parquet(args.data, tokenizer, chat_template_kwargs=json.loads(args.chat_template_kwargs))
    plan = tw.plan_continuations(rows, cuts, cap=args.cap, prompt_ids_for=lookup, per_cut=args.continuations_per_cut, seed=args.seed)
    payload = tw.manifest(cuts, plan, quantiles=quantiles, cap=args.cap, bf16_requests_count=len(rows))
    payload.update({"bf16_trace": str(args.bf_trace), "output_trace": str(args.output_trace), "model": args.model, "int4_model": args.int4_model})
    tw.write_manifest(args.output_trace.parent / "calibration_manifest.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    if args.dry_run:
        return 0
    engine = tw.VllmEngine(
        model=args.model,
        int4_model=args.int4_model,
        cap=args.cap,
        prompt_cap=args.prompt_max,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        lora_adapter=args.lora_adapter,
        lora_fast_path=not args.no_lora_fast_path,
        lora_dual_stream=not args.no_lora_dual_stream,
    )
    writer = tw.ContinuationTraceWriter(args.output_trace, log_tokens=args.log_tokens)
    try:
        results = tw.run_continuations(engine, plan, writer)
    finally:
        writer.close()
    finals = [r.generation_tokens for r in results]
    print(json.dumps({"continuations": len(results), "mean_continuation_tokens": float(sum(finals) / max(len(finals), 1)), "cap_hits": int(sum(1 for req, r in zip(plan, results) if r.generation_tokens >= req.max_tokens))}), flush=True)
    return 0


def cmd_fit_downstream(args: argparse.Namespace) -> int:
    if args.kind == "metrics":
        rows = read_metrics(args.inputs)
        steps = set(range(args.step_min, args.step_max + 1)) if args.step_min is not None else None
        result = regression.fit_from_metrics(rows, steps=steps)
        slope, n = regression.slope_from_metrics(rows)
        result["inline_slope_seconds_per_token"] = slope
        result["inline_rows"] = n
    elif args.kind == "points":
        if len(args.inputs) != 1:
            raise SystemExit("--kind points expects exactly one CSV")
        result = regression.fit_split(regression.read_points_csv(args.inputs[0]))
    else:
        summary, rows = regression.fit_replay_points(regression.load_replay_runs(args.inputs))
        result = summary
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            import csv

            with args.output.with_suffix(".points.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    if args.kind == "cells":
        rows = read_jsonl(args.inputs[0])
        batch_sizes = sorted({int(r["batch_size"]) for r in rows})
        seq_lens = sorted({int(r["seq_len"]) for r in rows})
        payload = matrix_payload(rows, batch_sizes, seq_lens)
        median = validate_heatmap(payload) if not args.skip_heatmap_guard else None
        if args.output:
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"cells": len(rows), "median_speedup": median, "output": str(args.output)}))
    elif args.kind == "gen1-csv":
        if len(args.inputs) != 2:
            raise SystemExit("--kind gen1-csv expects <bf16.jsonl> <w4.jsonl>")
        rows = legacy_grid_rows(read_jsonl(args.inputs[0]), read_jsonl(args.inputs[1]))
        write_legacy_grid_csv(args.output, rows)
        print(json.dumps({"rows": len(rows), "output": str(args.output)}))
    else:
        grid = TpotGrid.from_legacy_csv(args.inputs[0])
        args.output.write_text(json.dumps(grid.to_heatmap_json(), indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output)}))
    return 0


def cmd_build_fixed_frontier(args: argparse.Namespace) -> int:
    policy = fixed_frontier_policy(
        args.batch, args.cap, args.frontier, step=args.step, capture_max_batch=args.capture_max_batch
    )
    validate_policy(policy)
    write_policy_atomic(args.output, policy)
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m verl.experimental.precision_scheduler.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-policy", help="one offline global-search build from paired baseline traces")
    _add_grid_arguments(p)
    _add_calibration_arguments(p)
    p.add_argument("--revision", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_build_policy)

    p = sub.add_parser("watch-ema", help="online EMA watcher: rebuild the policy after every completed RL step")
    _add_grid_arguments(p)
    _add_calibration_arguments(p)
    p.add_argument("--run-dir", type=Path, required=True, help="rollout run directory (traces/, cohort JSONL)")
    p.add_argument("--policy", type=Path, required=True, help="policy JSON the scheduler reloads")
    p.add_argument("--steps", type=int, default=30, help="stop after this many completed rollouts")
    p.add_argument("--trace-name", default=DEFAULT_TRACE)
    p.add_argument("--cohort-name", default=DEFAULT_COHORTS, help="basename of VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS")
    p.add_argument("--poll-interval", type=float, default=0.25)
    p.add_argument("--initialize-only", action="store_true", help="write revision 0 and exit")
    p.add_argument("--no-cohort-gate", action="store_true", help="ingest cohorts regardless of completed steps")
    p.set_defaults(func=cmd_watch_ema)

    p = sub.add_parser("calib-tail-w4", help="second calibration path: continue BF16 prefixes under W4 from quantile cuts")
    p.add_argument("--bf-trace", type=Path, required=True, help="BF16 calibration trace written with token logging")
    p.add_argument("--output-trace", type=Path, required=True, help="continuation trace to write (manifest goes beside it)")
    p.add_argument("--calibration-requests", type=int, default=256)
    p.add_argument("--cap", type=int, required=True, help="response cap (tokens)")
    p.add_argument("--step", type=int, default=250)
    p.add_argument("--prompt-max", type=int, default=2048)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--cut-quantiles", type=lambda v: [float(x) for x in v.split(",")], default=list(DEFAULT_QUANTILES),
                   help="population quantiles of the BF16 lengths used as cut frontiers (default 0.667,0.75,0.8,0.9)")
    g.add_argument("--cut-tokens", type=lambda v: [int(x) for x in v.split(",")], help="explicit cut frontiers; 0 = uniform W4")
    p.add_argument("--data", type=Path, required=True, help="parquet the BF16 rollout used (prompt messages)")
    p.add_argument("--model", required=True, help="BF16 model path")
    p.add_argument("--int4-model", required=True, help="INT4 shadow checkpoint path")
    p.add_argument("--tokenizer", help="tokenizer path (default: --model)")
    p.add_argument("--chat-template-kwargs", default="{}", help='JSON, e.g. \'{"enable_thinking": true}\' (must match the rollout)')
    p.add_argument("--lora-adapter", help="LoRA adapter directory (the zero adapter the runs start from); omit for base model")
    p.add_argument("--no-lora-fast-path", action="store_true")
    p.add_argument("--no-lora-dual-stream", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--continuations-per-cut", type=int, default=None, help="uniform random sample of the requests alive at each cut (default: all)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--log-tokens", action="store_true", help="record continuation token ids in the trace")
    p.add_argument("--dry-run", action="store_true", help="plan and write the manifest only; no engine")
    p.set_defaults(func=cmd_calib_tail_w4)

    p = sub.add_parser("fit-downstream", help="downstream seconds vs tokens regressions")
    p.add_argument("--kind", choices=("metrics", "points", "replay"), default="metrics")
    p.add_argument(
        "--inputs", type=Path, nargs="+", required=True, help="metrics JSONL / points CSV / replay JSON files"
    )
    p.add_argument("--step-min", type=int)
    p.add_argument("--step-max", type=int)
    p.add_argument("--output", type=Path)
    p.set_defaults(func=cmd_fit_downstream)

    p = sub.add_parser("grid", help="heatmap.json from cells.jsonl, legacy Gen1 CSV, or CSV -> heatmap.json")
    p.add_argument("--kind", choices=("cells", "gen1-csv", "csv-to-heatmap"), default="cells")
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--skip-heatmap-guard", action="store_true")
    p.set_defaults(func=cmd_grid)

    p = sub.add_parser("build-fixed-frontier", help="fixed generation-frontier baseline policy JSON")
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--cap", type=int, required=True)
    p.add_argument("--frontier", type=int, required=True)
    p.add_argument("--step", type=int, default=250)
    p.add_argument("--capture-max-batch", type=int, default=32)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_build_fixed_frontier)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
