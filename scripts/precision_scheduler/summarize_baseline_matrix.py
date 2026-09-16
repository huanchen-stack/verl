"""Per-token normalization: the cells generated different token volumes, so raw
wall time per step is not comparable across precisions."""
import json, statistics
from pathlib import Path

ROOT = Path("/mnt/home/huanchen/ps_runs/baseline_matrix")
MODELS = ("qwen3_5_9b", "phi4_mini_reasoning")
PRECS = ("bf16", "w4a16", "nvfp4")

def load(model, prec):
    f = ROOT / f"{model}_{prec}" / "metrics" / "baseline_matrix" / f"{model}_{prec}.jsonl"
    recs = [{"step": json.loads(l)["step"], **json.loads(l).get("data", {})} for l in f.open()]
    w = [r for r in recs if r["step"] >= 2]           # step 1 = warm-up
    tot_tok = sum(r["perf/total_num_tokens"] for r in w)
    return {
        "steps": len(w),
        "gen_s": sum(r["timing_s/gen"] for r in w),
        "step_s": sum(r["timing_s/step"] for r in w),
        "tokens": tot_tok,
        "resp": statistics.mean(r["response_length/mean"] for r in w),
        "reward": statistics.mean(r["critic/score/mean"] for r in w),
        "gen_tps": tot_tok / sum(r["timing_s/gen"] for r in w),
        "step_tps": tot_tok / sum(r["timing_s/step"] for r in w),
    }

data = {(m, p): load(m, p) for m in MODELS for p in PRECS}

print(f"{'cell':<30}{'resp_len':>9}{'tokens':>10}{'gen tok/s':>11}{'step tok/s':>12}{'reward':>8}")
for m in MODELS:
    for p in PRECS:
        d = data[(m, p)]
        print(f"{m+'/'+p:<30}{d['resp']:>9.0f}{d['tokens']:>10}{d['gen_tps']:>11.0f}{d['step_tps']:>12.0f}{d['reward']:>8.3f}")
    print()

print("Throughput ratio vs bf16 (higher = quantized is faster per token):")
print(f"{'':<24}{'generation':>12}{'whole step':>12}{'reward delta':>14}")
for m in MODELS:
    b = data[(m, "bf16")]
    for p in ("w4a16", "nvfp4"):
        d = data[(m, p)]
        print(f"  {m+'/'+p:<22}{d['gen_tps']/b['gen_tps']:>12.2f}{d['step_tps']/b['step_tps']:>12.2f}"
              f"{d['reward']-b['reward']:>+14.3f}")
