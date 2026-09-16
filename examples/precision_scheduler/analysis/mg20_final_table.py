"""Final Megatron (mg20) table: per-arm means over steps WARM+1..20, speedups vs the BF16 Punica baseline, token inflation.
Seeds of the same arm are pooled (same prompts per step across arms; data.shuffle=False)."""
import json, glob, sys, numpy as np
R = "/data/huanchen/ps_runs"; WARM = 5
def rows(d): return [json.loads(l) for l in open(glob.glob(f"{R}/{d}/metrics/*/*.jsonl")[0]) if l.strip()]
def arm(dirs):
    out = []
    for d in dirs:
        if not glob.glob(f"{R}/{d}/COMPLETE"): continue
        for r in rows(d):
            if r["step"] > WARM:
                x = r["data"]; out.append((x["timing_s/gen"], x["timing_s/step"] - x["timing_s/gen"], x["timing_s/step"],
                                           x["response_length/mean"] * 32, x["critic/rewards/mean"], x["response_length/clip_ratio"], x["response_length/max"]))
    a = np.array(out); return a
def switches(d):
    p = f"{R}/{d}/switch_observations.jsonl"
    try: obs = [json.loads(l) for l in open(p)]
    except FileNotFoundError: return ""
    fr = [o["trigger"].get("committed_frontier") or o["trigger"].get("applied_response_tokens") for o in obs if o.get("trigger")]
    live = [o["trigger"].get("applied_live_requests") for o in obs if o.get("trigger")]
    fr = [f for f in fr if f]; live = [l for l in live if l is not None]
    return f"switch {min(fr)}-{max(fr)} (last {fr[-1]}), live {np.mean(live):.0f}" if fr else ""
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3_5_9b"
ARMS = [("BF16, Punica (baseline)", [f"mg20_{MODEL}_bf16_punica", f"mg20_{MODEL}_bf16_punica_seed43"]),
        ("uniform W4, Punica", [f"mg20_{MODEL}_full_w4_punica", f"mg20_{MODEL}_full_w4_punica_seed43"]),
        ("uniform W4, ours", [f"mg20_{MODEL}_full_w4_ours"]),
        ("fixed K=5000, ours", [f"mg20_{MODEL}_fixed_k5000_ours", f"mg20_{MODEL}_fixed_k5000_ours_seed43"]),
        ("live EMA slope 0, ours", [f"mg20_{MODEL}_ema_slope0"]),
        ("live EMA fitted slope, ours", [f"mg20_{MODEL}_ema_slope878", f"mg20_{MODEL}_ema_slope878_seed43", f"mg20_{MODEL}_ema_slope67"]),
        ("live EMA fitted slope + INT4 penalty 2e-4, ours", [f"mg20_{MODEL}_ema_slope878_pen2e4", f"mg20_{MODEL}_ema_slope67_pen2e4"])]
data = {n: arm(ds) for n, ds in ARMS}
b = data[ARMS[0][0]]
if len(b) == 0: sys.exit("no baseline")
bm = b.mean(0)
print(f"| arm ({MODEL}, Megatron TP1, B32, cap 16k, steps {WARM+1}-20) | runs×steps | rollout | downstream | step | token infl. | reward | cap hits/step | switch |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
for n, ds in ARMS:
    a = data[n]
    if len(a) == 0: continue
    m = a.mean(0); se = a.std(0, ddof=1) / np.sqrt(len(a))
    sw = "; ".join(s for s in (switches(d) for d in ds if glob.glob(f"{R}/{d}/COMPLETE")) if s)
    print(f"| {n} | {len(a)//15}×15 | {bm[0]/m[0]:.2f}x ({m[0]:.0f}±{se[0]:.0f} s) | {bm[1]/m[1]:.2f}x ({m[1]:.0f}±{se[1]:.0f} s) | {bm[2]/m[2]:.2f}x ({m[2]:.0f}±{se[2]:.0f} s) | {m[3]/bm[3]*100-100:+.0f}% | {m[4]:.3f}±{se[4]:.3f} | {m[5]*32:.1f} | {sw} |")
