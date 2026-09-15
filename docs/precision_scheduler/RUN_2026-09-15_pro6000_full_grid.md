# Qwen3.5-9B decode TPOT, full grid: BF16 vs W4A16 vs NVFP4 (RTX PRO 6000)

Date: 2026-09-15. Host: single NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB,
compute capability 12.0, driver 595.71.05, CUDA 13.2. Supersedes the 8-cell
sampled run of 2026-09-14, which is kept separately and should not be merged
with this one (see Protocol).

Payload, manifest and raw per-step timings archived under
`heatmaps/qwen35_9b_pro6000_3precision_full/`. Reproduce with
`scripts/precision_scheduler/run_tpot_heatmap_3precision.sh`.

## Protocol

The archived axes: batch {1, 2, 4, 8, 16, 32} by context {1024, 2048, 3072,
4096, 5120, 6144, 7168, 8192, 12288, 16384}. 60 cells for each of three
precisions, 180 in all, every one completing. Synthetic KV, decode barrier,
warmup 2, measure 9, repetitions 1, initial precision warmup 9, GPU memory
utilization 0.5, default compilation.

This is a fresh measurement, not an extension of the 8-cell run. Widening the
batch axis raises `max_num_seqs` from 8 to 32, which changes engine
configuration, so the earlier cells are not interchangeable with these. BF16 at
batch 1 and 1k context reads 12.89 ms here against 12.51 ms there, which is the
size of that effect.

As before, every row is a vanilla engine on its own checkpoint
(`--standalone-base-precision`), so the speedups compare numeric formats rather
than runtimes. The archived protocol instead ran INT4 inside the dual-precision
runtime.

## Results

Speedup over BF16, median across the 60 cells:

| precision | median | min (b32, 16k) | max |
|---|---|---|---|
| W4A16 | 1.786 | 1.255 | 2.008 (b2, 1k) |
| NVFP4 | 1.737 | 1.299 | 1.894 (b1, 1k) |

BF16 TPOT spans 12.89 to 25.82 ms/token, W4A16 6.80 to 20.58, NVFP4 6.81 to
19.87.

## Reading

**The two 4-bit formats cross over, and the full grid is what revealed it.**
The sampled run only carried batch 1 and 8 and concluded W4A16 was uniformly
ahead. That conclusion was too narrow:

| batch | contexts where NVFP4 beats W4A16 |
|---|---|
| 1, 2, 4, 8 | 0 of 10 |
| 16 | 6 of 10 |
| 32 | 10 of 10 |

At small batch the step is weight-bandwidth-bound, so what matters is bits per
weight, 4 either way, and NVFP4 loses the remainder to its slightly larger
footprint (9.4 GB against 8.6 GB). As batch grows the GEMMs get wider and
activation traffic and compute start to matter, which is exactly where NVFP4's
quantized activations begin to pay. By batch 32 it wins at every context
measured, by 2 to 6 percent.

**The peak is at batch 2, not batch 1.** W4A16 reaches 2.008 there and 1.894 at
batch 1. A single decode request cannot saturate the memory system, so the BF16
baseline is relatively less penalised at batch 1 than the 4-bit rows are helped.

**The decay along both axes is steep at the corner.** From 2.008 at batch 2 and
1k context down to 1.255 at batch 32 and 16k. Any policy priced off this grid
should interpolate rather than assume a constant multiplier; the archived
Phi-4-mini map has the same shape over a narrower range.

## Caveats

- One repetition per cell, so no within-cell variance. The archived Phi and
  Gemma lanes used five. The crossover at batch 32 is consistent across all ten
  contexts, which is reassuring, but the batch-16 row is genuinely mixed and a
  repeat would be worth it before leaning on that boundary.
- The three checkpoints come from three different quantizers, so some part of
  the W4A16 against NVFP4 difference is the quantizer, not the format.
- Throughput only. No quality measurement anywhere in this run.
