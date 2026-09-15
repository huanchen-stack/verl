# Qwen3.5-9B decode TPOT: BF16 vs W4A16 vs NVFP4 on RTX PRO 6000 Blackwell

Date: 2026-09-14. Host: single NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB, compute
capability 12.0, driver 595.71.05, CUDA 13.2. This is not the eight-GPU A100 host the
archived heatmaps came from, so nothing here is comparable to them cell by cell.

Payload and manifest archived under `heatmaps/qwen35_9b_pro6000_3precision/`.
Reproduce with `scripts/precision_scheduler/run_tpot_heatmap_3precision.sh`.

## Protocol and how it deviates from the archived runs

Eight grid cells, batch {1, 8} by context {1024, 4096, 8192, 16384}, measured for three
precisions. Synthetic KV, decode barrier, warmup 2, measure 9, repetitions 1, initial
precision warmup 9, GPU memory utilization 0.5, no LoRA adapter, default compilation
(the Qwen3.5 lane never set `--full-cudagraph-without-torch-compile`).

One deviation, deliberate. Every row is vanilla vLLM launched directly on that
precision's own checkpoint, via the new `--standalone-base-precision` flag. The archived
protocol instead ran the INT4 row inside the dual-precision runtime against a shadow
checkpoint. The dual-precision loader carries no FP4 shadow format, so a third row could
not have joined it, and mixing one runtime-measured row with two standalone rows would
have made the three numbers incomparable. What is measured here is therefore the decode
cost of each numeric format, not the cost of the scheduler's residency machinery.

Checkpoints:

| row | checkpoint | quantization vLLM selected |
|---|---|---|
| bf16 | `Qwen/Qwen3.5-9B` | none |
| int4 | `sanskar003/Qwen3.5-9B-AWQ` | compressed-tensors W4A16, group 128, Marlin kernel |
| nvfp4 | `AxionML/Qwen3.5-9B-NVFP4` | modelopt_fp4, group 16, activations also FP4 |

## Results

Decode TPOT in milliseconds per token, rows batch, columns context.

| precision | b1 c1k | b1 c4k | b1 c8k | b1 c16k | b8 c1k | b8 c4k | b8 c8k | b8 c16k |
|---|---|---|---|---|---|---|---|---|
| BF16 | 12.51 | 12.59 | 12.65 | 12.83 | 13.20 | 13.69 | 14.33 | 15.66 |
| W4A16 | 6.77 | 6.85 | 6.92 | 7.07 | 7.38 | 7.88 | 8.56 | 9.91 |
| NVFP4 | 6.81 | 6.90 | 6.94 | 7.11 | 7.60 | 8.11 | 8.73 | 10.08 |

Speedup over BF16:

| precision | median | min | max |
|---|---|---|---|
| W4A16 | 1.802 | 1.580 | 1.847 |
| NVFP4 | 1.771 | 1.553 | 1.835 |

Both medians are far from 1.0, so `tpot_grid.validate_heatmap` passes and neither
quantized row silently executed BF16.

## Reading

**Both 4-bit formats land on the same speedup, and W4A16 is marginally ahead.** NVFP4
quantizes activations as well as weights and still does not win. That is what a
weight-bandwidth-bound decode looks like: at these batch sizes the step time is set by
how many bytes of weight cross the memory bus, which is 4 bits per weight either way.
Activation traffic is negligible at decode, so NVFP4's extra quantization buys nothing,
and its slightly larger on-disk footprint (9.4 GB against 8.6 GB) shows up as a slightly
higher TPOT.

**The speedup decays along both axes, from 1.85 at batch 1 and 1k context to 1.58 at
batch 8 and 16k.** Same shape as the archived Phi-4-mini heatmap, same cause: as batch
and context grow, attention and KV traffic take a larger share of the step and the
weight-loading saving is amortized away. This is the gradient the tail-W4 policy prices.

**The absolute gain is much larger here than in the archived runs** (median 1.80 against
1.20 for Phi-4-mini). Do not read that as a Blackwell result alone. The model, the
checkpoint, and above all the protocol differ, since this run measures a standalone W4
engine rather than a W4 row inside the dual-precision runtime. Closing that gap is the
obvious next measurement: run the INT4 row both ways on this host and difference them.

## Caveats

- One repetition per cell, so no within-cell variance estimate. The archived Phi and
  Gemma lanes used five.
- Eight cells is a coarse grid. The decay is monotone in both directions across it, but
  two batch points cannot show curvature in batch.
- The three checkpoints come from three different quantizers, so a small part of any
  difference between W4A16 and NVFP4 may be the quantizer rather than the format.
- No quality measurement here at all. This is throughput only.
