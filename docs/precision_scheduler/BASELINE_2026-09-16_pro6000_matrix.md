# RL-step baseline: 2 models x 3 precisions, RTX PRO 6000 Blackwell

Date: 2026-09-16. Host: one NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB, sm_120.
Six cells, 5.8 hours end to end, every cell rc 0 with a COMPLETE marker.

Raw per-step metrics archived under `baselines/matrix_2026-09-16/`. Reproduce with
`scripts/precision_scheduler/run_baseline_matrix.sh`; aggregate with
`scripts/precision_scheduler/summarize_baseline_matrix.py`.

## Configuration

Full GRPO steps (rollout, old log-prob, reference, actor update, weight sync) on GSM8K,
32 concurrent requests (train batch 8 x rollout n 4), responses capped at 16384 tokens,
16 steps per cell with step 1 discarded as warm-up and steps 2-16 the measurement window.

**Vanilla punica LoRA in every cell.** `ps_resolve_policy` enables the fused rollout LoRA
fast path and the dual stream for W4 kinds, matching the archived launchers; both are
forced back off here, otherwise the quantized cells would measure a different LoRA kernel
from the BF16 cell and the comparison would be meaningless.

The quantized cells run the dual-precision runtime under a `uniform_w4` policy, so the
whole model decodes off the shadow. Binding was verified per cell: Qwen 152 of 152 shadow
layers active, Phi 128 of 128.

| model | bf16 | w4a16 | nvfp4 |
|---|---|---|---|
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | `Intel/Qwen3.5-9B-int4-AutoRound` | `AxionML/Qwen3.5-9B-NVFP4` |
| Phi-4-mini-reasoning | `microsoft/Phi-4-mini-reasoning` | `alishafique/...gptq-w4a16-llmcompressor` | built here, see `quantize_nvfp4.py` |

## Results

Cells generated different token volumes, because quantization changes the policy and so
its response lengths (Qwen: 4314 tokens mean at BF16 against 5564 at W4A16). Wall time per
step is therefore **not** comparable across precisions; everything below is per token.

| cell | resp len | tokens | gen tok/s | step tok/s | reward |
|---|---|---|---|---|---|
| Qwen3.5-9B bf16 | 4314 | 2103923 | 618 | 414 | 0.819 |
| Qwen3.5-9B w4a16 | 5564 | 2704131 | 1222 | 625 | 0.771 |
| Qwen3.5-9B nvfp4 | 4740 | 2308827 | 959 | 551 | 0.821 |
| Phi-4-mini bf16 | 1301 | 660629 | 887 | 664 | 0.952 |
| Phi-4-mini w4a16 | 1701 | 852760 | 842 | 640 | 0.902 |
| Phi-4-mini nvfp4 | 1903 | 949470 | 988 | 725 | 0.931 |

Ratios against each model's own BF16 cell:

| cell | generation | whole step | reward delta |
|---|---|---|---|
| Qwen3.5-9B w4a16 | 1.98x | 1.51x | -0.048 |
| Qwen3.5-9B nvfp4 | 1.55x | 1.33x | +0.002 |
| Phi-4-mini w4a16 | 0.95x | 0.96x | -0.050 |
| Phi-4-mini nvfp4 | 1.11x | 1.09x | -0.021 |

## Reading

**The standalone decode speedup does not survive contact with the RL step.** The TPOT
heatmap of 2026-09-15 measured 1.79x for W4A16 and 1.74x for NVFP4 on Qwen3.5-9B, with the
two formats within 3% of each other. Here the same model gives 1.98x and 1.55x on
generation alone, and 1.51x and 1.33x once the whole step is counted. Three things separate
the two measurements: the heatmap had no LoRA at all, it ran standalone engines rather than
the dual-precision runtime, and it measured pure steady-state decode rather than a rollout
whose live batch drains.

**On the small model the win disappears entirely.** Phi-4-mini W4A16 is *slower* per token
than its BF16 baseline, 0.95x. Vanilla punica adds BF16 LoRA GEMMs whose cost does not
shrink when the base weights do, and on a 3.8B model that fixed overhead is a far larger
share of the step than on a 9B one. Anyone budgeting a quantized rollout on a small model
should measure rather than assume.

**Training dominates, so rollout gains are capped.** Qwen's generation improves 1.98x but
the whole step only 1.51x, because the old log-prob, reference and actor passes are always
BF16 LoRA work and do not move. That ceiling is the number that matters for RL throughput.

**NVFP4 holds reward better than W4A16 in both models**, +0.002 against -0.048 on Qwen and
-0.021 against -0.050 on Phi. The two W4A16 checkpoints come from different quantizers
(AutoRound and llmcompressor GPTQ), so this is a checkpoint-quality observation as much as
a format one, but the direction is consistent across both models.

**The two formats swap places between models.** W4A16 wins clearly on Qwen (1.98x against
1.55x) while NVFP4 wins on Phi (1.11x against 0.95x). Neither format is uniformly better in
this setting, which the standalone heatmap would not have told you.

## Caveats

- One run per cell, no repeats, so no variance estimate on any of these numbers.
- Response lengths differ by up to 29% between precisions of the same model. Per-token
  normalization handles the timing, but the cells are not solving an identical workload,
  and reward deltas over 15 steps are not a quality evaluation.
- The two W4A16 checkpoints come from different quantizers, and the Phi NVFP4 one was built
  here while the Qwen NVFP4 came from the Hub. Format and checkpoint are entangled.
- 15 measurement steps is short for reward trends; treat the reward column as a guard
  against gross breakage, not as a learning-curve result.
