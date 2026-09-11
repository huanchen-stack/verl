# Reuse vs re-prefill: teacher-forced NLL study

Component C7 of the rollout precision scheduler. The runtime feature it evaluates is
`VLLM_DUAL_PRECISION_REPREFILL` in the vLLM branch (`docs/design/dual_precision_residency.md`,
section "Re-prefill after the switch"): at the BF16 to INT4 switch, preempt every surviving
request so its KV is recomputed under the INT4 base instead of continuing from the BF16 KV.

## Result: negative. Reusing the BF16 state is closer to BF16 than re-prefilling under INT4.

Archived run `qwen35_9b_w4qdq_longtail16` (Qwen3.5-9B, 16 long-tail traces, switch at output
token 4096, five 64-token windows per trace, fake INT4). `dNLL = NLL(reuse) - NLL(re-prefill)`
in nats per token; negative means reuse is better. CI95 is a 10000-resample bootstrap over the
16 traces (seed 20260812).

| window offset after switch | dNLL mean | dNLL CI95 | fraction of traces with dNLL > 0 | KL(BF16 ‖ reuse) | KL(BF16 ‖ re-prefill) | top-1 agreement with BF16, reuse | re-prefill |
|---|---|---|---|---|---|---|---|
| 0 | -0.0643 | [-0.0848, -0.0443] | 0.000 | 0.0405 | 0.0969 | 0.947 | 0.917 |
| 512 | -0.0304 | [-0.0470, -0.0162] | 0.125 | 0.0682 | 0.0986 | 0.937 | 0.923 |
| 1024 | -0.0322 | [-0.0559, -0.0104] | 0.250 | 0.0779 | 0.1061 | 0.913 | 0.902 |
| 2048 | -0.0451 | [-0.0615, -0.0286] | 0.125 | 0.0864 | 0.1200 | 0.910 | 0.903 |
| 4096 | -0.0424 | [-0.0581, -0.0281] | 0.000 | 0.1023 | 0.1362 | 0.903 | 0.880 |

At every offset the CI95 of dNLL is entirely negative and the reuse branch has the smaller KL to
the BF16 distribution and the higher top-1 agreement. The study's gate was therefore
`STOP: re-prefill did not improve NLL; verifier experiment not run`, and the runtime ships the
feature default-off as an ablation only.

What the study is and is not:

- It is a mechanism diagnostic with HF transformers and *fake* INT4 (symmetric round-to-nearest,
  per-row group-128 quantize-dequantize of the 248 language-model `nn.Linear` modules,
  6,918,504,448 weights, relative weight RMSE 0.123). It is not the GPTQ/Marlin production path;
  a paired reuse-vs-re-prefill experiment on the production kernels does not exist
  (`evidence_bundle/manifest.md`, "MISSING" list).
- Both INT4 branches use the identical W4 model and are teacher-forced on the identical canonical
  BF16 response tokens; only the state at the switch differs (BF16-produced vs W4-rebuilt).
  `identity.json` (`max_abs_logit_error 0.0`) checks that copying a cache is exact.
- The archived threshold studies that motivated it ran with re-prefill ON under the experimental
  scheduler: all 72 completed runs of the 120-run `temporal_guard_120` study
  (`evidence_bundle/threshold_runs.csv`, `reprefill=True`), the `tail-k*-reprefill` natural-behavior
  policies (`evidence_bundle/raw/natural_rollout_behavior.csv`) and the 27B `mixed_precision_reprefill`
  rollouts. Their rollout/reward numbers were measured with the survivors' KV recomputed under
  INT4; the later `best_t8_no_reprefill_gpu7` and `no_reprefill_rl_100step` studies and every
  headline dynamic-policy run ran with re-prefill OFF.

## Provenance

| item | path |
|---|---|
| run outputs (window metrics, summary, manifest, identity, quantization) | `/data/huanchen/verl/.codex-report/reprefill-study/runs/qwen35_9b_w4qdq_longtail16/` (the 5 GB of `bf16_intermediates/` stays archived) |
| committed copy of the small outputs (golden fixture, byte-identical) | `tests/precision_scheduler/golden/reprefill_nll_study/` |
| curated paired rows (80 = 16 traces x 5 offsets) | `/data/huanchen/verl/evidence_bundle/reprefill_paired_results.csv` |
| original scripts | `/data/huanchen/verl/.codex-report/reprefill-study/{run_experiment.py, run_experiment_single_gpu.py, analyze_results.py}` |
| report | `/data/huanchen/verl/.codex-report/reprefill-study/report.pdf` |
| BF16 traces (vLLM request-lifetime harness dumps, sha256 in `manifest.json`) | `/data/huanchen/vllm/.codex-reports/rollout/request-lifetime/results/{eurus,bigmath,gsm8k}/results-tp1/*-bf16-*textdump/responses.jsonl` |
| model | `/data/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

Archived command (from `manifest.json`; `CUDA_VISIBLE_DEVICES=2`, 2026-08-12):

```bash
python run_experiment_single_gpu.py --output runs/qwen35_9b_w4qdq_longtail16 \
    --limit 16 --switch-output-token 4096 --offsets 0 512 1024 2048 4096 --window 64 --chunk 256
```

## Files here

| file | purpose |
|---|---|
| `reprefill_teacher_forced.py` | the single-GPU scorer that produced the archived run, de-hard-coded: `--model`, `--inputs DATASET=PATH ...` and `--output` are required; trace selection by explicit id list |
| `archived_trace_ids.json` | the 16 `(dataset, request_id)` pairs of the archived run in archived `trace_index` order (10 EURUS, 4 BigMath, 2 GSM8K) |
| `analyze_reprefill_results.py` | recomputes the bootstrap summary from `window_metrics.jsonl`, decides the gate, draws `main_result.{pdf,png}` and `per_trace.{pdf,png}`; numpy-only without plots |

Trace selection changed on purpose. The original `load_traces` sorted candidates by
`Path(source).parts[-5]` (a fixed absolute-path depth, which happened to be the literal
`results` for all three dumps) and truncated to `--limit`, so the 16 scored traces were the 16
smallest `request_id`s over the union of the three dumps. That is reproduced here by the explicit
list; any other `--inputs` layout would have selected different traces silently.

## Running

Analyze the archived run (CPU, seconds; reproduces `summary.json` exactly, see
`tests/precision_scheduler/test_reprefill_nll_study_golden.py`):

```bash
python examples/precision_scheduler/analysis/reprefill_nll_study/analyze_reprefill_results.py \
    --run-dir /data/huanchen/verl/.codex-report/reprefill-study/runs/qwen35_9b_w4qdq_longtail16 \
    --out-dir /tmp/reprefill_measured            # analysis.json, summary.recomputed.json, figures
```

Rerun the scorer (one GPU, about 4 minutes of scoring after the 9B load; not deterministic to the
bit across GPUs and kernels: BF16 forwards over 4k-token contexts, so compare per-window metrics
with a tolerance, not byte-equal):

```bash
R=/data/huanchen/vllm/.codex-reports/rollout/request-lifetime/results
scripts/precision_scheduler/env/run_gpu.sh --gpus 2 -- python \
    examples/precision_scheduler/analysis/reprefill_nll_study/reprefill_teacher_forced.py \
    --model /data/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    --inputs eurus=$R/eurus/results-tp1/qwen35-9b-eurus-n128-cap32k-tp1-bf16-stoppinghint-textdump/responses.jsonl \
             bigmath=$R/bigmath/results-tp1/qwen35-9b-bigmath-n128-cap32k-tp1-bf16-textdump/responses.jsonl \
             gsm8k=$R/gsm8k/results-tp1/qwen35-9b-gsm8k-n128-cap32k-tp1-bf16-textdump/responses.jsonl \
    --output /data/huanchen/runs/reprefill_rerun
```

`--eos-token-id` defaults to 248046 (`<|im_end|>` of the Qwen3.5 tokenizer, the id the archived
run scored); set it for another model. The per-window `*_eos_logprob` columns are informational
and not part of the summary.
